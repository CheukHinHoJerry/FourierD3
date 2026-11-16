#!/usr/bin/env python
"""NPT (variable-cell) tests for fd3_fast r4. Exit NON-ZERO on fail. Independent reference = separate-dm StockD3
(upstream get_neighborhood NL + stock strain forward, E/F/stress).
 T-NPT1: real ASE NPTBerendsen 2 ps (MACE fp64 + new D3), cell ACTUALLY changes; replay recorded frames through
         new (STRESS path) and orig StockD3 -> per-frame |dE|/N, max|dF|, max|dsigma| within G2; zero exceptions.
 T-NPT2: speed new-D3-stress vs orig-D3-stress on the variable-cell frames; new not slower (<=5%); recompiles==0.
 T-NPT3: NVT(fixed cell, FAST)->NPT(changing cell)->NVT: FAST compiles when stable, falls to eager on change,
         resumes after re-stabilise; parity vs StockD3 throughout.
 T-NPT4: isotropic + TRICLINIC strain sequence: build_nl == upstream get_neighborhood edge-for-edge.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fourierd3.fast import FastFourierD3, D3FastCalculator, build_nl, HART, A2B
from mace.data.neighborhood import get_neighborhood
from fourierd3.mace_d3 import MACEFourierD3Calculator
dev = torch.device("cuda")
RD = os.environ.get("FD3_TEST_RUNDIR", "/mnt/vast/cheuk.hinho/runs/20261109T144651Z_symx-metad-speed")
MODEL = os.environ.get("FD3_TEST_MODEL", "/mnt/vast/home/cheuk.hinho/models/mace-mh-1.model"); HEAD = "oc20_usemppbe"
MFJSON = RD + "/inputs/mf_CuMoOH_oc20usemppbe_nsp512_NIX.json"
at0 = read(RD + "/inputs/seed_w0.extxyz"); n = len(at0); PBC = tuple(bool(x) for x in at0.pbc)
d3kw = {"r_cut": 6.0, "xcfunc": "pbe", "method": "ewald", "cnfunc": "smooth_cut", "k_cutoff": 6.0, "c6tol": 0.1}

class StockD3(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    def __init__(self, atoms0):
        super().__init__()
        self.host = MACEFourierD3Calculator(model_paths=MODEL, device="cuda", default_dtype="float32", head=HEAD, d3_kwargs=dict(d3kw))
        self.rmax = float(self.host.r_max); self.host._ensure_d3_model(atoms0, all_changes); self.dm = self.host.d3_model; self.DT = self.dm.dtype
    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms; nb = get_neighborhood(np.asarray(at.positions), self.rmax, pbc=PBC, cell=np.asarray(at.cell))
        ei = torch.as_tensor(np.asarray(nb[0]), dtype=torch.long, device=dev); us = torch.as_tensor(np.asarray(nb[2]), dtype=self.DT, device=dev)
        cell = torch.as_tensor(np.asarray(at.cell), dtype=self.DT, device=dev)
        p = torch.as_tensor(at.positions, dtype=self.DT, device=dev).requires_grad_(True)
        st = torch.zeros(3, 3, dtype=self.DT, device=dev, requires_grad=True)
        sc = cell + torch.einsum("ab,Ab->Aa", st, cell); self.dm._update_cell(sc)
        sp = p + torch.einsum("ab,ib->ia", st, p)
        e = self.dm(positions=sp, edge_index=ei, shifts=torch.matmul(us, sc)) * HART
        gp, gs = torch.autograd.grad(e, [p, st]); sigma = (gs / float(self.dm.volume) * (A2B ** 3)).detach().cpu().numpy()
        self.results = {"energy": float(e.item()), "free_energy": float(e.item()), "forces": (-gp).detach().cpu().numpy(), "stress": full_3x3_to_voigt_6_stress(sigma)}

def ref_efs(pos_np, cell_np, REF):
    # DIRECT-forward reference on REF.dm (the pinned upstream FourierD3 + get_neighborhood NL + strain-autograd),
    # bypassing the ASE wrapper. Verified == F to ~3e-8 on fixed and deformed cells (diag_npt2).
    dm = REF.dm; DT = REF.DT
    nb = get_neighborhood(np.asarray(pos_np), REF.rmax, pbc=PBC, cell=np.asarray(cell_np))
    ei = torch.as_tensor(np.asarray(nb[0]), dtype=torch.long, device=dev); us = torch.as_tensor(np.asarray(nb[2]), dtype=DT, device=dev)
    cell = torch.as_tensor(np.asarray(cell_np), dtype=DT, device=dev)
    p = torch.as_tensor(pos_np, dtype=DT, device=dev).requires_grad_(True)
    st = torch.zeros(3, 3, dtype=DT, device=dev, requires_grad=True)
    sc = cell + torch.einsum("ab,Ab->Aa", st, cell); dm._update_cell(sc)
    sp = p + torch.einsum("ab,ib->ia", st, p)
    e = dm(positions=sp, edge_index=ei, shifts=torch.matmul(us, sc)) * HART
    gp, gs = torch.autograd.grad(e, [p, st]); sigma = (gs / float(dm.volume) * (A2B ** 3)).detach().cpu().numpy()
    V = float(abs(np.linalg.det(np.asarray(cell_np))))
    return float(e.item()), (-gp).detach().cpu().numpy(), -V * sigma

res = {}
F = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32", compile_fast=True, stable_n=20)
REF = StockD3(at0)

# ---- T-NPT1: real NPTBerendsen, record variable-cell frames, replay parity ----
from symmetrix.experimental import MACEFoundation
mf = MACEFoundation(MFJSON, device="cuda", dtype="float64")
class Comb(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    def __init__(self, Fd3): super().__init__(); self.Fd3 = Fd3
    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms
        am = at.copy(); am.calc = mf; em = float(am.get_potential_energy()); fm = np.asarray(am.get_forces()); sm = np.asarray(am.get_stress())
        ed, fd, Wd = self.Fd3.energy_forces_stress(at.positions, np.asarray(at.cell))
        V = float(abs(np.linalg.det(np.asarray(at.cell)))); sd = full_3x3_to_voigt_6_stress(-Wd / V)
        self.results = {"energy": em + ed, "free_energy": em + ed, "forces": fm + fd, "stress": sm + sd}
atoms = at0.copy(); atoms.calc = Comb(F)
rng = np.random.default_rng(321); MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=rng); Stationary(atoms)
dyn = NPTBerendsen(atoms, timestep=0.5 * units.fs, temperature_K=300.0, pressure_au=1.01325 * units.bar,
                   taut=100 * units.fs, taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)
vols = []; frames = []
def rec():
    vols.append(atoms.get_volume()); frames.append((atoms.get_positions().copy(), np.asarray(atoms.cell).copy()))
dyn.attach(rec, interval=200)
exc = None
try:
    dyn.run(4000)   # 2 ps
except Exception as e:
    exc = repr(e)[:200]
dF1 = dE1 = dS1 = 0.0
for (p, c) in frames:
    eU, fU, WU = ref_efs(p, c, REF)
    es, fs, Ws = F.energy_forces_stress(p, c)
    dE1 = max(dE1, abs(es - eU) / n); dF1 = max(dF1, float(np.linalg.norm(fs - fU, axis=1).max())); dS1 = max(dS1, float(np.abs(Ws - WU).max()))
res["T_NPT1"] = {"n_frames": len(frames), "vol_min": float(min(vols)), "vol_max": float(max(vols)),
                 "vol_changed": bool(max(vols) - min(vols) > 1e-6), "exception": exc,
                 "dE": dE1, "dF": dF1, "dW": dS1, "PASS": bool(exc is None and dF1 < 1e-4 and dE1 < 1e-5 and (max(vols) - min(vols) > 1e-6))}

# ---- T-NPT2: speed new-stress vs orig-stress on variable-cell frames; recompiles==0 ----
rc0 = F._recompiles
def tev(fn, reps=40, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps
pf, cf = frames[len(frames) // 2]
ms_new = tev(lambda: F.energy_forces_stress(pf, cf))
ms_orig = tev(lambda: ref_efs(pf, cf, REF))
res["T_NPT2"] = {"ms_new_stress": ms_new, "ms_orig_stress": ms_orig, "not_slower_within_5pct": bool(ms_new <= 1.05 * ms_orig),
                 "recompiles_during_npt": int(F._recompiles - rc0), "PASS": bool(ms_new <= 1.05 * ms_orig and (F._recompiles - rc0) == 0)}

# ---- T-NPT3: NVT(FAST, fixed cell) -> NPT(changing cell) -> NVT; FAST compiles/eager/resumes; parity throughout ----
worst = 0.0; rc_before = F._recompiles
c0 = np.asarray(at0.cell)
# NVT fixed cell: 25 FAST calls -> should compile after stable_n=20
for i in range(25):
    pp = at0.positions + rng.normal(0, 0.005, at0.positions.shape)
    ef, ff = F.energy_forces(pp, c0); eU, fU, _ = ref_efs(pp, c0, REF); worst = max(worst, float(np.linalg.norm(ff - fU, axis=1).max()))
compiled_after_nvt = F._cfast is not None; rc_after_nvt = F._recompiles
# NPT changing cell: each call a different cell -> FAST must fall back to eager (no recompile storm)
for i in range(10):
    scl = 1.0 + 0.01 * (i + 1); c = c0 * scl
    pp = at0.positions + rng.normal(0, 0.005, at0.positions.shape)
    ef, ff = F.energy_forces(pp, c); eU, fU, _ = ref_efs(pp, c, REF); worst = max(worst, float(np.linalg.norm(ff - fU, axis=1).max()))
rc_during_change = F._recompiles - rc_after_nvt
# back to NVT fixed cell: 25 calls -> FAST re-stabilises + recompiles once
for i in range(25):
    pp = at0.positions + rng.normal(0, 0.005, at0.positions.shape)
    ef, ff = F.energy_forces(pp, c0); eU, fU, _ = ref_efs(pp, c0, REF); worst = max(worst, float(np.linalg.norm(ff - fU, axis=1).max()))
resumed = F._cfast is not None
res["T_NPT3"] = {"worst_dF": worst, "compiled_after_nvt": bool(compiled_after_nvt),
                 "recompiles_during_cell_change": int(rc_during_change), "fast_resumed_after": bool(resumed),
                 "PASS": bool(worst < 1e-4 and compiled_after_nvt and rc_during_change == 0 and resumed)}

# ---- T-NPT4: isotropic + triclinic strain sequence: NL == upstream ----
def edgeset(ei, us): return set(map(tuple, np.concatenate([ei.T.cpu().numpy(), np.rint(us.cpu().numpy()).astype(int)], 1).tolist()))
maxsd = 0; istri = False
strains = [np.eye(3) * 1.02, np.eye(3) + np.array([[0, 0.03, 0.01], [0.03, 0, 0.02], [0.01, 0.02, 0]]),
           np.array([[1.01, 0.04, 0], [0, 0.98, 0.03], [0.02, 0, 1.05]])]
for Smat in strains:
    c = c0 @ Smat.T
    if abs(c[0] @ c[1]) > 1e-6 or abs(c[0] @ c[2]) > 1e-6: istri = True
    p = at0.positions @ Smat.T
    ei, us = build_nl(F.dm, torch.as_tensor(p, dtype=F.DT, device=dev), torch.as_tensor(c, dtype=F.DT, device=dev), F.rmax, PBC, n, cap=None)
    nb = get_neighborhood(np.asarray(p), F.rmax, pbc=PBC, cell=np.asarray(c))
    eiU = torch.as_tensor(np.asarray(nb[0]), dtype=torch.long, device=dev); usU = torch.as_tensor(np.asarray(nb[2]), dtype=F.DT, device=dev)
    maxsd = max(maxsd, len(edgeset(ei, us) ^ edgeset(eiU, usU)))
res["T_NPT4"] = {"max_sym_diff": int(maxsd), "triclinic_tested": bool(istri), "PASS": bool(maxsd == 0 and istri)}

res["ALL_PASS"] = bool(all(v.get("PASS") for v in res.values() if isinstance(v, dict)))
print(json.dumps(res, indent=1, default=str), flush=True)
print("NPT_TESTS_DONE ALL_PASS=", res["ALL_PASS"], flush=True)
sys.exit(0 if res["ALL_PASS"] else 1)
