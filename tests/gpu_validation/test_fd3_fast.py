#!/usr/bin/env python
"""Tests for fd3_fast r3 — addresses Astra r2. Exit NON-ZERO on any fail.
ISOLATED reference (B2): a SEPARATE MACEFourierD3Calculator instance (its OWN d3 model, no shared tensors)
wrapped as a D3-only ASE calculator using the pinned upstream NL (mace.data.neighborhood.get_neighborhood)
and the stock strain-autograd forward; compared through PUBLIC ASE get_potential_energy/get_forces/get_stress.
Finiteness (M6) asserted on every compared quantity. 20 unwrapped manifest frames + triclinic-translation.
Gates: T0 NL edge-set; T1 FAST E/F vs ref; T2 STRESS E/F/stress(ASE)+virial vs ref; T3 FAST==STRESS;
T4 CAP overflow (eager+compiled) vs ref; T5 tiny+large cell change; T6 topology (species rebuild, count raise,
PBC raise, raw-API invalid guard); T7 compiled lifecycle (CAP overflow/cell/mesh/species/FAST->STRESS->FAST);
T8 padded==unpadded E/F finite.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from fourierd3.fast import FastFourierD3, D3FastCalculator, build_nl, HART, A2B

RD = os.environ.get("FD3_TEST_RUNDIR", "/mnt/vast/cheuk.hinho/runs/20261109T144651Z_symx-metad-speed")
MODEL = os.environ.get("FD3_TEST_MODEL", "/mnt/vast/home/cheuk.hinho/models/mace-mh-1.model"); HEAD = "oc20_usemppbe"
dev = torch.device("cuda")
man = json.load(open(os.environ.get("FD3_TEST_MANIFEST", "/mnt/vast/cheuk.hinho/runs/20261115T232018Z_fd3-cuda/results/realized_manifest.json")))
allf = read(RD + "/frames_parity/metad_mo_parity_frames_300.extxyz", index=":")
G1 = man["g1_frame_indices"]; frames = [allf[i] for i in G1]; at0 = frames[0]; n = len(at0)
from mace.data.neighborhood import get_neighborhood
from fourierd3.mace_d3 import MACEFourierD3Calculator
import fourierd3 as _fd3

def fin(*arrs):
    for a in arrs:
        if not np.all(np.isfinite(np.asarray(a))):
            raise AssertionError("non-finite value in a compared quantity")

def efs(calc, at):
    """Invoke a calculator's COMPUTE path freshly (bypass ASE result caching/staleness)."""
    calc.reset()
    calc.calculate(at.copy(), ["energy", "forces", "stress"], all_changes)
    r = calc.results
    return float(r["energy"]), np.asarray(r["forces"]), (np.asarray(r["stress"]) if "stress" in r else None)

class StockD3(Calculator):
    """Independent pinned-upstream D3-only ASE calculator: its OWN MACEFourierD3Calculator/d3 model,
    the upstream get_neighborhood NL, and the stock strain-autograd forward. No shared state with candidate."""
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    def __init__(self, d3_kwargs, atoms0, dtype="float32"):
        super().__init__()
        kw = dict(d3_kwargs)
        if str(dtype) in ("float64", "torch.float64"): kw["dtype"] = torch.float64
        self.host = MACEFourierD3Calculator(model_paths=MODEL, device="cuda", default_dtype=dtype, head=HEAD, d3_kwargs=kw)
        self.rmax = float(self.host.r_max); self.host._ensure_d3_model(atoms0, all_changes)
        self.dm = self.host.d3_model; self.DT = self.dm.dtype; self.pbc = tuple(bool(x) for x in atoms0.pbc)
    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms; nb = get_neighborhood(np.asarray(at.positions), self.rmax, pbc=self.pbc, cell=np.asarray(at.cell))
        ei = torch.as_tensor(np.asarray(nb[0]), dtype=torch.long, device=dev); us = torch.as_tensor(np.asarray(nb[2]), dtype=self.DT, device=dev)
        cell = torch.as_tensor(np.asarray(at.cell), dtype=self.DT, device=dev)
        p = torch.as_tensor(at.positions, dtype=self.DT, device=dev).requires_grad_(True)
        st = torch.zeros(3, 3, dtype=self.DT, device=dev, requires_grad=True)
        sc = cell + torch.einsum("ab,Ab->Aa", st, cell); self.dm._update_cell(sc)
        sp = p + torch.einsum("ab,ib->ia", st, p)
        e = self.dm(positions=sp, edge_index=ei, shifts=torch.matmul(us, sc)) * HART
        gp, gs = torch.autograd.grad(e, [p, st])
        sigma = (gs / float(self.dm.volume) * (A2B ** 3)).detach().cpu().numpy()
        self.results = {"energy": float(e.item()), "free_energy": float(e.item()),
                        "forces": (-gp).detach().cpu().numpy(), "stress": full_3x3_to_voigt_6_stress(sigma)}

d3kw = {"r_cut": 6.0, "xcfunc": "pbe", "method": "ewald", "cnfunc": "smooth_cut", "k_cutoff": 6.0, "c6tol": 0.1}
F = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
REF = StockD3(d3kw, at0, dtype="float32")   # fully separate instance
RMAX = F.rmax; PBC = F.pbc; DT = F.DT
res = {}; print("fourierd3 version:", getattr(_fd3, "__version__", "?"), flush=True)

def edgeset(ei, us):
    return set(map(tuple, np.concatenate([ei.T.cpu().numpy(), np.rint(us.cpu().numpy()).astype(int)], 1).tolist()))

# T0 NL vs upstream get_neighborhood, ALL 20 frames + triclinic 2-cell translation
t0 = {"max_sym_diff": 0}
for fr in frames:
    cell = torch.as_tensor(np.asarray(fr.cell), dtype=DT, device=dev); pos = torch.as_tensor(fr.positions, dtype=DT, device=dev)
    ei, us = build_nl(F.dm, pos, cell, RMAX, PBC, n, cap=None)
    nb = get_neighborhood(np.asarray(fr.positions), RMAX, pbc=PBC, cell=np.asarray(fr.cell))
    t0["max_sym_diff"] = max(t0["max_sym_diff"], len(edgeset(ei, us) ^ edgeset(torch.as_tensor(np.asarray(nb[0]), dtype=torch.long, device=dev), torch.as_tensor(np.asarray(nb[2]), dtype=DT, device=dev))))
frt = at0.copy(); p = frt.get_positions(); p[3] += 2 * np.asarray(at0.cell)[0] - 3 * np.asarray(at0.cell)[1]; frt.set_positions(p)  # triclinic multi-cell
cellt = torch.as_tensor(np.asarray(frt.cell), dtype=DT, device=dev); post = torch.as_tensor(frt.positions, dtype=DT, device=dev)
eit, ust = build_nl(F.dm, post, cellt, RMAX, PBC, n, cap=None); nbt = get_neighborhood(np.asarray(frt.positions), RMAX, pbc=PBC, cell=np.asarray(frt.cell))
t0["triclinic_translation_sym_diff"] = len(edgeset(eit, ust) ^ edgeset(torch.as_tensor(np.asarray(nbt[0]), dtype=torch.long, device=dev), torch.as_tensor(np.asarray(nbt[2]), dtype=DT, device=dev)))
t0["PASS"] = bool(t0["max_sym_diff"] == 0 and t0["triclinic_translation_sym_diff"] == 0)
res["T0_nl_vs_upstream"] = t0

# T1 FAST E/F vs ISOLATED ref (ASE public), all 20 frames, finite-gated
dF = dE = 0.0
for fr in frames:
    eU, fU, _ = efs(REF, fr)
    ef, ff = F.energy_forces(fr.positions, np.asarray(fr.cell)); fin(ff, ef, fU, eU)
    dF = max(dF, float(np.linalg.norm(ff - fU, axis=1).max())); dE = max(dE, abs(ef - eU) / n)
res["T1_FAST_vs_ref"] = {"dF": dF, "dE": dE, "PASS": bool(dF < 1e-4 and dE < 1e-5)}

# T2 STRESS E/F/virial + candidate ASE stress vs ref ASE stress
dF2 = dE2 = dW2 = dS2 = 0.0
candcalc = D3FastCalculator(F)
from ase.stress import voigt_6_to_full_3x3_stress
for fr in frames:
    eU, fU, sU = efs(REF, fr)
    es, fs, Ws = F.energy_forces_stress(fr.positions, np.asarray(fr.cell)); fin(fs, es, Ws, fU, eU, sU)
    _, _, sC = efs(candcalc, fr); fin(sC)
    dF2 = max(dF2, float(np.linalg.norm(fs - fU, axis=1).max())); dE2 = max(dE2, abs(es - eU) / n)
    V = float(abs(np.linalg.det(np.asarray(fr.cell)))); WU = -V * voigt_6_to_full_3x3_stress(sU)
    dW2 = max(dW2, float(np.abs(Ws - WU).max())); dS2 = max(dS2, float(np.abs(sC - sU).max()))
res["T2_STRESS_vs_ref"] = {"dF": dF2, "dE": dE2, "dW": dW2, "dStress_ASE": dS2, "PASS": bool(dF2 < 1e-4 and dE2 < 1e-5 and dW2 < 1e-3 and dS2 < 1e-4)}

# T3 FAST vs STRESS E/F
dF3 = dE3 = 0.0
for fr in frames[:8]:
    ef, ff = F.energy_forces(fr.positions, np.asarray(fr.cell)); es, fs, Ws = F.energy_forces_stress(fr.positions, np.asarray(fr.cell)); fin(ff, fs, ef, es)
    dF3 = max(dF3, float(np.linalg.norm(ff - fs, axis=1).max())); dE3 = max(dE3, abs(ef - es) / n)
res["T3_FAST_vs_STRESS"] = {"dF": dF3, "dE": dE3, "PASS": bool(dF3 < 1e-6 and dE3 < 1e-7)}

# T4 CAP overflow (eager) vs ref + finite
F.cap = 8; ef, ff = F.energy_forces(at0.positions, np.asarray(at0.cell))
eU, fU, _ = efs(REF, at0); fin(ff, ef, fU, eU)
res["T4_overflow"] = {"grown_cap": F.cap, "dF": float(np.linalg.norm(ff - fU, axis=1).max()), "dE": abs(ef - eU) / n,
                      "PASS": bool(F.cap > 8 and np.linalg.norm(ff - fU, axis=1).max() < 1e-4 and abs(ef - eU) / n < 1e-5)}

# T5 tiny (1e-4) AND large (3%) cell change -> FAST re-caches (lightweight _update_cell, construction ns, as the
# original code does per step) and matches the INDEPENDENT StockD3 reference at that cell (same construction-ns path)
ok5 = {}
for tag, scale in [("tiny", 1.0 + 1e-4), ("large", 1.03)]:
    a_s = at0.copy(); a_s.set_cell(np.asarray(at0.cell) * scale, scale_atoms=True)
    eU, fU, _ = efs(REF, a_s)
    ef_iv, ff_iv = F.energy_forces(a_s.positions, np.asarray(a_s.cell)); fin(ff_iv, ef_iv, fU, eU)
    ok5[tag] = {"dF": float(np.linalg.norm(ff_iv - fU, axis=1).max()), "dE": abs(ef_iv - eU) / n}
ok5["PASS"] = bool(ok5["tiny"]["dF"] < 1e-4 and ok5["large"]["dF"] < 1e-4 and ok5["tiny"]["dE"] < 1e-5 and ok5["large"]["dE"] < 1e-5)
res["T5_cell_change"] = ok5

# T6 topology: species rebuild matches fresh; count raise; PBC raise; raw-API invalid guard
ok6 = {}
F6 = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
at_sp = at0.copy(); nums = at_sp.numbers.copy(); oi = int(np.where(nums == 1)[0][0]); nums[oi] = 8; at_sp.set_atomic_numbers(nums)
Fspref = FastFourierD3(MODEL, HEAD, d3kw, at_sp, dtype="float32"); e_rs, f_rs = Fspref.energy_forces(at_sp.positions, np.asarray(at_sp.cell))
e_rb, f_rb = F6.energy_forces(at_sp); fin(f_rb, f_rs); ok6["species_rebuild_dF"] = float(np.linalg.norm(f_rb - f_rs, axis=1).max())
try: F6.energy_forces(np.zeros((n + 1, 3)), np.asarray(at0.cell)); ok6["count_raise"] = False
except RuntimeError: ok6["count_raise"] = True
at_pbc = at0.copy(); at_pbc.pbc = [True, True, False]
try: F6.energy_forces(at_pbc); ok6["pbc_raise"] = False
except RuntimeError: ok6["pbc_raise"] = True
F6._valid = False   # simulate a failed restore; raw-API must now refuse
try: F6.energy_forces(at0.positions, np.asarray(at0.cell)); ok6["raw_invalid_guard"] = False
except RuntimeError: ok6["raw_invalid_guard"] = True
ok6["PASS"] = bool(ok6["species_rebuild_dF"] < 1e-4 and ok6["count_raise"] and ok6["pbc_raise"] and ok6["raw_invalid_guard"])
res["T6_topology"] = ok6

# T7 compiled lifecycle (ewald+SPME): warm; CAP overflow AFTER warm; cell change; species rebuild; FAST->STRESS->FAST
import torch._dynamo as dyn
t7 = {}
for mname, kw in [("ewald", d3kw), ("spme", {"r_cut": 6.0, "xcfunc": "pbe", "method": "spme", "cnfunc": "smooth_cut", "mesh_spacing": 1.2, "interpolation_nodes": 5, "c6tol": 1.0})]:
    dyn.reset(); dyn.utils.counters.clear()
    Rf = StockD3(kw, at0, dtype="float32")
    Fc = FastFourierD3(MODEL, HEAD, kw, at0, dtype="float32", compile_fast=True, cap=60000)
    worst = 0.0; worstE = 0.0
    def chk(at, ref):   # ref MUST match at's cell+species (stock dm is per-config)
        global worst, worstE
        ef, ff = Fc.energy_forces(at); eU, fU, _ = efs(ref, at); fin(ff, ef, fU, eU)
        worst = max(worst, float(np.linalg.norm(ff - fU, axis=1).max())); worstE = max(worstE, abs(ef - eU) / n)
    for fr in frames[:3]: chk(fr, Rf)                  # warmed replay, changing pos/edges (same cell/species)
    Fc.cap = 8; chk(frames[3], Rf)                     # CAP overflow AFTER warmup -> regrow+recompile
    a_s = at0.copy(); a_s.set_cell(np.asarray(at0.cell) * 1.02, scale_atoms=True)
    chk(a_s, Rf)                                       # cell change: Rf (at0-built) evaluated at scaled cell = construction-ns _update_cell, matching FAST re-cache
    _ = Fc.energy_forces_stress(at0.positions, np.asarray(at0.cell)); chk(frames[1], Rf)      # FAST->STRESS->FAST
    at_sp2 = at0.copy(); nn = at_sp2.numbers.copy(); nn[int(np.where(nn == 1)[0][0])] = 8; at_sp2.set_atomic_numbers(nn)
    chk(at_sp2, StockD3(kw, at_sp2, dtype="float32"))  # species rebuild (fresh ref for new species)
    gb = 0
    for k, v in dyn.utils.counters.items():
        if "graph_break" in k: gb += (sum(v.values()) if isinstance(v, dict) else int(v))
    t7[mname] = {"worst_dF": worst, "worst_dE": worstE, "graph_breaks": int(gb), "compiled": Fc._cfast is not None,
                 "PASS": bool(worst < 1e-4 and worstE < 1e-5)}
t7["PASS"] = bool(t7["ewald"]["PASS"] and t7["spme"]["PASS"])
res["T7_compiled_lifecycle"] = t7

# T8 padded == unpadded E/F, finite
F8 = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
pos = torch.as_tensor(at0.positions, dtype=DT, device=dev); cell = F8._cached_cell
ei0, us0 = build_nl(F8.dm, pos, cell, RMAX, PBC, n, cap=None)
from fourierd3.fast import _pad
eip, usp = _pad(ei0, us0, ei0.shape[1] + 5000, dev, DT)
def ef_with(ei, us):
    p = pos.clone().requires_grad_(True); e = F8.dm(positions=p, edge_index=ei, shifts=torch.matmul(us, cell)) * HART
    g, = torch.autograd.grad(e, [p]); return float(e.item()), (-g).detach().cpu().numpy()
eU, fU = ef_with(ei0, us0); eP, fP = ef_with(eip, usp); fin(fU, fP, eU, eP)
res["T8_padding"] = {"dF": float(np.abs(fP - fU).max()), "dE": abs(eP - eU) / n, "finite": True,
                     "PASS": bool(np.abs(fP - fU).max() < 1e-6 and np.all(np.isfinite(fP)))}

res["ALL_PASS"] = bool(all(v.get("PASS") for v in res.values() if isinstance(v, dict)))
print(json.dumps(res, indent=1, default=str), flush=True)
print("TESTS_DONE ALL_PASS=", res["ALL_PASS"], flush=True)
sys.exit(0 if res["ALL_PASS"] else 1)
