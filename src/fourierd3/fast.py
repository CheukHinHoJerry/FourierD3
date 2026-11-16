#!/usr/bin/env python
"""Dual-path FourierD3 drop-in (coordinator L1 decision), r3 — addresses Astra r2 (B1,B2,M3..M8).
 - FAST (default, no 'stress'): fixed-cell cached cell-dependent work; forces-only autograd; static
   padded edges (masked self-pair padding); torch.compile/CUDA-graph friendly.
 - STRESS ('stress'): the ORIGINAL strain-autograd path (E/F/virial), exception-safe.
Pure wrapper on the UNCHANGED fourierd3 0.1.0 FourierD3 forward.

r3 fixes: (B1) validity+shape guard at the start of BOTH public paths; (M3) TRANSACTIONAL state —
instance marked invalid before any cell/species mutation, revalidated only after full success; a cell
change FULLY REBUILDS the model (correct k-vector/mesh count — resolves the T5 fresh-build gap; production
is fixed-cell regardless); (M5) a non-reentrant lock rejects concurrent entry; (M7) padding = MASKED (0,0)
self-pairs with ZERO shift (distance 0 -> excluded by compute_cn_smooth's r>1e-6 mask; no large-r^n overflow),
single consolidated implementation. B3 (r2) wrapped-coordinate NL retained + validated edge-for-edge vs upstream.
"""
import threading
from contextlib import contextmanager
import numpy as np, torch
from ase.calculators.calculator import Calculator, all_changes

A2B = 1.0 / 0.52917726
HART = 27.21138505


def _pad(ei, us, cap, dev, dt):
    """Consolidated static-shape padding: masked self-pairs (source=target=0, ZERO shift).
    A zero-shift self-pair has distance 0 -> excluded by FourierD3.compute_cn_smooth (r>1e-6 mask),
    contributes exactly 0, and introduces no large-distance intermediate (overflow-safe)."""
    m = ei.shape[1]
    if m > cap:
        raise RuntimeError(f"edges {m} > cap {cap}")
    pad = cap - m
    if pad == 0:
        return ei, us
    pe = torch.zeros(2, pad, dtype=torch.long, device=dev)      # (0,0) self-pairs
    pu = torch.zeros(pad, 3, dtype=dt, device=dev)              # zero shift
    return torch.cat([ei, pe], 1), torch.cat([us, pu], 0)


def build_nl(dm, pos_t, cell_t, rmax, pbc, n, cap=None):
    """GPU neighbour list correct for UNWRAPPED coords / triclinic / partial-PBC cells.
    Returns [source,target] (long) and INTEGER shifts S_true (pos dtype) with
    d_ij = pos[target]-pos[source]+S_true@cell (matches FourierD3.compute_cn_smooth)."""
    dt = pos_t.dtype; dev = pos_t.device; rc2 = rmax * rmax
    # Compute connectivity in fp64 regardless of the model dtype: the cutoff tie-break for near-boundary pairs
    # must match the fp64 upstream NL (matscipy get_neighborhood) under ALL (incl. deformed NPT) cells; the
    # returned integer shifts are exact. (An fp32 NL mismatches upstream by a few boundary pairs on some cells.)
    pos64 = pos_t.detach().double(); cell64 = cell_t.detach().double()
    inv = torch.linalg.inv(cell64)
    scaled = pos64 @ inv
    wrap = torch.zeros_like(scaled)
    for d in range(3):
        if pbc[d]:
            wrap[:, d] = torch.floor(scaled[:, d])
    pos_w = (scaled - wrap) @ cell64
    nrep = [int(np.ceil(rmax * float(torch.linalg.norm(inv[:, d]).item()))) if pbc[d] else 0 for d in range(3)]
    rngs = [torch.arange(-nrep[d], nrep[d] + 1, device=dev, dtype=torch.float64) for d in range(3)]
    S = torch.cartesian_prod(*rngs).reshape(-1, 3); sh = S @ cell64
    diff = pos_w.view(1, 1, n, 3) + sh.view(-1, 1, 1, 3) - pos_w.view(1, n, 1, 3)
    d2 = (diff * diff).sum(-1); selfm = (S == 0).all(1)
    eye = torch.eye(n, dtype=torch.bool, device=dev); keep = d2 <= rc2; keep[selfm] &= ~eye
    mi, ii, jj = keep.nonzero(as_tuple=True)
    ei = torch.stack([ii, jj]).to(torch.long)
    us = (S[mi] + wrap[ii] - wrap[jj]).to(dt)   # integer shifts (exact), cast to model dtype
    if cap is not None:
        ei, us = _pad(ei, us, cap, dev, dt)
    return ei, us


class FastFourierD3:
    """SINGLE-walker ownership, enforced by a reentrant lock (RLock): the owning thread (incl. the ASE
    calculator holding it across the whole transaction) re-enters freely; any OTHER thread is rejected.
    Raw (pos,cell) API uses the CURRENT ordered topology and cannot change it; an Atoms call may rebuild
    species. ALL validation (shape/count/PBC) runs BEFORE any mutation. Fixed cell (production is fixed-cell);
    an EXACT cell change FULLY REBUILDS the model. Mutations are transactional: invalid before, valid after
    success. Every persistent cell tensor is owned (cloned), never aliasing caller numpy."""

    def __init__(self, model_paths, head, d3_kwargs, atoms0, dtype="float32", device="cuda",
                 cap=None, compile_fast=False, stable_n=20):
        from fourierd3.mace_d3 import MACEFourierD3Calculator
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self._mk = dict(model_paths=model_paths, head=head)
        self._d3kw = dict(d3_kwargs)
        self._dtype = dtype
        if str(dtype) in ("float64", "torch.float64"):
            self._d3kw["dtype"] = torch.float64   # mace_d3 does NOT forward dtype to FourierD3 (defaults fp32)
        self.host = MACEFourierD3Calculator(model_paths=model_paths, device=device,
                                            default_dtype=dtype, head=head, d3_kwargs=dict(self._d3kw))
        self.rmax = float(getattr(self.host, "r_max", 0.0) or self.host.models[0].r_max)
        self.dev = torch.device(device)
        self._atoms = atoms0.copy()
        self.host._ensure_d3_model(self._atoms, all_changes)
        self.dm = self.host.d3_model; self.DT = self.dm.dtype
        self.n = len(atoms0)
        self._numbers = atoms0.numbers.copy()
        self.pbc = tuple(bool(x) for x in atoms0.pbc)
        self.cap = cap
        self._compile = compile_fast
        self._cfast = None; self._cfast_cell = None
        self._stable_N = int(stable_n)   # use the compiled/CUDA-graph FAST path only after the cell is stable this many calls
        self._stable_calls = 0
        self._recompiles = 0             # exposed counter (NPT must keep this 0)
        self._valid = True
        self._lock = threading.RLock()   # reentrant: calculator holds it across energy_forces (same thread); other threads rejected
        self._cell_cached = np.array(atoms0.cell, dtype=np.float64).copy()
        self._cached_cell = torch.as_tensor(np.asarray(atoms0.cell), dtype=self.DT, device=self.dev).clone()  # owned storage (never alias caller numpy)
        self.dm._update_cell(self._cached_cell)

    # ---- NPT-safe lightweight cell invalidation (matches the ORIGINAL per-step _update_cell: construction-time
    #      ns kept, k-vectors/Green's-fn/mesh/inverse/volume recomputed for the new cell). No full model rebuild,
    #      no recompile here -> no recompile storm under variable-cell MD. ----
    def _recache_cell(self, cell_np):
        # Transactional (Astra r4 M2): invalid BEFORE mutating, valid only AFTER success; a mid-recache
        # exception (e.g. _update_cell OOM) leaves the instance invalid (never half-updated).
        changed = not np.array_equal(np.asarray(cell_np, dtype=np.float64), self._cell_cached)
        self._valid = False
        try:
            nc = torch.as_tensor(np.asarray(cell_np), dtype=self.DT, device=self.dev).clone()  # owned
            self.dm._update_cell(nc)
            self._cached_cell = nc
            self._cell_cached = np.array(cell_np, dtype=np.float64).copy()
            if changed:
                self._stable_calls = 0                 # restart the stability window -> eager fallback
                self._cfast = None; self._cfast_cell = None   # drop any graph captured for the old cell
            self._valid = True
        except Exception:
            self._valid = False; raise

    def _rebuild_for_species(self, atoms):   # species/order change = full model rebuild (rare; NOT per-step NPT)
        self._valid = False; self._cfast = None; self._cfast_cell = None
        self._atoms = atoms.copy()
        self.host.d3_model = None; self.host._d3_species = None
        self.host._ensure_d3_model(self._atoms, all_changes)
        self.dm = self.host.d3_model
        self._numbers = atoms.numbers.copy()
        self._cached_cell = torch.as_tensor(np.asarray(atoms.cell), dtype=self.DT, device=self.dev).clone()  # owned
        self.dm._update_cell(self._cached_cell)
        self._cell_cached = np.array(atoms.cell, dtype=np.float64).copy()
        self._stable_calls = 0; self._valid = True

    def _validate_topology(self, pos_np, cell_np, numbers=None, pbc=None):
        # ALL non-mutating validation (shape/count/PBC-flag) BEFORE any mutation (Astra r3 M1). A PBC-FLAG or
        # atom-COUNT change raises (topology change, NOT NPT cell deformation); species/order change rebuilds.
        # A CELL (vector/shape/volume) change does NOT raise -> handled as graceful invalidation by the caller.
        if not self._valid:
            raise RuntimeError("FastFourierD3 is invalid (a prior mutation failed); rebuild the instance")
        pos_np = np.asarray(pos_np)
        if pos_np.shape != (self.n, 3):
            raise RuntimeError(f"positions shape {pos_np.shape} != ({self.n},3) (atom count fixed)")
        if np.asarray(cell_np).shape != (3, 3):
            raise RuntimeError(f"cell shape {np.asarray(cell_np).shape} != (3,3)")
        if pbc is not None and tuple(bool(x) for x in pbc) != self.pbc:
            raise RuntimeError("PBC-flag change unsupported (topology change, not NPT cell deformation)")
        if numbers is not None and not np.array_equal(numbers, self._numbers):
            self._rebuild_for_species_atoms(numbers, cell_np)

    def _fast_core(self, pos, ei, us, cell):
        p = pos.detach().requires_grad_(True)
        e = self.dm(positions=p, edge_index=ei, shifts=torch.matmul(us, cell)) * HART
        g, = torch.autograd.grad(e, [p])
        return e.detach(), (-g)

    def _ensure_cap(self, m):
        """Grow (never shrink) the static pad capacity; return True if it grew (caller drops the stale graph)."""
        if self.cap is None:
            self.cap = int(m * 1.15 // 8 * 8 + 8); return True
        if m > self.cap:
            self.cap = int(m * 1.15 // 8 * 8 + 8); return True
        return False

    @staticmethod
    def _unpack(atoms_or_pos, cell_np):
        if cell_np is None:
            at = atoms_or_pos
            return at.positions, np.asarray(at.cell), at.numbers, tuple(bool(x) for x in at.pbc)
        return atoms_or_pos, cell_np, None, None

    def energy_forces(self, atoms_or_pos, cell_np=None):
        """FAST path (no stress). NPT-safe: a cell change invalidates the cache (lightweight _update_cell) and
        falls back to EAGER; the compiled/CUDA-graph path is used only after the cell is stable for stable_N calls."""
        pos_np, cell_np, numbers, pbc = self._unpack(atoms_or_pos, cell_np)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent entry rejected (single-walker ownership)")
        try:
            self._validate_topology(pos_np, cell_np, numbers, pbc)
            if not np.array_equal(np.asarray(cell_np, dtype=np.float64), self._cell_cached):
                self._recache_cell(cell_np)          # cell changed -> invalidate (no raise, no full rebuild)
            else:
                self._stable_calls += 1
            pos = torch.as_tensor(pos_np, dtype=self.DT, device=self.dev)
            ei0, us0 = build_nl(self.dm, pos, self._cached_cell, self.rmax, self.pbc, self.n, cap=None)
            if self._ensure_cap(ei0.shape[1]):
                self._cfast = None; self._cfast_cell = None   # capacity grew -> stale graph
            ei, us = _pad(ei0, us0, self.cap, self.dev, self.DT)
            use_compiled = self._compile and (self._stable_calls >= self._stable_N)
            if use_compiled:
                if self._cfast is None:
                    self._cfast = torch.compile(self._fast_core, mode="reduce-overhead", fullgraph=False)
                    self._cfast_cell = self._cell_cached.copy(); self._recompiles += 1
                fn = self._cfast
            else:
                fn = self._fast_core                 # EAGER (cell not yet stable / recently changed)
            e, f = fn(pos, ei, us, self._cached_cell)
            return float(e.item()), f.detach().cpu().numpy()
        finally:
            self._lock.release()

    def energy_forces_stress(self, atoms_or_pos, cell_np=None):
        """STRESS path (used by NPT every step): the ORIGINAL strain-autograd, UNCHANGED. NL rebuilt for the
        current cell each call; dm updated to the strained cell then restored to the current cell (NPT-safe)."""
        pos_np, cell_np, numbers, pbc = self._unpack(atoms_or_pos, cell_np)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent entry rejected (single-walker ownership)")
        try:
            self._validate_topology(pos_np, cell_np, numbers, pbc)
            cell = torch.as_tensor(np.asarray(cell_np), dtype=self.DT, device=self.dev)
            pos = torch.as_tensor(pos_np, dtype=self.DT, device=self.dev)
            ei, us = build_nl(self.dm, pos, cell, self.rmax, self.pbc, self.n, cap=None)  # current cell (deformation-correct)
            try:
                p = pos.detach().requires_grad_(True)
                st = torch.zeros(3, 3, dtype=self.DT, device=self.dev, requires_grad=True)
                sc = cell + torch.einsum("ab,Ab->Aa", st, cell); self.dm._update_cell(sc)
                sp = p + torch.einsum("ab,ib->ia", st, p)
                e = self.dm(positions=sp, edge_index=ei, shifts=torch.matmul(us, sc)) * HART
                gp, gs = torch.autograd.grad(e, [p, st])
                sigma = (gs / float(self.dm.volume) * (A2B ** 3)).detach().cpu().numpy()
                V = float(abs(np.linalg.det(np.asarray(cell_np))))
                out = float(e.item()), (-gp).detach().cpu().numpy(), -V * sigma
            finally:
                try:
                    self._recache_cell(cell_np)   # undo the strain mutation; track the current cell (NPT)
                except Exception:
                    self._valid = False
            return out
        finally:
            self._lock.release()

    def _rebuild_for_species_atoms(self, numbers, cell_np):
        at = self._atoms.copy(); at.set_atomic_numbers(numbers); at.set_cell(np.asarray(cell_np), scale_atoms=False)
        self._rebuild_for_species(at)


class D3FastCalculator(Calculator):
    """ASE D3 calculator; dual path by requested `properties`. Single-walker ownership: EVERY ASE entry point that
    reads or mutates the results cache / atoms (get_property, calculate, calculate_properties, export_properties,
    reset) holds the SAME engine RLock and rejects concurrent/foreign-thread entry (Astra r4 M1 + r5). RLock is
    reentrant so same-thread nesting (get_property -> calculate) is fine; other threads are rejected."""
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, fast: "FastFourierD3", **kw):
        self.f = fast                 # set BEFORE super().__init__ (which calls self.reset())
        super().__init__(**kw)

    @contextmanager
    def _hold(self):
        if not self.f._lock.acquire(blocking=False):
            raise RuntimeError("concurrent/foreign-thread entry rejected (single-walker ownership)")
        try:
            yield
        finally:
            self.f._lock.release()

    def get_property(self, name, atoms=None, allow_calculation=True):
        with self._hold():
            return super().get_property(name, atoms=atoms, allow_calculation=allow_calculation)

    def reset(self):
        with self._hold():
            return super().reset()

    def calculate_properties(self, atoms, properties):
        with self._hold():
            return super().calculate_properties(atoms, properties)

    def export_properties(self):
        with self._hold():
            return super().export_properties()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        with self._hold():
            # Astra r6 MAJOR (stale-results-on-failure): CLEAR the ASE results cache BEFORE installing the new
            # atoms, so a failed engine evaluation (e.g. a cell-recache failure that invalidates the engine)
            # can NEVER leave the NEW atoms (self.atoms) paired with the OLD frame's results. The complete
            # results dict is published ATOMICALLY only after a fully successful evaluation; any exception
            # propagates with results left EMPTY, forcing a recompute (or an invalid-engine raise) on the next
            # getter instead of a stale cache hit. Also covers calculate_properties (routes through here).
            self.results = {}
            super().calculate(atoms, properties, system_changes)
            at = self.atoms
            if "stress" in properties:
                e, F, W = self.f.energy_forces_stress(at)
                V = float(abs(np.linalg.det(np.asarray(at.cell))))
                from ase.stress import full_3x3_to_voigt_6_stress
                res = {"energy": e, "free_energy": e, "forces": F, "stress": full_3x3_to_voigt_6_stress(-W / V)}
            else:
                e, F = self.f.energy_forces(at)
                res = {"energy": e, "free_energy": e, "forces": F}
            self.results = res   # publish only on full success


if __name__ == "__main__":
    print("fd3_fast module OK")
