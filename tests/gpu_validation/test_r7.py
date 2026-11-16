#!/usr/bin/env python
"""Astra r6 MAJOR regression (stale-results-on-failure). Exit NON-ZERO on fail.
A FAILED calculation must never leave the NEW atoms paired with the OLD frame's results.
 Scenario (both the calculate_properties path R7-1 and the direct calculate path R7-2):
  1. Successfully calculate frame A -> E/F/stress cached.
  2. Request frame B (CHANGED cell) with an INJECTED _update_cell/recache failure.
  3. The B calculation must RAISE and leave the ASE results cache EMPTY (never retain A's values paired with B
     in self.atoms).
  4. A subsequent getter on B must RAISE (invalid engine) or RECOMPUTE a fresh value -- NEVER return A's cached
     result (the exact round-6 hole: self.atoms==B, check_state==[], stale A value served from self.results).
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from fourierd3.fast import FastFourierD3, D3FastCalculator
RD = os.environ.get("FD3_TEST_RUNDIR", "/mnt/vast/cheuk.hinho/runs/20261109T144651Z_symx-metad-speed")
MODEL = os.environ.get("FD3_TEST_MODEL", "/mnt/vast/home/cheuk.hinho/models/mace-mh-1.model"); HEAD = "oc20_usemppbe"
d3kw = {"r_cut": 6.0, "xcfunc": "pbe", "method": "ewald", "cnfunc": "smooth_cut", "k_cutoff": 6.0, "c6tol": 0.1}
at0 = read(RD + "/inputs/seed_w0.extxyz")


def _boom(*a, **k):
    raise RuntimeError("injected _update_cell failure")


def run_case(trigger):
    F = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
    calc = D3FastCalculator(F)
    # 1. prime A
    a = at0.copy(); a.calc = calc
    eA = float(a.get_potential_energy()); fA = a.get_forces().copy(); sA = a.get_stress().copy()
    # 2. B with a changed cell + injected recache/_update_cell failure
    B = at0.copy(); B.set_cell(np.asarray(at0.cell) * 1.05, scale_atoms=True)
    orig = F.dm._update_cell
    F.dm._update_cell = _boom
    raised = False
    try:
        trigger(calc, B)
    except Exception:
        raised = True
    # 3. the ASE results cache must be EMPTY and hold NONE of A's values
    r = dict(calc.results)
    stale = (("forces" in r and np.allclose(np.asarray(r["forces"]), fA)) or
             ("stress" in r and np.allclose(np.asarray(r["stress"]), sA)) or
             ("energy" in r and np.isclose(float(r["energy"]), eA)))
    results_empty = (len(r) == 0)
    # 4. a subsequent getter must raise, or recompute != A (never return A's cached value)
    F.dm._update_cell = orig                    # restore engine fn; the INSTANCE still stays invalid
    getter_safe = False
    try:
        val = calc.get_property("forces", atoms=B)
        getter_safe = not np.allclose(np.asarray(val), fA)   # a fresh recompute, not A
    except Exception:
        getter_safe = True                      # raised -> never returned A
    return {"raised": bool(raised), "results_empty": bool(results_empty),
            "no_stale_A_in_cache": bool(not stale), "getter_never_returns_A": bool(getter_safe),
            "PASS": bool(raised and results_empty and (not stale) and getter_safe)}


res = {}
res["R7_1_calculate_properties"] = run_case(lambda c, B: c.calculate_properties(B, ["stress"]))
res["R7_2_direct_calculate"] = run_case(lambda c, B: (B.__setattr__("calc", c), B.get_stress())[1])
res["ALL_PASS"] = bool(all(v.get("PASS") for v in res.values() if isinstance(v, dict)))
print(json.dumps(res, indent=1), flush=True)
print("R7_TESTS_DONE ALL_PASS=", res["ALL_PASS"], flush=True)
sys.exit(0 if res["ALL_PASS"] else 1)
