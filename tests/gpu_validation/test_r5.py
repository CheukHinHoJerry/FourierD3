#!/usr/bin/env python
"""Astra r4 exception-injection tests. Exit NON-ZERO on fail.
 R5-1 (M1 lock scope / foreign-thread): while another thread holds the engine lock, a get_property / energy_forces
       entry RAISES (concurrent/foreign-thread entry rejected) — not silently evaluated.
 R5-2 (M2 recache failure-safety): a mid-recache exception leaves the instance INVALID (never half-updated);
       a subsequent call on the invalid instance raises; a fresh instance (clean retry) succeeds.
"""
import os, sys, json, threading, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from fourierd3.fast import FastFourierD3, D3FastCalculator
RD = os.environ.get("FD3_TEST_RUNDIR", "/mnt/vast/cheuk.hinho/runs/20261109T144651Z_symx-metad-speed")
MODEL = os.environ.get("FD3_TEST_MODEL", "/mnt/vast/home/cheuk.hinho/models/mace-mh-1.model"); HEAD = "oc20_usemppbe"
d3kw = {"r_cut": 6.0, "xcfunc": "pbe", "method": "ewald", "cnfunc": "smooth_cut", "k_cutoff": 6.0, "c6tol": 0.1}
at0 = read(RD + "/inputs/seed_w0.extxyz"); c0 = np.asarray(at0.cell)
res = {}

# R5-1 foreign-thread / lock-scope
F = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
calc = D3FastCalculator(F)
held = threading.Event(); release = threading.Event()
def hog():
    F._lock.acquire()  # simulate a foreign thread inside the engine
    held.set(); release.wait(timeout=30); F._lock.release()
t = threading.Thread(target=hog, daemon=True); t.start(); held.wait(timeout=10)
rej_ef = rej_gp = False
try:
    F.energy_forces(at0.positions, c0)
except RuntimeError as e:
    rej_ef = "rejected" in str(e)
try:
    a = at0.copy(); a.calc = calc; a.get_potential_energy()
except RuntimeError as e:
    rej_gp = "rejected" in str(e)
release.set(); t.join(timeout=5)
# after release, a normal call works
ok_after = False
try:
    e, f = F.energy_forces(at0.positions, c0); ok_after = np.all(np.isfinite(f))
except Exception:
    ok_after = False
res["R5_1_foreign_thread"] = {"energy_forces_rejected": bool(rej_ef), "get_property_rejected": bool(rej_gp),
                              "works_after_release": bool(ok_after), "PASS": bool(rej_ef and rej_gp and ok_after)}

# R5-2 mid-recache failure-safety
F2 = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
orig_update = F2.dm._update_cell
def boom(*a, **k): raise RuntimeError("injected _update_cell failure")
F2.dm._update_cell = boom
raised = invalid = False
try:
    F2.energy_forces(at0.positions, c0 * 1.05)   # changed cell -> _recache_cell -> _update_cell boom
except RuntimeError:
    raised = True
invalid = (F2._valid is False)
# a subsequent call on the invalid instance must raise
blocked_after = False
try:
    F2.dm._update_cell = orig_update              # even if the engine is "fixed", the instance stays invalid
    F2.energy_forces(at0.positions, c0)
except RuntimeError as e:
    blocked_after = "invalid" in str(e)
# clean retry with a FRESH instance succeeds
F3 = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
e3, f3 = F3.energy_forces(at0.positions, c0)
retry_ok = bool(np.all(np.isfinite(f3)))
res["R5_2_recache_failsafe"] = {"recache_raised": bool(raised), "instance_invalid_after": bool(invalid),
                                "blocked_after_failure": bool(blocked_after), "fresh_retry_ok": retry_ok,
                                "PASS": bool(raised and invalid and blocked_after and retry_ok)}

res["ALL_PASS"] = bool(all(v.get("PASS") for v in res.values() if isinstance(v, dict)))
print(json.dumps(res, indent=1), flush=True)
print("R5_TESTS_DONE ALL_PASS=", res["ALL_PASS"], flush=True)
sys.exit(0 if res["ALL_PASS"] else 1)
