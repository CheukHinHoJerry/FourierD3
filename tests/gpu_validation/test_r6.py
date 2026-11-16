#!/usr/bin/env python
"""Astra r5 test: the remaining base-Calculator cache entry points (reset / calculate_properties /
export_properties) hold the engine lock -> foreign-thread entry is REJECTED, and works after release.
Exit NON-ZERO on fail."""
import os, sys, json, threading, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from fourierd3.fast import FastFourierD3, D3FastCalculator
RD = os.environ.get("FD3_TEST_RUNDIR", "/mnt/vast/cheuk.hinho/runs/20261109T144651Z_symx-metad-speed")
MODEL = os.environ.get("FD3_TEST_MODEL", "/mnt/vast/home/cheuk.hinho/models/mace-mh-1.model"); HEAD = "oc20_usemppbe"
d3kw = {"r_cut": 6.0, "xcfunc": "pbe", "method": "ewald", "cnfunc": "smooth_cut", "k_cutoff": 6.0, "c6tol": 0.1}
at0 = read(RD + "/inputs/seed_w0.extxyz")
F = FastFourierD3(MODEL, HEAD, d3kw, at0, dtype="float32")
calc = D3FastCalculator(F)
# populate results once (for export_properties)
a = at0.copy(); a.calc = calc; a.get_potential_energy()

# entry points to test (only those present on the base class)
entrypoints = {
    "reset": (lambda: calc.reset()),
    "get_property": (lambda: calc.get_property("energy", atoms=at0.copy())),
}
if hasattr(calc, "calculate_properties"):
    entrypoints["calculate_properties"] = (lambda: calc.calculate_properties(at0.copy(), ["energy"]))
if hasattr(calc, "export_properties"):
    entrypoints["export_properties"] = (lambda: calc.export_properties())

held = threading.Event(); release = threading.Event()
def hog():
    F._lock.acquire(); held.set(); release.wait(timeout=30); F._lock.release()
t = threading.Thread(target=hog, daemon=True); t.start(); held.wait(timeout=10)
rejected = {}
for nm, fn in entrypoints.items():
    try:
        fn(); rejected[nm] = False
    except RuntimeError as e:
        rejected[nm] = "rejected" in str(e)
    except Exception as e:
        rejected[nm] = "rejected" in str(e)   # any other -> not a clean rejection
release.set(); t.join(timeout=5)
# after release, each works
works_after = {}
for nm, fn in entrypoints.items():
    try:
        fn(); works_after[nm] = True
    except Exception as e:
        works_after[nm] = False
res = {"tested_entrypoints": sorted(entrypoints), "rejected_while_locked": rejected, "works_after_release": works_after,
       "PASS": bool(all(rejected.values()) and all(works_after.values()))}
print(json.dumps(res, indent=1), flush=True)
print("R6_TESTS_DONE ALL_PASS=", res["PASS"], flush=True)
sys.exit(0 if res["PASS"] else 1)
