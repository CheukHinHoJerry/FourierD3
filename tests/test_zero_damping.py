"""Tests for the D3 zero-damping option (damping='zero') and the unchanged BJ default.

Run with `pytest tests/` from the repository root (pyproject sets pythonpath = src).
All tests run on CPU in float64 and take a few seconds each.
"""
import math

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

from fourierd3 import FourierD3ASECalculator
from fourierd3.utils import load_r0ab

CPU = torch.device("cpu")
REVPBE_BJ = [1.0, 2.3550, 0.5238, 3.5016]
REVPBE_ZERO = {"s6": 1.0, "rs6": 0.923, "s8": 1.010, "rs8": 1.0, "alpha": 14.0}
HARTREE, BOHR = 27.21138505, 0.52917726


def calc(damping="bj", **kw):
    args = dict(r_cut=6.0, method="spme", device=CPU, dtype=torch.float64)
    if damping == "bj":
        args["params"] = REVPBE_BJ
    else:
        args.update(damping="zero", zero_params=REVPBE_ZERO)
    args.update(kw)
    return FourierD3ASECalculator(**args)


def ethanol():
    a = molecule("CH3CH2OH")
    a.set_cell([11.0, 12.0, 13.0])
    a.center()
    a.positions += [0.37, -0.21, 0.13]
    a.set_pbc(True)
    return a


def dimer(sym1, sym2, r, box=40.0):
    a = Atoms(f"{sym1}{sym2}", positions=[[0, 0, 0], [r, 0, 0]], cell=[box] * 3, pbc=True)
    a.positions += box / 2 - r / 2
    return a


def evaluate(atoms, c):
    atoms = atoms.copy()
    atoms.calc = c
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


# ---------------------------------------------------------------------------------------------------------------- BJ
def test_bj_default_unchanged():
    """damping='bj' (the default) reproduces values frozen from the unmodified code (commit ccc89fb)."""
    e, f, s = evaluate(ethanol(), calc())
    assert abs(e - (-0.31614323148235024)) < 1e-12
    np.testing.assert_allclose(f[0], [-0.014641051372293669, 0.006752421798933296, -8.685225549499744e-06], rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[-1], [-0.011398904972728958, 0.011764952939526944, 0.0069108995892179685], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s, [8.274041597900078e-05, 3.6579652675799836e-05, 1.5406260542182824e-05,
                                   1.4462925779372956e-10, -4.4671385082431186e-09, -2.3493921098127774e-05], rtol=0, atol=1e-14)


def test_bj_explicit_equals_default():
    """Passing damping='bj' explicitly is bit-identical to not passing it."""
    a = evaluate(ethanol(), calc())
    b = evaluate(ethanol(), calc(damping="bj"))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# -------------------------------------------------------------------------------------------------------------- zero
def test_r0ab_table_values():
    """Spot values of the pairwise R0 table (Angstrom), Grimme et al. 2010 r0ab."""
    r0 = load_r0ab([1, 6, 8, 29]).numpy()
    assert abs(r0[0, 0] - 2.1823) < 1e-12      # H-H
    assert abs(r0[2, 0] - 2.1768) < 1e-12      # O-H
    assert abs(r0[2, 2] - 2.4817) < 1e-12      # O-O
    assert abs(r0[3, 3] - 3.3008) < 1e-12      # Cu-Cu
    assert abs(r0[3, 1] - 3.1058) < 1e-12      # Cu-C
    assert np.array_equal(r0, r0.T)


def _hand_pair_energy(c, atoms, z1, z2, r_ang):
    """-[s6 C6 f6/r^6 + 3 sqrt(Q1 Q2) s8 C6 f8/r^8] in eV for one isolated pair, C6 from the model's own CN/C6."""
    model = c._model
    pos = torch.tensor(atoms.positions, dtype=torch.float64) / BOHR
    ei, sh = c.neighbor_list.get(atoms)
    cn = model.compute_cn_smooth(pos, ei, torch.matmul(sh, torch.tensor(atoms.cell.array, dtype=torch.float64)) / BOHR)
    w = model.compute_c6_weights(cn)
    c6 = torch.einsum("nk,nkr->nr", w, model.v_q_reshaped[model.species])
    c6ij = float(torch.einsum("r,r,r->", c6[0], model.eigs, c6[1]))
    r = r_ang / BOHR
    r0 = float(load_r0ab([z1, z2])[0, 1]) / BOHR
    q = float(model.zero_qq[model.species[0], model.species[1]])
    p = REVPBE_ZERO
    f6 = 1.0 / (1.0 + 6.0 * (r / (p["rs6"] * r0)) ** (-p["alpha"]))
    f8 = 1.0 / (1.0 + 6.0 * (r / (p["rs8"] * r0)) ** (-(p["alpha"] + 2)))
    return -(p["s6"] * c6ij * f6 / r ** 6 + 3.0 * q * p["s8"] * c6ij * f8 / r ** 8) * HARTREE


def test_zero_pair_energy_hand():
    """Isolated pairs (40 A box): FourierD3 zero damping equals the hand-evaluated Grimme-2010 pair formula."""
    for s1, s2, z1, z2, r in (("Ar", "Ar", 18, 18, 3.8), ("C", "O", 6, 8, 3.0), ("Cu", "O", 29, 8, 2.6)):
        a = dimer(s1, s2, r)
        c = calc(damping="zero", c6tol=0.01)
        e = evaluate(a, c)[0]
        ref = _hand_pair_energy(c, a, z1, z2, r)
        # tolerance = SPME accuracy of the reciprocal-space part (~1e-6 eV); periodic images at >= 40 A are negligible
        assert abs(e - ref) < 1e-4 * abs(ref) + 2e-6, (s1, s2, e, ref)


def test_zero_equals_bj_tail():
    """Far apart (14 A) zero and BJ with the same s6, s8 give the same energy: both reduce to C6/r^6 + C8/r^8."""
    a = dimer("Ar", "Ar", 14.0)
    ez = evaluate(a, calc(damping="zero"))[0]
    eb = evaluate(a, calc(params=[REVPBE_ZERO["s6"], REVPBE_ZERO["s8"], 0.5238, 3.5016]))[0]
    assert abs(ez - eb) < 2e-3 * abs(eb)


def _distorted_cell():
    a = bulk("Cu", "fcc", a=3.62, cubic=True)
    a = a + Atoms("CO", positions=[[0.9, 0.8, 1.95], [0.95, 0.85, 3.10]])
    a.positions += np.array([[0.03, -0.02, 0.01], [-0.02, 0.04, 0.0], [0.01, 0.0, -0.03], [0.0, 0.02, 0.02],
                             [0.0, 0.0, 0.0], [0.01, -0.01, 0.0]])
    a.set_cell(a.cell.array @ np.array([[1.0, 0.03, 0.0], [0.0, 1.0, 0.02], [0.01, 0.0, 1.0]]), scale_atoms=False)
    a.set_pbc(True)
    return a


def test_zero_forces_finite_difference():
    """Analytic zero-damping forces agree with central finite differences of the energy."""
    a = _distorted_cell()
    _, f, _ = evaluate(a, calc(damping="zero"))
    h = 1e-4
    for i in range(len(a)):
        for k in range(3):
            p, m = a.copy(), a.copy()
            p.positions[i, k] += h
            m.positions[i, k] -= h
            fd = -(evaluate(p, calc(damping="zero"))[0] - evaluate(m, calc(damping="zero"))[0]) / (2 * h)
            assert abs(fd - f[i, k]) < 1e-6, (i, k, fd, f[i, k])


def test_zero_stress_finite_difference():
    """Analytic zero-damping stress agrees with finite differences of the energy under homogeneous strain."""
    a = _distorted_cell()
    _, _, s = evaluate(a, calc(damping="zero"))
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    h = 1e-5
    for n, (i, j) in enumerate(voigt):
        e = []
        for sign in (1, -1):
            eps = np.zeros((3, 3))
            if i == j:
                eps[i, i] = sign * h
            else:                       # engineering shear h split symmetrically
                eps[i, j] = eps[j, i] = sign * h / 2
            b = a.copy()
            b.set_cell(a.cell.array @ (np.eye(3) + eps), scale_atoms=True)
            e.append(evaluate(b, calc(damping="zero"))[0])
        fd = (e[0] - e[1]) / (2 * h) / a.get_volume()
        assert abs(fd - s[n]) < 1e-7, (n, fd, s[n])


def test_zero_r_zero_convergence():
    """The real-space correction is converged at the default r_zero = 20 A (dense metal, worst case: large Cu BJ radius)."""
    a = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((2, 2, 2))
    e20 = evaluate(a, calc(damping="zero", r_zero=20.0))[0]
    e26 = evaluate(a, calc(damping="zero", r_zero=26.0))[0]
    assert abs(e20 - e26) / len(a) < 1e-6, (e20, e26, (e20 - e26) / len(a))
