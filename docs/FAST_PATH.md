# FourierD3 fast path (`fourierd3.fast`)

A drop-in wrapper around the **unchanged** FourierD3 model (`FourierD3.forward`) that removes per-step overhead in
molecular dynamics. The dispersion physics code is not modified.

## What it does
- **Fixed-cell cache (FAST path, used when `stress` is not requested).** Cell-dependent setup (Ewald k-vectors and
  Green's function, or the SPME mesh interpolator and k-space influence function, cell inverse/volume) is computed once
  per cell instead of every step. Forces come from autograd w.r.t. positions only (no strain graph).
- **CUDA graphs.** With the cell-dependent work cached, the D3 forward + backward is capturable, and
  `torch.compile(mode="reduce-overhead")` replays the whole step as one graph. The neighbour list is padded to a fixed
  capacity with masked zero-shift self-pairs, which the CN mask excludes (they contribute exactly 0).
- **STRESS path (used when `stress` is requested, e.g. NPT).** The original strain-autograd code, unchanged.
- **NPT-safe.** A cell change invalidates the cache; while the cell keeps changing the step runs eager (no re-capture,
  no recompilation storm); the compiled path resumes after `stable_n` (default 20) unchanged-cell calls.
- **GPU neighbour list** correct for unwrapped coordinates, triclinic cells and partial PBC (fp64 connectivity,
  integer image shifts), checked edge-for-edge against `mace.data.neighborhood.get_neighborhood`.
- **Safety.** Cache keyed on cell (exact), atomic numbers and order, atom count and PBC; transactional updates
  (invalid during mutation, valid only after success); the ASE property transaction is locked; results are cleared
  before every calculation so a failed step never returns a previous frame's results.

## Results (production 684-atom Cu/water/Mo slab, H200, fp32 D3, symmetrix MACE)
| | original | fast path | |
|---|---|---|---|
| D3 per walker, 2 walkers/GPU (skin-reused NL) | ~9.9 ms | ~1.5 ms | ~6.6x |
| D3 with NL rebuilt every step, 2 walkers | 17.2 ms | 6.7 ms | 2.6x |
| D3 in NPT (stress every step) | 15.9 ms | 8.0 ms | ~2x, 0 recompiles |
| full MetaD step, 2 walkers/GPU (estimate) | 19.4 ms | ~9.4 ms | ~2x |

Accuracy against an independent upstream calculator: max |dF| 4.5e-8 eV/Å, |dE| 2.2e-8 eV/atom; stress path within the
fp32 run-to-run noise floor. fp64 finite-difference self-consistency passes. Paired 20 ps NVE drift (3 states x 2
timesteps) is indistinguishable from the original. A 2 ps NPT replay stays within the parity limits.

References: NVIDIA-BioNeMo FourierD3 0.3.0 (JAX) is ~1.3x faster still and physics-consistent; NVIDIA ALCHEMI
`fourier_dftd3` was ~1.5x slower than this path on the same system.

## Tests
`tests/gpu_validation/` are GPU integration tests on the production system. They need CUDA, a MACE model and the test
frames: set `FD3_TEST_RUNDIR`, `FD3_TEST_MODEL` and `FD3_TEST_MANIFEST`. Each exits non-zero on failure.
