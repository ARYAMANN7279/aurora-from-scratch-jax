# Stage 5 Report: 3-D windowed and shifted-window attention

Spec: `handouts/stage05_swin.md`. Code: `src/aurora/swin.py`. Tests: `tests/test_stage5_gates.py`.
Reference: Aurora's own `Swin3DTransformerBlock` (microsoft-aurora 2.0.1, torch 2.14.0; see
`reference_env.txt`), run by `make_reference.py` in a separate environment. Its outputs are in
`tests/fixtures/swin3d_ref.npz`, so the tests need no PyTorch.

## Verdict

**Accepted.** Our block matches Aurora's to **2.8e-7 in float32** (the plan's bar is 1e-5) and
**5.2e-16 in float64**, which is machine precision, on four cases (two shapes × unshifted/shifted).
All 9 gates pass, and both predictions HIT. (Gate W10 was added on 28 Sep 2026; see Bugs fixed.)

## Gates

Command: `.venv/bin/python -m pytest tests/test_stage5_gates.py -q -s` → 9 passed.

| Gate | Measured | Bar |
|---|---|---|
| W1 window partition / reverse vs Aurora | bitwise | bitwise |
| W2 shifted-window mask, periodic (F, F, T), vs Aurora's `warped=True` mask | exact | exact |
| W3 **block vs Aurora, float32** (4 cases) | **2.8e-7** | ≤ 1e-5 |
| W4 block vs Aurora, float64 (4 cases) | 5.2e-16 | ≤ 1e-12 |
| W5 doubly periodic shifted block == roll ∘ unshifted ∘ roll⁻¹ | 0 | ≤ 1e-12 |
| W6 translation by whole windows commutes (3 shift settings) | 0 | ≤ 1e-12 |
| W7 identity at initialisation (zero AdaLN modulation) | exact | exact |
| W8 gradient vs finite differences | 1.0e-7 | ≤ 1e-6 |
| W9 `adjust` vs Aurora on 6 cases; a non-divisible shape raises | exact; raises | — |

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| T1 | W3 ≤ 1e-5 (guess 1e-6) | 2.8e-7 | **HIT** |
| T2 | W4 ≤ 1e-12 (guess 1e-14) | 5.2e-16 | **HIT** |

## Bugs fixed
* **Reference fixture (harness bug).** The first fixture saved the weights after a float64 →
  float32 → float64 round trip, while the float64 reference output used the unrounded weights. W4
  showed a systematic 4e-8 gap. `make_reference.py` now saves the weights before any cast and runs
  the float32 case on a deep copy. After the fix W4 is 5e-16.

* **Masked shift under `jit` (found 28 Sep 2026).** `shift_mask` converted a `jnp` array back to
  numpy. That works eagerly (all of W1–W9) but fails inside `jax.jit`. The path is reached only when
  a non-periodic axis is shifted, which in the full model means the level axis with L > 3: E4's
  L = 6 variant. A real-data smoke test of the E4 configs found it before any GPU run reached it.
  The mask is now built with numpy alone, and the values are identical. New gate **W10**: the masked
  block under `jit` matches Aurora's reference to < 1e-12 (float64), and jit equals eager with a
  shifted level axis. The Stage 5 verdict is unchanged.

## Design note
Aurora's mask is periodic only in longitude (W). This project's domain is periodic in both x and y,
so the mask is generalised by a per-axis `periodic` flag (a periodic axis forms a single group).
The generalisation reproduces Aurora's mask exactly for `(F, F, T)` (W2). The doubly periodic
setting is verified independently by W5 and W6.
