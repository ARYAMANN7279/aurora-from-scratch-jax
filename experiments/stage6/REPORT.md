# Stage 6 Report: 3-D Swin U-Net backbone

Spec: `handouts/stage06_backbone.md` (including amendment A1). Code: `src/aurora/backbone.py`.
Tests: `tests/test_stage6_gates.py`. Reference: Aurora's `Swin3DTransformerBackbone`
(microsoft-aurora 2.0.1), via `make_reference.py` → `tests/fixtures/backbone_ref.npz`.

## Verdict

**Architecture verified; the plan's overfit test is not yet met.** The whole backbone reproduces
Aurora's to **3e-16 (float64)** and 1.9e-7 (float32) on two configurations, and all 6 gates pass.
The pre-registered overfit target was missed: one-step error 4.1e-2 after 4000 steps, against
< 2e-2. The loss was still falling when the cosine schedule ended. A 16k-step diagnostic run is
queued to establish whether the model can reach near-zero; its result will be added here.

## Gates

Command: `.venv/bin/python -m pytest tests/test_stage6_gates.py -q -s` → 6 passed.

| Gate | Measured | Bar |
|---|---|---|
| U1 patch merging and splitting vs Aurora | 1.7e-16 | ≤ 1e-12 |
| U2 **full backbone vs Aurora, float64** (2 configurations) | **3.1e-16** | ≤ 1e-12 |
| U3 full backbone vs Aurora, float32 | 1.9e-7 | ≤ 1e-5 |
| U4 output width 2D; concatenated skip independent of decoder weights | exact | exact |
| U5 translation by 2^{S−1}·window patches commutes (doubly periodic mode) | 0 | ≤ 1e-12 |
| U6 gradient vs finite differences (A1: h = 1e-4, mixed tolerance) | 3.3e-8 | ≤ 1e-6 |

**U6 history:** the pre-registered form (h = 1e-6, pure relative) **failed** at 5.5e-5. A first
fix (mixed tolerance, h = 1e-6) **also failed**, at 6.7e-6. Both failures are the finite-difference
roundoff floor: the absolute error grows as h shrinks, while the analytic gradient is unchanged.
The amended check (h = 1e-4, mixed) passes at 3.3e-8. See the spec, amendment A1.

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| V1 | U2 ≤ 1e-12 | 3.1e-16 | **HIT** |
| V2 | U3 ≤ 1e-5 | 1.9e-7 | **HIT** |
| V3 | persistence error on the training triples in [0.05, 0.4] | 0.215 | **HIT** |
| V4 | trained one-step error < 0.1 × persistence and < 2e-2 (4000 steps) | 4.08e-2 (0.19 × persistence) | **MISS** |

Overfit run: 5.0M parameters (D = 64, S = 3), 1183 s on the shared CPU.
Learning curve (training triples): 0.130 at step 250, 0.067 at 1500, 0.049 at 2500, 0.041 at 4000.

## Commands
```
.venv/bin/python -m pytest tests/test_stage6_gates.py -q -s
refenv/bin/python experiments/stage6/make_reference.py      # reference environment only
.venv/bin/python experiments/stage6/overfit.py               # V3/V4
.venv/bin/python experiments/stage6/overfit.py --steps 16000 # diagnostic (queued)
```
