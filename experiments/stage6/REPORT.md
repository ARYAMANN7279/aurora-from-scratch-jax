# Stage 6 Report: 3-D Swin U-Net backbone

Spec: `handouts/stage06_backbone.md` (including amendment A1). Code: `src/aurora/backbone.py`.
Tests: `tests/test_stage6_gates.py`. Reference: Aurora's `Swin3DTransformerBackbone`
(microsoft-aurora 2.0.1), via `make_reference.py` → `tests/fixtures/backbone_ref.npz`.

## Verdict

**Accepted on the plan's test, with the pre-registered step budget missed.** The whole backbone
reproduces Aurora's to **3e-16 (float64)** and 1.9e-7 (float32) on two configurations, and all 6
gates pass. The pre-registered overfit target was missed at 4000 steps (4.1e-2 vs < 2e-2; V4 MISS).
A diagnostic run with 16,000 steps (same model and recipe, cosine schedule over 16k) **overfits the
trajectory to 1.50e-2, which is 0.070× persistence**. That meets both parts of the original
target, so the miss was the step budget, not the model's capacity.

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
**Diagnostic, 16,000 steps** (`results/overfit_steps16000.json`): 1.50e-2 (0.070 × persistence),
4700 s. Curve: 0.068 at 1500, 0.045 at 3500, 0.026 at 7250, 0.020 at 9750, 0.016 at 12750,
0.015 at 16000.
Learning curve (training triples): 0.130 at step 250, 0.067 at 1500, 0.049 at 2500, 0.041 at 4000.

## Commands
```
.venv/bin/python -m pytest tests/test_stage6_gates.py -q -s
refenv/bin/python experiments/stage6/make_reference.py      # reference environment only
.venv/bin/python experiments/stage6/overfit.py               # V3/V4
.venv/bin/python experiments/stage6/overfit.py --steps 16000 # diagnostic
```
