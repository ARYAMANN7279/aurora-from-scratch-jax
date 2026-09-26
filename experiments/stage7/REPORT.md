# Stage 7 Report: Full model (encoder → backbone → decoder)

Spec: `handouts/stage07_model.md` (including amendment A1). Code: `src/aurora/model.py`.
Tests: `tests/test_stage7_gates.py`. Training: `train_onestep.py` → `results/onestep_{mse,mae}.json`.

## Verdict

**Accepted.** All gates pass. The model is exactly persistence at initialisation, and after 3000
steps it predicts one step ahead on **unseen trajectories** at **0.32×** (Re100) and **0.46×**
(Re300) the persistence error. That meets the pre-registered N3 (< 0.5). N4 (absolute error
< 0.1) is met at Re100 but **missed** at Re300 (0.146).

## Gates

Command: `.venv/bin/python -m pytest tests/test_stage7_gates.py -q -s` → 5 passed (M2 and M6 are
one test).

| Gate | Measured | Bar |
|---|---|---|
| M1 identity at initialisation (64², 128²) | exact | exact |
| M2 all four variables out, for input subsets {all}, {u, v}, {ω}, {u, v, p} | exact | exact |
| M3 one parameter set on 64² and 128² under jit | shapes exact | exact |
| M4 translation equivariance with the position encoding zeroed (A1), 128², 64-point shift | 5.0e-15 | ≤ 1e-10 |
| M5 gradient vs finite differences (h = 1e-4, mixed) | 1.5e-7 | ≤ 1e-6 |
| M6 an absent variable gets residual base 0 | exact | exact |

**M4 history:** as pre-registered (no zeroing) M4 **failed** at 8.28. The model carries an absolute
position encoding, which it should, because the forcing breaks y-translation symmetry. The spec had
overlooked this. See amendment A1.

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| N1 | parameter count (reported) | 5,106,688 | — |
| N2 | persistence error: Re100 in [0.1, 0.35], Re300 in [0.15, 0.45] | 0.218 / 0.315 | **HIT** |
| N3 | trained / persistence < 0.5, each regime | 0.322 / 0.463 | **HIT** |
| N4 | trained error < 0.1, each regime | 0.070 / 0.146 | **MISS** (Re300) |

Learning curve (ratio to persistence on the evaluation seeds, Re100 / Re300):
250: 0.73 / 0.80 · 750: 0.52 / 0.63 · 1250: 0.43 / 0.54 · 1750: 0.37 / 0.50 · 2250: 0.34 / 0.48 ·
2750: 0.33 / 0.46 · final: 0.32 / 0.46. Still improving slowly at the end of the cosine schedule.

The same configuration trained with **MAE** (for Stage 8) reaches 0.067 / 0.148, a ratio of
0.31 / 0.47.

## Run notes
* 1.03 s per step (batch 8, 64², float32) on the shared 4-core CPU; 3748 s in total.
* The first attempt was killed by a container restart at about step 1500. Training now checkpoints
  every 250 steps. The relaunched run reproduced the killed run's step-250 loss bitwise
  (0.018555916845798492), so resuming does not change results.

## Commands
```
.venv/bin/python -m pytest tests/test_stage7_gates.py -q -s
.venv/bin/python experiments/stage7/train_onestep.py --loss mse
.venv/bin/python experiments/stage7/train_onestep.py --loss mae
```
