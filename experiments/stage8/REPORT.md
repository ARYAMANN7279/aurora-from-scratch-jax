# Stage 8 Report: Autoregressive rollout and MAE loss

Spec: `handouts/stage08_rollout.md`. Code: `src/aurora/rollout.py`. Tests: `tests/test_stage8_gates.py`.
Measurement: `rollout_eval.py` → `results/rollout.json`.

## Verdict

**The plan's test is met by the MSE model; the pre-registered Z1 is a MISS because of the MAE
model.** The MSE-trained model rolls out 50 steps (10 time units, ~7–12 decorrelation times) from
all 8 initial conditions with energy and enstrophy staying physical (E 0.54–1.34×, Z 0.56–1.91×
climatology). The **MAE-trained** model has the same one-step accuracy, but **3 of its 4 Re300
rollouts blow up** after step 25–38 (E up to 19× climatology). Z1 predicted both models stable.
Gates Q1–Q4 pass.

## Gates

| Gate | Measured | Bar |
|---|---|---|
| Q1 `rollout(n)` vs an explicit loop | 8.4e-15 | ≤ 1e-12 |
| Q2 jit with static n; n = 1 equals one `apply` | exact | ≤ 1e-12 |
| Q3 `mae` vs NumPy (with and without variable weights) | 4.4e-16 | ≤ 1e-14 |
| Q4 `mae` gradient = w·sign(d)/(count·Σw) | 0 | ≤ 1e-14 |

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| Z1 | both models: finite; E in [0.5, 2] and Z in [0.3, 3] at all 50 steps | MSE: E [0.54, 1.34], Z [0.56, 1.91] (all 8 stable). MAE: E up to 19.2, Z up to 8.5 (3 of 4 Re300 rollouts leave the bounds) | **MISS** |
| Z2 | one-step error MAE / MSE in [0.7, 1.3] | 0.96 (Re100), 1.01 (Re300) | **HIT** |
| Z3 | step-5 error below persistence, each model and regime | MSE 0.263 vs 0.810, 0.386 vs 0.971; MAE 0.246 vs 0.810, 0.415 vs 0.971 | **HIT** |
| Z4 | step-50 error ≤ 1.6 | MSE 1.50 / 1.35; MAE 1.31 / **2.78** | **MISS** (MAE, Re300) |

## The MAE instability, per rollout

| Model | Regime | seed / t₀ | E at step 10 / 25 / 50 | first step out of bounds |
|---|---|---|---|---|
| MAE | Re300 | 6 / 50 | 0.72 / 2.11 / 19.2 | 25 |
| MAE | Re300 | 6 / 250 | 1.10 / 1.51 / 8.84 | 32 |
| MAE | Re300 | 7 / 250 | 0.78 / 0.98 / 7.10 | 38 |
| MAE | Re300 | 7 / 50 | 0.73 / 0.73 / 1.48 | — |
| MAE | Re100 | all 4 | within [0.79, 1.31] | — |
| MSE | both | all 8 | within [0.54, 1.34] | — |

**Reading.** At equal one-step accuracy, the MSE-trained model is stable over 50 steps and the
MAE-trained model is not, at the more turbulent regime. One plausible mechanism: MAE penalises
rare large errors less than MSE, so the high-amplitude errors that seed energy growth under
feedback are not suppressed. **Caveat:** this is one training seed per loss, so it is a lead for
E4 (MAE vs MSE, 3 seeds) and for Stage 9 (does pushforward remove it?), not a conclusion.
Aurora trains with MAE, which makes this worth testing properly.

## Commands
```
.venv/bin/python -m pytest tests/test_stage8_gates.py -q -s
.venv/bin/python experiments/stage8/rollout_eval.py
```
