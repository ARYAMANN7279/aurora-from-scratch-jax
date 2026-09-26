# Stage 8 Spec: Autoregressive rollout and MAE loss

**Plan acceptance test (PROJECT_PLAN.md §5):** a stable 50-step rollout without blow-up.
**Written:** 26 Sep 2026, before any Stage 8 code. §§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Design

* `rollout(p, X^{−1}, X^0, n)`: `lax.scan` over n steps, carrying `(X^{t−1}, X^t)`; each step
  applies the Stage 7 model and feeds its own prediction back. Returns `(n, 4, N, N)`.
* **Loss:** `mae(pred, target, w) = Σ_v w_v · mean|pred_v − target_v| / Σ_v w_v` on normalised
  fields, equal weights by default. Aurora trains with a (weighted) MAE; MSE is kept as the
  alternative for the E4 ablation.
* **Models compared:** the Stage 7 model (MSE, 3000 steps) and the same configuration trained with
  MAE (same data, schedule, steps and key).

## 2. Gates — `tests/test_stage8_gates.py` (float64)

| Gate | Test | Pass if |
|---|---|---|
| Q1 | `rollout(n)` equals n explicit calls of `model.apply` feeding back | ≤ 1e-12 |
| Q2 | `rollout` under `jit` with n static; n = 1 equals one `apply` | exact / ≤ 1e-12 |
| Q3 | `mae` equals a NumPy reference, with and without variable weights | ≤ 1e-14 |
| Q4 | the gradient of `mae` with respect to the prediction equals `w_v · sign(d) / (count · Σw)` | ≤ 1e-14 |

## 3. Measurement

* **Initial conditions:** evaluation seeds 6 and 7 of re100_kf4 and re300_kf4 (64²), start frames
  t₀ ∈ {50, 250}: 8 rollouts of 50 steps (10 time units) per model.
* **Per step:** relative L2 error against the truth; kinetic energy `½⟨u² + v²⟩` and enstrophy
  `½⟨ω²⟩` of the prediction (de-normalised), each divided by the regime's climatological value
  (the mean over the evaluation trajectories); all values finite.
* **Baselines:** persistence (`X^{t₀}` held fixed) and climatology (the time-mean field; relative
  error ≈ 1 by construction).

| # | Measurement | Prediction |
|---|---|---|
| Z1 | every rollout, both models: finite at all 50 steps; E ratio in [0.5, 2] and Z ratio in [0.3, 3] at every step | HIT for both (the acceptance test) |
| Z2 | one-step relative error, MAE model / MSE model, each regime | in [0.7, 1.3] |
| Z3 | step-5 relative error, model vs persistence, each regime (mean over rollouts) | model < persistence |
| Z4 | step-50 relative error, each regime (mean over rollouts) | ≤ 1.6 (two independent samples give √2 ≈ 1.41) |

**Stage 8 acceptance:** Q1–Q4 pass, and Z1 HIT.
