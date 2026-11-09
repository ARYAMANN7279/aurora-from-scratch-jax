# E11 Spec: Predict ω only, derive u, v and p exactly

**Status: pre-registered 6 Nov 2026, before any E11 run.** E10 found that ω holds 62–76% of the
one-step error and that the model's four outputs break the flow's constraints by about 10%
(divergence 9–15%, curl mismatch 10–17%). On this periodic grid u, v and p are fixed by ω:
ψ = −∇⁻²ω, u = ∂ψ/∂y, v = −∂ψ/∂x, and ∇²p = 2(u_x v_y − u_y v_x). The owner asked to test the
first idea E10 pointed to: let the model predict ω only and derive the other three fields exactly.

---

## 1. Questions

* **(a)** Does deriving u, v and p from the predicted ω lower the one-step error over all four
  fields, at the same training objective?
* **(b)** Does a physically consistent output also keep rollouts closer to the truth?
* **(c)** Should the loss then sit on ω alone, or on all four derived fields?

## 2. The change (`src/aurora`, off by default)

* **`model.derive_from_omega: true`.** The network is unchanged (same 5,106,688 parameters, same
  weights for a given seed). Its ω output is kept; its u, v and p outputs are replaced, inside
  `model.apply`, by the fields derived from that ω in physical units (de-normalised with the
  harness's mean and std, derived spectrally on the 2π grid, then normalised again). The u, v and p
  heads then get no gradient.
  * The harness passes the normalisation in `ModelConfig.derive`; the model uses it only with this
    flag.
  * Every prediction path uses it: training, one-step evaluation and every rollout.
* **`train.loss_fields`:** the fields the MSE averages over (default: all four).

## 3. Gates (all must pass before any run)

| # | Gate |
|---|---|
| D1 | Defaults: `ModelConfig().derive` is empty, and a 40-step harness run without the new options reproduces the previous commit's harness bitwise (parameters, final loss, metrics) |
| D2 | From the true ω (normalised as in training), the derived u and v match the true ones to 1e-5 and p to 1e-3 (relative, normalised units) |
| D3 | A model with the flag outputs fields whose divergence ratio and curl mismatch are below 1e-4 |
| D4 | With the flag and `loss_fields: ["omega"]`, every backbone gradient is finite and the ω head's is non-zero; the u, v and p heads get exactly zero |
| D5 | Both new arms run through the harness for a few steps |

`tests/test_e11_omega.py`.

## 4. Task and arms

* **Task:** E1's scratch task, exactly as in E7–E9: train on `re300_kf4` trajectories 0–1, evaluate
  on 2–5 (`e1_v1`, 64², 501 frames each); 22,000 steps, batch 16, lr 1e-3, cosine schedule; the E1
  model. Evaluation as in E8: one-step error and a 50-step rollout from every 20th frame from 50 on
  (84 rollouts per run).
* **Arms,** 3 seeds each (0–2), same starting weights and batches per seed:

| Arm | Change from E8's arm B |
|---|---|
| **B** | none (the control, rerun now) |
| **W** | `derive_from_omega`; loss over all four (derived) fields, as B |
| **WO** | `derive_from_omega`; loss on ω only |

* **Where:** Kaggle, two T4 cards, software pinned as for E6–E9; float32 with matmul precision
  `highest`. Partner arms of a seed run at the same time on the two cards.
* **Cross-check:** B is E8's arm B rerun; its one-step errors are compared with E8's rows.

## 5. Metrics

* **One-step relative error** over all four fields (E1's `rel_err`), and per field (from the saved
  weights, with E10's `diag.py` measures).
* **Drift:** E8's 20-step drift from 84 rollouts (`rollout_drift_many`) and the 50-step drift.
* **Advantage** of arm Y over B: 1 − m(Y) / m(B), paired by seed, mean ± SE over 3 seeds.
* **Physics:** E10's divergence ratio and curl mismatch.
* **Cost:** `train_s`.

## 6. Predictions (fixed before any run)

* **P1.** W lowers the one-step `rel_err` against B by at least 5% in every seed.
* **P2.** W lowers the 20-step drift (`rollout_drift_many`) against B, in every seed.
* **P3.** WO has a lower ω one-step error than W, and W a lower u one-step error than WO, in every
  seed.
* **P4.** WO's overall one-step `rel_err` is not better than W's by more than 2% (mean over seeds).
* **P5.** B reproduces E8's arm-B one-step error within 1% in every seed.
* **P6.** W's `train_s` is within 10% of B's (mean over seeds).

A miss stays a miss; any change after the first run is dated as an amendment below.

## Amendments

* **A1 (6 Nov 2026, before any run).** Gate D3 failed on random test outputs: a predicted ω can
  carry a mean and Nyquist modes, which no velocity can reproduce (a periodic ω has zero mean, and
  the data's Nyquist modes are zero). `derive_from_omega` now first removes both from the predicted
  ω and returns that projected ω with the derived u, v and p, so all four fields are mutually
  consistent. Gate D2 accordingly asks the projected true ω to match the true ω to 1e-5 (it was
  "unchanged"). Arms, metrics and predictions are unchanged.
