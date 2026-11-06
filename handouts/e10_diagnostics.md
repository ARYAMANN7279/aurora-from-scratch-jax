# E10 Spec: Diagnostics of E1's trained models (where the error lives, what limits it)

**Status: pre-registered 6 Nov 2026, before any E10 measurement.** After the project review, Prashant
suggested diagnosing what limits accuracy rather than only reporting it: which loss term limits it,
whether the output layer is converged, where the remaining error lives, and checks built on the
physics. The owner chose the three cheapest, plus a physics check. E10 trains nothing: it measures
the 30 E1 models whose weights are kept on the lab machine.

---

## 1. Questions

* **(a) Where does the remaining error live?** Split the one-step error of each model by field
  (u, v, p, ω) and by scale (the wavenumber shell |k| of a 2-D Fourier transform). Does pretraining
  remove error at every scale, or mainly at some?
* **(b) Which loss term limits accuracy?** The training loss is the mean squared error over the four
  normalised fields, weighted equally. Which field's term is largest at the end of training? Is the
  limit the fit to the training trajectories or the step from them to unseen ones (the gap between
  training and evaluation error)? That decides whether longer training or a different weighting can
  help.
* **(c) Is the output layer converged?** The model's last layer is linear: four heads, one per
  field, each a 64 × 16 matrix and a bias, applied to every level's decoded tokens and summed over
  the levels. Holding the rest of the network fixed, the best heads for the training loss are a
  least-squares problem with an exact solution. How far is each trained model from it?
* **(d) Does the output respect the physics?** The true fields are divergence-free
  (∂u/∂x + ∂v/∂y = 0) and ω = ∂v/∂x − ∂u/∂y. The model predicts all four fields separately, with
  neither built in. How far do its predictions break them?

## 2. Models and data

* **Models:** E1's 30 runs with weights kept (`data/runs/<config_hash>/params.npz`):
  * scratch, seeds 0–2 (`e1_scratch_s*`);
  * fine-tuned, 9 cells (n = 2, 8, 32 pretraining regimes × T = 32, 64, 128 pretraining
    trajectories in all) × seeds 0–2 (`e1_ft_*`).
* **Data:** `e1_v1`, `re300_kf4`, 64², 501 frames per trajectory. Training trajectories 0–1,
  evaluation trajectories 2–5, as in E1. Each model's own normalisation (mean and std per field,
  stored with its weights) is used.
* **Pairs:** every one-step pair (frames k − 1, k → k + 1), as in the harness's evaluation.
* **Where:** CPU (no GPU is needed for evaluation only).

## 3. Measures

* **Field shares (a, b):** for field f, S_f = Σ (prediction − truth)² over that field's normalised
  values, divided by the sum over all four fields. The shares sum to 1; equal weighting would give
  0.25 each. Computed on training and on evaluation pairs.
* **Error by scale (a):** the 2-D FFT of the normalised error and of the normalised truth, summed
  over fields, pairs and the integer shell |k| (rounded). The **shell error ratio** R(k) =
  error power in shell k / truth power in shell k.
* **Generalisation gap (b):** G = evaluation MSE / training MSE, both in normalised units, all
  fields.
* **Output-layer refit (c):** the exact least-squares heads on all training pairs (the network's
  other weights fixed). Reported: the change in training MSE, and in the evaluation relative error
  (E1's `rel_err`), as 1 − refit / trained.
* **Physics (d):** on evaluation pairs, in physical units (de-normalised), with spectral derivatives
  on the 2π-periodic grid:
  * **divergence ratio** D = ‖∂u/∂x + ∂v/∂y‖ / ‖∇u‖, with ‖∇u‖² = Σ of the four squared velocity
    gradients;
  * **curl mismatch** C = ‖ω − (∂v/∂x − ∂u/∂y)‖ / ‖ω‖.
  * For the truth both are at float32 round-off, which gate G2 checks.
* **Pretraining's advantage at a scale:** 1 − R_ft(k) / R_scratch(k), paired by seed, for the cell
  E1 reports as best (n = 32).

## 4. Gates (all must pass before any measurement)

| # | Gate |
|---|---|
| G1 | The evaluation code reproduces E1's stored `rel_err` for all 30 models to 1e-4 (relative) |
| G2 | On the true fields D and C are below 1e-4 |
| G3 | On a synthetic field the shell spectrum sums to the field's total power (Parseval) to 1e-6 |
| G4 | The refit reproduces the trained model exactly when the heads are already the least-squares solution (a model refit twice changes by < 1e-6 the second time) |
| G5 | The refit never raises the training MSE |

`tests/test_e10_diagnostics.py`.

## 5. Predictions (fixed before any measurement)

* **P1 (field shares).** ω has the largest share of the evaluation error and p the smallest, in all
  30 models; ω's share is above 0.35 in every model.
  * Why: normalised to unit spread, ω carries most small-scale content and changes most per step;
    p is the smoothest field.
* **P2 (error by scale).** R(k) at 12 ≤ |k| < 20 is at least 3× R(k) at 1 ≤ |k| < 5, in all 30
  models. The remaining error is a small-scale error.
* **P3 (pretraining by scale).** For n = 32 (all three T, all seeds), pretraining's advantage in
  8 ≤ |k| < 20 is larger than in 1 ≤ |k| < 5.
* **P4 (gap).** Scratch's G is above 2 in every seed, and every fine-tuned model's G is below the
  scratch model's of the same seed. Scratch is limited by generalisation from two trajectories,
  not by the fit; longer training alone would not help it.
* **P5 (output layer).** The refit lowers the training MSE by less than 3%, and changes the
  evaluation `rel_err` by less than 1% in either direction, in every model. The output layer is
  converged.
* **P6 (physics).** D is between 1e-3 and 1e-1 in every model, and fine-tuned models have lower D
  than scratch (same seed) for n = 32. C is below ω's own one-step relative error in every model.

A miss stays a miss; any change after the first measurement is dated as an amendment below.
