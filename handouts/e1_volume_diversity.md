# Experiment E1 Spec: Volume vs diversity of pretraining data (RQ1)

**Status: FROZEN 28 Sep 2026.** Launched on the owner's instruction; the open points in the last section were not resolved before launch and stay open for the report.
**Question (PROJECT_PLAN.md RQ1):** at a fixed number of pretraining trajectories, is it better to
have many trajectories from few flow regimes, or few trajectories from many regimes?
**Tools:** Stage 2 generator, Stage 7 model, Stage 11 harness (one config → one run → one row).
**Written:** 27 Sep 2026.

---

## 1. Regimes

A regime is `(Re, k_f)`, with `A = 1` and `α = 0.1`. Distance between regimes is measured in
`(log Re, k_f)`.

* **Held-out target (interior, decision of 26 Sep):** `(Re, k_f) = (300, 4)`. It is never used for
  pretraining.
* **Pretraining regime sets, nested** (2 ⊂ 8 ⊂ 32), each enclosing the target:

| n | Regimes | How the target sits |
|---|---|---|
| 2 | (200, 3), (450, 5) | exact midpoint in log Re and in k_f |
| 8 | Re ∈ {160, 200, 450, 560} × k_f ∈ {3, 5} | inside, no neighbour closer than the 2-set's |
| 32 | Re ∈ {125, 160, 200, 250, 350, 450, 560, 700} × k_f ∈ {3, 4, 5, 6} | inside, with neighbours (250, 4) and (350, 4) |

**Known confound:** in this design, more regimes also means a closer nearest neighbour to the
target (only in the 32-set). Section 5 reports the result against nearest-neighbour distance as
well. A follow-up can control distance explicitly.

## 2. Data

* **Generation grid per regime:** measured with the Stage 2 rule. The relative difference between
  N and 2N must be ≤ 1e-3 at t = 1 in the 64² output band, with candidate grids {128, 256, 512}.
  `experiments/e1/measure_ngen.py` does this.
* **Per trajectory:** spin-up 50, record 100 time units, frame every Δ = 0.2 (501 frames), **64²
  only**. The post-spin-up state is saved with each trajectory (Stage 2 finding).
* **Cells:** total trajectories `T ∈ {32, 64, 128}` × `n ∈ {2, 8, 32}`, i.e. `T/n` trajectories per
  regime, seeds `0 … T/n − 1`. Seeds are nested, so a smaller cell uses a prefix of a larger one.
* **Trajectories to generate:**
  * the 2-set needs 64 per regime: 128;
  * the 6 further 8-set regimes need 16 each: 96;
  * the 24 further 32-set regimes need 4 each: 96;
  * the target needs 6: seeds 0–1 for fine-tuning, seeds 2–5 for evaluation.
  * **Total: 326 trajectories, ≈ 10.7 GB at 64².**

## 3. Runs (all through the Stage 11 harness)

The Stage 7 default model (D = 64, L = 3, 5.1M parameters), MSE loss (Stage 8: MAE was unstable
in rollout), float32, batch 16, cosine learning-rate decay.

| Run | Data | Steps | lr |
|---|---|---|---|
| **pretrain(n, T, s)** | the cell's regimes; target listed as `test`, so the harness refuses any leak | 20,000 | 1e-3 |
| **finetune(n, T, s)** | init_from pretrain; target seeds 0–1 | 2,000 | 1e-4 |
| **scratch(s)** | target seeds 0–1 only | 22,000 (matched optimiser steps) | 1e-3 |

A fine-tune keeps the per-channel normalisation of its pretrain run (`data.norm_from_init`);
pretrain and scratch runs normalise with their own training data.

Training seeds `s ∈ {0, 1, 2}`: 27 pretrain + 27 finetune + 3 scratch = **57 runs**. Every
evaluation is on target seeds 2–5: the one-step relative error, and the rollout drift (mean
relative error over steps 1–20 from t₀ = 50).

**Pretraining advantage** of a cell: `adv = 1 − e_finetune / e_scratch`, paired by seed. Report the
mean ± SE over the 3 seeds, for both the one-step metric and the drift metric.

**Compute estimate (GPU box, one RTX PRO 6000):** generation ≈ 1–2 min per trajectory, about
8 GPU-hours in total; training ≈ 20–30 ms per step, about 7 GPU-hours. Over 4 cards, roughly
half a day end to end.

## 4. Pre-registered predictions

| # | Prediction |
|---|---|
| E1-P1 | pretraining helps in every cell: adv > 2 SE on the one-step metric |
| E1-P2 | within each row (fixed T), adv rises with diversity: adv(2) < adv(8) < adv(32) |
| E1-P3 | diversity matters more than volume: going from n = 2 to 32 at fixed T raises adv more than going from T = 32 to 128 at fixed n (mean over the matching rows and columns) |
| E1-P4 | best cell adv in [10%, 40%]; the (n = 2, T = 32) cell adv in [0%, 15%] |
| E1-P5 | zero-shot on the target (pretrained, before fine-tuning): the n = 32 cells beat the n = 2 cells at every T |
| E1-P6 | the ordering of cells by the drift metric matches the one-step ordering (Spearman ρ ≥ 0.7 over the 9 cells) |

## 5. Analysis and report

* A 3 × 3 table of adv ± SE, one-step and drift, plus the zero-shot error.
* adv against distance from the target to its nearest pretraining regime, to expose the confound
  in §1.
* Every number traced to its row in `experiments/results.jsonl` (config hash).

## 6. Tooling

| Step | Command |
|---|---|
| N_gen per regime | `experiments/e1/measure_ngen.py` (resumable; several `--out` files may be merged) |
| configs | `experiments/e1/make_configs.py --ngen experiments/e1/results/ngen_*.json` → `experiments/e1/configs/` |
| data | `experiments/stage2/generate.py --config experiments/e1/configs/e1_v1.json` (lanes: `--shard i --n-shards k`), then `--finalize` |
| runs | `experiments/run.py experiments/e1/configs/pretrain.json experiments/e1/configs/scratch.json --skip-done --shard i --n-shards k`, then the same for `finetune.json` |

The design (regimes, cells, budgets) lives in one place, `experiments/e1/e1_design.py`, and
`tests/test_e1_tooling.py` checks it against this spec: 326 trajectories, 57 runs, target never in
pretraining, every fine-tune pointing at its own pretrain run.

## 7. Open points for review
1. **Regime grid:** the Re range 125–700 and k_f 3–6. Enough spread, or wider?
2. **Budgets:** 20k pretraining steps, the same for every cell, i.e. compute is fixed and data varies.
   Alternative: scale the steps with T.
3. **Distance confound:** accept and report it (the current plan), or add a distance-matched
   32-regime control that excludes the target's nearest neighbours?

## F1. Training curves requested on 17 Oct 2026

**Registered before the follow-up runs.** The original E1 predictions and verdicts are unchanged.
Prashant requested training curves and contours of the truth, zero-shot prediction and fine-tuned
prediction. Rerun cells (n = 2, T = 32) and (n = 32, T = 32), plus scratch, with seeds 0–2:
6 pretrains, 6 fine-tunes, 3 scratch runs, **15 runs**. All data, batches, budgets, initialisation,
normalisation and optimisers are E1's. Each fine-tune starts from its own rerun pretrain.

Log the mean training objective and the complete held-out one-step error every 500 pretraining or
scratch steps and every 100 fine-tuning steps, including step zero and the final step. Logging must
consume no training randomness and must leave the parameters unchanged (a correctness gate).
`curve_every` cannot be combined with `scan_chunk`. Curves are diagnostic: no checkpoint selection
or hyperparameter tuning uses the held-out curves.

| Prediction | Test |
|---|---|
| F1-P1 | Every rerun's final one-step relative error is within 2% of its own original E1 run. |
| F1-P2 | Each scratch rerun's final held-out error is within 2% of the minimum logged on its own curve. |

The curve figure shows all three seeds, pretraining error on the target, fine-tuning versus scratch,
and training loss. Differences in normalisation mean training losses across cells are not directly
comparable. The original target evaluation trajectories are 2–5; pretraining never uses this regime.
`make_curves.py` writes `configs/curves.json`; `score_curves.py` matches results by config hash and
writes `results/f1_analysis.json` and `figs/e1_curves.png`.

**Exploratory figures and diagnosis (no new predictions).** Use original E1 weights where available;
otherwise label the rerun explicitly. Fix evaluation trajectory 2, frame 50 (physical time 60), model
seed 0, cells n = 2 and 32 at T = 32, and horizons 1, 5, 20. Plot physical vorticity with a shared
colour scale for truth and predictions and a separate shared error scale. Include scratch and
zero-shot as well as fine-tuning. Compare rollout amplitude and uncentred alignment on all four
evaluation trajectories at E1's original frame 50 for every n at T = 32 and all model seeds.
Report the exact identity e² = 1 + a² − 2 a c per forecast, where a = ||prediction|| / ||truth|| and
c is their cosine similarity in each model's training-normalised coordinates. This is a diagnosis,
not a causal experiment separating diversity, distance and optimisation.
