# E6 Spec: Does pretraining on simulations buy accuracy on real flow measurements? (RealPDE foil)

**Status: pre-registered 1 Oct 2026, before any E6 run.** Advisor's notes (1 Oct 2026): test the
model on RealPDE data. The owner chose the sim → real pretraining test.

---

## 1. Question

UGP-I (the RealPDE Track 1 competition) ended on a claim it could not test: the remaining gap needed
a better *pretrained* model. Outside weights were banned, and only 81 real trajectories existed.
RealPDEBench pairs real particle-image-velocimetry (PIV) measurements of the flow past an airfoil
with simulations of the same conditions. That makes the claim testable:

* Does pretraining our model on the simulations improve its forecasts of real flows at held-out
  conditions, compared with training on the real data alone?
* Does it help further when the simulations include the test conditions themselves?
* How does it compare with the organisers' FNO?

This is E1's question (what does pretraining data buy?) with a real distribution shift: the
sim-to-real gap.

## 2. Data

The UGP-I competition copy on the lab machine, with UGP-I's preprocessing reused unchanged.

* **Real:** the 81 trajectories of `UGP/local_harness/tr_frames.npy` and `tr_meta.json`, built by
  `prep_train2.py`.
  * u and v, strided `[::2, ::2]` from 64 × 128 to **32 × 64**, as the competition evaluated.
  * Frame spacing 0.02. Lengths 607–868 frames.
  * About 11% of pixels are exactly zero (the airfoil and its shadow); they stay in the loss and in
    the scores, as in the competition.
* **Simulated:** the 100 trajectories of `UGP/data/comp_sim/train_sim`.
  * u, v and p, strided the same way, 1000 frames each, spacing 0.02002.
  * They cover all 81 real conditions and 19 more. Units differ from the real data: u reaches 1.6
    against 0.17.
  * Each source is therefore normalised with its own per-variable mean and standard deviation,
    computed on its training trajectories.
* **Held-out test: UGP-I's `re_lohi` split.** The 2 lowest and 2 highest Reynolds numbers (3750,
  5025, 25425, 26700) at every angle of attack: 16 real trajectories.
  * Training uses the other 65 real trajectories.
  * The test windows are UGP-I's **900 `re_lohi` windows**: `r12_eval.val_starts(900, "re_lohi")`
    with random seed 0. Each window is 20 input frames followed by 20 target frames.
  * UGP-I found this split reproduces the live competition's transfer rate (UGP-I record, §21).
* **Simulation subsets:**
  * **disjoint:** every simulated trajectory except those at the 4 held-out Reynolds numbers;
  * **all:** every simulated trajectory, including the held-out conditions.

## 3. Model

The E1 configuration (D = 64, 4 heads, L = 3, 12 Swin blocks, windows 3 × 4 × 4), adapted to this
data:

* **Grid:** 32 × 64, patches P = 2, so a token grid of 3 × 16 × 32. The backbone stages are
  3 × 16 × 32, 3 × 8 × 16 and 3 × 4 × 8.
* **Levels:** spectral octave bands of a rectangular field. The band index is max(|k_y|, |k_x|·H/W),
  which gives 4 bands at 32 × 64, with the Nyquist row and column removed. The bands still sum
  exactly to the field. FFT bands of a non-periodic field ring at the edges, but the split stays
  exact.
* **Not periodic:**
  * shifted windows are masked along both H and W (`periodic = (False, False, False)`, the case
    gate W10 covers);
  * position encodings use half a period, so the two edges get different codes.
* **Variables:**
  * real inputs are (u, v), simulated inputs (u, v, p);
  * the loss and every metric use the variables present (u and v on real data).
* The square, periodic Kolmogorov path stays bitwise unchanged. The existing gates must keep
  passing, and new gates cover the rectangular, non-periodic case.

## 4. Arms

3 seeds each. Every run on real data trains on the 65 training trajectories. Budgets follow E1:

| Arm | Pretraining | Then |
|---|---|---|
| **S** | — | real data from scratch, 22,000 steps, lr 1e-3, cosine |
| **PD** | simulations, disjoint subset, 20,000 steps, lr 1e-3 | fine-tune on real data, 2,000 steps, lr 1e-4 |
| **PA** | simulations, all, 20,000 steps, lr 1e-3 | fine-tune on real data, 2,000 steps, lr 1e-4 |
| **ZD, ZA** | the PD and PA pretrained models | none: zero-shot sim → real, with the real data's normalisation |
| **FNO** | the organisers' `sim_fno.pth` (simulations only) and `sim_real_fno.pth` | evaluated once on the same windows |

* Batch 16, MSE.
* `sim_real_fno.pth` was fine-tuned by the organisers on real data that may include the held-out
  conditions, so its score is an upper reference, not an honest one. `sim_fno.pth` never saw real
  data.

## 5. Metrics

On the 900 test windows, at 32 × 64, on u and v:

* **Rollout (primary):** the model reads the last 2 of the 20 input frames and rolls out 20 steps.
  * Relative L2 error over the 20 target frames, averaged over windows. This is the competition's
    `rel_l2`.
  * TKE error and mean-velocity-profile error, from the competition kit's `scoring.py` (imported,
    not rewritten).
  * The kit's 0–100 subscores, so the numbers sit next to UGP-I's.
* **One-step:** relative error of the first predicted frame, and its ratio to persistence.
* **Advantage:** adv(P) = 1 − err(P) / err(S), paired by seed. Mean ± SE over the 3 seeds.

## 6. Predictions (fixed before any run)

| # | Prediction |
|---|---|
| E6-P1 | Pretraining on simulations helps at held-out real conditions: adv(PD) on rollout `rel_l2` is at least 10% and more than 2 SE |
| E6-P2 | Simulations of the test conditions help further: err(PA) < err(PD) on rollout `rel_l2` by more than 2 SE |
| E6-P3 | Simulation alone beats the small real set: zero-shot ZA has lower rollout `rel_l2` than S. Anchor: in UGP-I the organisers' simulation-only FNO beat every model trained from scratch on the real data |
| E6-P4 | Fine-tuning on real data still adds: err(PD) is at least 10% below err(ZD) on rollout `rel_l2` |
| E6-P5 | None of our arms reaches the kit FNO's `rel_l2` subscore on these windows (95.47, `sim_real_fno.pth`, re-measured here) |

## 7. Cost and order

* About 1,500 tokens per sample, twice E1's count. On an idle card, roughly 15–25 ms per step, so
  5–8 min per 20,000-step run.
* 15 training runs (S, PD and PA, the last two each a pretrain plus a fine-tune, at 3 seeds),
  about 2 GPU-hours. A CPU is too slow here: about 2 s per step.
* Order:
  1. the model generalisation and its gates;
  2. data preparation;
  3. the FNO baselines (CPU is enough);
  4. the runs, on idle cards only.

## 8. Amendments

**A1 (1 Oct 2026, before any E6 run): pressure in simulated pretraining, and where the zero-shot
numbers come from.**
* **p is dropped in half the pretraining.** Real data has no p, so a model pretrained only with p
  in its inputs would face an input it never trained on at zero-shot time. The competition kit
  handled this by zeroing p at random during simulated training (`mask_prob` 0.5). We drop p from
  the inputs of a random half of the 100-step training chunks (seeded). Our embedding takes any
  subset of variables, so p is absent rather than zero. The loss always covers u, v and p, whose
  targets the simulations have.
* **The zero-shot arms ZD and ZA are not separate runs.** Every pretraining run is evaluated on the
  real test windows when it finishes, and that evaluation is the zero-shot result.

**A2 (1 Oct 2026, before any E6 run): E6-P3's anchor misread UGP-I. The prediction stands as
registered.** The anchor says the organisers' simulation-only FNO beat every model trained from
scratch on the real data. It does not hold, as the baselines measured on the 900 windows
(`results/fno_baseline.json`) show:
* `sim_fno.pth`, simulations only, scores rel_l2 error 1.23 (subscore 61.95): it does not
  transfer to real data zero-shot. Its own training log shows real-data validation errors of 4–6,
  and UGP-I measured subscore 73.7 for it on another holdout.
* The FNO that beat UGP-I's from-scratch U-Net and MWT was `sim_real_fno.pth`: 0.0948 here
  (subscore 95.47, reproducing UGP-I's selftest). The organisers fine-tuned it on real data.
* UGP-I's evidence therefore supports simulation pretraining followed by real fine-tuning. From
  `sim_fno.pth`, fine-tuning on 55 real trajectories reached subscore 95.16 (UGP-I, M55). That is
  E6-P1's claim, not E6-P3's.

**A3 (2 Oct 2026, after the seed-0 S run and before any other E6 run): the runs move to Kaggle.**
* **Why.** Other users' jobs have held the lab machine's cards 1–3 since 1 Oct, except for about an
  hour on card 2. Under our runs that card passed 93 °C within seconds. The owner chose to run E6 on
  their own Kaggle account.
* **Change.**
  * **All 15 runs on Kaggle,** the seed-0 S run included, so that every arm shares one device.
    Kaggle's NVIDIA T4 cards run two at a time, one queue per card (`kaggle_run.py`).
  * **Same software and settings:** the same code, with JAX 0.10.2 and optax 0.2.8 pinned to the
    lab machine's versions; float32 with matmul precision `highest` (a T4 has no TF32).
  * **Which rows count:** `results/rows.jsonl`, from Kaggle, is scored. The lab's seed-0 S row
    (`results/rows_lab.jsonl`) is reported as a cross-device check.
  * **The scoring file:** `run_e6.py` reads the competition kit's `scoring.py` from `E6_KIT` when it
    is set (Kaggle), and from the lab path otherwise.
  * Nothing else changes: the data, splits, arms, budgets, metrics and predictions stand as
    registered.

## 9. Files

* `src/aurora`: rectangular, non-periodic support (square path unchanged).
* `experiments/e6/prep_sim.py`: strided simulation frames.
* `experiments/e6/run_e6.py`: training and evaluation.
* `experiments/e6/fno_baseline.py`: the kit's FNOs on the same windows.
* `experiments/e6/REPORT.md`.
