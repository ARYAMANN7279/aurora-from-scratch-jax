# E8 Spec: Follow-ups on E1's scratch task (the attention bias, q/k normalisation, scan chunks)

**Status: pre-registered 3 Oct 2026, before any E8 run.** The owner asked for the smaller follow-ups
listed after E7, run on Kaggle overnight. They share one task and one control, so they run as one
experiment.

---

## 1. Questions

* **(a) The relative-position bias, measured better and combined with pushforward** (follow-up to
  E7).
  * E7's bias lowered the one-step error by 6.9% ± 0.7%, but not the 20-step drift (−0.9% ± 1.6%).
    That drift came from only 4 rollouts per run, one per evaluation trajectory.
  * Is there a drift effect that a finer measure, from many rollout starts, can see? How does the
    one-step gain fade along a rollout?
  * E4's lever for drift was pushforward training. Does pushforward lower drift on this task, and
    does the bias keep its one-step gain under pushforward?
* **(b) q/k normalisation** (the Stage 12 report's second attention idea, §8). A LayerNorm on each
  head's queries and keys, as in ViT-22B, bounds the attention logits. That is what lets attention
  train stably in bfloat16. Does it cost accuracy in float32?
* **(c) Scan chunks in the harness** (the Stage 12 recommendation). Does training 50 steps per jitted
  call (`lax.scan`) on the same batches leave the results unchanged, and how much faster is it on a
  T4?

## 2. The changes (`src/aurora`; each off by default)

* **q/k normalisation:** `backbone.Config(qk_norm=True)`.
  * Each block gets a gain and a bias for its queries ("qn") and keys ("kn"), each of the head
    dimension (32) and shared by the heads: 1536 parameters in all.
  * They start at 1 and 0 and draw no random numbers. For a given seed every other weight is the
    same as without them, and at initialisation the model computes the same function, because
    Aurora's AdaLN modulation starts at zero.
* **Harness: scan chunks:** `train.scan_chunk = K` runs K steps per jitted call.
  * The batch indices are drawn on the host in the loop's exact order, and the batches are gathered
    on the card, so the run sees the same batches as without the option.
  * Steps beyond the last full chunk run in the ordinary loop.
* **Harness: drift from many starts:** `eval.drift_every`, `eval.drift_steps` and `eval.drift_from`
  (default 50).
  * A `drift_steps`-step rollout starts at every `drift_every`-th frame from `drift_from` on, in
    each evaluation trajectory.
  * New metrics:
    * `rollout_err_by_step`: the relative error at each step, averaged over all rollouts;
    * `rollout_drift_many`: its mean over the first `rollout_steps` steps (E1's 20-step drift,
      measured from many starts);
    * `rollout_drift_many_long`: the mean over all `drift_steps` steps.
  * The existing `rollout_drift`, a single rollout from frame 50, is unchanged.
* **Harness: training time:** every row gains `train_s`, the training loop alone. `wall_s` also covers
  compilation and evaluation.

## 3. Gates (all must pass before any run)

| # | Gate |
|---|---|
| Q1 | The default config has no q/k normalisation |
| Q2 | The flag changes no other weight; 12 blocks × 2 × (32 + 32) = 1536 parameters, all at 1 and 0 |
| Q3 | Window attention with q/k normalisation matches a NumPy reference to 1e-12, with and without a mask and a bias |
| Q4 | It makes attention blind to the scale of q and k: scaling their projections by 10 changes the output by < 1e-3 (by > 0.1 without it) |
| Q5 | At initialisation the backbone output is exactly that without it; with a random AdaLN modulation it differs |
| Q6 | Every gain and bias gets a finite, non-zero gradient under jit |
| H1 | Scan chunks reproduce the loop (7 steps: three chunks of 2 and one step), two regimes drawn, with and without pushforward: parameters within 1e-5 (relative) |
| H2 | The many-start drift equals a rollout-by-rollout computation, and with one start at frame 50 equals the existing `rollout_drift` |
| H3 | The defaults add no metric, and 0 < `train_s` ≤ `wall_s` |
| H4 | The bias, q/k normalisation and scan chunks run together through the harness |

`tests/test_e8_qknorm.py` (Q1–Q6) and `tests/test_harness_options.py` (H1–H4).
* **Measured on the lab machine's CPU:**
  * all 115 tests pass: the 104 earlier ones and these 11;
  * H1's largest parameter difference is 0 without pushforward and 2 × 10⁻¹⁰ with it;
  * Q4 gives 4 × 10⁻⁵ with the normalisation and 3.2 without.
* **Default path unchanged:** with none of the options, the harness reproduces the previous commit's
  harness bitwise. That holds for the parameters, the final loss and the metrics, with and without
  pushforward, on a 40-step run with a 20-step rollout.

## 4. Task and arms

* **Task:** E1's scratch task, as in E7.
  * Train on `re300_kf4` trajectories 0–1, evaluate on 2–5 (`e1_v1`, 64², 501 frames each).
  * 22,000 steps, batch 16, lr 1e-3 with the cosine schedule, MSE.
  * The E1 model: D = 64, 12 Swin blocks, windows 3 × 4 × 4.
* **Arms,** 3 seeds each (0–2):

| Arm | Change from E1's scratch configuration |
|---|---|
| **B** | none (the control) |
| **R** | relative-position bias (E7's arm R) |
| **BP** | pushforward |
| **RP** | relative-position bias and pushforward |
| **Q** | q/k normalisation |
| **BS** | scan chunks of 50 steps |

  * For each seed every arm starts from the same weights. The new parameters start at zero, or at
    one for the gains.
  * B, R, Q and BS draw the same batches. BP and RP draw their own, the same for both: pushforward
    samples from one frame fewer.
  * Pushforward runs at matched steps (22,000), so BP and RP cost about twice as much per step.
* **Evaluation (every arm):** E1's metrics, plus a 50-step rollout from every 20th frame from frame 50
  on. That is 21 starts per trajectory and 84 rollouts per run (`drift_every` 20, `drift_steps` 50).
* **Where:** Kaggle, two T4 cards, with the software pinned as for E6 and E7. Float32 with matmul
  precision `highest`.
  * Each card runs 9 runs. Every arm has seeds on both cards.
  * Each run runs at the same time as its partner arm for the same seed on the other card: B with
    R, Q with BS, and BP with RP.
  * The trained weights are kept this time.
* **Cross-check:** B and R train exactly as E7's B and R, on the same type of card with the same
  software. Their one-step errors are compared with E7's rows.

## 5. Metrics

* **One-step relative error** on the evaluation trajectories, and its ratio to persistence.
* **Drift, from 84 rollouts:**
  * the 20-step drift, `rollout_drift_many`;
  * the 50-step drift, `rollout_drift_many_long`;
  * the error at each step k, e(k), from `rollout_err_by_step`.
* **Advantage of Y over X:** adv_X(Y) = 1 − m(Y) / m(X), paired by seed, for a metric m. Mean ± SE
  over the 3 seeds.
  * adv(k) is the advantage on e(k).
* **Cost:** `train_s`, the training loop alone.

## 6. Predictions (fixed before any run)

| # | Part | Prediction |
|---|---|---|
| E8-P1 | a | The bias's one-step gain replicates: adv_B(R) ≥ 3% and > 2 SE |
| E8-P2 | a | Measured from 84 rollouts, the bias still leaves the 20-step drift unchanged: \|adv_B(R)\| < 3% |
| E8-P3 | a | The gain fades along the rollout: adv_B(R) at step 1 is ≥ 3%, and at step 20 less than half its step-1 value |
| E8-P4 | a | Pushforward lowers the 50-step drift: adv_B(BP) ≥ 10% and > 2 SE |
| E8-P5 | a | At matched steps pushforward keeps the one-step accuracy: \|adv_B(BP)\| < 3% |
| E8-P6 | a | Under pushforward the bias still helps one step: adv_BP(RP) ≥ 3% and > 2 SE |
| E8-P7 | b | q/k normalisation changes the one-step error by less than 3%: \|adv_B(Q)\| < 3% |
| E8-P8 | b | It costs little: mean `train_s` of Q within 10% of B's |
| E8-P9 | c | Scan chunks leave the results unchanged within device noise: in every seed, BS's one-step error within 1% of B's, and its 20-step drift within 3% |
| E8-P10 | c | Scan chunks train at least 10% faster on a T4: mean `train_s` of B / that of BS ≥ 1.10 |

* **Anchors:**
  * P1–P3: E7.
  * P4–P5: E4, where pushforward cut the 50-step drift by 31–50% at matched steps and left the
    one-step error unchanged (−0.5% ± 2.1%).
  * P9: E7's cross-device check. The same code and seed on another card landed within 0.7% on
    one-step error.
  * P10: the Stage 12 supplement, where scan was 1.16× faster than the step loop on a T4.

Scored by `experiments/e8/score.py`, written before any run.

## 7. Cost

* **B, R, Q and BS:** about 21 min each on a T4 (E7).
* **BP and RP:** about 40 min each.
* **In all:** 18 runs, about 4.1 h per card, run on the two cards at the same time.
* **Order:** after the Stage 4 F3 job, if Kaggle does not run two sessions at once.

## 8. Files

* `src/aurora/{swin,backbone,harness}.py`: the options, each off by default.
* `tests/test_e8_qknorm.py` and `tests/test_harness_options.py`: the gates.
* `experiments/e8/`:
  * `make_configs.py` and `configs.json`: the 18 configs;
  * `kaggle_e8.py`: the job;
  * `score.py`;
  * `results/`;
  * `REPORT.md`.
