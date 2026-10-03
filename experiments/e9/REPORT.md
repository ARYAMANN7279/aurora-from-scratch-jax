# E9 Report: Do the relative-position bias and q/k normalisation stack?

Spec: `handouts/e9_rpb_plus_qknorm.md` (pre-registered 3 Oct 2026, before any E9 run). No code changes: both
options exist since E7 and E8. Gates: `tests/test_e9_rq.py`. Runs: `experiments/e9/`.
* Scored by `score.py` → `results/analysis.json`, from `results/rows.jsonl` and E8's rows.
* Exploratory attention check: `attention.py` → `results/attention.json`.

## Verdict

**4 of 6 predictions HIT (E9-P2, P3, P4, P5). The two gains stack only partly.**
* **Together, the one-step error is 8.9% ± 0.4% lower than B's.**
  * That beats either change alone: E8 measured 6.9% for the bias and 7.3% for q/k normalisation,
    and RQ's error is below E8's R and Q in every seed (P3 HIT).
  * It is far short of the 13.7% that independent gains would give (P2 HIT).
  * It is just below P1's 9% bar (P1 MISS).
* **About a quarter of the possible extra.** RQ lands 25% of the way from the better single change
  (7.3%) to the independent value (13.7%). With each gain measured against B in its own session,
  RQ adds 2.0 ± 0.5 points to the bias and 1.6 ± 0.5 to q/k normalisation.
* **The rollout gain persists:** the 20-step drift is 4.2% ± 1.7% lower (P4 HIT). Like each part, it
  fades: 11% at step 3, 2% at step 20, under 1% from step 30.
* **Cost:** 11.3% more training time (P5 HIT).
* **The session check misses (P6).** B rerun here differs from E8's B by up to 0.8% in one-step error,
  against a 0.5% bar. That much noise sits in the cross-session comparisons with E8's R and Q (P3).
  It does not touch the within-session comparison with B (P1, P2).
* **Exploratory: the spec's overlap hypothesis does not describe the trained models.**
  * B's attention is not near-uniform overall. It is near-uniform at the fine stages and nearly
    one-hot at the coarsest stage, where its logits reach a standard deviation of 8–11.
  * q/k normalisation evens this out across stages. The bias leaves it as it is.
  * The two changes act on attention in different ways, yet their gains overlap.

## Set-up

* **Task and evaluation:** exactly as in E8.
  * E1's scratch task: `re300_kf4` trajectories 0–1, evaluated on 2–5.
  * 22,000 steps, batch 16, lr 1e-3, cosine schedule, MSE.
  * 84 rollouts of 50 steps from every 20th frame from frame 50 on.
* **Arms,** seeds 0–2:
  * **RQ:** both `rel_pos_bias` and `qk_norm`;
  * **B:** the control, rerun as a session check.
  * RQ starts from the same weights as B, R and Q for the same seed, and draws the same batches.
* **References:** E8's B, R and Q rows.
* **Where:** Kaggle job `ugp2-e9-runs`, two T4 cards, commit 0a953e3, 3 Oct 12:17–13:27 IST.
  * Software pinned as for E6–E8; float32 with matmul precision `highest`.
  * Each RQ run ran at the same time as the B run of the same seed on the other card: card 0 ran RQ0,
    B1 and RQ2; card 1 ran B0, RQ1 and B2.
  * Smoke runs of RQ and B (60 steps) passed first.
  * The trained weights are kept, on the lab machine in `data/e9_params/`.

## Results

Mean ± SE over 3 seeds (`results/analysis.json`).

| Arm | One-step error | By seed | d20 | d50 | Training (s) |
|---|---|---|---|---|---|
| **B** (this job) | 0.1495 ± 0.0027 | 0.1444, 0.1508, 0.1534 | 0.744 ± 0.016 | 1.100 ± 0.014 | 1195 ± 2 |
| **RQ** | **0.1363 ± 0.0027** | 0.1308, 0.1386, 0.1394 | **0.712 ± 0.017** | 1.080 ± 0.013 | 1332 ± 6 |
| B (E8) | 0.1500 ± 0.0025 | 0.1451, 0.1520, 0.1529 | 0.743 ± 0.017 | 1.098 ± 0.021 | 1205 ± 21 |
| R (E8) | 0.1397 ± 0.0024 | 0.1352, 0.1432, 0.1406 | 0.718 ± 0.013 | 1.091 ± 0.015 | 1258 ± 21 |
| Q (E8) | 0.1392 ± 0.0033 | 0.1327, 0.1421, 0.1428 | 0.715 ± 0.016 | 1.082 ± 0.012 | 1327 ± 28 |

One-step gain over B, each measured against B in its own job, by seed:

| | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| RQ (E9) | 9.4% | 8.1% | 9.1% | **8.9% ± 0.4%** |
| R (E8) | 6.8% | 5.8% | 8.1% | 6.9% ± 0.7% |
| Q (E8) | 8.6% | 6.6% | 6.6% | 7.3% ± 0.7% |
| independent combination of R and Q | | | | 13.7% |

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| E9-P1 | adv_B(RQ) on one-step error ≥ 9% and > 2 SE | 8.9% ± 0.4% | **MISS** |
| E9-P2 | adv_B(RQ) < 13.7% | 8.9% | **HIT** |
| E9-P3 | RQ's one-step error below E8's R and E8's Q in every seed | by 2.5% ± 0.8% (R) and 2.1% ± 0.4% (Q); every seed | **HIT** |
| E9-P4 | adv_B(RQ) on the 20-step drift ≥ 3% and > 2 SE | 4.2% ± 1.7% | **HIT** |
| E9-P5 | RQ's training time within 15% of B's (within each card) | 1.112 and 1.114 | **HIT** |
| E9-P6 | this job's B within 0.5% of E8's B in every seed | −0.52%, −0.79%, +0.28% | **MISS** |

## Readings

* **Mostly overlapping gains.**
  * Each change alone buys about 7%. Both together buy 8.9%, so the second one adds 1.6–2.0 points
    instead of about 6.
  * Together they still beat either alone in every seed, so the combination is the best one-step
    model of E7–E9: 0.136, against 0.150 for B.
  * For scale, doubling the depth bought 6.7% ± 3.1% on E4's task.
* **The rollout profile is unchanged.** RQ's gain over B is 9% at step 1 and peaks at 11% at step 3.
  It then fades as each part's did: 4% at step 10, 2% at step 20, under 1% from step 30. Combining
  the changes does not extend the horizon over which attention helps.
* **Run-to-run noise on the GPU is up to 0.8%.**
  * B rerun gave one-step errors 0.5% and 0.8% lower than E8's in two seeds, and 0.3% higher in the
    third. It is the same configuration on the same type of card with the same software, so the
    difference is run-to-run noise.
  * E7 and E8 had agreed within 0.34%, so that agreement was on the lucky side.
  * Comparisons within one job, against its own B, are the reliable ones. RQ's margin over E8's R in
    seed 2 (0.9%) is within this noise.
* **Cost.** Both together cost 11.3% more training time. The parts cost 3% and 10% (E8), so the
  costs roughly add.

## Exploratory: how selective the trained attention is

Not a pre-registered prediction. `attention.py` ran each kept model eagerly on 8 evaluation inputs
with the window attention instrumented, on the lab machine's CPU. For each Swin block it averaged
over windows, heads and queries:
* the attention entropy over the keys a query may see, divided by the log of their number (1 is
  uniform, 0 is one key);
* the standard deviation of the logits over those keys.

Means over 3 seeds; enc1 to enc3 run at 16 × 16, 8 × 8 and 4 × 4 tokens, and dec1 to dec3 back up.

| Arm | enc1 | enc2 | enc3 | dec1 | dec2 | dec3 | All blocks |
|---|---|---|---|---|---|---|---|
| B: entropy | 0.93, 0.92 | 0.86, 0.84 | **0.13, 0.13** | **0.15, 0.16** | 0.70, 0.66 | 0.86, 0.56 | 0.58 |
| R: entropy | 0.93, 0.89 | 0.86, 0.85 | 0.12, 0.13 | 0.16, 0.16 | 0.80, 0.75 | 0.78, 0.60 | 0.59 |
| Q: entropy | 0.85, 0.70 | 0.69, 0.68 | 0.63, 0.59 | 0.58, 0.59 | 0.62, 0.67 | 0.58, 0.48 | 0.64 |
| RQ: entropy | 0.84, 0.71 | 0.68, 0.68 | 0.68, 0.61 | 0.59, 0.63 | 0.65, 0.65 | 0.58, 0.52 | 0.65 |
| B: logit SD | 0.8, 0.9 | 1.1, 1.2 | **11.2, 10.7** | **8.9, 7.7** | 1.8, 1.9 | 1.1, 2.5 | 4.2 |
| R: logit SD | 0.8, 0.8 | 1.1, 1.2 | 11.6, 9.9 | 8.2, 7.2 | 1.1, 1.4 | 1.1, 1.3 | 3.8 |
| Q: logit SD | 1.3, 2.2 | 2.1, 2.1 | 2.3, 2.6 | 2.5, 2.5 | 2.5, 2.1 | 3.0, 4.8 | 2.5 |
| RQ: logit SD | 1.3, 1.8 | 2.1, 2.2 | 2.0, 2.4 | 2.6, 2.3 | 2.2, 2.1 | 2.4, 2.7 | 2.2 |

* **Without q/k normalisation, attention splits into two regimes.**
  * At the fine stages (enc1, enc2) it is close to uniform: entropy 0.84–0.93, logits barely spread.
    That is close to a plain average over each window.
  * At the coarsest stage (enc3, dec1) it is nearly one-hot: entropy 0.13–0.16, logit SD 8–11. There
    one window covers the whole 3 × 4 × 4 grid.
  * The bias leaves both regimes as they are.
* **q/k normalisation evens the regimes out:** entropy 0.58–0.85 and logit SD 1.3–4.8 in every block.
* **The spec's overlap hypothesis does not fit.** The spec guessed that both changes let attention
  escape near-uniformity. The trained models show q/k normalisation reshaping attention at every
  stage, while the bias does not change its sharpness. Their gains still overlap by three quarters.
  Why is not measured here. One possibility is that both relieve the same bottleneck in different
  ways: position-aware weighting at the fine stages (the bias), or softer coarse-stage attention
  (q/k normalisation).
* These are end-of-training snapshots on 8 inputs. How attention evolves during training was not
  recorded.

## Files

* `tests/test_e9_rq.py`: gates RQ1–RQ3 (with E7's and E8's attention gates, 15 pass on the CPU).
* `experiments/e9/`:
  * `make_configs.py`, `configs.json`, `kaggle_e9.py`, `score.py` and `attention.py`;
  * `results/rows.jsonl` (6 rows), `analysis.json`, `attention.json` and `kaggle_env.json`.
