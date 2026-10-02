# E7 Report: Does a relative-position bias inside the attention windows help?

Spec: `handouts/e7_attention_rpb.md` (pre-registered 2 Oct 2026, before any E7 run). Code: the bias in
`src/aurora/swin.py` and `backbone.py` (off by default), gates in `tests/test_e7_rpb.py`, the runs in
`experiments/e7/`. Scored by `score.py` → `results/analysis.json`, from `results/rows.jsonl`.

## Verdict

**2 of 3 predictions HIT (E7-P1, P3).** A learned relative-position bias inside the attention windows
makes the model more accurate one step ahead, at almost no cost. It does not help over a 20-step
rollout.
* **One step: 6.9% ± 0.7% lower error** (E7-P1 HIT).
  * Every seed gains, by 5.6–8.1%. Each of the three runs with the bias beats each of the three
    without it.
  * The error falls from 0.150 to 0.140, that is from 0.489 to 0.455 of persistence's.
* **Rollout: no gain** (E7-P2 MISS). The 20-step drift changes by −0.9% ± 1.6% (seeds +0.1%, +1.2%,
  −4.1%).
* **Cost: 1.2% more time** (E7-P3 HIT), for 13,720 parameters (0.27% of the model).

For scale, on E4's task doubling the depth to 24 blocks lowered the one-step error by 6.7% ± 3.1%, for
twice the backbone's arithmetic.

## Set-up

* **Task:** E1's scratch task.
  * Train on `re300_kf4` trajectories 0–1, evaluate on 2–5 (dataset `e1_v1`, 64²).
  * 22,000 steps, batch 16, lr 1e-3 with the cosine schedule, MSE.
  * The E1 model: D = 64, 12 Swin blocks, windows 3 × 4 × 4.
* **Arms** (seeds 0–2):
  * **B:** the E1 scratch configuration, unchanged;
  * **R:** the same with the bias. Each block learns a table of 245 offsets per head and adds the
    entry for each (query, key) offset to the attention logits.
  * For each seed, R starts from B's exact weights with zero tables, and the two draw the same
    batches. Gate G2 shows that the two models then compute the same function. The 20-step smoke runs
    of the two arms agree to 5 × 10⁻¹⁰ in one-step error.
* **Gates:** G1–G6 and the existing suite (104 tests) passed before any run.
* **Where:** Kaggle, two NVIDIA T4 cards (`kaggle_e7.py`, commit ee4669e).
  * JAX 0.10.2 and optax 0.2.8, pinned to the lab machine's versions. Float32 with matmul precision
    `highest`.
  * The arms alternate across the cards: card 0 ran B0, R1 and B2; card 1 ran R0, B1 and R2.
  * Each run took 21 min. The six took about 65 min on the two cards.
* **Metrics** (as in E1, on trajectories 2–5):
  * **One-step relative error** (primary), over every frame;
  * **20-step drift:** the relative error of a 20-step rollout from frame 50 of each trajectory,
    averaged over the 20 steps and then over the 4 trajectories;
  * **Advantage:** adv(R) = 1 − err(R) / err(B), paired by seed. Mean ± SE over the 3 seeds.

## Results

Mean ± SE over 3 seeds (`results/analysis.json`).

| Arm | One-step error | / persistence | 20-step drift | Wall time (s) |
|---|---|---|---|---|
| **B**, no bias | 0.1499 ± 0.0024 | 0.489 | 0.684 ± 0.016 | 1256 ± 4 |
| **R**, relative-position bias | **0.1396 ± 0.0025** | **0.455** | 0.690 ± 0.026 | 1272 ± 3 |

By seed:

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| One-step error, B | 0.1451 | 0.1518 | 0.1530 |
| One-step error, R | 0.1348 | 0.1433 | 0.1406 |
| adv(R), one step | +7.1% | +5.6% | +8.1% |
| Drift, B | 0.687 | 0.655 | 0.709 |
| Drift, R | 0.687 | 0.647 | 0.738 |
| adv(R), drift | +0.1% | +1.2% | −4.1% |

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| E7-P1 | adv(R) on one-step error ≥ 3% and > 2 SE | **6.9% ± 0.7%** (9.4 SE) | **HIT** |
| E7-P2 | the mean paired drift advantage is > 0 | −0.9% ± 1.6% | **MISS** |
| E7-P3 | mean wall(R) within 10% of mean wall(B) | R / B = 1.012 | **HIT** |

## Readings

* **The bias helps one step ahead, and the gain is robust.**
  * The mean gain (0.010) is larger than the spread between B's seeds (0.008).
  * It shows on training data too. On the last training batch, which is the same batch for both arms
    of a seed, R's loss is 8–11% lower. That is one batch, so it only indicates that the bias helps
    the model fit the update, rather than acting only as a regulariser.
  * It costs 13,720 parameters. On E4's task, doubling the depth bought about as much (6.7% ± 3.1%).
* **The gain does not carry into rollouts.**
  * In every seed the drift advantage is 4–12 points below the one-step advantage. If the rollout
    error were only the accumulated one-step errors, the two would fall together.
  * E1 found the same split at a larger scale: pretraining halved the one-step error but raised the
    drift by 51–123% (E1-P6). In this harness, one-step error and drift measure different things.
  * The drift is a coarse measure: one rollout per evaluation trajectory, so 4 per run. B's own drift
    spans 0.655–0.709 across seeds, so a drift effect of a few percent would not show with 3 seeds.
  * E4's lever for drift was pushforward training, which cut the 50-step drift by 31–50%. The bias
    and pushforward act on different things, and nothing stops them being combined.
* **The cost is small and the same on both cards.** Measured within each card, R takes 1.0% longer
  (each card ran both arms). The wall time includes compilation and evaluation.

## Cross-device check

E1's own scratch rows (lab machine, RTX PRO 6000, commit b940e74) have B's configuration and seeds:
* one-step error 0.1461 / 0.1517 / 0.1532, against B's 0.1451 / 0.1518 / 0.1530 on the T4s: within
  0.7%;
* drift 0.705 / 0.664 / 0.727, against 0.687 / 0.655 / 0.709: 1.3–2.6% lower on the T4s.

The drift's device difference is smaller than its seed spread. Both arms ran on both T4s, so it does
not affect the comparison.

## Notes

* **Job version 1** stopped at its smoke run, before any E7 run: its data archive lacked the dataset's
  manifest, which the harness reads to check the splits. Commit ee4669e added it, and version 2 made
  all six runs.
* **The trained weights were not kept.** The harness writes them to `data/runs/` in the job's scratch
  folder, outside Kaggle's output. The rows record each run's configuration hash and commit, so the
  runs can be repeated. A follow-up that needs the weights would rerun them.
* **Run time:** 21 min per run, against the spec's estimate of about 18 min plus evaluation.

## Files

* `src/aurora/swin.py`, `backbone.py`, `harness.py`: the bias, off by default (`rel_pos_bias`).
* `tests/test_e7_rpb.py`: gates G1–G6.
* `experiments/e7/`:
  * `configs.json`, `kaggle_e7.py` and `score.py`;
  * `results/rows.jsonl` (6 rows), `results/analysis.json`, and `results/kaggle_env.json` (devices
    and package versions).
