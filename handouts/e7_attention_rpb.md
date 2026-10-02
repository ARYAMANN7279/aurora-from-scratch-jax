# E7 Spec: Does a relative-position bias inside the attention windows help?

**Status: pre-registered 2 Oct 2026, before any E7 run.** Advisor's notes (1 Oct 2026): find ideas
to improve the multi-head attention. The owner chose the top idea of the Stage 12 report (§8).

---

## 1. Question

* **No positional term in attention.** Neither our window attention nor Aurora's has one. A token's
  place in its window reaches attention only through the position encodings added at the input.
* **What Swin adds.** Swin's relative-position bias gives each head a learned term per offset
  between query and key. Attention can then weigh neighbours by where they are, as a
  finite-difference stencil does. That is a natural inductive bias for a translation-invariant flow.
* **The question.** Does the bias lower the error on a small-data task, and at what cost?

## 2. The change (`src/aurora`)

* **Switch:** `backbone.Config(rel_pos_bias=True)`.
* **The table:** each of the 12 Swin blocks gets a learned table of (2·3 − 1)(2·4 − 1)(2·4 − 1) =
  245 offsets per head. The block adds table[offset(query, key)] to every window's logits, before
  the shift mask.
* **Zero start, same weights:** the table starts at zero and draws no random numbers. For a given
  seed, every other weight is identical to the model without it, and at initialisation the two
  compute the same function.
* **Size:** 13,720 parameters (56 heads × 245), 0.27% of the model's 5.1 M.
* **Off by default:** every existing config, and every model trained so far, is unchanged.

## 3. Gates (`tests/test_e7_rpb.py`; all must pass before any run)

| # | Gate |
|---|---|
| G1 | The default config has no tables |
| G2 | The flag changes no other weight, and a zero table gives the same backbone output (≤ 1e-12 relative), both at initialisation and with a random AdaLN modulation |
| G3 | The offset index matches a brute-force loop, for the full window and for adjusted (smaller) windows |
| G4 | Window attention with a bias matches a NumPy reference to 1e-12, with and without a mask |
| G5 | A non-zero table changes the backbone output |
| G6 | Every table gets a finite, non-zero gradient under jit |

The existing test suite must still pass. Measured: all 104 tests pass, and G2's difference is exactly 0.

**Why G2, G5 and G6 use a random modulation.** At initialisation Aurora's AdaLN modulation is zero, so
every block is the identity. Attention, the bias included, then gets no gradient until the first update
of the modulation switches it on, as for every attention weight. So those gates also run in a state
where attention contributes.

## 4. Task and arms

* **Task:** E1's scratch task (`experiments/e1/configs/scratch.json`).
  * Train on `re300_kf4` trajectories 0–1, evaluate on 2–5 (dataset `e1_v1`, 64², stride 1).
  * 22,000 steps, batch 16, lr 1e-3 with the cosine schedule, MSE.
  * The E1 model: D = 64, 12 blocks, windows 3 × 4 × 4.
* **Arms,** 3 seeds each (0–2):
  * **B:** that config unchanged.
  * **R:** the same with `"rel_pos_bias": true`.
  * For each seed, B and R start from identical weights (R's tables are zero) and draw the same
    batches.
* **Where:** Kaggle's two T4s, because other users hold the lab machine's cards. The software is
  pinned as for E6, with matmul precision `highest`.
  * B and R alternate across the two cards, so each arm runs on both.
  * E1's own scratch rows (lab card, 2 trajectories, same config) are reported as a cross-device
    check of B.

## 5. Metrics

* **One-step relative error** on the evaluation trajectories (primary), and **20-step rollout
  drift**, both as in E1 (`metrics["val/re300_kf4"]`).
* **Advantage:** adv(R) = 1 − err(R) / err(B), paired by seed. Mean ± SE over the 3 seeds.
* **Cost:** each run's wall time. It covers compile, training and evaluation, which the two arms
  share apart from the bias.

## 6. Predictions (fixed before any run)

| # | Prediction |
|---|---|
| E7-P1 | The bias lowers the one-step error: adv(R) ≥ 3% and > 2 SE |
| E7-P2 | It lowers the 20-step drift too: the mean paired drift advantage is > 0 |
| E7-P3 | It costs little: the mean wall time of R is within 10% of B's |

Scored by `experiments/e7/score.py`, written before any run.

## 7. Cost

* **Per run:** on a T4, the harness's training loop takes about 50 ms per step (Stage 12 supplement,
  J0), so 22,000 steps take about 18 min, plus evaluation.
* **In all:** 6 runs, 3 per card, about 1 h on the two cards.

## 8. Files

* `src/aurora/{swin,backbone,harness}.py`: the bias, off by default.
* `tests/test_e7_rpb.py`: gates G1–G6.
* `experiments/e7/`:
  * `configs.json`: the 6 configs;
  * `kaggle_e7.py`: the Kaggle job;
  * `score.py`;
  * `results/rows.jsonl`;
  * `REPORT.md`.
