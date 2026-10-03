# E9 Spec: Do the relative-position bias and q/k normalisation stack?

**Status: pre-registered 3 Oct 2026, before any E9 run.** The owner chose this follow-up to E8.

---

## 1. Question

In E8, two different changes to the attention each lowered the one-step error on E1's scratch task
by the same amount:
* the relative-position bias (E7), by 6.9% ± 0.7%;
* q/k normalisation, by 7.3% ± 0.7%.

They also gave the same rollout profile: the gain peaks at steps 2–3 and fades by step 30. Do the two
gains add up when both are switched on, or do the changes fix the same weakness?

* **Why they might overlap.** At initialisation the q, k and v weights are small (standard deviation
  0.02), so the logits are tiny and attention is close to uniform over each window. Either change
  lets attention become selective early in training:
  * the bias, by position;
  * q/k normalisation, by putting the logits at order one from the start.
* **Why they might add.** The bias carries the offset between tokens, which content-based attention
  can only reconstruct from the absolute position encodings added at the input.

## 2. Arms and set-up

* **Task, model, training and evaluation:** exactly as in E8 (`handouts/e8_followups.md`, §4).
  * E1's scratch task: train on `re300_kf4` trajectories 0–1, evaluate on 2–5.
  * 22,000 steps, batch 16, lr 1e-3 with the cosine schedule, MSE.
  * 84 rollouts of 50 steps from every 20th frame from frame 50 on.
* **Arms,** seeds 0–2:
  * **RQ:** both `rel_pos_bias` and `qk_norm`;
  * **B:** the control, rerun in the same job as a session check.
* **Same start and batches.** Both options add only deterministic parameters (tables at zero, gains
  at one, biases at zero). For each seed, RQ therefore starts from the same weights as B, R and Q, and
  draws the same batches.
* **References:** E8's rows for B, R and Q (same configurations, same software, same type of card).
  * E8's one-step errors reproduced E7's within 0.34% for the same configurations.
  * This job's B checks that again (E9-P6), so that RQ can be compared with E8's R and Q.
* **Where:** Kaggle, two T4 cards, with the software pinned as for E6–E8.
  * Each RQ run runs at the same time as the B run of the same seed on the other card, and each arm
    has seeds on both cards: card 0 runs RQ0, B1 and RQ2; card 1 runs B0, RQ1 and B2.
  * The trained weights are kept.
* **No code changes.** The options exist since E8.
  * New gates (`tests/test_e9_rq.py`): with both flags, the other weights are those of the same seed
    and the backbone at initialisation equals B's exactly; with a random AdaLN modulation it differs
    from R's and Q's; every table, gain and bias gets a gradient under jit.
  * E8's gate H4 already ran the two together through the harness.
  * Measured on the lab machine's CPU: the three new gates pass, with E7's and E8's attention gates
    (15 in all).

## 3. Metrics

As in E8:
* the one-step relative error;
* d20 and d50, the 20- and 50-step drift from the 84 rollouts;
* the error at each rollout step;
* `train_s`, the training loop's time.

adv_X(Y) = 1 − m(Y) / m(X), paired by seed, mean ± SE over the 3 seeds.

## 4. Predictions (fixed before any run)

The independent-combination value is 13.7%: if the two gains did not interact, RQ would gain
1 − (1 − 0.069)(1 − 0.073).

| # | Prediction |
|---|---|
| E9-P1 | The gains stack at least partly: adv_B(RQ) on one-step error ≥ 9% and > 2 SE (B from this job) |
| E9-P2 | But not fully: adv_B(RQ) on one-step error < 13.7% |
| E9-P3 | RQ beats each part alone: its one-step error is below E8's R and below E8's Q, in every seed |
| E9-P4 | The rollout gain persists: adv_B(RQ) on the 20-step drift ≥ 3% and > 2 SE |
| E9-P5 | It costs less than the sum of the two parts' costs (1.03 × 1.10): RQ's training time within 15% of B's, measured within each card and averaged over the two cards |
| E9-P6 | The session check holds: this job's B is within 0.5% of E8's B on one-step error, in every seed |

Scored by `experiments/e9/score.py`, written before any run.

## 5. Cost

6 runs of about 22–25 min, 3 per card, so about 80 min with set-up and smoke runs.

## 6. Files

* `tests/test_e9_rq.py`: the gates.
* `experiments/e9/`:
  * `make_configs.py` and `configs.json`;
  * `kaggle_e9.py`;
  * `score.py`;
  * `results/`;
  * `REPORT.md`.
