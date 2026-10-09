# Stage 9 Report: The pushforward trick

Spec: `handouts/stage09_pushforward.md`. Training: `experiments/stage7/train_onestep.py --pushforward`.
Evaluation: `pf_eval.py` → `results/pushforward.json`; exploratory analysis in
`results/exploratory_matched_steps.txt`.

## Verdict

**The plan's acceptance test is not met: at matched compute, pushforward did not reduce rollout
drift.** The pre-registered F1 is a MISS in both regimes. Drift over rollout steps 1–20 was 13.7%
*worse* at Re100 and 5.6% worse at Re300 (the Re300 difference is within noise). Spending the same
budget on plain one-step training was better.

**Exploratory (not pre-registered):** with twice the compute (the same 3000 optimiser steps as the
baseline), pushforward **reduces drift by 14% (Re100) and 11% (Re300)**. It wins 4/4 paired
rollouts in each regime at 6–9 standard errors, with almost no one-step cost (0.073 vs 0.070;
0.150 vs 0.146). So the trick does improve rollout robustness here, but it does not pay for its
extra compute at this scale and budget.

Addendum (9 Oct 2026): E4 later met this test with 3 seeds and a longer budget; see the end of
this report.

## Predictions

Paired by initial condition, n = 4 rollouts per regime (seeds 6–7 × t₀ ∈ {50, 250}).

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| F1 | baseline − pushforward (matched compute) drift > 2 SE, each regime | Re100 −0.080 ± 0.039; Re300 −0.036 ± 0.039 | **MISS** |
| F2 | drift reduction 5–30% | −13.7% / −5.6% (an increase) | **MISS** |
| F3 | one-step error, pushforward / baseline in [0.9, 1.5] | 1.34 / 1.21 | **HIT** |
| F4 | matched-steps drift < matched-compute drift | 0.500 vs 0.661; 0.581 vs 0.691 | **HIT** |
| F5 | all pushforward rollouts stable (Stage 8 Z1 criterion) | both pushforward models stable | **HIT** |

## Exploratory comparison (baseline vs each pushforward model)

| Regime | Model | Mean drift (baseline → model) | Paired difference ± SE | Model better |
|---|---|---|---|---|
| Re100 | matched compute (1500 steps) | 0.581 → 0.661 | −0.080 ± 0.039 (−13.7%) | 0/4 |
| Re100 | matched steps (3000 steps, 2× compute) | 0.581 → 0.500 | **+0.082 ± 0.014 (14.0%)** | **4/4** |
| Re300 | matched compute | 0.654 → 0.691 | −0.036 ± 0.039 (−5.6%) | 2/4 |
| Re300 | matched steps | 0.654 → 0.581 | **+0.074 ± 0.008 (11.2%)** | **4/4** |

## Deviations and spec notes
* **Rollout count.** The spec's §3 says "8 rollouts per regime". As implemented (and as in Stage 8)
  there are **4 per regime**, 8 in total. The spec's wording was wrong; the standard errors use
  n = 4.
* The first pushforward launch was killed by a container restart. It restarted from its last
  checkpoint (training checkpoints every 250 steps and resumes exactly).

## Consequences for the experiments
* The pushforward ablation in E4 must report both matched-steps and matched-compute comparisons.
  The answer depends on which one is asked.
* Stage 8 found the MAE-trained model unstable under rollout. Whether pushforward fixes the MAE
  instability is a natural E4 cell.

## Addendum (written 9 Oct 2026): the test repeated in E4 (the verdicts above stand)
E4 (28–29 Sep) repeated the comparison with 3 seeds, 10,000 steps (5,000 at matched compute) and the
50-step drift (`experiments/e4/REPORT.md`):
* **At matched compute,** pushforward cut the 50-step drift by 44.6 ± 4.6% (Re 100) and
  47.1 ± 6.9% (Re 300), about 10 SE, at the price of a 21.5 ± 3.8% higher one-step error.
* **At matched steps,** it cut the drift by 49.8 ± 3.8% and 31.3 ± 8.1%, with no one-step cost
  (−0.5 ± 2.1%).

So the plan's test, lower drift at matched compute, is met in E4, and the README shows Stage 9 as met
there. The single-seed, 3000-step result above did not hold up. On E1's two-trajectory task (E8),
the gain is smaller: 6.2 ± 1.6% in 50-step drift at matched steps, about two-thirds of it from damped
amplitude.
