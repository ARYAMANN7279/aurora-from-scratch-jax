# Stage 9 Spec: The pushforward trick

**Plan acceptance test (PROJECT_PLAN.md §5):** rollout drift measurably lower than without it, at
matched compute.
**Written:** 26 Sep 2026, before any Stage 9 code and before any Stage 7/8 result.
§§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Method

The pushforward trick (Brandstetter, Worrall & Welling, ICLR 2022) trains the model on inputs from
its own prediction distribution without backpropagating through the unroll:

    X̂^{t+1} = Φ(X^{t−1}, X^t)
    X̃^{t+2} = Φ(X^t, stop_gradient(X̂^{t+1}))
    loss = ℓ(X̂^{t+1}, X^{t+1}) + ℓ(X̃^{t+2}, X^{t+2})

The gradient flows through the second application only via its parameters, not through the input
produced by the first. ℓ is the Stage 7 loss (MSE).

## 2. Matched compute

Cost unit: one forward + backward of Φ on one sample ≈ 3 forward passes.
* **Baseline:** the Stage 7 MSE model, 3000 steps × batch 8 = 24,000 one-step samples ≈ 72,000
  forward-equivalents.
* **Pushforward:** each sample costs two forward+backward passes (the first step's forward is
  reused for its own loss) ≈ 6 forward-equivalents. Matched budget: **1500 steps × batch 8**.
  The learning-rate schedule is cosine decay over 1500 steps; everything else is identical (data,
  normalisation, key 0).
* A second, **matched-steps** pushforward run (3000 steps) is also reported, to separate "better
  per step" from "better per FLOP". The acceptance test uses the matched-compute run.

## 3. Measurement

Rollouts exactly as in Stage 8 §3 (8 rollouts per regime, 50 steps, evaluation seeds 6 and 7,
t₀ ∈ {50, 250}), for the baseline and both pushforward models.

* **Drift metric:** the mean relative error over rollout steps 1–20 (the window before errors
  saturate at the chaotic decorrelation level), per rollout.
* **Comparison:** paired by rollout (same initial condition), difference = baseline − pushforward,
  mean ± SE over the 8 rollouts of each regime.

| # | Measurement | Prediction |
|---|---|---|
| F1 | matched-compute pushforward vs baseline, drift metric, each regime | pushforward lower, paired mean difference > 2 SE |
| F2 | same, relative reduction of the drift metric | 5–30% |
| F3 | one-step error, pushforward (matched compute) / baseline | in [0.9, 1.5] (slightly worse one-step is the expected price) |
| F4 | matched-steps pushforward vs baseline, drift metric | lower than the matched-compute pushforward |
| F5 | stability (Stage 8 Z1 criterion) for all pushforward rollouts | HIT |

**Stage 9 acceptance:** F1 HIT in both regimes.
