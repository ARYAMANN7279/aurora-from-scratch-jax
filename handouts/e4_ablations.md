# Experiment E4 Spec: Which architectural components matter? (RQ4)

**Status: DRAFT for advisor review.** Frozen on approval; later changes go in dated amendments.
**Question:** how much do shifted windows, depth, the number of latent levels, the pushforward
trick and the loss choice each contribute?
**Tools:** the dataset `acc_v1` (Stage 2), the Stage 7 model, and the Stage 11 harness.
**Written:** 27 Sep 2026.

---

## 1. Design
* **Base:** the Stage 7 default model (D = 64, L = 3, 12 Swin blocks, window (3, 4, 4)).
  * Data: trained on re100_kf4 and re300_kf4, seeds 0–5; evaluated on seeds 6–7 of both regimes.
  * Training: MSE loss, 10,000 steps, batch 16, lr 1e-3 with cosine decay.
  * Metrics: the one-step error ratio to persistence, and the 50-step rollout drift from t₀ = 50.
* **Variants.** Each changes one thing from the base, with 3 seeds each, for **27 runs** (about
  2.5 GPU-h):

| Variant | Change |
|---|---|
| base | none |
| noshift | every Swin block uses unshifted windows (`backbone.shift = false`) |
| depth6 / depth24 | encoder and decoder depths (1,1,1) / (4,4,4): 6 or 24 blocks instead of 12 |
| L1 / L6 | 1 or 6 latent levels instead of 3 |
| pf_steps | pushforward, same number of steps (≈ 2× compute) |
| pf_compute | pushforward, half the steps (matched compute) |
| mae | MAE loss instead of MSE |

* **Effect of a variant:** the relative change in each metric against base, paired by seed,
  mean ± SE.

## 2. Pre-registered predictions

| # | Prediction |
|---|---|
| E4-P1 | MAE: Re 300 rollout drift > 1.5× base. Anchor: Stage 8, where the MAE model blew up in 3 of 4 Re 300 rollouts |
| E4-P2 | pushforward: drift lower than base by ≥ 5% at matched steps, and not lower by more than 2 SE at matched compute (anchor: Stage 9) |
| E4-P3 | noshift: one-step error 3–20% higher than base |
| E4-P4 | depth: depth6 worse than base by > 3%; depth24 not better than base by > 3% (12 training trajectories) |
| E4-P5 | levels: L1 worse than base by > 3%; L6 within ±3% of base |

## 3. Open points for review
1. **Budget:** 10k steps (3.3× Stage 7). Ablation effects under-trained models show may not hold at
   convergence. The alternative is 20k steps, at 2× the cost.
2. **Data:** only two regimes. Running E4 on an E1 pretraining set (the 32-regime cell) would test
   the components where diversity is high.
