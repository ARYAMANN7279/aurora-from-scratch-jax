# Experiment E5 Spec: LoRA vs full fine-tuning across fine-tuning set sizes (RQ5)

**Status: DRAFT for advisor review.** Frozen on approval; later changes go in dated amendments.
**Question:** does LoRA match full fine-tuning when fine-tuning data is scarce, and how does
that change as data grows? This picks up Stage 10, where two trajectories were too few to adapt
with at all.
**Tools:** the dataset `acc_v1`, the Stage 7 model, the Stage 10 LoRA, and the Stage 11 harness.
**Written:** 27 Sep 2026.

---

## 1. Design
* **Pretrain:** Re 100 and Re 300 (k_f 4), seeds 0–7. 10,000 steps, lr 1e-3, batch 16, MSE loss.
  Re 1000 is held out as `test`; its error on seeds 6–7 is the zero-shot reference.
* **Fine-tune on Re 1000:** k ∈ {1, 2, 4, 6} trajectories (seeds 0 … k−1), 2,000 steps, keeping the
  pretrain normalisation.
  * **full:** all parameters, lr 1e-4.
  * **lora:** rank 4 on the backbone attention (0.84% of the parameters), lr 1e-3.
* **Scratch:** on the same k trajectories for 12,000 steps (matched optimiser steps).
* **Runs:** with seeds {0, 1, 2}, 3 pretrain + 24 fine-tune + 12 scratch = **39 runs** (about
  1.5 GPU-h). Every evaluation is on Re 1000 seeds 6–7 (one-step error, 20-step drift).
* **Metrics:**
  * gain(method, k) = 1 − e_method / e_zero-shot;
  * adv(k) = 1 − e_full / e_scratch;
  * mean ± SE over the seeds.

## 2. Pre-registered predictions

| # | Prediction |
|---|---|
| E5-P1 | gain(full, k) rises with k (monotone over the 4 sizes, seed means) |
| E5-P2 | gain(full, 2) ∈ [2%, 10%] and gain(full, 6) ≥ 8%. Anchor: Stage 10 measured 3.0% at k = 2 with 1,000 steps |
| E5-P3 | gain(lora, k) / gain(full, k) falls with k: LoRA keeps up when data is scarce and falls behind when it is not |
| E5-P4 | adv(k) falls with k: pretraining matters most when fine-tuning data is scarce; adv(1) > 2 SE |

## 3. Open points for review
1. **LoRA rank:** rank 4 only. A rank sweep (1, 4, 16) at k = 2 would add 6 runs.
2. **Re 1000 frames:** they come from the 512² generation grid, down-sampled to 64². At 64² the
   Re 1000 small scales are not resolved in the saved frames. The regime is still learnable, but
   harder than the others.
