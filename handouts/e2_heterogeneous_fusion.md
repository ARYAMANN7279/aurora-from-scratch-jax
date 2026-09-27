# Experiment E2 Spec: Does the Perceiver encoder earn its place? (RQ2)

**Status: DESIGN DRAFT for advisor review. Code not built yet (section 4).**
**Question:** Aurora's Perceiver encoder lets one model train on datasets with different
resolutions and different variable sets. Does it fuse such data better than the simple
alternatives?
**Written:** 27 Sep 2026.

---

## 1. Heterogeneous training sets (from `acc_v1`, no new data)
Two sources, each with a different format:
* **A:** re100_kf4, seeds 0–5;
* **B:** re300_kf4, seeds 0–5.

| Set | A | B |
|---|---|---|
| (a) mixed resolution | 128², all variables (u, v, p, ω) | 64², all variables |
| (b) mixed variables | 64², all variables | 64², (u, v) only |
| (c) both | 128², all variables | 64², (u, v) only |
| (h) homogeneous reference | 64², all variables | 64², all variables |

Evaluation is on seeds 6–7 of each source in its own format: the one-step error ratio to
persistence, and 20-step rollout drift. The loss covers only the variables a source provides.

## 2. Methods
1. **Perceiver** (this model): the spectral-band levels and the Perceiver aggregator handle any
   N, and per-variable embeddings handle any subset. This already works in `model.apply`.
2. **Pad-and-mask:** bring every sample to one format:
   * upsample 64² to 128² spectrally (zero-padding the modes);
   * zero-fill the missing variables, adding a per-variable "present" mask channel;
   * fixed C = 6 levels.
3. **Per-dataset encoders:** a separate embed and Perceiver per source format, and a shared
   backbone.

Each set × method uses 3 seeds, with (h) run once per seed as the reference: 3 × 3 × 3 + 3 = 30 runs.

## 3. Pre-registered predictions (to be finalised with the design)

| # | Prediction |
|---|---|
| E2-P1 | Perceiver on (a), (b), (c) is within 5% of the homogeneous reference (h) on the shared format |
| E2-P2 | pad-and-mask is worse than the Perceiver on B (the upsampled or zero-filled source) by > 5% in (a) and (c) |
| E2-P3 | per-dataset encoders are worse than the Perceiver on the smaller-format source by > 5%, because each encoder sees half the data |

## 4. Code needed (after approval)
* **Harness:** per-source data format (N and a variable list) in the data config; batches drawn
  per source, so one batch never mixes shapes; a loss masked to the variables present.
* **The two baselines:** pad-and-mask (a spectral upsample and a mask channel in the embedding) and
  per-dataset encoders (a dict of embed/Perceiver pytrees keyed by format).
* **Gates:**
  * pad-and-mask on a homogeneous set equals the base model with an all-ones mask;
  * per-dataset encoders with one source equal the base model;
  * a masked loss ignores the missing variables exactly.

## 5. Open points for review
1. Is a 2× resolution gap (64 vs 128) enough heterogeneity? 32² vs 128² would stress it more.
2. Should the evaluation include an unseen format, e.g. Re 1000 at 128² with (u, v) only, to test
   generalisation of the fusion itself?
