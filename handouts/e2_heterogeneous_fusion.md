# Experiment E2 Spec: Does the Perceiver encoder earn its place? (RQ2)

**Status: FROZEN 28 Sep 2026.** Launched on the owner's instruction; the open points in the last section were not resolved before launch and stay open for the report. The
code is built and tested (section 4).
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

## 3. Pre-registered predictions
Every comparison is the relative change in one-step relative error on validation seeds 6–7, in the
source's own format, paired by seed (mean over seeds 0–2).

| # | Prediction |
|---|---|
| E2-P1 | The Perceiver is within ±5% of the homogeneous reference (h) on every source whose format is the same as in (h): B in set (a), A in set (b) |
| E2-P2 | Pad-and-mask is worse than the Perceiver by > 5% on B (the upsampled source) in sets (a) and (c) |
| E2-P3 | Per-dataset encoders are worse than the Perceiver by > 5% on B in sets (a), (b) and (c), since each encoder sees half the data |

## 4. Implementation (built; `tests/test_e2_fusion.py`)
* **`src/aurora/fusion.py`:**
  * sources with their own N and variables;
  * per-variable normalisation over the training frames that provide the variable;
  * batches drawn per source, so one batch never mixes shapes;
  * loss and metrics over the source's own variables at its own N;
  * the three methods, through one `predict` function.
* **Pad-and-mask:** spectral zero-padding to the largest N, zero-filled missing variables, and one
  presence field per variable (1 where present), embedded as an extra input. The prediction is
  spectrally truncated back to the source's N.
* **Per-dataset encoders:** an embedding and a Perceiver per source; the backbone, frame
  embeddings and mid-projection are shared.
* **Gates:**
  * spectral up and down round trip;
  * per-source encoders with one source equal the base model bitwise;
  * pad-and-mask with zero mask embeddings, on a source already at the largest N, equals the base
    model;
  * every method predicts exactly the source's variables at its own N;
  * each method runs end to end;
  * a train/val seed overlap is refused.
* **Configs:** `experiments/e2/configs/runs.json`, 30 runs (10,000 steps, batch 16, lr 1e-3 with
  cosine decay, 20-step rollout). Run with `experiments/run.py`, which routes configs that carry
  `method` to `aurora.fusion`.
* **Analysis:** `experiments/analyse_e345.py --exp e2`.
* **Cost:** about 3 GPU-h. Sources at 128² cost about 4× per step.

## 5. Open points for review
1. Is a 2× resolution gap (64 vs 128) enough heterogeneity? 32² vs 128² would stress it more.
2. Should the evaluation include an unseen format, e.g. Re 1000 at 128² with (u, v) only, to test
   generalisation of the fusion itself?
