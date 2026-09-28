# Stage 4 Spec: Perceiver level aggregator and level decoder

**Plan acceptance test (PROJECT_PLAN.md §5):** inputs with C = 3 and C = 8 levels both map to L = 3
latent levels, and the decoder recovers either.
**Levels** are spectral octave bands (decision of 26 Sep), so C = log2(N) − 1: C = 3 at 16²,
5 at 64², 6 at 128², 8 at 512².
**Stack:** pure JAX pytrees + Optax. **Written:** 26 Sep 2026, before any Stage 4 code. §§2–4 are
fixed, and the Stage 1 rules apply.

---

## 1. Scope

    tokens (C, T, D) ─ aggregator (per position t: L latent queries attend over the C level tokens) ─▶ latents (L, T, D)
    latents (L, T, D) ─ decoder (per position t: C_out level queries attend over the L latents) ─▶ tokens (C_out, T, D)

This follows Aurora, where a Perceiver maps a variable number of pressure levels to a fixed number
of latent levels, and back. Attention runs **across levels, separately at each patch position**.
Spatial mixing is the backbone's job (Stages 5–6).

## 2. Design

### 2.1 Multi-head attention (from primitives)
`mha(params, q_in (Q, D), kv_in (K, D), mask (K,) or None, H heads)`:
`q = q_in W_q`, `k = kv_in W_k`, `v = kv_in W_v`, each split into H heads of size D/H; per head
`softmax(q kᵀ / √(D/H) + log-mask) v`; concatenate the heads, then `W_o`. No biases in the q/k/v
projections; bias in `W_o`. Masked keys get −∞ before the softmax. At least one key must be
unmasked.

### 2.2 Perceiver block (res-post-norm, as in Aurora)
    h = LN(q_in + mha(q_in, kv_in, mask))
    out = LN(h + MLP(h)),    MLP = Linear(D, 4D) → GELU → Linear(4D, D)
LN is LayerNorm over D with learned scale and shift, ε = 1e-5.

### 2.3 Aggregator
* Latent queries: a learned `Z ∈ R^{L×D}`, the same at every position.
* Keys/values: the C level tokens at that position. They already carry their level encodings
  (Stage 3), which is what lets the attention tell levels apart.
* `n_blocks` Perceiver blocks, default 1. Block i > 1 uses the latents from block i−1 as queries
  and the same keys/values.
* `vmap` over the T positions. Output `(L, T, D)`.

### 2.4 Decoder
* Queries: the Fourier level encodings (Stage 3 `level`) of the requested output levels, through the
  decoder's own learned map `W_ql ∈ R^{D×D}` with bias `q_b ∈ R^D`. Shape `(C_out, D)`, the same at every position.
* Keys/values: the L latents at that position.
* `n_blocks` Perceiver blocks. Output `(C_out, T, D)`, which goes to the Stage 3 heads.
* C_out can differ from the input C, so the decoder can be asked for levels the input did not have.

### 2.5 Masked levels (for padded batches of mixed C)
Inputs may be padded to `C_max` with a level mask. A masked level must have no effect on the
latents.

---

## 3. Gates — `tests/test_stage4_gates.py` (float64)

| Gate | Test | Pass if |
|---|---|---|
| K1 | `mha` vs a NumPy loop reference (explicit heads, explicit softmax), H ∈ {1, 4} | ≤ 1e-12 |
| K2 | level-permutation invariance: permuting the C input level tokens leaves the latents unchanged | ≤ 1e-12 |
| K3 | position equivariance: permuting the T positions permutes the output identically | ≤ 1e-12 |
| K4 | masking: padding C = 5 to C_max = 8 with arbitrary masked tokens gives the same latents as the unpadded input | ≤ 1e-12 |
| K5 | one parameter set runs on C ∈ {3, 5, 6, 8} → L = 3 and decodes to any requested C_out, under `jit` | shapes exact |
| K6 | softmax rows sum to 1; fully masked rows are rejected (ValueError at trace time when the mask is static) | ≤ 1e-14 / raises |
| K7 | `jax.grad` of a full autoencoder loss (embed → aggregate → decode → heads) vs central finite differences, 5 entries | ≤ 1e-6 relative |
| K8 | `init` is deterministic per key; the parameter count matches the formula in §4 | exact |

## 4. Measurements and predictions

**Model for the measurements:** D = 128, H = 4, L = 3, n_blocks = 1, P = 4, all four variables.
Exact parameter count (aggregator + decoder, excluding Stage 3), checked by K8:
* one Perceiver block: attention `4D² + D` (W_q, W_k, W_v without bias; W_o with bias) + two
  LayerNorms `4D` + MLP `8D² + 5D` = **12D² + 10D**;
* aggregator: latents `L·D` + one block;
* decoder: its own level-query map `D² + D` (the bias is q_b) + one block;
* total **25D² + 21D + L·D** = 412,672 for D = 128, L = 3.

**Data:**
* 16² (C=3), 64² (C=5) and 128² (C=6) frames: `acc_v1`, regime re300_kf4, seeds 0–5 for training
  and seed 6 for evaluation. 16² comes from spectral truncation of the 64² frames.
* 512² (C=8): 40 frames at Re = 1000 on the 512 generation grid (Δ = 0.2, after `acc_v1`'s
  spin-up of seed 0), produced by `experiments/stage4/make_512.py`; frames 0–31 train, 32–39
  evaluate. Normalisation is per channel, from the training frames of each resolution.

**Training:** Adam (Optax), lr 3e-4, batch 4 frames of one resolution per step, cycling through the
resolutions in turn, 6000 steps, float32, key 0. Loss: MSE on normalised fields. One model, with
shared parameters, for all four resolutions.

| # | Measurement | Prediction |
|---|---|---|
| S1 | relative L2 reconstruction error on evaluation frames, C = 3 (16²) | < 2e-2 |
| S2 | same, C = 5 (64²) | < 2e-2 |
| S3 | same, C = 6 (128²) | < 3e-2 |
| S4 | same, C = 8 (512²) | < 5e-2 |
| S5 | control: the same model without latent compression (L = C, a separate run per C, same steps) beats S1–S4 | each control error < the L = 3 error |
| S6 | control: with the level encodings ablated (all levels get the same encoding), the aggregator cannot tell bands apart; C = 5 error | > 2 × S2 |

**Stage 4 acceptance:** K1–K8 pass, and S1 and S4 are both HITs (C = 3 and C = 8 recovered through
L = 3).

## 5. Amendments

**A1 (29 Sep 2026): S5 and S6 are confounded with the training budget.** Both controls are
separate single-resolution runs with all 6000 steps, while S1–S4 come from the shared run, which
gives each resolution 1500 steps. S5 therefore measures compression and budget together, and S6's
bar (> 2 × S2) compares the ablation with a model that had a quarter of its per-resolution budget.
The original verdicts (S5 HIT, S6 MISS) are kept. A matched comparison needs single-resolution
runs with and without the component at equal steps (for S6: L = 3, 64² only, level encodings on).
