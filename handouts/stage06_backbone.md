# Stage 6 Spec: 3-D Swin U-Net backbone with skips

**Plan acceptance test (PROJECT_PLAN.md §5):** overfits a single trajectory to near-zero loss.
**Reference:** Aurora's `Swin3DTransformerBackbone` (microsoft-aurora 2.0.1), in the separate
reference environment, as in Stage 5. **Stack:** pure JAX pytrees + Optax.
**Written:** 26 Sep 2026, before any Stage 6 code. §§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Design (Aurora's backbone, deterministic mode, no LoRA, no drop-path)

* **Conditioning:** `c = Linear(SiLU(Linear(t_emb)))`, both linears D → D with bias, where
  `t_emb (B, D)` is the lead-time Fourier embedding. The backbone takes `t_emb` as input; the
  project computes it with Stage 3's `lead_time` encoding (the reference test feeds Aurora's
  `lead_time_expansion` values instead).
* **Encoder:** S stages. Stage i has width `D·2^i`, `depth_i` Swin3D blocks (Stage 5) with shift 0
  on even blocks and `(Wc//2, Wh//2, Ww//2)` on odd ones, and `heads_i` heads. Every stage except
  the last is followed by **patch merging**: 2×2 in (H, W) →
  `LayerNorm(4·w) → Linear(4·w → 2·w, no bias)`. The level axis C is never merged. Each stage saves
  its pre-merge output as a skip.
* **Decoder:** S stages with widths `D·2^{S−1−i}`. Every stage except the last is followed by
  **patch splitting**: `Linear(w → 2·w, no bias)` → reshape each token into 2×2 tokens of width w/2
  → `LayerNorm(w/2)` → `Linear(w/2 → w/2, no bias)`.
* **Skips (Aurora):** after decoder stage i, add the matching encoder skip for `0 < i < S−1`.
  After the last stage, **concatenate** the first encoder skip, so the output width is **2D**.
* The AdaLN conditioning dimension is D in every stage. Windows are the same in every stage, and
  `adjust` shrinks them when a grid is smaller than the window.
* Grid sizes must allow S−1 halvings with windows dividing every stage's grid (no padding in this
  project; a non-divisible shape raises ValueError).
* The Swin blocks use `periodic = (False, True, True)` in this project, and `(False, False, True)`
  for the reference comparison.
* **Initialisation:** as Aurora: truncated normal (std 0.02) for linears, LayerNorm (1, 0), and zero
  AdaLN modulation.

## 2. Gates — `tests/test_stage6_gates.py`

`experiments/stage6/make_reference.py` (reference environment) builds Aurora's backbone with random
non-zero weights (including the AdaLN modulations) for two configurations, runs it on random tokens
and lead times, and stores the weights, the `lead_time_expansion` values, the inputs and the
outputs in `tests/fixtures/backbone_ref.npz`.

| Gate | Test | Pass if |
|---|---|---|
| U1 | patch merging and splitting vs Aurora (from the fixture) | ≤ 1e-12 (f64) |
| U2 | **full backbone vs Aurora**, two configurations, float64 | ≤ 1e-12 relative to max\|out\| |
| U3 | same in float32 | ≤ 1e-5 |
| U4 | output shape `(B, C·H·W, 2D)`; skip wiring (additive in the middle, concatenated at the end) checked by zeroing one skip | exact |
| U5 | doubly periodic mode: translation by `2^{S−1}·window` patches in H and W commutes with the backbone | ≤ 1e-12 |
| U6 | `jax.grad` vs central finite differences, 5 entries | ≤ 1e-6 relative |

## 3. Overfit measurement (the plan's acceptance test)

* **Model:** Stage 3 embedding (bands **off**, one level; encodings on) → backbone → Stage 3 heads,
  reading the backbone's 2D-wide output through a `Linear(2D → D)`. Inputs are two frames
  `(X^{t−1}, X^t)`, embedded separately and summed with a learned per-frame vector. The target is
  `X^{t+1}`, predicted as a residual on `X^t`.
* **Configuration:** D = 64, S = 3, depths (2, 2, 2) / (2, 2, 2), heads (2, 4, 8) / (8, 4, 2),
  window (1, 4, 4), P = 4, 64² frames → token grid (1, 16, 16) → (1, 8, 8) → (1, 4, 4).
* **Data:** `acc_v1` re100_kf4 seed 0, 64², normalised per channel; all 499 consecutive triples,
  stride 1 (Δt = 0.2). Adam, lr 1e-3 with cosine decay to 1e-5, batch 8, float32, key 0,
  4000 steps.
* **Metric:** relative L2 error of the one-step prediction on the training triples (overfitting is
  the point), compared with persistence (`X^{t+1} ≈ X^t`).

| # | Measurement | Prediction |
|---|---|---|
| V1 | U2 worst float64 error | ≤ 1e-12 |
| V2 | U3 worst float32 error | ≤ 1e-5 |
| V3 | persistence relative error on the training triples | in [0.05, 0.4] |
| V4 | trained one-step relative error on the training triples, after 4000 steps | < 0.1 × persistence, and < 2e-2 |

**Stage 6 acceptance:** U1–U6 pass, and V4 HIT.

---

## Amendments (after results; the original verdicts stand in the report)

**A1 (26 Sep, after the first gate run): U6 criterion.** The pre-registered check (h = 1e-6,
relative error ≤ 1e-6 per entry) **failed** at 5.5e-5. Diagnosis: at all 5 sampled entries the
absolute finite-difference error is 1e-12 to 2e-11. It grows as h shrinks (roundoff ∝ ε·loss/h),
while the analytic gradient is unchanged. The failing entry has |g| = 3.1e-7, so a roundoff-level
error of 1.7e-11 reads as 5.5e-5 relative. The gradient is correct. A first fix (the mixed
criterion below, keeping h = 1e-6) **also failed**, at 6.7e-6: at h = 1e-6 the roundoff floor
(~1.5e-10 absolute, ≈ ε·loss/h) is itself near the median gradient scale (2.2e-5). The same
diagnostic run shows that at h = 1e-4 every sampled entry's absolute error is ≤ 3.9e-12. h = 1e-4
is also the near-optimal central-difference step, ≈ ε^(1/3) × the parameter scale. From now on U6
uses h = 1e-4 and asserts `|fd − g_i| ≤ 1e-6 · max(|g_i|, median|g|)`.
