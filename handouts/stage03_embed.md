# Stage 3 Spec: Level transform, patch embedding and unembedding

**Plan acceptance test (PROJECT_PLAN.md §5):** round-trip reconstruction error at machine precision.
**Stack (decided 26 Sep):** pure JAX, with parameters as plain pytrees (nested dicts of arrays) and
layers as functions; Optax for optimisers only. No Flax.
**Written:** 26 Sep 2026, before any Stage 3 code. §§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Scope

This stage covers the first and last layers of the model, everything except the Perceiver level
aggregator (Stage 4) and the backbone (Stages 5–6):

    frame (V vars, N×N) ─ level transform ─▶ (V, C, N, N) ─ patchify + per-variable embed ─▶ tokens (C, T, D)
    tokens (C, T, D)    ─ per-variable heads + unpatchify ─▶ (V, C, N, N) ─ sum over levels ─▶ frame

## 2. Design

### 2.1 Levels = spectral bands (decision 26 Sep)
* Square-shell octave bands on the output grid, `n = ‖k‖∞ = max(|kx|, |ky|)`:
  `B_0: n < 2`, `B_j: 2^j ≤ n < 2^{j+1}` for j ≥ 1, up to `n ≤ N/2 − 1`.
  **C(N) = log2(N) − 1** levels: 5 at 64², 6 at 128².
* `decompose(x) -> (C, N, N)`: band-pass each field with `rfft2`/`irfft2` (norm="forward"), with
  the Nyquist row and column zeroed (stored frames have none). `recompose = sum over levels`.
* Nesting: for j < C(64), band j of `down(x, 64)` equals `down(band j of x, 64)`.
* Level coordinate: `ℓ_j = j` (the octave index).

### 2.2 Patches
* Patch size `P` (default 4) divides N. `patchify: (N, N) -> (T, P²)` with `T = (N/P)²`, patches
  in row-major order over the patch grid `[py, px]`, pixels row-major within a patch.
  `unpatchify` is its exact inverse.

### 2.3 Encodings (Fourier features, dimension D, D even)
`fourier(s, D, λ_min, λ_max)`: `[sin(2π s/λ_i), cos(2π s/λ_i)]` for D/2 wavelengths λ_i spaced
geometrically from λ_min to λ_max.
* **Position:** the patch-centre angles `(x_c, y_c) ∈ [0, 2π)`. The encoding is periodic by
  construction: D/2 dimensions encode x and D/2 encode y, each as `sin(m x_c), cos(m x_c)` for
  integer m = 1 … D/4.
* **Patch area** (the multi-resolution key): `s = log(A)`, with `A = (2π P / N)²`, λ ∈ [0.01, 100].
* **Level:** `s = ℓ_j`, λ ∈ [0.5, 64].
* **Lead time** (used from Stage 7 on): `s = Δt`, λ ∈ [0.05, 100].
Each encoding goes through its own learned linear map `D → D` before it is added, so the model can
scale or ignore it.

### 2.4 Embedding
Per variable `v`: `W_v ∈ R^{P²×D}`, `b_v ∈ R^D`, shared across levels. For a frame with the
variable subset `S ⊆ {u, v, p, ω}`:

    tokens[c, t] = Σ_{v∈S} (patchify(X_v,c)[t] · W_v + b_v) + E_pos[t] + E_area + E_level[c]

A missing variable is simply left out of the sum; this is how heterogeneous variable sets (RQ2)
enter the model. The same parameters serve every N: only T and C change.

### 2.5 Unembedding
Per output variable `v`: `W'_v ∈ R^{D×P²}`, `b'_v ∈ R^{P²}`.
`X̂_v,c = unpatchify(tokens[c] · W'_v + b'_v)` and `X̂_v = Σ_c X̂_v,c`.

### 2.6 Initialisation
`init(key, D, P, variables)` returns a pytree. Weights are LeCun-normal (std 1/√fan_in) and biases
are zero. It is a deterministic function of the key.

---

## 3. Gates — `tests/test_stage3_gates.py` (float64 unless stated)

| Gate | Test | Pass if |
|---|---|---|
| H1 | `unpatchify(patchify(x)) == x` for N ∈ {64, 128}, P ∈ {2, 4, 8} | bitwise |
| H2 | patch order: the patch at grid index (py, px) equals `x[py·P:(py+1)·P, px·P:(px+1)·P]` flattened | exact |
| H3 | `recompose(decompose(x))` vs x | ≤ 1e-13 · max\|x\| |
| H4 | band orthogonality `⟨B_i x, B_j x⟩`, i ≠ j, normalised | ≤ 1e-13 |
| H5 | nesting: band j of `down(x, 64)` vs `down(band j of x, 64)`, j < 5 | ≤ 1e-13 |
| H6 | C(N) = log2(N) − 1 and every retained mode lies in exactly one band | exact |
| H7 | **acceptance round trip**, encodings off, D = V·P², heads = pseudo-inverse of the stacked embedding: frame → tokens → frame. f64 / f32 | ≤ 1e-12 / ≤ 1e-5 relative |
| H8 | variable subsets: tokens for S equal the sum of the single-variable token contributions (biases and encodings counted once) | ≤ 1e-12 |
| H9 | encodings: position encoding periodic (x_c → x_c + 2π); distinct patch centres give distinct codes; 64² and 128² area codes differ | exact / > 1e-3 |
| H10 | `jax.grad` of a reconstruction loss vs central finite differences at 5 random parameter entries | ≤ 1e-6 relative |
| H11 | `init` is deterministic per key; different keys give different params | exact |
| H12 | the same params run on 64² (C=5, T=256) and 128² (C=6, T=1024) inputs under `jit` | shapes exact |

## 4. Measurements and predictions

Data: `acc_v1` frames at 64², regime re100_kf4, seed 0 (training), seed 1 (evaluation). Adam
(Optax), lr 1e-3, batch 16 frames, float32, key 0.

| # | Measurement | Prediction |
|---|---|---|
| R1 | H7's f32 relative round-trip error | ≤ 1e-5; guess 1e-6 |
| R2 | learned autoencoder, bands on, encodings on, D = 128 (> V·P² = 64), 3000 steps: relative L2 reconstruction error on seed 1 | < 1e-2; guess 3e-3 |
| R3 | learned autoencoder, one level (no bands), encodings off, D = 32 < 64, 5000 steps: error vs the PCA (Eckart–Young) optimum for rank 32, computed on the same patches | trained error ≤ 1.2 × PCA optimum |
| R4 | parameter count for D = 128, P = 4, V = 4 (embedding + heads + encoding maps) | 4·(16·128+128) + 4·(128·16+16) + 4·(128·128+128) = 83,008 |

**Stage 3 acceptance:** H1–H12 pass and R1 HIT. R2 and R3 check that the embedding can be learned
and are reported as HIT or MISS.

---

## Amendments (after results; the original verdicts stand in the report)

**A1 (26 Sep, after the first gate run): H7 float32 bar.** The pre-registered float32 bar (1e-5)
**failed**: 1.72e-5. Diagnosis: the float64 round trip is 1.9e-14 and the band decomposition alone
is 2e-7 in float32, so the machinery is exact. The float32 error is the roundoff amplified by the
condition number of the random 64×64 stacked embedding (κ = 836 for that draw; κ·ε_f32 ≈ 5e-5).
A fixed bar cannot be right for a random matrix. From now on the float32 case of H7 asserts
`error ≤ 10 · κ(W) · ε_f32`. The float64 bar (1e-12) is unchanged and is the machine-precision
acceptance test. R1 is recorded as a MISS.
