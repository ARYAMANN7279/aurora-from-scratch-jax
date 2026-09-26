# Stage 7 Spec: Full model (encoder → backbone → decoder)

**Plan acceptance test (PROJECT_PLAN.md §5):** identity-ish at initialisation; trains one-step
prediction.
**Stack:** pure JAX pytrees + Optax. **Written:** 26 Sep 2026, before any Stage 7 code. §§2–4 are
fixed, and the Stage 1 rules apply.

---

## 1. Design

    (X^{t−1}, X^t), variables S ⊆ {u, v, p, ω}, N×N
      ─ Stage 3 embed, per frame (bands on: C = log2 N − 1 levels) + learned frame vector, summed
        ─▶ tokens (C, T, D)
      ─ Stage 4 aggregate ─▶ latents (L, T, D)
      ─ Stage 6 backbone on the token grid (L, N/P, N/P), conditioned on the lead-time embedding ─▶ (L, T, 2D)
      ─ Linear(2D → D) ─ Stage 4 decode ─▶ (C_out, T, D) ─ Stage 3 heads (all 4 variables) ─ sum over levels ─▶ Δ
    X̂^{t+1} = X^t + Δ   (for variables absent from the input, the residual base is 0)

* **History.** Summing per-frame linear embeddings plus a learned per-frame vector is exactly a
  linear embedding of the concatenated two-frame history. Aurora does the same thing with a joint
  patch embedding over time.
* **Lead time.** `t_emb = lead_time(Δt)` (Stage 3 Fourier encoding). Δt = stride × 0.2.
* **Identity at initialisation.** The heads' weights and biases start at **zero**, so Δ = 0 and
  X̂^{t+1} = X^t (persistence) exactly. The backbone is also the identity at initialisation (zero
  AdaLN, Stage 5), apart from its patch merging/splitting.
* **Resolution-agnostic:** one parameter set serves 64² (C = 5, token grid 16×16) and 128²
  (C = 6, 32×32). The backbone window `(Wc, Wh, Ww)` must divide `(L, H', W')` at every stage.
* **Default configuration:** D = 64, P = 4, L = 3, Perceiver H = 4 with 1 block per side, backbone
  S = 3, depths (2, 2, 2) / (2, 2, 2), heads (2, 4, 8) / (8, 4, 2), window (3, 4, 4),
  `periodic = (False, True, True)`.

## 2. Gates — `tests/test_stage7_gates.py` (float64 unless stated)

| Gate | Test | Pass if |
|---|---|---|
| M1 | at initialisation, X̂^{t+1} == X^t, for 64² and 128² inputs | exact |
| M2 | shapes: all four variables out, for every input subset S (including S = {ω}) | exact |
| M3 | the same params run on 64² and 128² under `jit` | shapes exact |
| M4 | translation equivariance: shifting both input frames by 16 patches (64 grid points, half the domain) in x and y shifts the output identically, on 128² input. The band transform, Perceiver and heads are translation-covariant; the backbone needs shifts of 2^{S−1}·window = 16 patches. (At 64², 16 patches is the whole domain, so the test would be vacuous.) | ≤ 1e-10 |
| M5 | `jax.grad` vs central finite differences (h = 1e-4, mixed tolerance as in Stage 6 A1), 5 entries | ≤ 1e-6 |
| M6 | a variable absent from the input gets residual base 0 (checked by feeding S = {u, v} and verifying `X̂_p = Δ_p`) | exact |

## 3. Measurement: one-step training

* **Data:** `acc_v1`, regimes re100_kf4 and re300_kf4, 64². Training seeds 0–5, evaluation seeds 6–7
  (in-distribution, disjoint trajectories). Windows `(X^{t−1}, X^t, X^{t+1})`, stride 1 (Δt = 0.2).
  Normalisation per channel from the training frames of both regimes together.
* **Training:** Adam, lr 1e-3 with cosine decay to 1e-5, batch 8, **3000 steps**, float32, key 0.
  Loss MSE (the MAE ablation is Stage 8/E4).
* **Metric:** relative L2 error of one-step predictions on the evaluation windows, per regime,
  against persistence.

| # | Measurement | Prediction |
|---|---|---|
| N1 | parameter count of the default configuration | reported (exact, from `n_params`) |
| N2 | persistence relative error on the evaluation windows | Re100 in [0.1, 0.35]; Re300 in [0.15, 0.45] |
| N3 | trained one-step relative error / persistence, each regime | < 0.5 |
| N4 | trained one-step relative error, each regime | < 0.1 |

**Stage 7 acceptance:** M1–M6 pass, and N3 HIT ("trains one-step prediction": clearly better than
persistence on unseen trajectories).

---

## Amendments (after results; the original verdicts stand in the report)

**A1 (26 Sep, after the first gate run): M4.** As written, M4 **failed** (max error 8.28). The
model is not translation-equivariant, and should not be. The spec overlooked that the Stage 3
embedding adds an **absolute position encoding**. The forcing `−A k_f cos(k_f y)` breaks
translation symmetry in y (except by multiples of 2π/k_f), so the model needs absolute position.
The gate was meant to verify that every other component (band transform, embedding, Perceiver,
backbone, heads) is translation-covariant. From now on M4 checks that property with the
position-encoding map set to zero, which removes the only position-dependent term.
