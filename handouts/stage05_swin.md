# Stage 5 Spec: 3-D windowed and shifted-window attention (Swin3D block)

**Plan acceptance test (PROJECT_PLAN.md §5):** matches a reference Swin block on random input to 1e-5.
**Reference implementation:** Aurora's own PyTorch code, `microsoft-aurora==2.0.1`
(`aurora.model.swin3d.Swin3DTransformerBlock`), installed in a separate reference environment
(`python3.11 -m venv refenv && refenv/bin/pip install microsoft-aurora`), never imported by the
project code. **Stack:** pure JAX pytrees. **Written:** 26 Sep 2026, before any Stage 5 code.
§§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. What the block does

Tokens live on a 3-D grid `(C, H, W)`: latent levels × patch rows × patch columns. The block does
self-attention inside non-overlapping 3-D windows `(Wc, Wh, Ww)`. Alternate blocks shift the windows
by half a window, so information crosses window boundaries.

## 2. Design (matching Aurora's block exactly, with one generalisation)

For input `x (B, C·H·W, D)` and conditioning `c (B, D_t)`:

1. `ws, ss = adjust(window, shift, (C, H, W))`: any axis where `res ≤ window` gets window = res
   and shift = 0 (Aurora's `maybe_adjust_windows`).
2. View as `(B, C, H, W, D)`. If any shift is non-zero, cyclically roll by `−ss` along (C, H, W)
   and build the attention mask (below).
3. Partition into windows `(B·nW, Wc·Wh·Ww, D)`, in window order `(B, C1, H1, W1)` and token order
   `(Wc, Wh, Ww)` (Aurora's `window_partition_3d`). **Every dimension must be divisible by its
   window**; there is no padding, and a non-divisible shape raises ValueError. (Aurora pads; this
   project picks window sizes that divide the grid.)
4. Window attention: `qkv = x W_qkv + b_qkv` (with bias), split in the order `(qkv, heads, head_dim)`
   as `"B N (qkv H D)"`; `softmax(q kᵀ/√d_h + mask) v`; merge heads; output projection `W_p, b_p`.
5. Reverse the partition, roll back by `+ss`, and flatten.
6. **Res-post-norm with adaptive LayerNorm (FiLM):**
   `x = x_in + AdaLN_1(attn_out, c)`, then `x = x + AdaLN_2(MLP(x), c)`, where
   `AdaLN(y, c) = LN_no_affine(y) · (scale_bias + s) + t`, `[t, s] = Linear(SiLU(c))`, split
   **shift first, then scale** (Aurora's `chunk(2)`), and
   `MLP = Linear(D, 4D) → GELU (exact, erf) → Linear(4D, D)`.

**Shifted-window mask, generalised by periodic axes.** Each axis is periodic or not. For a
non-periodic axis with shift s > 0, tokens fall into three groups by position after the roll:
`[0, −w)`, `[−w, −s)`, `[−s, end)`. A periodic axis has a single group. A token's group is the
tuple of its axis groups. Two tokens may attend to each other only if their groups are equal. The
masked logit is −∞.
* `periodic = (False, False, True)` reproduces Aurora's `warped=True` mask (its merge groups make W
  periodic), up to Aurora using −100 in place of −∞.
* **This project uses `periodic = (False, True, True)`:** levels are not periodic; x and y are
  (a doubly periodic domain).

**Parameters (pytree):** `qkv {W (D, 3D), b (3D)}`, `proj {W, b}`,
`norm1 {W (D_t, 2D), b (2D)}`, `norm2 {…}`, `mlp {W1, b1, W2, b2}`. Initialisation follows Aurora:
truncated normal (std 0.02) for linear layers and **zeros for the AdaLN modulation**, so each block
is the identity at initialisation.

---

## 3. Gates — `tests/test_stage5_gates.py`

The reference outputs come from `experiments/stage5/make_reference.py`, run in the reference
environment. It writes `tests/fixtures/swin3d_ref.npz` (random weights, including non-zero AdaLN
modulation, inputs and Aurora outputs), which is committed so the tests need no PyTorch.

| Gate | Test | Pass if |
|---|---|---|
| W1 | window partition and reverse vs Aurora's `window_partition_3d` / `window_reverse_3d` output (from the fixture) | bitwise |
| W2 | shifted-window mask with `periodic = (F, F, T)` vs Aurora's `compute_3d_shifted_window_mask(warped=True)` (−100 ↔ masked) | exact |
| W3 | **acceptance:** block output vs Aurora's `Swin3DTransformerBlock` for 4 cases (unshifted/shifted × two shapes) in float32 | ≤ 1e-5 max-abs relative to max\|out\| |
| W4 | same in float64 | ≤ 1e-12 |
| W5 | doubly periodic mode, shift (0, s_h, s_w): block equals roll ∘ unshifted-unmasked-block ∘ roll⁻¹ | ≤ 1e-12 |
| W6 | doubly periodic mode: cyclic translation by whole windows in H and W commutes with the block | ≤ 1e-12 |
| W7 | identity at initialisation (zero modulation, scale_bias = 0) | exact |
| W8 | `jax.grad` vs central finite differences, 5 entries | ≤ 1e-6 relative |
| W9 | a non-divisible shape raises ValueError; `adjust` matches Aurora's `maybe_adjust_windows` on 6 cases | raises / exact |

## 4. Predictions

| # | Quantity | Prediction |
|---|---|---|
| T1 | W3 worst float32 error | ≤ 1e-5; guess 1e-6 |
| T2 | W4 worst float64 error | ≤ 1e-12; guess 1e-14 |

**Stage 5 acceptance:** W1–W9 pass (W3 is the plan's test).

## Amendment A1 (28 Sep 2026): gate W10
Adds **W10**: the masked shifted-window block (a shifted non-periodic axis) must run under
`jax.jit` and match the eager result and Aurora's reference (< 1e-12, float64). W1–W9 exercised
the mask only eagerly, and the first jitted use (E4's L = 6 variant) failed. W1–W9 and T1–T2 are
unchanged.
