# How the model works

A walkthrough of the JAX model in `src/aurora/` at the configuration every experiment uses (E1–E5):
64² fields, width D = 64, patch P = 4, L = 3 latent levels, 12 Swin blocks. It sits next to Aurora
(Bodnar et al., *Nature* 2025, and its code `microsoft-aurora` 2.0.1). The parts are the same; the
sizes and the meaning of "levels" differ.

## 1. What the model computes

A **one-step simulator**: given two consecutive snapshots of the flow, X(t−Δt) and X(t), predict
X(t+Δt), with Δt = 0.2. Each snapshot has 4 fields on a 64 × 64 grid: velocity u, v, pressure p
and vorticity ω, each normalised to zero mean and unit variance. The network predicts only the
**change**: output = X(t) + Δ, and Δ is exactly zero at initialisation (the heads start at zero),
so an untrained model is persistence. Applied repeatedly to its own output, it forecasts further;
that is a **rollout** (`src/aurora/rollout.py`).

## 2. The forward pass, with shapes

Per sample; training adds a batch axis in front (16 in E1).

| # | Step | Code | Shape |
|---|---|---|---|
| 1 | input: two snapshots | `model.apply` | 2 × (4, 64, 64) |
| 2 | split each field into **spectral bands** with the FFT (‖k‖∞ in [0,2), [2,4), [4,8), [8,16), [16,32)); the bands sum back to the field | `levels.decompose` | (4, 5, 64, 64) per snapshot |
| 3 | cut into 4 × 4 patches: 256 patches of 16 values | `patches.patchify` | (4, 5, 256, 16) |
| 4 | per-variable linear embedding, summed over variables | `embed.embed` | (5, 256, 64) = (levels C, positions T, D) |
| 5 | add encodings: patch position, patch area, level | `embed.encoding_terms` | (5, 256, 64) |
| 6 | the two snapshots: embed each, add a learned frame vector, sum | `model.apply` | (5, 256, 64) |
| 7 | **Perceiver encoder**: at each position, 3 learned latent queries attend over the 5 level tokens | `perceiver.aggregate` | (3, 256, 64) |
| 8 | lead-time encoding of Δt → time MLP → conditioning vector c | `encodings.lead_time`, `backbone.apply` | (64,) |
| 9 | **Swin U-Net backbone** on the 3 × 16 × 16 token grid (below) | `backbone.apply` | (768, 64) → (768, 128) |
| 10 | linear 128 → 64 | `model.apply` (`mid`) | (3, 256, 64) |
| 11 | **Perceiver decoder**: at each position, 5 level queries attend over the 3 latents | `perceiver.decode` | (5, 256, 64) |
| 12 | per-variable heads (64 → 16), un-patch, sum over levels | `embed.unembed` | (4, 64, 64) = Δ |
| 13 | residual: X(t) + Δ | `model.apply` | (4, 64, 64) |

**The backbone** (`backbone.py`, `swin.py`) works on the token grid (C, H, W) = (3, 16, 16):

| Stage | Grid | Tokens | Width | Blocks | Heads | Windows of 3 × 4 × 4 = 48 tokens |
|---|---|---|---|---|---|---|
| encoder 1 | 3 × 16 × 16 | 768 | 64 | 2 | 2 | 16 |
| encoder 2 | 3 × 8 × 8 | 192 | 128 | 2 | 4 | 4 |
| encoder 3 | 3 × 4 × 4 | 48 | 256 | 2 | 8 | 1 |
| decoder 1 | 3 × 4 × 4 | 48 | 256 | 2 | 8 | 1 |
| decoder 2 | 3 × 8 × 8 | 192 | 128 | 2 | 4 | 4 |
| decoder 3 | 3 × 16 × 16 | 768 | 64 | 2 | 2 | 16 |

* Between encoder stages, **patch merging** joins 2 × 2 neighbouring tokens and doubles the width.
  Between decoder stages, **patch splitting** reverses it.
* Encoder 1's output is added after decoder 2 and concatenated after decoder 3. These are the
  U-Net skips, and the concatenation is why the output width is 128.
* Every block is "res-post-norm": x ← x + AdaLN(attention(x)), then x ← x + AdaLN(MLP(x)). AdaLN is
  a LayerNorm whose scale and shift come from the lead-time vector c.
* Attention runs inside windows of 48 tokens. Every second block shifts the windows by half a
  window (0, 2, 2) so that information crosses window borders. The level axis has only 3 entries,
  so its window covers it whole and it is never shifted. x and y are periodic, so the shift wraps
  around and no window needs a mask.

## 3. Part by part, against Aurora

| Part | Aurora (1.3 billion parameters) | Ours (5.1 million) |
|---|---|---|
| Levels | atmospheric pressure levels | spectral octave bands of each field (5 at 64², 6 at 128², 8 at 512²) |
| History | 2 steps inside each patch: a 3-D convolution over (time, patch), separate weights per step | each snapshot embedded with the same weights, plus a learned frame vector |
| Encodings | position, patch area, level, lead time, absolute time (Fourier features) | position, patch area, level, lead time |
| Encoder | Perceiver: levels → 4 latent levels | Perceiver: levels → 3 latent levels |
| Backbone | Swin U-Net, width 512, depths 6, 10, 8 / 8, 10, 6 (48 blocks), windows 2 × 6 × 12 | Swin U-Net, width 64, depths 2, 2, 2 / 2, 2, 2 (12 blocks), windows 3 × 4 × 4 |
| Shifted windows | wrap around in longitude only, masked in latitude | wrap around in x and y (the box is periodic) |
| Attention kernel | PyTorch's fused `scaled_dot_product_attention` | explicit softmax; on the lab card a fused kernel makes no difference in PyTorch and is 8% slower in JAX (Stage 12, T3 and J5) |
| Position inside windows | no relative-position term: positions enter only through the input encodings | the same by default; optionally a learned bias per head for each offset (`rel_pos_bias`; E7 and E8: −6.9% one-step error, a gain that fades over 20–30 rollout steps) |
| Query and key scale | unnormalised | the same by default; optionally a LayerNorm on each head's queries and keys (`qk_norm`; E8: −7.3% one-step error, at 10% more time) |
| Loss | MAE | MSE (MAE tested in E4) |
| Fine-tuning | full, and LoRA for long rollouts | full and LoRA (Stage 10, E5) |

The Swin block and the backbone were checked against Aurora's own code. They agree to 2.8e-7
(float32) and 3e-16 (float64) when given the same weights (Stages 5 and 6).

## 4. Where the parameters are

| Component | Parameters | Share |
|---|---:|---:|
| backbone, deepest stage (encoder 3 + decoder 1, width 256) | 3,421,184 | 67.0% |
| backbone, middle stage (encoder 2 + decoder 2, width 128) | 924,160 | 18.1% |
| backbone, first stage (encoder 1 + decoder 3, width 64) | 265,472 | 5.2% |
| patch merging and splitting | 350,080 | 6.9% |
| Perceiver encoder and decoder (with latents and level queries) | 103,936 | 2.0% |
| embeddings, encodings, heads, frame vectors, time MLP, `mid` | 41,856 | 0.8% |
| **total** | **5,106,688** | |

## 5. Where the computation is

A rough count of multiply-adds in one forward pass of one sample (2 FLOP each). Elementwise work
(LayerNorm, softmax, GELU, reshapes) adds little arithmetic, but it moves memory and launches
kernels.

| Component | ≈ MFLOP |
|---|---:|
| spectral split (FFTs) and patch embedding, both snapshots | 35 |
| Perceiver encoder | 85 |
| backbone: 12 blocks of about 80 MFLOP each, plus merging and splitting | 1,040 |
| `mid`, Perceiver decoder and heads | 140 |
| **total** | **≈ 1,300** |

* The linear layers (q, k, v, output projection and the MLP) do most of the arithmetic. The
  attention products are about 7% of the backbone's (11% in the first stage), because windows of
  48 tokens are small.
* A training step (forward and backward, about 3× a forward pass) at batch 16 is therefore about
  60 GFLOP. At the measured ~10 ms per step that is ~6 TFLOP/s, far below the card's float32 peak
  (on the order of 100 TFLOP/s).
* Turning TF32 off cost only 7% (`project_memory.md` §3). Both point to a step bounded by kernel
  launches and host work rather than by arithmetic.
* Stage 12 confirmed it (`experiments/stage12/REPORT.md`):
  * on the lab card, the training loop keeps the card about half busy;
  * running 50 steps per jitted call (`lax.scan`) trains 2.07× faster;
  * TF32 and bfloat16 gain 2% and 12%.
  * On a T4, a card about 15× slower in float32, the same step is bound by the card instead.

## 6. Reading guide

| Part | Our code | Our spec | Aurora's code (`microsoft-aurora` 2.0.1) |
|---|---|---|---|
| spectral bands | `src/aurora/levels.py` | `handouts/stage03_embed.md` | (pressure levels; no counterpart) |
| patches, embedding, encodings | `patches.py`, `embed.py`, `encodings.py` | `stage03_embed.md` | `model/patchembed.py`, `model/fourier.py`, `model/posencoding.py`, `model/encoder.py` |
| Perceiver | `perceiver.py`, `attention.py` | `stage04_perceiver.md` | `model/perceiver.py`, `model/encoder.py`, `model/decoder.py` |
| Swin block | `swin.py` | `stage05_swin.md` | `model/swin3d.py` (`Swin3DTransformerBlock`, `WindowAttention`) |
| backbone | `backbone.py` | `stage06_backbone.md` | `model/swin3d.py` (`Swin3DTransformerBackbone`) |
| full model | `model.py` | `stage07_model.md` | `model/aurora.py` |
| rollout, pushforward | `rollout.py` | `stage08_rollout.md`, `stage09_pushforward.md` | `rollout.py` |
| LoRA | `lora.py` | `stage10_lora.md` | `model/lora.py` |

A good order: the paper's architecture section first, then steps 1–13 above with `model.py` open,
then each row of the reading guide.
