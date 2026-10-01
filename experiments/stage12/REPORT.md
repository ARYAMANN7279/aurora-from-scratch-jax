# Stage 12 Report: Speed — what makes training faster, in JAX and against PyTorch

Spec: `handouts/stage12_speed.md` (amendments A1 and A2).

**Status: in progress (1 Oct 2026).** The re-timed arms (amendment A2) wait for an idle GPU. So far
this file has the set-up (§2) and the ideas for the attention (§9). The results, the verdicts and the
other sections will be added when the timings land. The first-pass verdicts (S12-P1 and P2 HIT, P4
MISS) are in the README and `project_memory.md`.

## 2. Set-up

* **Card:** one idle NVIDIA RTX PRO 6000 Blackwell (Workstation Edition) of the lab machine, from
  cards 1–3 (card 0 is excluded). The lane claims a card only when nobody else's process is on it
  and it is below 85 °C.
* **Software:**
  * JAX 0.10.2 with CUDA 12;
  * PyTorch 2.11.0+cu128;
  * microsoft-aurora 2.0.1, for A0.
* **Precision:** float32 with TF32 off on both sides (JAX matmul precision `highest`), unless an
  arm says otherwise.
* **One number:** wall time per step = (time for 500 steps) / 500.
  * The device is synchronised at both ends, after 100 warm-up steps.
  * Each arm runs 3 repeats in one process; the median is reported with the spread.
  * Compile time (the first call) is reported separately.
* **Same computation (A1):** loaded with the JAX weights, the PyTorch port reproduces the JAX model
  in float64:
  * forward pass within 2.8e-13;
  * every gradient within 8.4e-10 (`results/torch_equivalence_cpu_f64.json`).

  In float32 each framework is about 1e-4 from its own float64 result. The spec's float32 bars were
  set below that rounding floor (A1).
* **Host load (A2):**
  * The first J0–J3 timings ran while the shared host was at load 11–14 on 16 cores. These are in
    `results/first_pass/`.
  * Every arm was then timed again in one lane, one job after another, with J0 first and last.
    The host's load average and CPU pressure were logged every 30 s (`logs/hostload.log` on the
    lab machine).
* **No overlap with other jobs:** the lane runs under `glane4.sh` with `PREEMPT_S=0`.
  * It checks the card every 5 s.
  * If another process appears on the card, or the card reaches 87 °C, the job is killed at once
    and rerun from the start on the next idle card.
  * So no reported timing overlaps another process: the spec's §2 discard rule, applied to the
    whole arm.

## 9. Ideas for the multi-head attention

The advisor asked for ideas to improve the multi-head attention. Three facts shape the list:
* **Small windows.** Attention runs in windows of 48 tokens (3 × 4 × 4), with head dimension 32 at
  every stage. Its two products, QKᵀ and AV, are about 7% of the backbone's arithmetic
  (`docs/architecture.md` §5). The q, k, v and output projections and the MLPs do the rest.
* **Host-bound step.** The training step is bound by the host and by kernel launches, not by
  arithmetic. J0 keeps the card 26% busy, and 4× the batch costs 1.16× the time (first timings).
* **No positional term.** Neither our attention nor Aurora's own (`aurora/model/swin3d.py`, fused
  `scaled_dot_product_attention`) has a relative-position term. A token's place in its window
  reaches attention only through the encodings added at the input.

A faster attention kernel alone can therefore buy little. The ideas fall into two kinds.

**The same function, faster.** None of these changes the model's output, so each can be timed like
J5.
1. **A fused attention kernel.** J5 (`jax.nn.dot_product_attention`) and T3 (PyTorch's
   `scaled_dot_product_attention`) measure it. With bfloat16 inputs (J6), cuDNN's fused kernel
   becomes available (`implementation="cudnn"` needs half precision).
2. **Shifted windows by one gather.** A shifted block today rolls the grid, partitions it,
   attends, reverses the partition and rolls back: four passes over the activations besides the
   attention itself. A precomputed index gather and its scatter do the same in two passes. XLA may
   already fuse some of this, so profile before writing it.
3. **Projections before partitioning.** q, k and v are per-token. Computing them on the whole grid
   and partitioning once avoids reshaping a tensor three times as wide.

These save memory passes and launches. They start to matter only once the step is no longer
host-bound, as in J3.

**A different function.** Each of these changes accuracy as well as speed, so each needs its own spec
with accuracy predictions (spec §6). They are ranked by expected value for this project.
1. **Relative position inside windows.** Two ways to do it:
   * a learned bias per head for each of the (2·3 − 1)(2·4 − 1)(2·4 − 1) = 245 offsets, as in Swin;
   * rotary embeddings on q and k along the three axes.

   Either makes attention depend on the offset between tokens. That is the natural bias for a
   translation-invariant flow, whose update at a point depends on its neighbours' relative
   positions, as a finite-difference stencil does. The cost is negligible.
2. **Normalising q and k** (a LayerNorm per head, as in ViT-22B, or Swin V2's cosine attention).
   This keeps the logits bounded, which is what lets attention train stably in bfloat16. J6 tests
   bfloat16 for the forward pass only, so this pairs with an end-to-end bfloat16 training spec.
3. **A global mixer at the fine stages.**
   * **Where attention is already global:** at encoder 3 and decoder 1 one window covers the whole
     3 × 4 × 4 grid.
   * **Where it is not:** at the 16 × 16 stages a token reaches at most 3 × 8 × 8 tokens after two
     blocks.
   * **Larger windows in stage 1** (3 × 8 × 8 = 192 tokens) would raise the attention products from
     about 11% to about 33% of that stage's arithmetic, a small cost in a host-bound step.
   * **Fourier mixing**, in the style of AFNO or FNO: global, at O(N log N). The organisers' FNO is
     the strongest model UGP-I measured on the RealPDE windows (subscore 95.47, E6).
   * **Softmax-free "Galerkin" attention** from operator learning, linear in the number of tokens.
4. **Fewer key and value heads** (multi-query or grouped-query attention). This saves part of the k
   and v projections: about 8% of a block's arithmetic at stage 1 and 15% at stage 3. A host-bound
   step would hardly notice, so this comes last.
