# Stage 12 Report: Speed — what makes training faster, in JAX and against PyTorch

Spec: `handouts/stage12_speed.md` (amendments A1, A2 and A3; supplement §9). Scored by `score.py`:
`results/analysis.json` (the sequential lane) and `results/first_pass/analysis.json` (A2).

## Verdict

**Stage 12 acceptance MET.** Running 50 training steps per jitted call (`lax.scan`, J3) trains the
E1 model **2.07×** faster than the E-runs' loop (J0): 4.94 ms per step against 10.19 ms, with the
same numerics. **6 of 10 predictions HIT** (P1, P2, P5, P7, P8, P9); P3, P4, P6 and P10 MISS.

The step is **bound by the host, not by the card**:
* J0 keeps the card about half busy (49%).
* 4× the batch costs only 1.78× the time.
* J0's time follows the host's load. It ran at 19.2 ms with the shared host at load 11–14, and at
  10.2 ms at load 4.5.

Everything that helps removes host work per step:
* **scan (J3):** 2.07×. Its time did not move with the host load (4.92 ms against 4.94 ms).
* **CUDA graphs in PyTorch (T2):** 1.36× faster than J0.
* **Buffer donation (J2):** 1.13×, or 1.58× on the loaded host.
* **Training 3 seeds at once (J4):** 1.67× per model.

Everything that speeds up arithmetic does little:
* bfloat16 (J6): 1.12×.
* TF32 (J6t): 1.02×.
* A fused attention kernel (J5): 0.92×.

**Against PyTorch:**
* Eager PyTorch (T0) is 2.42× slower than J0.
* `torch.compile` (T1) is within 11% of J0.
* `torch.compile` with CUDA graphs (T2) beats J0, but JAX with scan is still 1.52× faster than T2.

**On a T4 (supplement, §7) the picture inverts. 12 of 13 supplement predictions HIT.**
* The same step is bound by the card: it is 78% busy, and 4× the batch costs 3.31× the time.
* So the host-side tools barely matter there: scan gives 1.16×, donation 1.00×, vmap over seeds
  0.90× per model.
* Two T4s (J7) train 1.24× faster than one.
* Compiled PyTorch is within 9% of JAX.

## 1. Results on the lab card

The sequential lane: card 3 (RTX PRO 6000 Blackwell), 2 Oct 13:53–14:19, batch 16, float32 (TF32 off),
median of 3 × 500 steps. ×J0 is J0's time divided by the arm's, so above 1 is faster. Util is the
mean card utilisation during the repeats.

| Arm | What changes | ms per step | ×J0 | Util | Spread |
|---|---|---|---|---|---|
| J0 | the E-runs' loop (`harness.py`): numpy sampling, one jitted step per iteration | 10.19 | 1.00 | 49% | 4.4% |
| J1 | batch 32 | 12.19 | (1.20× the time) | 58% | 0.6% |
| J1 | batch 64 | 18.11 | (1.78× the time) | 67% | 0.5% |
| J2 | parameters and optimiser state donated (`donate_argnums`) | 8.99 | 1.13 | 57% | 3.4% |
| J3 | 50 steps per jitted call (`lax.scan`), sampling on the device | **4.94** | **2.07** | 100% | 0.2% |
| J4 | 3 seeds in one step (`vmap`): 18.34 ms per step of 3 models | 6.11 per model | 1.67 per model | 57% | 0.7% |
| J5 | attention through `jax.nn.dot_product_attention` | 11.03 | 0.92 | 44% | 3.7% |
| J6 | bfloat16 matmuls (float32 parameters) | 9.13 | 1.12 | 44% | 1.0% |
| J6t | TF32 matmuls (diagnostic) | 9.97 | 1.02 | 40% | 10.4% |
| fwd | forward pass alone, JAX | 1.09 | | 50% | 0.3% |
| T0 | the PyTorch port, eager | 24.69 | 0.41 | 49% | 0.7% |
| T1 | T0 under `torch.compile` | 11.35 | 0.90 | 68% | 0.3% |
| T2 | `torch.compile(mode="reduce-overhead")`, CUDA graphs | 7.52 | 1.36 | 93% | 4.4% |
| T3 | T1 with `scaled_dot_product_attention` | 11.16 | 0.91 | 77% | 1.1% |
| fwd_eager | forward pass alone, PyTorch eager | 5.54 | | 57% | 0.3% |
| fwd_compile | forward pass alone, `torch.compile` | 1.97 | | 84% | 0.2% |
| A0 | Aurora's own backbone (forward and backward), eager / compiled | 14.41 / 7.07 | | 46% / 68% | |
| B0 | our ported backbone on the same input, eager | 18.29 | | 47% | 0.6% |
| J0, again at the end | the lane's drift check | 9.78 | (−4.0%) | 51% | 4.0% |

* **bfloat16 does not cost accuracy.** Matmuls in bfloat16 change the trained E1 model's one-step
  test error by +0.002% (0.083990 → 0.083991; `results/accuracy_bf16.json`).
* **First pass (A2), 1 Oct, host load 11–14** (`results/first_pass/`): J0 19.23, J1 19.69 (32) and
  22.25 (64), J2 12.18, J3 4.92 ms.
* **Card 2 (A3), 1 Oct, host load 4.7:** J0 10.30 ms (`results/card2/`).
* **A0 and B0 are not the same computation.** Aurora masks shifted windows along H and uses SDPA;
  ours wraps both axes. They are a reference point, not a prediction.

## Predictions

| # | Prediction | Scored on | Measured | Verdict |
|---|---|---|---|---|
| S12-P1 | J1 at batch 64 < 2× J0 | first pass | 22.25 vs 19.23 ms: 1.16× (sequential lane: 1.78×) | **HIT** |
| S12-P2 | J3 ≤ 0.75 × J0 | first pass | 0.26× (sequential lane: 0.48×) | **HIT** |
| S12-P3 | one 3-model J4 step < 1.67 × J0 | sequential lane | 1.80× | **MISS** |
| S12-P4 | donation (J2) changes the time by < 5% | first pass | −37% (sequential lane: −12%) | **MISS** |
| S12-P5 | `dot_product_attention` (J5) changes the time by < 10% | sequential lane | +8.2% | **HIT** |
| S12-P6 | bfloat16 (J6) ≥ 1.3× faster, and its forward error changes by < 1% | sequential lane | 1.12×; +0.002% | **MISS** |
| S12-P7 | PyTorch eager (T0) ≥ 1.5× slower than J0 | sequential lane | 2.42× | **HIT** |
| S12-P8 | `torch.compile` (T1) within ±25% of J0 | sequential lane | +11% | **HIT** |
| S12-P9 | CUDA graphs (T2) ≥ 20% faster than T1 | sequential lane | 34% faster | **HIT** |
| S12-P10 | the forward pass within ±25% under JAX jit and `torch.compile` | sequential lane | `torch.compile` 1.81× slower | **MISS** |

**Acceptance:** a JAX configuration ≥ 1.5× faster than J0 with numerics unchanged. **MET** by J3
(2.07×). The spec counts J3 as the same numerics: the same step, with batches drawn from a different
random stream.

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
* **Card 3 only (A3).** Card 2 passed 87 °C within seconds under J1, and a timing run cannot be
  throttled.
* **No overlap with other jobs:** the lane runs under `glane4.sh` with `PREEMPT_S=0`.
  * It checks the card every 5 s.
  * If another process appears on the card, or the card reaches 87 °C, the job is killed at once
    and rerun from the start on the next idle card.
  * So no reported timing overlaps another process: the spec's §2 discard rule, applied to the
    whole arm.
  * This happened three times: J6t twice and B0 once, each time because another user's process
    appeared on the card. Each arm was rerun from the start.
* **Commit:** the lab machine's clone moved from 651344a to 56ba61c during the lane, between J6 and
  J6t, when the three copies of the repository were synced. No file the lane runs changed between the
  two (`bench_jax.py`, `bench_torch.py`, `torch_model.py`, `src/`, the E1 config).

## 3. Where the time goes

* **The step is bound by the host.**
  * J0 keeps the card 49% busy, and 4× the batch costs 1.78× the time.
  * J0's time follows the host's load: 19.2 ms at load 11–14 (first pass), 10.2 ms at load 4.5.
  * The E-runs measured "~10 ms per step" on an idle card. That is the same loop on a quiet host.
  * By the FLOP count of `docs/architecture.md` (~60 GFLOP per training step at batch 16), J0
    reaches about 6 TFLOP/s and J3 about 12, far below the card's float32 peak.
* **Removing host work per step is what pays.**
  * **scan (J3), 2.07×.** It moves the sampling onto the device and dispatches once per 50 steps.
    The card goes from 49% to 100% busy, and the time did not move with host load (4.92 ms then,
    4.94 ms now).
  * **Donation (J2), 1.13×.** It lets XLA reuse the parameter and optimiser buffers instead of
    allocating new ones every step. That is host-side work, so the gain was larger on the loaded host
    (1.58×). S12-P4's "less than 5%" missed both times.
  * **vmap over 3 seeds (J4), 1.67× per model.** It shares one dispatch among three models. A J4 step
    took 1.80× a J0 step, just above the predicted 1.67× (S12-P3 MISS).
* **Faster arithmetic does little.**
  * bfloat16 (J6) gives 1.12×, TF32 (J6t) 1.02×.
  * A fused attention kernel (J5) is 8% slower. The attention products are only 7% of the
    arithmetic, so a fused kernel has little to save. Plausibly, the reshape into its layout costs
    more than that (not profiled).
  * J6 was timed during a host-load spike. The 1-minute load was 8–16 during the arm and 15–16
    during its timed repeats, with CPU pressure up to 20%, against 4.5 for J0
    (`results/hostload_s12.txt`). On a host-bound step that inflates J6's time, so its
    speedup may be understated. The verdict stays as measured.
* **bfloat16 is safe for the forward pass.** It changes the test error by 0.002%. The speed case for
  it will come only once the step stops being host-bound (J3), or on a card whose float32 is slow
  (§7).

## 4. JAX against PyTorch

Both run the same model: the PyTorch port reproduces the JAX model to 2.8e-13 in float64 (A1).
* **Eager PyTorch (T0) is 2.42× slower than J0.** Every kernel of the step is launched from
  Python, one at a time, with no fusion.
* **`torch.compile` (T1) is within 11% of jitted JAX (J0).** Both fuse elementwise work and both
  pay one launch sequence per step.
* **CUDA graphs (T2) replay the whole step's launches at once.** That is the host-overhead fix in
  PyTorch, and it beats J0 by 1.36×. JAX's equivalent, scan (J3), is faster still: 4.94 ms against
  7.52.
* **The forward pass alone is where JAX leads most.** It takes 1.09 ms in JAX, 1.97 under
  `torch.compile` and 5.54 eager. S12-P10's "within ±25%" missed. With only 1 ms of work, a
  fixed per-call cost in compiled PyTorch would be a large share; that reading is not profiled.
* **SDPA (T3) changes nothing (−2%).** The windows hold 48 tokens.
* **Aurora's own backbone (A0)** runs forward and backward in 14.41 ms eager and 7.07 compiled. Our
  port of the backbone takes 18.29 ms eager (B0). Aurora uses SDPA and a masked rather than wrapped
  shift, so this is a reference, not a test.

## 5. Parallelism

* **Across seeds (J4):** one `vmap`ped step trains 3 models at 1.67× the per-model throughput. This
  is the cheapest parallelism here, because the step is host-bound.
* **Across cards (J7, data-parallel):** the lab machine never had two idle cards at once, so J7 was
  timed on Kaggle's two T4s (§7).
  * Splitting batch 16 into 8 + 8 trains **1.24×** faster than one T4.
  * The gradient reduction across the cards and the smaller per-card batch take most of the ideal
    2×.
  * At batch 16, a second card is worth less than scan on one fast card.
* **Across jobs:** our lanes already run independent jobs, one per idle card.

## 6. Decorators and transformations: what each bought here

| Tool | Arm | Effect on this model |
|---|---|---|
| `@jax.jit` (the E-runs' loop) | J0 | 10.19 ms per step, host-bound |
| `jax.jit(..., donate_argnums=(0, 1))` | J2 | 1.13× (1.58× on a loaded host) |
| `jax.lax.scan` over 50 steps inside one `jit` | J3 | **2.07×** |
| `jax.vmap` over seeds | J4 | 1.67× per model |
| `jax.default_matmul_precision("bfloat16")` | J6 | 1.12×, accuracy unchanged |
| `jax.sharding` (data-parallel) | J7 | 1.24× on two T4s (§7) |
| `torch.compile` / `mode="reduce-overhead"` | T1 / T2 | 2.2× / 3.3× over eager |

**Recommendation for future runs: train with scan chunks.** `run_e6.py` already does (100 steps
per call), and E1–E5 did not. Adding donation and running seeds under vmap stacks further. bfloat16
can wait until the step is no longer host-bound.

## 7. Supplement: the same arms on Kaggle's T4

Spec §9 (predictions S12-K1 to K13, fixed before any T4 timing); `score_t4.py` → `results/t4/analysis.json`.
* **Where and when:** Kaggle job `ugp2-s12-t4`, version 2 (commit 215f5e1), 2 Oct 13:40–14:27.
  Two NVIDIA T4s on a quiet host (1-minute load 0.2–2.3).
* **Software:** JAX 0.10.2 and PyTorch 2.11.0, in separate environments.
* **Version 1 is not a result.** It ran JAX on the CPU: a fresh environment's JAX fell back silently.
  The runner now refuses to start without a GPU.

| Arm | T4, ms per step | T4, ×J0 | Lab card, ms | Lab card, ×J0 |
|---|---|---|---|---|
| J0 | 49.49 | 1.00 | 10.19 | 1.00 |
| J1, batch 32 / 64 | 88.27 / 163.82 | (1.78× / 3.31× the time) | 12.19 / 18.11 | (1.20× / 1.78× the time) |
| J2, donation | 49.52 | 1.00 | 8.99 | 1.13 |
| J3, scan | 42.58 | 1.16 | 4.94 | 2.07 |
| J4, 3 seeds, per model | 54.85 | 0.90 | 6.11 | 1.67 |
| J5, `dot_product_attention` | 52.83 | 0.94 | 11.03 | 0.92 |
| J6, bfloat16 | 50.97 | 0.97 | 9.13 | 1.12 |
| J6t, TF32 | 50.83 | 0.97 | 9.97 | 1.02 |
| **J7, two cards** | **39.92** | **1.24** | not run | |
| fwd, JAX | 14.87 | | 1.09 | |
| T0, eager | 69.36 | 0.71 | 24.69 | 0.41 |
| T1, `torch.compile` | 45.13 | 1.10 | 11.35 | 0.90 |
| T2, CUDA graphs | 44.61 | 1.11 | 7.52 | 1.36 |
| T3, SDPA | 60.90 | 0.81 | 11.16 | 0.91 |
| fwd, eager / compiled | 22.23 / 13.10 | | 5.54 / 1.97 | |
| A0 (eager / compiled), B0 | 40.52 / 27.15, 47.81 | | 14.41 / 7.07, 18.29 | |
| J0 at the end | 51.26 (+3.6%) | | 9.78 (−4.0%) | |

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| S12-K1 | J1 at batch 64 ≥ 2.5× J0 | 3.31× | **HIT** |
| S12-K2 | J0 / J3 < 1.25 | 1.16 | **HIT** |
| S12-K3 | one 3-model J4 step ≥ 2.5× J0 | 3.32× | **HIT** |
| S12-K4 | donation changes the time by < 10% | +0.1% | **HIT** |
| S12-K5 | `dot_product_attention` changes the time by < 10% | +6.7% | **HIT** |
| S12-K6 | J0 / J6 < 1.1 and the bfloat16 error change < 1% | 0.97; 5 × 10⁻⁹ | **HIT** (trivially, below) |
| S12-K7 | 1.2 ≤ J0 / J7 < 1.9 | 1.24 | **HIT** |
| S12-K8 | eager PyTorch (T0) ≥ 1.3× slower than J0 | 1.40× | **HIT** |
| S12-K9 | `torch.compile` (T1) within ±25% of J0 | −8.8% | **HIT** |
| S12-K10 | CUDA graphs gain little: T1 / T2 < 1.1 | 1.01 | **HIT** |
| S12-K11 | SDPA changes T1's time by < 10% | +35% | **MISS** |
| S12-K12 | the forward pass within ±25%, JAX against `torch.compile` | `torch.compile` 0.88× JAX | **HIT** |
| S12-K13 | J0 drifts by < 5% over the job | +3.6% | **HIT** |

* **On a T4 the step is bound by the card.**
  * J0 keeps it 78% busy, and 4× the batch costs 3.31× the time.
  * Per step the T4 is 4.9× slower than the lab card under J0, and 8.6× slower under J3. The faster
    the card, the more the host matters.
* **So the host-side tools fade.**
  * scan gives 1.16×, against 2.07× on the lab card.
  * Donation does nothing.
  * vmap over seeds costs 10% per model, where it gained 67% on the lab card.
* **The frameworks meet.**
  * `torch.compile` is 9% faster than jitted JAX, and CUDA graphs add nothing on a busy card.
  * Eager PyTorch is 1.40× slower, against 2.42× on the lab card.
  * In the forward pass, `torch.compile` is 12% faster than JAX, where JAX was 1.8× faster on the
    lab card.
* **SDPA is 35% slower on the T4 (the one miss).** On the lab card it changed nothing. Plausibly,
  PyTorch's fused attention kernels need a newer card, and the fallback kernel is slower than plain
  matmul and softmax for 48-token windows (not profiled).
* **bfloat16 is not bfloat16 on a T4.**
  * J6's forward error matches float32 to 5 × 10⁻⁹, against 2 × 10⁻⁵ on the lab card, so these
    matmuls ran in float32. A T4 has no bfloat16 matrix units.
  * K6 HITS as registered, but J6 says nothing about bfloat16 on this card.

## 8. Ideas for the multi-head attention

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

## Files

* **Scripts:**
  * `bench_jax.py`: J0–J7, the forward pass, and the bfloat16 accuracy check;
  * `bench_torch.py`: T0–T3, the forward passes, A0 and B0;
  * `torch_model.py` and `check_torch.py`: the PyTorch port and its equivalence gate;
  * `score.py` and `score_t4.py`: scoring, each written before its results;
  * `kaggle_s12.py`: the T4 job.
* **Results:**
  * `results/`: the sequential lane on the lab card, `analysis.json` and `hostload_s12.txt`;
  * `results/first_pass/`: A2;
  * `results/card2/`: A3;
  * `results/t4/`: the supplement, with Kaggle's environment and host load.
* **Lab machine queue file:** `ugp2_scratch/lanes/s12.cmds` (`glane4.sh s12 ... "3"`, with
  `PREEMPT_S=0` and `POLL_S=5`).

