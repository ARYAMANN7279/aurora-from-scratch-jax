# Stage 12 Spec: Speed — what makes training faster, in JAX and against PyTorch

**Status: pre-registered 1 Oct 2026, before any timing run.** Advisor's notes (1 Oct 2026): JAX
speedups as the focus, with PyTorch as the reference; the same run with `torch.compile`;
decorators; parallelism; ideas for the multi-head attention.

---

## 1. Question

The E-runs train at about 10 ms per step on an idle card, and turning TF32 off cost only 1.07×
(`project_memory.md` §3). That suggests the step is not limited by matrix multiplies. Where does the
time go, which changes make training faster, and how does the same model run in PyTorch, eager and
compiled?

## 2. What is timed

* **Model:** the E1 pretraining configuration. D = 64, P = 4, L = 3, 4 Perceiver heads; 12 Swin
  blocks (encoder depths 2, 2, 2 with heads 2, 4, 8; decoder depths 2, 2, 2 with heads 8, 4, 2);
  windows (3, 4, 4); 64² inputs. In this configuration no shifted window needs a mask: the level
  axis is never shifted and both spatial axes are periodic.
* **Steps timed:**
  * the training step: forward, backward and Adam update, batch 16, MSE loss, cosine schedule, with
    the sampling of each batch included as in `harness.py`;
  * the forward pass alone (inference), batch 16.
* **Measure:** wall time per step = (time for N steps) / N, with the device synchronised at both
  ends, N = 500 after 100 warm-up steps. Each arm runs 3 such repeats in one process; we report
  the median and the spread. Compile time is reported separately.
* **Where:** one idle card of the lab machine, claimed by `glane3.sh` (nobody else's process on it,
  below 85 °C; card 0 excluded). Temperature and utilisation are logged during the run. A repeat
  during which another process appears on the card is discarded and run again.
* **Precision:**
  * JAX: float32 with matmul precision `highest` (TF32 off), as in every E-run;
  * PyTorch: float32 with TF32 disabled for matmuls and cuDNN;
  * unless an arm says otherwise.
* **Same computation:** before any PyTorch arm is timed, the PyTorch port, loaded with the JAX
  weights, must reproduce the JAX forward pass to ≤ 1e-5 relative and its parameter gradients to
  ≤ 1e-4 relative, in float32 on a fixed batch. Otherwise it is not the same model and the arm is
  not timed.

## 3. Arms

Each arm changes one thing from J0 (or from T0 for the PyTorch arms).

| Arm | What changes |
|---|---|
| J0 | the E-run training loop of `harness.py`: numpy sampling on the host, four gathers and one jitted step per iteration |
| J1 | J0 at batch 32 and 64 (diagnostic: is the step launch-bound?) |
| J2 | J0 with the parameters and optimiser state donated to the jitted step (`donate_argnums`) |
| J3 | 50 steps per jitted call (`jax.lax.scan`), with the batches sampled on the device |
| J4 | 3 models (seeds 0–2) trained together in one jitted step (`jax.vmap` over seeds); reported per model |
| J5 | window attention through `jax.nn.dot_product_attention` (XLA implementation, float32) |
| J6 | bfloat16 compute (float32 parameters and optimiser state) |
| J7 | data-parallel over 2 cards (`jax.sharding`). Run only if two cards are idle at the same time; otherwise reported as not run |
| T0 | the same model in PyTorch (a port of `src/aurora`), eager, plain matmul-and-softmax attention as in J0 |
| T1 | T0 under `torch.compile` (default mode) |
| T2 | T0 under `torch.compile(mode="reduce-overhead")`, which uses CUDA graphs |
| T3 | T1 with `torch.nn.functional.scaled_dot_product_attention`, the counterpart of J5 |
| A0 | reference point only: Aurora's own PyTorch backbone (microsoft-aurora 2.0.1) at our backbone configuration, forward and backward, eager and compiled. Not the same computation: it masks shifted windows along H, while ours are periodic. |

## 4. Predictions (fixed before any timing)

| # | Prediction |
|---|---|
| S12-P1 | the step is launch-bound: time per step at batch 64 is less than 2× that at batch 16 (J1) |
| S12-P2 | scan (J3) cuts time per step by at least 25% against J0 |
| S12-P3 | vmap over 3 seeds (J4) trains each model at ≥ 1.8× J0's throughput, i.e. one 3-model step takes < 1.67× a J0 step |
| S12-P4 | donation (J2) changes time per step by less than 5% |
| S12-P5 | `dot_product_attention` (J5) changes time per step by less than 10% |
| S12-P6 | bfloat16 (J6) speeds the step up by ≥ 1.3×. Run in bfloat16, the forward pass of a trained E1 pretrain model changes its one-step test error by less than 1% |
| S12-P7 | PyTorch eager (T0) is at least 1.5× slower per step than J0 |
| S12-P8 | `torch.compile` (T1) is within ±25% of J0 |
| S12-P9 | CUDA graphs (T2) are at least 20% faster than T1 |
| S12-P10 | the forward pass alone takes within ±25% under JAX jit and under `torch.compile` |

## 5. Acceptance

Stage 12 is met if some JAX configuration trains the E1 model at least 1.5× faster than J0, either
with numerics identical up to float rounding or, for a reduced-precision arm, with the accuracy
check of S12-P6 passed.

## 6. Out of scope

* Changes to the multi-head attention that alter the model (more heads, other attention forms)
  change accuracy as well as speed. Each needs its own spec with accuracy predictions; candidates
  are listed in the report.
* The accuracy of a model trained end to end in bfloat16 is not tested here; S12-P6 checks only the
  forward pass of a model trained in float32.

## 7. Files

* `experiments/stage12/bench_jax.py`: J0–J7.
* `experiments/stage12/torch_model.py`: the PyTorch port.
* `experiments/stage12/check_torch.py`: the same-computation gate.
* `experiments/stage12/bench_torch.py`: T0–T3 and A0.
* Results in `experiments/stage12/results/`; report in `experiments/stage12/REPORT.md`.
