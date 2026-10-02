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

## 7. Amendments

**A1 (1 Oct 2026, before any timing): the float32 bars of the same-computation gate are ill-posed,
so the gate is applied in float64.** Measured with `check_torch.py` on the CPU
(`results/torch_equivalence_cpu*.json`):
* **float64, both sides:** forward 2.8e-13, loss 4.6e-14, every gradient within 8.4e-10. The
  PyTorch port computes the same function as the JAX model, to machine precision.
* **float32:** forward 4.9e-5 and worst gradient 148% (`bb/enc/2/blocks/1/qkv/b`), a FAIL as
  written. But JAX's own float32 result is just as far from its float64 result: forward 1.1e-4,
  and gradients up to 72% off on leaves whose gradient is tiny. In that bias, the key part has an
  exactly zero gradient (6e-21 in float64), because softmax ignores a shift shared by all keys. Its
  float32 value is rounding noise on both sides. The bars were set below the float32 rounding floor
  of this 12-block network.
* **Gate from now on:** float64 forward within 1e-10 and gradients within 1e-6 relative (met). The
  float32 verdict stays FAIL as recorded, and timing proceeds in float32 as planned.
* The first float32 check also flagged `emb/enc/time` as having no PyTorch gradient. The model never
  uses that parameter, so its JAX gradient is exactly zero; the check now skips such parameters.

**A2 (1 Oct 2026, after J0–J3 were timed and before any other arm): host load. Every arm is timed
again in one sequential lane.**
* **Why.**
  * J0 ran at 19.2 ms per step with the card 26% busy, so its step is bound by the host, not the
    card. The same loop trained the E-runs at under 10.5 ms per step: a 20,000-step pretrain took a
    median 3.5 min, compile and evaluation included.
  * The shared host was near saturation during the first timings and still is: load average 11–14
    on 16 cores, CPU pressure (PSI "some", 60 s) 27–40%. The load comes from other users' jobs and
    from our CPU runs (Stage 4 F2 on 4 cores, the E6 smoke test on 4 more).
  * A host-bound step slows with host load. Arms timed hours apart, or by two lanes running at once,
    are therefore not comparable, and the JAX-against-PyTorch predictions (S12-P7, P8, P10) compare
    exactly such arms.
* **Change.**
  * One lane times every arm, one job at a time: J0; J1 at 32 and 64; J2–J6 and J6t; the JAX
    forward pass; T0–T3; the PyTorch forward passes; A0 and B0; then J0 again. The bfloat16
    accuracy check runs last.
  * The host's load average and CPU pressure are logged every 30 s to `logs/hostload.log`.
  * J0's drift between the start and the end of the lane is reported.
* **Scoring.**
  * S12-P1, P2 and P4 are scored as registered, on the first timings (`results/first_pass/`): all
    their arms were timed before this amendment. Their re-timed values are reported alongside.
  * Every other prediction, and the acceptance, is scored on the sequential lane, against the J0
    that opens it.
  * `score.py` takes the results directory as an argument; its rules are unchanged.

**A3 (1 Oct 2026, after the sequential lane's first J0 and before any other arm in it): card 3
only.**
* **Why.**
  * The sequential lane's first card was card 2, freed by another user at 21:06. J0 ran there at
    10.3 ms per step, with the host load at 4.7 and the card reaching 86 °C.
  * J1 (batch 32) then took card 2 from below 85 °C to 88 °C within 10 s, and the lane killed it
    twice.
  * Card 2 cannot hold a sustained job below the lane's 87 °C limit. Card 1 runs nearly as hot
    (88–90 °C under its current user). Card 3 stays at 52–68 °C under load.
  * A timing run cannot be throttled to keep a card cool.
* **Change.**
  * The lane restarts on card 3 only, from J0.
  * The J0 that opens the restarted lane is the reference for scoring (A2's rule, now on the same
    card as every other arm).
  * The card-2 J0 is reported alongside (`results/card2/`).

## 8. Files

* `experiments/stage12/bench_jax.py`: J0–J7.
* `experiments/stage12/torch_model.py`: the PyTorch port.
* `experiments/stage12/check_torch.py`: the same-computation gate.
* `experiments/stage12/bench_torch.py`: T0–T3 and A0.
* Results in `experiments/stage12/results/`; report in `experiments/stage12/REPORT.md`.

## 9. Supplement: every arm on Kaggle's T4

**Fixed 2 Oct 2026, before any T4 timing.** The lab machine's cards were busy. The owner asked to use
their Kaggle account meanwhile, without skipping any GPU run. This supplement does not score S12-P1
to P10 or the acceptance, which stay on the lab card (A2, A3). It asks how the answers change on a much
smaller card.

* **Card:** an NVIDIA T4 (Turing, 2018).
  * About 8 TFLOP/s in float32, a fifteenth of the lab card's.
  * Memory at 320 GB/s, about a sixth of the lab card's.
  * No TF32 and no bfloat16 matrix units.
  * Two cards per Kaggle session, on a 4-core host.
* **How:** as §2. Each arm runs on one T4, except J7, which uses both. All arms run in one
  sequential job (`kaggle_s12.py`), with J0 first and last and the host load logged.
  * The software is pinned to the lab machine's: JAX 0.10.2; PyTorch 2.11.0; microsoft-aurora 2.0.1.
  * JAX and PyTorch get separate environments, as on the lab machine.
* **J7 runs here.** Data-parallel over the two T4s: batch 16 split 8 + 8, gradients reduced across
  the cards (`bench_jax.py --arm J7`). The lab machine never had two idle cards at once.
* **Expectation:** on a T4 the step is bound by arithmetic and memory traffic, not by launches.

| # | Prediction (fixed before any T4 timing) |
|---|---|
| S12-K1 | The step is device-bound: J1 at batch 64 takes at least 2.5× J0 |
| S12-K2 | Scan gains little: J0 / J3 < 1.25 |
| S12-K3 | vmap over 3 seeds (J4): one 3-model step takes at least 2.5× a J0 step |
| S12-K4 | Donation (J2) changes the time per step by less than 10% |
| S12-K5 | `dot_product_attention` (J5) changes the time per step by less than 10% |
| S12-K6 | bfloat16 (J6) is not faster: J0 / J6 < 1.1. Its forward-pass error changes by less than 1% |
| S12-K7 | Two T4s (J7) are faster than one but not twice as fast: 1.2 ≤ J0 / J7 < 1.9 |
| S12-K8 | PyTorch eager (T0) is at least 1.3× slower than J0: unfused elementwise work costs more on a memory-bound card |
| S12-K9 | `torch.compile` (T1) is within ±25% of J0 |
| S12-K10 | CUDA graphs (T2) gain little: T1 / T2 < 1.1 |
| S12-K11 | SDPA (T3) changes T1's time by less than 10% |
| S12-K12 | The forward pass takes within ±25% under JAX jit and under `torch.compile` |
| S12-K13 | J0 drifts by less than 5% between the start and the end of the job |

Scored by `score_t4.py` on `results/t4/`, written before any T4 timing.
