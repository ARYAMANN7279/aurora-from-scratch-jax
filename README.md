# Aurora from Scratch in JAX

**What does pretraining data actually buy a neural PDE surrogate?**

A from-primitives JAX reimplementation of the [Aurora](https://www.nature.com/articles/s41586-025-09005-y)
architecture (a 3-D Swin Transformer U-Net with Perceiver encoders and decoders), built at small
scale and used as an instrument. The goal is to isolate **which properties of pretraining data**
produce the pretraining advantage.

---

## The question

Aurora reports that its pretrained model **beats training from scratch by 25%**. The paper cannot
say which of its pretraining datasets earned that gain. Those datasets differ in variables,
resolution, physics and volume all at once, and rerunning the ablation would cost 32 × A100 for
150k steps.

**On 2-D Kolmogorov flow the question becomes answerable.** The data is generated, so volume,
diversity, resolution and distribution shift are independent knobs, and each can be turned alone.
The data is also noise-free with exact ground truth, so a validation score measures what it claims
to measure.

## Research questions

| | Question | Experiment |
|---|---|---|
| **RQ1** | **Volume vs diversity.** At a fixed number of training samples, is it better to have many trajectories from a few flow regimes, or a few from many? | 3 × 3 grid of regime count × trajectories per regime, scored on a held-out regime |
| **RQ2** | **Does the Perceiver encoder earn its place?** | Deliberately mixed resolutions and variable sets; Perceiver vs pad-and-mask vs per-dataset encoders |
| **RQ3** | **How far does pretraining transfer out of distribution?** | Pretrain on a Reynolds band; measure the advantage against distance beyond it |
| **RQ4** | **Which architectural components matter?** | Ablate shifted windows, depth, latent level count, the pushforward trick, MAE vs MSE |
| **RQ5** | **Parameter-efficient adaptation.** Does LoRA match full fine-tuning when fine-tuning data is scarce? | LoRA vs full fine-tuning across fine-tuning set sizes |

## Approach

**Testbed:** 2-D incompressible Navier–Stokes in vorticity form, doubly periodic, with sinusoidal
(Kolmogorov) forcing and linear drag:

    ∂ω/∂t + u·∇ω = ν∇²ω − αω − A k_f cos(k_f y)

The Reynolds number (via ν) and the forcing wavenumber k_f define the flow regimes.

**Model:** a scaled-down Aurora, 5.1 M parameters in the default configuration (D = 64, 12 Swin blocks):
* **Encoder:** per-variable patch embeddings of the field's spectral octave bands (the "levels"),
  with position, patch-area and level encodings. A Perceiver maps any number of levels to L = 3
  latent levels. The time step enters through the backbone's adaptive LayerNorm.
* **Backbone:** Swin Transformer U-Net with shifted-window attention.
* **Decoder:** mirrors the encoder.
* **Training:** a one-step simulator, rolled out autoregressively. The experiments train with an
  MSE loss. Aurora uses MAE; a single-seed MAE model blew up in rollout (Stage 8), so MAE was
  tested as an E4 ablation, where it stayed stable over 3 seeds.

**Protocol**, applied to every experiment:
* Every run has a prediction written down **before** it is launched.
* Test sets are held out by **flow regime**, never by sample.
* Every cell uses at least three seeds, and the spread is reported alongside the mean.
* Any fit made on the data it is scored on is treated as inflated until it is scored on disjoint data.

## Roadmap

Each stage has an acceptance test and is not passed until that test is met.

| # | Deliverable | Acceptance test | Status |
|---|---|---|---|
| 1 | Pseudo-spectral Kolmogorov solver | energy budget balances to < 1%; expected inertial-range scaling | ✅ **done** |
| 2 | Dataset generator + loader | reproducible from seed; spectra stable across trajectories | ✅ spectra stable (all HIT); ⚠️ bitwise regeneration from the seed holds at Re 100 and 300 but not for a full-length Re 1000 trajectory (512²). **Follow-up F1** repeated the test on one fixed machine and Re 1000 still differs, so the cause is not a change of machine; it is not yet identified. Chaos turns any last-bit difference into a different trajectory (~e²⁸ over the spin-up), so each trajectory stays a valid sample and the data's statistics are unaffected |
| 3 | Spectral-band levels, patch embed / unembed | round trip at machine precision | ✅ **done** (1.9 × 10⁻¹⁴) |
| 4 | Perceiver level aggregator | variable level counts map to fixed latent levels and back | ✅ **met with 32× the training and a decaying learning rate (F4)**: the one shared model reconstructs to **0.008 / 0.002 / 0.001 / 0.001** relative error at C = 3 / 5 / 6 / 8 (16² / 64² / 128² / 512²; bars 0.02 / 0.02 / 0.03 / 0.05), in both runs, and holds S1's 16² bar from step ~90,000 on. Earlier it was not met: at the registered 6000 steps the errors were 0.167 / 0.085 / 0.063 / 0.060 (that verdict stays in the stage report). F1 showed S1's bar is reachable at C = 3. F2 (4× steps) met S4 but not S1. F3 (32× at a constant rate) reached S1 only intermittently, between loss spikes |
| 5 | Windowed + shifted-window attention | matches a reference Swin block to 1e-5 | ✅ **done**: matches Aurora's own block to 2.8 × 10⁻⁷ |
| 6 | U-Net backbone with skips | overfits a single trajectory | ✅ **done**: matches Aurora's backbone to 3 × 10⁻¹⁶; overfits to 0.07× persistence (16k steps) |
| 7 | Full encoder → backbone → decoder | trains one-step prediction | ✅ **done**: 0.32× / 0.46× persistence error on unseen trajectories (Re 100 / 300) |
| 8 | Autoregressive rollout + MAE loss | stable 50-step rollout | ✅ MSE model stable over all 50-step rollouts. The single-seed MAE blow-up at Re 300 did **not** reproduce in E4 (3 seeds, 10k steps): MAE drift −11 ± 6.5% vs MSE |
| 9 | Pushforward trick | lower rollout drift at matched compute | ✅ **met in E4** (3 seeds, 10,000 steps): at matched compute pushforward cuts 50-step drift by 45 ± 5% / 47 ± 7% (Re 100 / 300). At Stage 9's own 3000-step budget it was not met earlier (that verdict stays in the stage report). On E1's 2-trajectory task (E8) the drift gain is smaller and partly damped amplitude |
| 10 | LoRA fine-tuning | adapts to unseen ν with < 1% of parameters trainable | ⚠️ **not met**: 0.84% trainable, but with 2 trajectories neither LoRA nor full fine-tuning adapts much (−0.5% / −3%); zero-shot is already 0.58× persistence. **F1:** rank is not the limit (r = 64 gains 1.0%, full fine-tuning 2.7%). **F2** (3 seeds): still not met at Re 3000, ν 10× below the training band (LoRA −0.5%, full −3.2%); at Re 30, where the base is worse than persistence, LoRA cuts the error by 71%. **F3** (6 trajectories): a model trained from scratch beats the bar at both targets (−26% at Re 3000, −30% at Re 1000), while full fine-tuning gets −12% and LoRA −1%; an exact solver that sees only the 64² band predicts at 0.05× persistence. So the miss lies in adapting this base, not in the data or the physics |
| 11 | Ablation harness | one config → one run → one row of results | ✅ **done**: reproduces Stage 7 exactly from a config file |
| 12 | Speed: what makes training faster, in JAX and against PyTorch | a JAX configuration at least 1.5× faster than the E-run loop, with numerics unchanged | ✅ **met**: running 50 steps per jitted call (`lax.scan`) trains the E1 model **2.07×** faster than the E-runs' loop, numerics unchanged; 6 of 10 predictions HIT. The step is bound by the host, not the card: it runs at 10.2 ms on a quiet host and 19.2 ms on a loaded one, with the card 49% busy. Scan, donation (1.13×), vmap over seeds (1.67× per model) and CUDA graphs pay; bfloat16 (1.12×) and fused attention (0.92×) do not. Eager PyTorch is 2.42× slower than jitted JAX, `torch.compile` is within 11%, and CUDA graphs beat jitted JAX by 1.36× but trail JAX with scan by 1.52×. A T4 supplement (Kaggle, two-card J7 included) is in the report |

Stages 1–8 build the instrument. Stages 9–11 and experiments RQ1–RQ5 are the contribution.

**Experiments.** Every experiment has a spec with pre-registered predictions, committed in git before any
run (E1–E5 on 28 Sep 2026, E6 on 1 Oct, E7 on 2 Oct, E8 and E9 on 3 Oct). E1–E5 ran on the lab GPU box, E6–E9
on Kaggle's T4 cards. All are reported against those predictions:

| | Spec | Runs | Data | Status |
|---|---|---|---|---|
| E1 volume vs diversity | [`handouts/e1_volume_diversity.md`](handouts/e1_volume_diversity.md) | 57 | new `e1_v1`, 326 trajectories (generated on CPU) | ✅ **done** ([report](experiments/e1/REPORT.md)): 2/6 predictions HIT; pretraining halves one-step error in every cell (advantage 42–50%), but neither diversity nor volume changes it; zero-shot follows the nearest regime at the target's k_f; fine-tuned models drift more than scratch |
| E2 heterogeneous fusion | [`handouts/e2_heterogeneous_fusion.md`](handouts/e2_heterogeneous_fusion.md) | 30 | `acc_v1` | ✅ **done** ([report](experiments/e2/REPORT.md)): 1/3 predictions HIT; pad-and-mask beats the Perceiver by 23% on the upsampled source; fusion costs the Perceiver 13–39% even on sources whose format did not change |
| E3 out-of-distribution transfer | [`handouts/e3_ood_transfer.md`](handouts/e3_ood_transfer.md) | 33 | `e1_v1` | ✅ **done** ([report](experiments/e3/REPORT.md)): 3/4 predictions HIT; zero-shot degrades slowly (0.30× → 0.41× persistence out to 2.8× the band's top Re); the fine-tuned advantage stays at 36–41% at every distance |
| E4 component ablations | [`handouts/e4_ablations.md`](handouts/e4_ablations.md) | 27 | `acc_v1` | ✅ **done** ([report](experiments/e4/REPORT.md)): 1/5 predictions HIT; pushforward cuts 50-step drift by 31–50% at matched steps and by 45–47% at matched compute; MAE does not blow up; depth 24 helps (−6.7% one-step) |
| E5 LoRA vs full fine-tuning | [`handouts/e5_lora_vs_full.md`](handouts/e5_lora_vs_full.md) | 39 | `acc_v1` | ✅ **done** ([report](experiments/e5/REPORT.md)): 2/4 predictions HIT; the pretraining advantage over scratch falls from 41% (1 trajectory) to 5% (6); fine-tuning gains ≤ 5.3% (full) and ≤ 1.7% (LoRA) |
| E6 sim-to-real on real data | [`handouts/e6_realpde_sim2real.md`](handouts/e6_realpde_sim2real.md) | 15, plus the kit's FNOs | RealPDE foil (UGP-I's competition copy), 32 × 64 | ✅ **done** ([report](experiments/e6/REPORT.md)): 2/5 predictions HIT (15 runs on Kaggle T4s, amendment A3). From scratch on 65 real trajectories: rollout rel_l2 0.112 (subscore 94.71, within 0.8 of the organisers' fine-tuned FNO). Pretraining on the simulations and then fine-tuning for 2000 steps is **21–23% worse** (P1 MISS). Zero-shot transfer barely beats persistence (P3 MISS); simulating the test conditions does not help (P2 MISS). Fine-tuning recovers 49% over zero-shot (P4 HIT); no arm reaches the FNO (P5 HIT). **Follow-up F1** (3 of 3 HIT): given scratch's full 22,000 real-data steps, the pretrained starts match scratch (+0.4% ± 1.8% disjoint, +2.0% ± 0.9% all simulations), so the deficit was the short fine-tune and simulation pretraining buys no accuracy here |
| E7 attention: relative-position bias | [`handouts/e7_attention_rpb.md`](handouts/e7_attention_rpb.md) | 6 | `e1_v1` (E1's scratch task) | ✅ **done** ([report](experiments/e7/REPORT.md)): 2/3 predictions HIT; a learned bias per head for each offset inside the attention windows lowers one-step error by **6.9% ± 0.7%** (every seed; 0.27% more parameters, 1.2% more time) but not the 20-step drift (−0.9% ± 1.6%, P2 MISS); off by default |
| E8 follow-ups: attention bias, q/k normalisation, scan chunks | [`handouts/e8_followups.md`](handouts/e8_followups.md) | 18 | `e1_v1` (E1's scratch task) | ✅ **done** ([report](experiments/e8/REPORT.md)): 5/10 predictions HIT. q/k normalisation lowers one-step error by **7.3% ± 0.7%**, as much as E7's bias (6.9% ± 0.7%, replicated). Measured from 84 rollouts, both gains fade from 8–10% at step 2–3 to about zero by step 30. Pushforward costs 12.9% one-step error on this task, and about two-thirds of its 50-step drift gain is damped amplitude (exploratory). Scan chunks give the same models 1.21× faster |
| E9 bias and q/k normalisation together | [`handouts/e9_rpb_plus_qknorm.md`](handouts/e9_rpb_plus_qknorm.md) | 6 | `e1_v1` (E1's scratch task) | ✅ **done** ([report](experiments/e9/REPORT.md)): 4/6 predictions HIT. Together the two lower one-step error by **8.9% ± 0.4%**, beating either alone (6.9%, 7.3%) in every seed but far short of the 13.7% that independent gains would give: they overlap. The best one-step model so far (0.136 against 0.150). Exploratory: without q/k normalisation, attention is near-uniform at the fine stages and nearly one-hot at the coarsest; q/k normalisation evens this out, and the bias does not |

## Stage 1 results: the data generator

The solver is written from scratch in JAX. It uses 2/3-rule dealiasing, Lawson integrating-factor
RK4 time stepping and float64, with energy and enstrophy budgets tracked at every step.
Full report: [`experiments/stage1/REPORT.md`](experiments/stage1/REPORT.md).

| Check | Target | Measured |
|---|---|---|
| Correctness gates (FFT, dealiasing, conservation, exact solutions, …) | 11 gates | **all pass** |
| Temporal convergence order | 4 | 3.998 – 4.015 |
| Energy budget residual, Re = 100 / 300 / 1000 | < 1% | ≤ 1.7 × 10⁻⁹ |
| Enstrophy budget residual | < 0.1% | ≤ 1.1 × 10⁻⁹ |
| Forward-cascade spectral slope, Re = 1000, k ∈ [8, 16] | −6 to −2.8 | **−4.38 ± 0.03** |
| Largest Lyapunov exponent, Re = 100 / 300 / 1000 | > 0, rising with Re | 0.39 / 0.47 / 0.57 |
| Pre-registered predictions | — | 13 of 16 hit; 2 misses analysed in the report |

<p align="center">
  <img src="experiments/stage1/figs/spectra.png" width="520" alt="Time-mean energy spectra at Re 100, 300 and 1000">
  <br><img src="experiments/stage1/figs/snapshots.png" width="760" alt="Vorticity snapshots at Re 100, 300 and 1000">
</p>

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install jax numpy scipy sympy pytest matplotlib optax

# correctness gates (98 tests, about 7-10 min on 4 CPU cores)
.venv/bin/python -m pytest tests -q

# one config -> one run -> one row: reproduces the Stage 7 result (needs the acc_v1 dataset)
.venv/bin/python experiments/run.py experiments/configs/h1_stage7_repro.json

# a long run: Re = 1000 on a 256² grid, 250 time units (resumable; raw output in data/)
.venv/bin/python experiments/stage1/run_long.py --name L1000 --re 1000 --n 256
.venv/bin/python experiments/stage1/analyse_long.py --name L1000
```

## Repository layout

| Path | Contents |
|---|---|
| `PROJECT_PLAN.md` | Research questions, testbed, architecture, staged deliverables, experiments |
| `project_memory.md` | Decision record and open issues. Read §2 (protocol rules) before running any experiment |
| `docs/architecture.md` | How the model works: the forward pass with shapes, each part next to Aurora's, where the parameters and the computation are, and a reading guide |
| `handouts/` | Per-stage specs: equations, tests, tolerances and pre-registered predictions |
| `src/kolmogorov/` | Pseudo-spectral Kolmogorov flow solver |
| `src/kdata/` | Dataset generator (resumable, checksummed) and regime-split window loader |
| `src/aurora/` | The model in pure JAX: spectral-band levels, embedding, Perceiver, Swin3D blocks, U-Net backbone, full model, rollout, LoRA; the ablation harness (`harness.py`) and E2's multi-source trainer (`fusion.py`) |
| `tests/` | Correctness gates (`pytest tests`) |
| `experiments/stageN/` | Per-stage run scripts, result JSONs, figures and reports |
| `experiments/eN/` | Per-experiment configs, result rows, `analysis.json`, figures and `REPORT.md` |
| `experiments/e1/` | E1 design, N_gen measurement (all 33 regimes), configs and analysis |
| `experiments/designs.py`, `make_configs.py`, `analyse_e345.py` | E2–E5 designs, the config writer (`experiments/e{2,3,4,5}/configs/`), and the analysis (verdicts per prediction) |
| `experiments/run.py` | Runs configs (lists, sweeps, `--skip-done`, sharding); one row per run, by default in `experiments/results.jsonl` |

## Status

**Complete.** All twelve stages and all nine experiments have their results.
* **Met:** stages 1, 3, 5, 6, 7, 8, 11 and 12 met their acceptance tests. Two more are met after
  follow-ups, and missed at their registered budgets earlier:
  * Stage 9 (pushforward) is met in E4, with 3.3× its budget and 3 seeds;
  * Stage 4 (the Perceiver) is met with 32× its training steps and a decaying learning rate (F4).
* **Not met:** Stage 10. LoRA adapts little from two trajectories, at Re 1000 and at Re 3000, and rank
  is not the limit (F1). At Re 30, where the base is systematically wrong, LoRA cuts the error by
  71% (F2). With six trajectories, training from scratch beats the bar at both targets, but
  fine-tuning the base does not (F3).
* **A known limit:** Stage 2's data are statistically sound, but a full-length Re 1000 trajectory does
  not regenerate bit for bit, even on one machine (F1).
* **Experiments:** E1 hit 2 of 6 predictions, E2 1 of 3, E3 3 of 4, E4 1 of 5, E5 2 of 4, E6 2 of 5 (and
  its follow-up 3 of 3), E7 2 of 3, E8 5 of 10 and E9 4 of 6. Each is reported in
  `experiments/eN/REPORT.md`.
* **Records:** every stage is built and passes its correctness gates (118 tests on CPU, including the
  experiment tooling; GPU notes in `project_memory.md`). Each stage and experiment has a spec with
  pre-registered predictions in `handouts/`, and a report.

## Advisors

Prof. Rajesh Ranjan (ACAL, IIT Kanpur) · Prashant Kumar (PhD)

## References

* C. Bodnar et al., *A foundation model for the Earth system*, **Nature** (2025).
  [arXiv:2405.13063](https://arxiv.org/abs/2405.13063)
* D. Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, **PNAS** 118 (2021).
