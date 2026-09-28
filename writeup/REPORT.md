# What Does Pretraining Data Actually Buy a Neural PDE Surrogate?

*UGP-II write-up. Draft of 28 Sep 2026. Sections 1–4 are complete. Sections 5–6 are filled in as
the E1–E5 results come in from the GPU runs; nothing in them is written ahead of a measurement.*

Aryamann Srivastava. Advisors: Prof. Rajesh Ranjan (ACAL, IIT Kanpur) and Prashant Kumar.

---

## 1. Question

Aurora (Bodnar et al., *Nature* 2025) reports that pretraining beats training from scratch by
about 25%. It cannot say which of its eight pretraining datasets earned that gain. Those datasets
differ at the same time in variables, resolution, physics and volume, and rerunning the ablation
at Aurora's scale is out of reach (32 × A100 for 150k steps).

We rebuild the Aurora architecture from primitives in JAX at small scale (5.1M parameters), and
use it as an instrument on 2-D Kolmogorov flow. Because the data is generated:
* volume, diversity, resolution and distribution shift are independent knobs;
* the data is noise-free with exact ground truth, so a validation score measures what it claims to.

Five research questions:

| | Question | Experiment |
|---|---|---|
| RQ1 | At a fixed number of trajectories, is it better to have many from few regimes, or few from many? | E1 |
| RQ2 | Does the Perceiver encoder fuse heterogeneous data better than simple alternatives? | E2 |
| RQ3 | How far beyond the pretraining distribution does the advantage survive? | E3 |
| RQ4 | Which architectural components matter? | E4 |
| RQ5 | Does LoRA match full fine-tuning when fine-tuning data is scarce? | E5 |

## 2. Protocol

Every stage and experiment follows the same protocol, carried over from the predecessor project
(UGP-I), where each rule was learned from a failure:
1. **A written spec comes before any code**, with gates as tests and numeric predictions
   (`handouts/`). The experiment specs E1–E5 were frozen in git (28 Sep 2026) before any E-run.
2. **Hold out by regime, never by sample.** The harness refuses a config whose held-out regime
   appears in training.
3. **Three seeds per cell**; mean ± SE.
4. **A failed prediction is reported as a MISS** and kept. An ill-posed spec gets a dated
   amendment, and the original verdict stays in the report.
5. **One config gives one run and one result row**, keyed by a hash of the config (the Stage 11
   harness).

## 3. Testbed and data

2-D incompressible Navier–Stokes in vorticity form on the doubly periodic square [0, 2π)², with
Kolmogorov forcing and linear drag:

    ∂ω/∂t + u·∇ω = ν∇²ω − αω − A k_f cos(k_f y),    ν = 1/Re, α = 0.1, A = 1.

A regime is (Re, k_f).

**Solver (Stage 1):**
* pseudo-spectral, 2/3-rule dealiasing, Lawson integrating-factor RK4, float64;
* temporal order 3.998–4.015;
* energy and enstrophy budgets closed to ≤ 1.7e-9 relative;
* forward-cascade slope −4.38 ± 0.03 at Re 1000;
* largest Lyapunov exponents 0.39 / 0.47 / 0.57 at Re 100 / 300 / 1000.

**Data (Stage 2):**
* A trajectory is 50 time units of spin-up, then 100 time units recorded every 0.2 (501 frames of
  u, v, p, ω), saved at 64² (and 128²).
* The generation grid per regime is the smallest N whose output-band difference against 2N is
  ≤ 1e-3: 128 / 256 / 512 at Re 100 / 300 / 1000. For E1's 33 regimes it is 128 or 256.
* Spectra are statistically stable across seeds.
* Bitwise regeneration holds on a fixed machine configuration, but failed once across machine
  state at Re 1000. At λ ≈ 0.57, 50 time units amplify a last-bit difference by about e²⁸, and the
  result depends on the CPU instruction set XLA compiles for. Every trajectory therefore stores its
  post-spin-up state and the host description.

## 4. The instrument: a from-scratch Aurora, validated piece by piece

| Stage | Component | Result |
|---|---|---|
| 3 | Spectral-band levels (C = log₂N − 1 octave bands), patch embedding | round trip 1.9e-14 |
| 4 | Perceiver level aggregator and decoder (C → L = 3 → C) | gates pass; learning runs S1–S6 on the GPU (**pending**) |
| 5 | 3-D Swin block (res-post-norm, AdaLN, shifted windows) | matches Aurora 2.0.1's block to 2.8e-7 (f32), 5.2e-16 (f64) |
| 6 | Swin U-Net backbone | matches Aurora's backbone to 3e-16 (f64); overfits one trajectory to 0.070× persistence |
| 7 | Full model, one-step simulator | 0.32× / 0.46× persistence error on unseen trajectories (Re 100 / 300) |
| 8 | Autoregressive rollout | MSE model stable over 50 steps; **MAE model blows up in 3 of 4 Re 300 rollouts** |
| 9 | Pushforward | **no gain at matched compute**; −14% / −11% drift at matched steps (2× compute, exploratory) |
| 10 | LoRA (r = 4, 0.84% of parameters) | **little adaptation from 2 trajectories**: −0.5% (LoRA), −3.0% (full); zero-shot already 0.58× |
| 11 | Harness | reproduces Stage 7 exactly from a config file |

Three findings from building the instrument shaped the experiments:
* **MAE vs MSE is a stability question, not an accuracy one** (Stage 8). The two losses give the
  same one-step error, but only the MSE model rolls out stably. E4 tests this with 3 seeds.
* **Pushforward pays only if its extra compute is free** (Stage 9). E4 reports it at both matched
  steps and matched compute.
* **Two fine-tuning trajectories are too few to adapt with** (Stage 10), and zero-shot transfer
  from Re 100/300 to Re 1000 is already reasonable. E5 sweeps the fine-tuning set size.

## 5. Experiments

Designs are summarised here. The full specs, with pre-registered predictions, are in `handouts/`.
Results are filled in only from `experiments/eN/results/analysis.json`.

### 5.1 E1: volume vs diversity (RQ1)
* **Design:**
  * target (300, 4), held out;
  * nested pretraining sets of 2 ⊂ 8 ⊂ 32 regimes that enclose it;
  * cells T ∈ {32, 64, 128} total trajectories × n ∈ {2, 8, 32} regimes;
  * each pretrain run is fine-tuned on 2 target trajectories and compared with a scratch model at
    matched optimiser steps;
  * advantage `adv = 1 − e_ft / e_scratch`;
  * 326 trajectories, 57 runs.
* **Predictions:** E1-P1 … P6 (diversity beats volume; best cell adv in [10%, 40%]).
* **Results:** *pending.*

### 5.2 E2: heterogeneous fusion (RQ2)
* **Design:**
  * two sources, Re 100 and Re 300, in deliberately mismatched formats: (a) 128² vs 64²;
    (b) all variables vs (u, v); (c) both; plus a homogeneous reference;
  * compares the Perceiver with pad-and-mask (spectral upsampling, zero-filled variables, presence
    fields) and with per-dataset encoders;
  * 30 runs.
* **Predictions:** E2-P1 … P3.
* **Results:** *pending.*

### 5.3 E3: out-of-distribution transfer (RQ3)
* **Design:**
  * pretrain on Re 125–250 (16 regimes);
  * zero-shot, fine-tuned and scratch errors at Re 300, 350, 450, 560 and 700 (k_f = 4), i.e.
    `d = ln(Re/250)` from 0.18 to 1.03;
  * 33 runs.
* **Predictions:** E3-P1 … P4.
* **Results:** *pending.*

### 5.4 E4: component ablations (RQ4)
* **Design:** single-change variants of the Stage 7 model: no shifted windows; depth 6 / 24;
  L = 1 / 6; pushforward at matched steps and at matched compute; MAE. 27 runs.
* **Predictions:** E4-P1 … P5.
* **Results:** *pending.*

### 5.5 E5: LoRA vs full fine-tuning (RQ5)
* **Design:**
  * pretrain on Re 100 / 300;
  * fine-tune on k ∈ {1, 2, 4, 6} Re 1000 trajectories with LoRA (r = 4) or all parameters;
  * a scratch baseline at matched steps;
  * 39 runs.
* **Predictions:** E5-P1 … P4.
* **Results:** *pending.*

## 6. Discussion

*To be written from the results.* Points it must address:
* the E1 distance confound (more regimes also puts a pretraining regime nearer the target), read
  against E3's advantage-vs-distance curve;
* whether E4's component effects hold at the scale and budget of E1's pretraining;
* float32 matrix-multiply precision on the GPU (TF32 vs full) and its effect, if any, on the E4 /
  E5 comparisons anchored on CPU results;
* the limits of the analogy to Aurora: one PDE family, a 5.1M-parameter model, and 64² fields.

## 7. Reproducing

Everything is in the repository:
* `experiments/stageN/REPORT.md` and `experiments/eN/`;
* each experiment runs from its configs with `experiments/run.py`;
* each is analysed with `experiments/e1/analyse.py` or `experiments/analyse_e345.py`;
* the GPU procedure is in `experiments/e1/GPU_RUNBOOK.md` and `experiments/stage4/GPU_RUNBOOK.md`.
