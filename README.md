# Aurora-from-Scratch in JAX

**What does pretraining data actually buy a neural PDE surrogate?**

A from-primitives JAX reimplementation of the [Aurora](https://www.nature.com/articles/s41586-025-09005-y)
architecture (a 3-D Swin Transformer U-Net with Perceiver encoders and decoders), built at small
scale and used as an instrument. The goal is to isolate **which properties of pretraining data**
produce the pretraining advantage.

---

## The question

Aurora reports that its pretrained model **beats training from scratch by 25%**. The paper cannot
say which of its eight pretraining datasets earned that gain. Those datasets differ in variables,
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

**Model:** a scaled-down Aurora with 2–10 M parameters:
* **Encoder:** Perceiver with patch embeddings, patch-area and time encodings, and a fixed set of
  latent levels.
* **Backbone:** Swin Transformer U-Net with shifted-window attention.
* **Decoder:** mirrors the encoder.
* **Training:** a one-step simulator, rolled out autoregressively and trained with an MAE loss.

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
| 2 | Dataset generator + loader | reproducible from seed; spectra stable across trajectories | next |
| 3 | Patch embed / unembed | round trip at machine precision | |
| 4 | Perceiver level aggregator | variable level counts map to fixed latent levels and back | |
| 5 | Windowed + shifted-window attention | matches a reference Swin block to 1e-5 | |
| 6 | U-Net backbone with skips | overfits a single trajectory | |
| 7 | Full encoder → backbone → decoder | trains one-step prediction | |
| 8 | Autoregressive rollout + MAE loss | stable 50-step rollout | |
| 9 | Pushforward trick | lower rollout drift at matched compute | |
| 10 | LoRA fine-tuning | adapts to unseen ν with < 1% of parameters trainable | |
| 11 | Ablation harness | one config → one run → one row of results | |

Stages 1–8 build the instrument. Stages 9–11 and experiments RQ1–RQ5 are the contribution.

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
.venv/bin/pip install jax numpy scipy sympy pytest matplotlib

# correctness gates (~5 s)
.venv/bin/python -m pytest tests -q

# a long run: Re = 1000 on a 256² grid, 250 time units (resumable; raw output in data/)
.venv/bin/python experiments/stage1/run_long.py --name L1000 --re 1000 --n 256
.venv/bin/python experiments/stage1/analyse_long.py --name L1000
```

## Repository layout

| Path | Contents |
|---|---|
| `PROJECT_PLAN.md` | Research questions, testbed, architecture, staged deliverables, experiments |
| `project_memory.md` | Decision record and open issues. Read §2 (protocol rules) before running any experiment |
| `handouts/` | Per-stage specs: equations, tests, tolerances and pre-registered predictions |
| `src/kolmogorov/` | Pseudo-spectral Kolmogorov flow solver |
| `tests/` | Correctness gates (`pytest tests`) |
| `experiments/stage1/` | Stage 1 run scripts, result JSONs, figures and report |

## Status

**Ongoing.** Stage 1 is complete. Next: fix the generation resolution for the dataset (see
report §8.1), then build the Stage 2 dataset generator and loader.

## Advisors

Prof. Rajesh Ranjan (ACAL, IIT Kanpur) · Prashant Kumar (PhD)

## References

* C. Bodnar et al., *A foundation model for the Earth system*, **Nature** (2025).
  [arXiv:2405.13063](https://arxiv.org/abs/2405.13063)
* D. Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, **PNAS** 118 (2021).
