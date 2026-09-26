# Aurora-from-Scratch in JAX

**What does pretraining data actually buy a neural PDE surrogate?**

A from-primitives JAX reimplementation of the [Aurora](https://www.nature.com/articles/s41586-025-09005-y)
architecture (3-D Swin Transformer U-Net with Perceiver encoders/decoders), at small scale, used to
isolate **which properties of pretraining data** produce the pretraining advantage.

## The question

Aurora reports that its pretrained model **beats training from scratch by 25%**. It cannot say which
of its eight heterogeneous pretraining datasets earned that — they differ in variables, resolution,
physics and volume simultaneously, and rerunning the ablation costs 32×A100 for 150k steps.

On **2-D Kolmogorov flow** the data is generated, so volume, diversity, resolution and distribution
shift become independent knobs. The ablation becomes tractable.

## Research questions

1. **Volume vs diversity** — at fixed total samples, many trajectories from few regimes, or few from many?
2. **Does the Perceiver encoder earn its place?** — test it on deliberately heterogeneous data (mixed resolutions, mixed variable sets) against pad-and-mask and per-dataset encoders.
3. **How far does pretraining transfer out of distribution?** — advantage as a function of distance beyond the training Reynolds band.
4. **Which architectural components matter?** — shifted windows, depth, latent level count, the pushforward trick, MAE vs MSE.
5. **Parameter-efficient adaptation** — does LoRA match full fine-tuning in the small-data regime?

## Layout

| Path | Contents |
|---|---|
| `PROJECT_PLAN.md` | Research questions, testbed, architecture, 11 staged deliverables with acceptance tests, 5 experiments |
| `project_memory.md` | Decision record. **Read §2 (carry-over rules) before running any experiment.** |
| `handouts/` | Per-stage specs: equations, tests, tolerances, pre-registered predictions |
| `src/kolmogorov/` | Stage 1: pseudo-spectral 2-D Kolmogorov flow solver (JAX) |
| `tests/` | Correctness gates (`pytest tests`) |
| `experiments/stage1/` | Stage 1 run scripts, result JSONs, figures and `REPORT.md` |

## Status

**Stage 1 (pseudo-spectral Kolmogorov solver): complete, meets its acceptance test.** Energy budget
closes to ≤ 2e-9 relative; the Re = 1000 forward-cascade slope is −4.38 over k ∈ [8, 16]; all 11
correctness gates pass. Full results: [`experiments/stage1/REPORT.md`](experiments/stage1/REPORT.md).

**Ongoing:** the resolution to use for dataset generation (Stage 2) is being decided (see report §8.1), followed by
the dataset generator and loader.

### Running the solver

```
python3.11 -m venv .venv && .venv/bin/pip install jax numpy scipy sympy pytest matplotlib
.venv/bin/python -m pytest tests -q
.venv/bin/python experiments/stage1/run_long.py --name L1000 --re 1000 --n 256
```

## Advisors

Prof. Rajesh Ranjan (ACAL, IIT Kanpur) · Prashant Kumar (PhD)
