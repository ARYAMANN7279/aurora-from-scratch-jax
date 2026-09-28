# Stage 4 measurements on the GPU box (S1–S6)

The Stage 4 training measurements (`handouts/stage04_perceiver.md` §4) are too heavy for a CPU.
Measured on the 4-core CPU: 0.17 / 0.35 / 7.8 s per step at 16² / 64² / 128², and minutes per step
at 512². Run them on the lab VM. Everything below is deterministic from seeds. GPU floating point
differs from CPU, so regenerated data is statistically, not bitwise, identical to the CPU copy.

## VM rules (project_memory.md §3)
* Check `nvidia-smi` first: do not launch at ≥ 90 °C or with thermal slowdown active.
* Set `CUDA_VISIBLE_DEVICES` explicitly on every command. Kill only by explicit PID.
* Work under the project directory, never `/tmp` or the root partition.

## GPU notes (measured 28 Sep 2026)
* Run the tests and every training run with `JAX_DEFAULT_MATMUL_PRECISION=highest`. At the
  default precision, float32 matrix multiplies use TF32, and three float32 gates fail (Stage 3 H7,
  Stage 5 W3/W4, Stage 6 U2/U3). `highest` costs only 1.07× the step time.
* Gate D9 fails on the GPU for a diagnostics-only reason (`experiments/stage2/REPORT.md`,
  addendum). The other gates pass: 90/91 were measured before W10 was added, so expect 91/92.
* To run several processes on one card, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` on each.
  Otherwise the first process reserves 75% of the card's memory.

## Steps

```bash
git clone https://github.com/ARYAMANN7279/aurora-from-scratch-jax.git && cd aurora-from-scratch-jax
python3.11 -m venv .venv && .venv/bin/pip install "jax[cuda12]" numpy scipy sympy pytest matplotlib optax
export CUDA_VISIBLE_DEVICES=0            # pick a cool, idle card
JAX_DEFAULT_MATMUL_PRECISION=highest .venv/bin/python -m pytest tests -q   # see the GPU notes above

# data: the re300_kf4 trajectories of acc_v1 (seeds 0-7), and the 512^2 frames for C = 8
.venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/acc_v1.json
.venv/bin/python experiments/stage4/make_512.py

# S1-S4: one model, L = 3, all four resolutions
.venv/bin/python experiments/stage4/train_perceiver.py --run main
# S5: no compression (L = C), one run per resolution
for n in 16 64 128 512; do .venv/bin/python experiments/stage4/train_perceiver.py --run nocompress --res $n; done
# S6: level encodings ablated, C = 5
.venv/bin/python experiments/stage4/train_perceiver.py --run nolevelenc --res 64
```

Results land in `experiments/stage4/results/*.json` (small). Commit those files only: no data,
no checkpoints.
