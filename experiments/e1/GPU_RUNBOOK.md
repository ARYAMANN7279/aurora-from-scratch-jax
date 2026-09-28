# E1 on the GPU box

Runs E1 (`handouts/e1_volume_diversity.md`) end to end on the lab VM. The spec was frozen on
28 Sep 2026, before any E1 run. Every step is resumable: rerun the same command after an
interruption.

## VM rules (project_memory.md §3)
* Check `nvidia-smi` first. Do not launch at ≥ 90 °C or with thermal slowdown active.
* Set `CUDA_VISIBLE_DEVICES` explicitly on every command. Kill only by explicit PID.
* Work under the project directory, never `/tmp` or the root partition.

## 0. Setup (once)
```bash
git clone https://github.com/ARYAMANN7279/aurora-from-scratch-jax.git && cd aurora-from-scratch-jax
python3.11 -m venv .venv && .venv/bin/pip install "jax[cuda12]" numpy scipy sympy pytest matplotlib optax
CUDA_VISIBLE_DEVICES=<card> JAX_DEFAULT_MATMUL_PRECISION=highest .venv/bin/python -m pytest tests -q
# measured 90/91 on the GPU before W10 was added (expect 91/92); D9 fails for a diagnostics-only reason (experiments/stage2/REPORT.md, addendum)
```

## 1. Data: 326 trajectories, 64² frames, ≈ 10.7 GB
Measured cost (28 Sep):
* **GPU:** 132 s / 304 s per trajectory at N_gen 128 / 256 on an RTX PRO 6000, so about 21 GPU-h
  (the spec guessed 8).
* **CPU:** about 13 h on 4 pinned single-thread cores. This is bitwise reproducible on a fixed
  machine, and it is the route used for `e1_v1`:
```bash
mkdir -p logs
for i in 0 1 2 3; do
  JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    nohup taskset -c $i .venv/bin/python experiments/stage2/generate.py \
    --config experiments/e1/configs/e1_v1.json --shard $i --n-shards 4 > logs/e1_gen_cpu_$i.log 2>&1 &
done
```
The configs are already in the repo (`experiments/e1/configs/`, built from the measured N_gen). For
GPU generation, use one lane per free card:
```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python experiments/stage2/generate.py \
    --config experiments/e1/configs/e1_v1.json --shard $g --n-shards 4 > logs/e1_gen_$g.log 2>&1 &
done
# when all lanes of either route have finished (all checksums are verified here):
.venv/bin/python experiments/stage2/generate.py --config experiments/e1/configs/e1_v1.json --finalize
```

## 2. Runs: 57
At the measured ~120 ms per step on a shared card, the 660k optimiser steps take about 22 GPU-h
(the spec guessed 7). Some rules for every lane:
* Export `JAX_DEFAULT_MATMUL_PRECISION=highest`.
* Use as many lanes as there are free cards, and keep `--n-shards` fixed for the whole experiment.
  `--skip-done` reads only the lane's own rows file.
* To run 2–3 lanes per card, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Pretrain and scratch runs first. Every fine-tune reads its pretrain run's weights.
```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python experiments/run.py \
    experiments/e1/configs/pretrain.json experiments/e1/configs/scratch.json \
    --skip-done --shard $g --n-shards 4 --results experiments/e1/results/rows_$g.jsonl > logs/e1_pre_$g.log 2>&1 &
done
# once all four lanes have finished:
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python experiments/run.py experiments/e1/configs/finetune.json \
    --skip-done --shard $g --n-shards 4 --results experiments/e1/results/rows_ft_$g.jsonl > logs/e1_ft_$g.log 2>&1 &
done
```
Each lane writes its own rows file, so parallel lanes never append to the same file. Parameters
go to `data/runs/<config_hash>/`, which all lanes share.

## 3. Analysis and results
```bash
cat experiments/e1/results/rows_[0-9].jsonl experiments/e1/results/rows_ft_*.jsonl > experiments/e1/results/all.jsonl
.venv/bin/python experiments/e1/analyse.py --results experiments/e1/results/all.jsonl
git add experiments/e1/results/*.jsonl experiments/e1/results/analysis.json && git commit -m "E1 results" && git push
```
Commit only the result rows and `analysis.json`: no data, no parameters.
