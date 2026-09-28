# Stage 2 Report: Dataset generator and loader

Spec: `handouts/stage02_dataset.md` (including amendment A1). Code: `src/kdata/`.
Tests: `tests/test_stage2_gates.py`. Results: `results/m1_Re*.json`, `results/acc_v1_analysis.json`,
`results/repro_acc_v1.json`, `results/loader_acc_v1.json`, `results/predictions.json`.

## Verdict

**Spectra are stable across trajectories; bitwise reproducibility holds for Re 100 and 300 but
failed for the one Re 1000 trajectory tested.** The acceptance dataset `acc_v1` is complete:
3 regimes × 8 seeds × 501 frames at 64² and 128², with all 24 SHA-256 checksums verified. All 9
gates pass. The spectral-stability predictions all HIT (Q4–Q8), and so does the loader throughput
(Q9). Q3 is a **MISS**: regenerated in a fresh process, the Re 100 and Re 300 trajectories match the
originals bitwise, but the Re 1000 trajectory (512² grid) does not. The investigation is below. It
does not affect the statistics of the data; it affects whether a single trajectory can be
regenerated bit for bit on a different machine or process state.

## Gates

`.venv/bin/python -m pytest tests/test_stage2_gates.py -q` → 8 passed (D1–D9, with D3/D4 in one test).
D2 (generator vs the Stage 1 solver) 5.8e-16; D3 1e-7; D4 3.4e-7; D5 4.9e-8; D7 1.1e-16; D1, D6, D8,
D9 exact.

## M1: generation grid (the fidelity bar in the saved output band)

`F_out(N)` at t = 1: the relative difference between N and 2N, restricted to |k| < N_out/2.

| Regime | N vs 2N | F (64² band) | F (128² band) |
|---|---|---|---|
| Re100 | 64 vs 128 | 1.7e-2 | 1.7e-2 |
| Re100 | **128** vs 256 | **2.9e-5** | **3.0e-4** |
| Re300 | 128 vs 256 | 5.4e-3 | 1.2e-2 |
| Re300 | **256** vs 512 | **8.1e-7** | **1.3e-5** |
| Re1000 | 256 vs 512 | 1.5e-3 | 5.0e-3 |
| Re1000 | **512** vs 1024 | **5.8e-7** | **1.6e-5** |

Rule (≤ 1e-3): **N_gen = 128 / 256 / 512 for Re 100 / 300 / 1000**, for both bands.

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| Q1 | N_gen (64 band): 128 / 256 / 256 | 128 / 256 / **512** | **MISS** (Re1000 at 256 is 1.5e-3) |
| Q2 | N_gen (128 band): 256 / 256 / 512 | **128** / 256 / 512 | **MISS** (Re100 needs less) |
| Q3 | cross-process sha256, one trajectory per regime | 2 / 3 equal (Re1000 differs) | **MISS** |
| Q4 | reduced χ² of per-seed ⟨E⟩ in [0.2, 3] | 2.87 / 1.44 / 1.25 | **HIT** |
| Q5 | per-seed slope within ±0.15 of the regime mean | 0.149 / 0.103 / 0.042 | **HIT** |
| Q6 | max shell deviation, k ≤ 31, < 0.3 | 0.299 / 0.225 / 0.230 | **HIT** |
| Q7 | regime mean vs Stage 1, every shell within 15% | 0.148 / 0.048 / 0.069 | **HIT** |
| Q8 | first vs second half ⟨E⟩ within 3 SE | 0.23 / 0.61 / 0.16 SE | **HIT** |
| Q9 | loader ≥ 2000 windows/s (64²), ≥ 500 (128²) | 28,959 / 8,864 | **HIT** |

Q5, Q6 and Q7 at Re100 sit right at their bars (0.149, 0.299, 0.148). The Re100 spread across seeds
is the largest (χ² 2.87), consistent with its longer decorrelation time (τ_e ≈ 1.4) giving fewer
independent samples per 100-unit record.

Figures: `figs/spectra_by_seed.png` (the 8 seeds per regime overlap), `figs/energy_by_seed.png`.

## The Re 1000 reproducibility miss

* Regenerated in a fresh process, Re1000 seed 7 already differs at **frame 0**, i.e. at the end of
  the 50-unit spin-up (max |Δ| 5.6e-3 on fields of order 10). At λ ≈ 0.57, 50 time units amplify
  a last-bit difference by ~e²⁸, so a single rounding difference anywhere in the 105,500 spin-up
  steps is enough.
* **Ruled out** (all bitwise identical at 512²):
  * two separate processes (2,000 steps);
  * single- vs multi-threaded XLA (2,000 steps);
  * three processes running concurrently under contention (21,100 steps each).
* **Demonstrated mechanism:** the result depends on the CPU instruction set XLA compiles for.
  Capping XLA at SSE4.2 changes the 512² hash; AVX2 gives the same hash as the default. So bitwise
  reproducibility is tied to the machine configuration.
* **Not identified:** what differed for the original seed-7 process. It ran for ~3.3 h inside a
  VM that is paused and restored whenever this session idles (see project_memory). A change of
  host or CPU state across a restore is the leading suspect, but it could not be proven.
* **Consequences:**
  * every trajectory is a valid sample of the dynamics (Q4–Q8 hold for Re1000 as for the others);
  * regeneration is exact on a fixed machine configuration (D1, D9, and Q3 for Re 100 / 300), and
    statistical, not bitwise, across machines, which is expected for chaotic flows (GPU vs CPU
    likewise);
  * for the experiment datasets: store the post-spin-up generation-grid state (`state_t0.npy`,
    4 MB at 512²) with each trajectory, so any trajectory can be re-derived exactly from its own
    stored state; and record the XLA ISA and host CPU in `meta.json`.

## Deviations
* Generation ran on the 4-core CPU container in parallel lanes and survived two container
  restarts through per-chunk checkpoints (amendment A1, gate D9).
* The loader throughput was measured with the dataset in the page cache (warm).

## Addendum (28 Sep 2026): gates on the GPU box
All gates pass on CPU. On the lab VM (RTX PRO 6000, JAX 0.10.2, CUDA 12.8), **gate D9 fails**:
* After an interrupted-then-resumed trajectory, both frame files are bitwise identical to the
  uninterrupted run, but `diag.npz` differs.
* **Likely cause (not yet confirmed on the GPU):** `shell_spectrum`
  (`src/kolmogorov/diagnostics.py`) bins energy into shells with a scatter-add. On GPUs, XLA
  implements scatter-add with atomics, so the summation order, and hence the last bits, can change
  from run to run. The frames involve no scatter.
* **Confirmation check:** rerun D9 with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, which is
  expected to pass. Or compare two *uninterrupted* GPU runs, which are expected to differ in
  `diag.npz` only.
* **Consequences:**
  * the training data (the frames) is unaffected;
  * the gate is unchanged;
  * the experiment datasets are generated on CPU where possible (`e1_v1` entirely; `acc_v1` for
    Re 100 / 300), and a GPU-generated trajectory's checksums are only reproducible bitwise for the
    frames.

## Commands
```
.venv/bin/python -m pytest tests/test_stage2_gates.py -q
.venv/bin/python experiments/stage1/run_resolution.py --src L1000 --grids 128 256 512 1024 --bands 64 128 --out experiments/stage2/results/m1_Re1000.json   # likewise L100, L300
.venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/acc_v1.json   # (--only tag:seed ... per lane)
.venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/acc_v1.json --finalize
.venv/bin/python experiments/stage2/repro.py --dataset acc_v1 --pick re100_kf4:3 re300_kf4:5 re1000_kf4:7
.venv/bin/python experiments/stage2/loader_bench.py --dataset acc_v1
.venv/bin/python experiments/stage2/analyse.py --dataset acc_v1
.venv/bin/python experiments/stage2/make_figs.py --dataset acc_v1
```
