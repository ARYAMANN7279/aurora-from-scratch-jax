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

**Follow-up F1 (pre-registered 4 Oct 2026; the verdicts above stand).** M3 was repeated on one fixed
machine: the lab machine's CPU, with the same cores, code and settings for both processes and the
persistent compilation cache off.
* Re 100 and Re 300 again regenerate bitwise.
* **Re 1000 does not:** all three of its files differ.

F1-Q3 is a **MISS**, so Stage 2's acceptance is not met on a fixed machine either. This refutes the
leading suspect below, a change of host under the original run. A full-length Re 1000 trajectory at
512² does not regenerate bit for bit even on one machine. See "Follow-up F1" below.

**Follow-up F2 (pre-registered 5 Oct 2026; the verdicts above stand): Stage 2's acceptance is met for
single-threaded generation on a fixed machine.**
* With single-threaded XLA, two fresh processes generated M3's three trajectories identically,
  including the full-length Re 1000 one: every file and the post-spin-up state.
* The gates pass with single-threaded XLA.
* With XLA's default multi-threading, F1's miss replicates. The two Re 1000 runs already differ
  after the spin-up (6.7e-8), so they part during it.

All three predictions HIT. By the acceptance fixed in F2 before the runs, Stage 2 is met for
single-threaded generation on a fixed machine. Multi-threaded generation at 512² does not regenerate
bit for bit, and across machines regeneration stays statistical. See "Follow-up F2" below.

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
  * single- vs multi-threaded XLA (2,000 steps; too short, see the correction of 5 Oct below);
  * three processes running concurrently under contention (21,100 steps each).
* **Demonstrated mechanism:** the result depends on the CPU instruction set XLA compiles for.
  Capping XLA at SSE4.2 changes the 512² hash; AVX2 gives the same hash as the default. So bitwise
  reproducibility is tied to the machine configuration.
* **Not identified:** what differed for the original seed-7 process. It ran for ~3.3 h inside a
  cloud VM that is paused and restored whenever it idles (see project_memory). A change of
  host or CPU state across a restore is the leading suspect, but it could not be proven.
* **Consequences:**
  * every trajectory is a valid sample of the dynamics (Q4–Q8 hold for Re1000 as for the others);
  * regeneration is exact on a fixed machine configuration (D1, D9, and Q3 for Re 100 / 300), and
    statistical, not bitwise, across machines, which is expected for chaotic flows (GPU vs CPU
    likewise);
  * for the experiment datasets: store the post-spin-up generation-grid state (`state_t0.npy`,
    4 MB at 512²) with each trajectory, and record the XLA ISA and host CPU in `meta.json`.
    Correction (30 Sep 2026): this lets a trajectory be re-derived bitwise without redoing the
    spin-up, but only on a machine with the same XLA ISA. The recording phase (100 time units at
    λ ≈ 0.57) amplifies a last-bit difference just as the spin-up does, so bitwise regeneration
    across machines is not achievable for these chaotic flows. Across machines, trajectories agree
    statistically (Q4–Q8).
  * Correction (4 Oct 2026, follow-up F1): regeneration is exact on a fixed machine for Re 100 and
    Re 300, but not for a full-length Re 1000 trajectory. So the stored state does not guarantee a
    bitwise re-derivation of the recording at 512², even on the same machine.
  * Correction (5 Oct 2026, follow-up F2): threading is the cause at 512². Over a full run,
    multi-threaded XLA does not give the same bits from one run to the next, while single-threaded
    XLA regenerates the full Re 1000 trajectory bitwise. The 2,000-step check was too short to show
    it. With single-threaded XLA (one core per job), regeneration is exact on a fixed machine at all
    three Re.

## Deviations
* Generation ran on the 4-core CPU container in parallel lanes and survived two container
  restarts through per-chunk checkpoints (amendment A1, gate D9).
* The loader throughput was measured with the dataset in the page cache (warm).

## Addendum (28 Sep 2026): gates on the GPU box
All gates pass on CPU. On the lab VM (RTX PRO 6000, JAX 0.10.2, CUDA 12.8), **gate D9 fails**:
* After an interrupted-then-resumed trajectory, both frame files are bitwise identical to the
  uninterrupted run, but `diag.npz` differs.
* **Likely cause (as stated on 28 Sep; confirmed on 9 Oct, below):** `shell_spectrum`
  (`src/kolmogorov/diagnostics.py`) bins energy into shells with a scatter-add. On GPUs, XLA
  implements scatter-add with atomics, so the summation order, and hence the last bits, can change
  from run to run. The frames involve no scatter.
* **Confirmation check:** rerun D9 with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, which is
  expected to pass. Or compare two *uninterrupted* GPU runs, which are expected to differ in
  `diag.npz` only.
* **Confirmed (9 Oct 2026),** on lab card 3 (RTX PRO 6000, JAX 0.10.2):
  * D9 fails in both of two runs at the default settings, and passes in both of two runs with
    `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, as expected.
  * Rebuilding D9's interrupted and uninterrupted trajectories at the default settings,
    `frames_16.npy` and `frames_32.npy` are identical and only `diag.npz` differs.
  * So the GPU failure comes from non-deterministic operations in the diagnostics, and disappears
    when XLA is made deterministic. The frames do not depend on it.
* **Consequences:**
  * the training data (the frames) is unaffected;
  * the gate is unchanged;
  * the experiment datasets are generated on CPU where possible (`e1_v1` entirely; `acc_v1` for
    Re 100 / 300), and a GPU-generated trajectory's checksums are only reproducible bitwise for the
    frames.

## Follow-up F1: M3 on one fixed machine (pre-registered 4 Oct 2026)

Spec: F1 in `handouts/stage02_dataset.md`, committed before the run (a6a5510). Result:
`results/repro_s2f1_v1.json`.

**Set-up as registered.**
* **Machine:** the lab machine's CPU (Intel Xeon w3-2525; JAX 0.10.2; float64), pinned to cores 8–15,
  with GPUs hidden.
* **Two passes,** one lane, one after the other:
  * **First pass:** `generate.py --only` generated M3's three picks into `s2f1_v1` (`acc_v1`'s
    settings) in one process: Re 100 in 54 s, Re 300 in 12 min and Re 1000 in 2.3 h. Then
    `--finalize --partial` verified the manifest's 3 trajectories.
  * **Second pass:** `repro.py`, unchanged, regenerated them in a fresh process (13:25–15:56 IST).
  * Both passes called the generator with the same arguments, at the same commit.
* **Compilation cache:** the persistent compilation cache was off in both processes. Each compiled
  its own programs, as in the original M3.

| Trajectory | Grid | Steps | `frames_64.npy` | `frames_128.npy` | `diag.npz` |
|---|---|---|---|---|---|
| Re 100, seed 3 | 128² | 36,000 | equal | equal | equal |
| Re 300, seed 5 | 256² | 120,000 | equal | equal | equal |
| Re 1000, seed 7 | 512² | 316,500 | **differs** | **differs** | **differs** |

**F1-Q3 (all three identical): MISS.**

**Readings**
* **The miss is not a change of machine.** It reproduces on one fixed machine, with nothing changed
  between the processes. That refutes the original report's leading suspect, a host change across
  the container's pause and restore.
* **The cause is not identified** (F2, below, traces it to multi-threading).
  * The earlier checks at 512² were bitwise identical across processes: 2,000 steps, single against
    multiple threads, and 21,100 steps under contention.
  * So whatever differs is rare per step, or shows up only in a full-length run.
  * At λ ≈ 0.57, one last-bit difference anywhere in the 316,500 steps is enough to change the saved
    frames.
  * A test that would locate it: run two fresh processes of the 512² spin-up side by side, and
    compare their states at every chunk, with multiple threads and with one.
* **What holds and what does not:**
  * Every trajectory is still a valid sample of the dynamics (Q4–Q8), so the data and every result
    trained on them are unaffected.
  * Bitwise regeneration from the seed holds on one machine at Re 100 and Re 300. It does not hold
    at Re 1000, on one machine or across machines.
  * Stage 2's acceptance stays not met.

## Follow-up F2: where the Re 1000 regeneration parts, and single-threaded XLA (pre-registered 5 Oct 2026)

Spec: F2 in `handouts/stage02_dataset.md`, committed before the runs (3ca04eb). Results:
`results/f2_divergence_m.json`, `results/f2_divergence_s.json`, `results/f2_analysis.json` and
`results/f2_gates_single_thread.txt`.

**Set-up as registered.**
* **Settings:** the lab machine's CPU, GPUs hidden, the persistent compilation cache off, the lowest
  scheduling priority, and `acc_v1`'s dataset settings (`configs/s2f2_v1.json`). All lanes started
  at 01:45 IST.
* **M:** `re1000_kf4:7` with F1's settings (cores 8–15, default threading), in 4.3 h. It is compared
  with F1's first pass (`s2f1_v1`).
* **S-a and S-b:** M3's three picks with single-threaded XLA, side by side on cores 6 and 7, each in
  a fresh process. Re 100 took 1–1.5 min, Re 300 14–15 min and Re 1000 4.1–4.3 h.
* **Gates:** with single-threaded XLA, 8 passed.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| F2-Q1 | M differs from F1's first pass | all three files differ | **HIT** |
| F2-Q2 | S-a and S-b identical for all three trajectories (every file, post-spin-up state) | identical, the full-length Re 1000 one (512², 316,500 steps) included | **HIT** |
| F2-Q3 | the M pair parts during the spin-up | post-spin-up states differ by 6.7e-8 | **HIT** |

**Where the multi-threaded pair parts.**
* After the spin-up (t = 50) the two states differ by 6.7e-8 (relative).
* Over the recording, the difference grows:
  * in the 128² vorticity, at about 0.66 per unit time, from 7.6e-8 at t = 50 to 3.7e-2 at t = 70;
  * in E, at 0.70 (fitted over 88 frames). Stage 1's Lyapunov exponent at Re 1000 is 0.565.
  * From t ≈ 80 the two vorticity fields are uncorrelated (difference ≈ √2).
* Extrapolated back, a last-bit difference (1e-16) reaches 6.7e-8 by t = 50 if it starts at t ≈ 14
  at Stage 1's rate, or t ≈ 21 at the fitted rate.
  * That is roughly 30,000–44,000 steps into the 105,500-step spin-up. It is consistent with the
    earlier 21,100-step checks (t = 10) being identical.
  * Extrapolating E's difference alone gives t ≈ 28. That estimate is biased late: E's relative
    difference is about 100 times smaller than the state's (5.8e-10 against 6.7e-8 at t = 50).

**Single- against multi-threaded (not pre-registered;** `results/f2_divergence_s_vs_f1.json`**).**
Comparing S-a with F1's first pass:
* **Re 100 and Re 300: identical.** On the 128² and 256² grids threading changed no bits, which is
  consistent with these trajectories regenerating in M3 and F1.
* **Re 1000: differs,** by 3.2e-2 already after the spin-up. That is far more than the
  multi-threaded pair's 6.7e-8, so on the 512² grid threading changes the bits much earlier.

**Readings**
* **The cause is XLA's multi-threaded execution on the 512² grid.**
  * With one thread, the full-length Re 1000 trajectory regenerates bit for bit in a fresh process.
  * With eight threads, two runs on the same machine part during the spin-up.
  * Which operation varies was not isolated. The FFT is the leading suspect.
* **Stage 2's acceptance is met for single-threaded generation on a fixed machine,** by the rule
  fixed in F2 before the runs. That covers D1–D8, M3's criterion, Q5 and Q6.
  * The earlier verdicts (Q3, F1-Q3) stay as recorded.
  * Single-threaded generation takes longer per trajectory but needs only one core: Re 1000 took
    4.1–4.3 h on one core, against 2.3 h on eight cores in F1.
* **What does not change:**
  * Multi-threaded generation at 512² does not regenerate bit for bit.
  * Across machines, regeneration stays statistical (Q4–Q8), as for any chaotic flow.
  * The existing datasets and every result trained on them are unaffected.

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
# follow-up F1, on the lab machine's CPU (cores 8-15, GPUs hidden, persistent compilation cache off)
.venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f1_v1.json --only re100_kf4:3 re300_kf4:5 re1000_kf4:7
.venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f1_v1.json --finalize --partial
.venv/bin/python experiments/stage2/repro.py --dataset s2f1_v1 --pick re100_kf4:3 re300_kf4:5 re1000_kf4:7
# follow-up F2, on the lab machine's CPU (GPUs hidden, persistent compilation cache off, nice -n 19)
# M: cores 8-15, default threading
taskset -c 8-15 .venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f2_v1.json --root data/stage2/s2f2_v1_m --only re1000_kf4:7
taskset -c 8-15 .venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f2_v1.json --root data/stage2/s2f2_v1_m --finalize --partial
# S-a (core 6) and S-b (core 7, --root data/stage2/s2f2_v1_b), single-threaded XLA
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 6 .venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f2_v1.json --root data/stage2/s2f2_v1_a --only re100_kf4:3 re300_kf4:5 re1000_kf4:7
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 6 .venv/bin/python experiments/stage2/generate.py --config experiments/stage2/configs/s2f2_v1.json --root data/stage2/s2f2_v1_a --finalize --partial
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 5 .venv/bin/python -m pytest -p no:cacheprovider tests/test_stage2_gates.py -q > experiments/stage2/results/f2_gates_single_thread.txt
.venv/bin/python experiments/stage2/divergence_f2.py --a data/stage2/s2f2_v1_m --b data/stage2/s2f1_v1 --picks re1000_kf4:7 --out f2_divergence_m
.venv/bin/python experiments/stage2/divergence_f2.py --a data/stage2/s2f2_v1_a --b data/stage2/s2f2_v1_b --picks re100_kf4:3 re300_kf4:5 re1000_kf4:7 --out f2_divergence_s
.venv/bin/python experiments/stage2/divergence_f2.py --a data/stage2/s2f2_v1_a --b data/stage2/s2f1_v1 --picks re100_kf4:3 re300_kf4:5 re1000_kf4:7 --out f2_divergence_s_vs_f1
.venv/bin/python experiments/stage2/score_f2.py
```
