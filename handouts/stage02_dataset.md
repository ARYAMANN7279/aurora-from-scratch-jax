# Stage 2 Spec: Dataset generator and loader

**Plan acceptance test (PROJECT_PLAN.md §5):** reproducible from seed; spectra stable across
trajectories.
**Builds on:** the Stage 1 solver (`src/kolmogorov/`), used unchanged.
**Written:** 26 Sep 2026, before any Stage 2 code. §§2–5 (design, gates, measurements,
predictions) are fixed. The rules of the Stage 1 spec §0 apply: fix code, not tolerances; two
attempts, then report; every number comes from a script output plus its command.

---

## 1. Why this stage matters

Every experiment (E1–E5) trains and scores on these files. The generator decides what "a regime",
"a trajectory" and "a sample" mean. The loader decides what the model sees and which data is held
out. Both must be exactly reproducible, and the held-out data must be separated by regime
(project_memory §2.2).

---

## 2. Design

### 2.1 Units of data
* **Regime:** `(Re, k_f)`, with `A = 1`, `α = 0.1` fixed. Tag `re{Re}_kf{k_f}`, e.g. `re1000_kf4`.
* **Trajectory:** `(regime, seed)`. Its IC is the Stage 1 recipe `random_ic(g, seed)` on the
  generation grid (white noise, `1 ≤ |k| ≤ 8`, masked, `E = 1`).
* **Spin-up:** integrate `T_spin = 50` and discard it (Stage 1: the transient ends within ~40).
* **Record:** `T_rec` time units after spin-up. One frame every `Δ = 0.2`, i.e. 1 + T_rec/Δ frames
  including the first. With `τ_e ≈ 0.8–1.4` (Stage 1) that gives 4–7 frames per decorrelation time.
  Training strides that are multiples of Δ come from the loader.

### 2.2 Frames
* **Channels:** `(u, v, p, ω)`, float32, arrays indexed `[y, x]`.
* **Output resolutions:** `N_out ∈ {64, 128}`, produced from the same run by spectral truncation
  of the generation-grid coefficients (`resample`, keeping `|kx|, |ky| < N_out/2`).
* `u, v, ω` are linear in `ω̂`, so truncation commutes with them: the stored `u, v` are exactly those
  implied by the stored `ω`. `p` is quadratic, so it is solved on the generation grid (Stage 1
  `pressure_hat`: `∇²p = 2(u_x v_y − u_y v_x)`, which is the full pressure for this forcing,
  since `∇·F = 0`) and then truncated.
* Per frame, also stored: `E, Z, P, CFL` and the generation-grid shell spectrum.

### 2.3 Generation grid N_gen
The Stage 1 resolution metric compared full vorticity fields. The saved training data, though, only
contains `|k| < N_out/2`. The rule therefore compares the **output band**:

    F_out(N) = ‖B(ω_N) − B(ω_2N)‖ / ‖B(ω_2N)‖    at t = 1, common IC, same dt (finest grid's)
    B = truncation to |kx|, |ky| < N_out/2

**Rule:** `N_gen(regime, N_out)` = the smallest tested N with `F_out(N) ≤ 1e-3`. A dataset that
stores N_out = 128 uses `N_gen(regime, 128)`.
Measurement M1 (§4) sets it for Re ∈ {100, 300, 1000}, k_f = 4. For any other regime the generator
requires an explicit `N_gen`, until a later measurement extends the table.

### 2.4 Storage (under `data/stage2/<dataset>/`, git-ignored)
    dataset.json                           the config (regimes, seeds, T_spin, T_rec, Δ, N_out, N_gen)
    <tag>/seed_<SSSS>/frames_<N_out>.npy   (n_frames, 4, N_out, N_out) float32
    <tag>/seed_<SSSS>/diag.npz             t, E, Z, P, cfl, spec (n_frames × n_shells, float64)
    <tag>/seed_<SSSS>/meta.json            regime, seed, N_gen, dt, m, T_spin, Δ, t0, sha256 of each file, wall time
    manifest.json                          every trajectory's meta, written by `finalize`

A trajectory whose `meta.json` exists is complete. Generation skips it, so a run can resume at
trajectory granularity. Any CFL > 0.7 voids the trajectory and aborts generation.

### 2.5 Loader
* `WindowDataset(root, N_out, regimes, seeds=None, stride=1, n_in=2, n_out=1)`: indexes every window
  of `n_in + n_out` consecutive frames spaced by `stride` inside a trajectory, never across
  trajectories. It reads frames through `np.load(mmap_mode="r")`.
* **Splits** are lists of regimes, optionally with seeds within a regime. `make_splits` refuses a
  regime that is both held out and in training.
* **Normalisation:** per-channel mean and std over the training split only, stored alongside the
  split.
* **Batches:** `batches(batch_size, seed, shuffle=True, drop_last=True)`. The order is a
  deterministic function of `seed`; each epoch passes its own seed.
* Output: `x` of shape `(B, n_in, 4, N, N)` and `y` of shape `(B, n_out, 4, N, N)`, float32 numpy.

---

## 3. Gates — `tests/test_stage2_gates.py` (pytest)

Run on a tiny dataset: 2 regimes × 2 seeds, N_gen = 64, N_out ∈ {16, 32}, T_spin = 1, T_rec = 2.

| Gate | Test | Pass if |
|---|---|---|
| D1 | regenerate one trajectory in-process | frames bitwise equal |
| D2 | generator ω at the generation grid vs a Stage 1 `make_advance` run from the same IC for the same steps | ≤ 1e-12 relative |
| D3 | stored `u, v` vs `u, v` derived spectrally from stored `ω` (float64 arithmetic on float32 data) | max-abs ≤ 1e-5 · max\|u\| |
| D4 | spectral divergence of stored `(u, v)` | ≤ 1e-5 · rms(ω) |
| D5 | stored `p` vs pressure recomputed independently on the generation grid and truncated | max-abs ≤ 1e-5 · max\|p\| |
| D6 | loader windows equal the right frames; count = Σ_traj (n_frames − (n_in+n_out−1)·stride); no window crosses a trajectory | exact |
| D7 | splits: overlapping regimes rejected; same seed → same order, different seed → different order; normalisation stats equal those of the training frames only | exact / ≤ 1e-6 relative |
| D8 | the manifest sha256 of each file matches the file on disk | exact |

---

## 4. Measurements

* **M1: generation grid.** `run_resolution.py`, extended with output-band metrics for
  N_out ∈ {64, 128}. Common IC from the Stage 1 t = 50 state; T = 2; criterion at t = 1.
  Grids: Re100 {64, 128, 256, 512}; Re300 {64, 128, 256, 512}; Re1000 {128, 256, 512, 1024}.
* **M2: acceptance dataset `acc_v1`.** Re ∈ {100, 300, 1000}, k_f = 4, seeds 0–7, T_spin = 50,
  T_rec = 100, Δ = 0.2, N_out ∈ {64, 128}, N_gen from M1 (the 128 band).
* **M3: cross-process reproducibility.** In a fresh process, regenerate 3 trajectories of `acc_v1`
  (one per regime) and compare sha256.
* **M4: spectral stability.** For each regime of `acc_v1`: per-trajectory ⟨E⟩ over the record with
  batch-means SE (5 batches of 20); reduced χ² of the 8 ⟨E⟩ about their weighted mean; each
  trajectory's time-mean spectrum vs the regime mean for `1 ≤ k ≤ 31`; slope over k ∈ [8, 16];
  first-half vs second-half ⟨E⟩; the regime-mean spectrum vs the Stage 1 long run at the same
  regime.
* **M5: loader throughput.** Windows/s for batch 32, N_out = 64 and 128, n_in = 2, shuffled,
  memmap from the local disk, best of 3 epochs.

---

## 5. Pre-registered predictions

| # | Quantity | Prediction |
|---|---|---|
| Q1 | N_gen for the 64 band | Re100: 128 · Re300: 256 · Re1000: 256 |
| Q2 | N_gen for the 128 band | Re100: 256 · Re300: 256 · Re1000: 512 |
| Q3 | M3 cross-process sha256 | all equal |
| Q4 | reduced χ² of the per-trajectory ⟨E⟩, each regime | in [0.2, 3] |
| Q5 | per-trajectory slope, k ∈ [8, 16] | within ±0.15 of the regime mean, every trajectory |
| Q6 | max over 1 ≤ k ≤ 31 of \|E_i(k)/Ē(k) − 1\| | < 0.3, every trajectory |
| Q7 | regime-mean spectrum vs Stage 1 long run, 1 ≤ k ≤ 31 | every shell within 15% |
| Q8 | first-half vs second-half ⟨E⟩ (regime mean over seeds) | within 3 combined SE, each regime |
| Q9 | loader throughput, batch 32, n_in = 2 | ≥ 2000 windows/s at 64; ≥ 500 at 128 |

**Stage 2 acceptance:** D1–D8 pass, Q3 HIT, and spectra stable across trajectories, meaning Q5 and
Q6 HIT. A MISS on Q1, Q2, Q4, Q7, Q8 or Q9 is reported and reviewed, but does not by itself block
acceptance.

## 6. Report — `experiments/stage2/REPORT.md`

Same structure as Stage 1: gates with measured values, predictions HIT/MISS, the M1 table, the
dataset table (per trajectory: ⟨E⟩ ± SE, slope, max spectral deviation, max CFL, wall time,
storage), figures (per-regime spectra of all 8 seeds, ⟨E⟩ per seed with error bars), bugs fixed,
deviations, spec concerns, commands.
