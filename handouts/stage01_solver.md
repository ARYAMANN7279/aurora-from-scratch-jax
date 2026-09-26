# Stage 1 Spec: Pseudo-spectral Kolmogorov solver

**Machine:** Mac, M2, 8 GB, **CPU only**. No GPU for this stage.

This spec is self-contained. Everything in §§1–6 (equations, numerics, tests, tolerances,
runs, predictions) is **fixed**. If any of it appears incorrect, record the concern in the report (§8, "Spec concerns") and carry on with the spec as written.

---

## 0. Rules — read first

1. **Spec, tolerances and predictions are fixed.** If a test fails, fix the code, not the
   tolerance. Never loosen, skip, xfail or delete a gate.
2. **Two attempts, then report.** If a gate or run still fails after two honest fix attempts,
   stop working on it, record exactly what fails (command + output), and move on or stop.
3. **Every number in the report comes from a script output**, and the report quotes the command
   that produced it. No hand-computed or remembered numbers.
4. **Own solver.** Write it from the equations below. Do not paste code from jax-cfd, Dedalus,
   blog posts or any other solver. (jax-cfd is allowed only in the optional cross-check, §5.4,
   in its own venv.)
5. **Write only** `src/`, `tests/`, `experiments/stage1/`. Do not touch `PROJECT_PLAN.md`,
   `project_memory.md`, `README.md`, `handouts/`.
6. **Local commits only. Do not push.** Push happens after spec author's review of the report.
7. **Stop after Stage 1.** Do not start the dataset generator or any model code.
8. Large outputs (`.npy`, `.npz`, checkpoints) go under `data/` (git-ignored). Only small
   result JSONs, the report and small PNG figures are committed, under `experiments/stage1/`.

---

## 1. Environment

* Use **Python 3.11** (`python3.11 -m venv .venv`). 3.14 may lack JAX wheels. Install
  `jax` (CPU), `numpy`, `scipy`, `sympy`, `pytest`, `matplotlib`. Record exact versions
  (`pip freeze`) in the report.
* **float64 everywhere** (`jax.config.update("jax_enable_x64", True)` before any array is made),
  except the one precision run in §5.3.
* Compile the step with `jax.jit`; advance with `jax.lax.scan`. Outer Python loop over
  **10-time-unit chunks**; inside a chunk, a scan over save intervals, each an inner scan of `m`
  steps (below). Write outputs and a full-state checkpoint after every chunk, so a run can resume.
* Wrap long runs in `caffeinate -i` so the Mac does not sleep.
* Stage 1 uses **jax + jax.numpy only**. No Flax, no Optax. (Whether later stages use Flax is an
  open question to be decided before Stage 3.)

---

## 2. The equations

2-D incompressible Navier–Stokes in vorticity form, doubly periodic box `[0, 2π)²`:

    ∂ω/∂t + u·∇ω = ν∇²ω − αω + f,       f(y) = −A k_f cos(k_f y)
    u = ∂ψ/∂y,   v = −∂ψ/∂x,   ω = −∇²ψ

The vorticity forcing `f` is the curl of the body force `F = (A sin(k_f y), 0)`.

**Parameters:** `A = 1`, `α = 0.1`, `k_f = 4`, `ν = 1/Re`.
Re ∈ {100, 300, 1000}. Re = 1000 (ν = 1e-3) is the Kochkov et al. (2021) configuration.

Define `λ = ν k_f² + α`. The laminar steady state is

    ω_L = −(A k_f / λ) cos(k_f y),   u_L = (A/λ) sin(k_f y),   E_lam = A² / (4 λ²)

    Re   ν        λ        E_lam
    100  1e-2     0.26     3.698
    300  3.333e-3 0.15333  10.63
    1000 1e-3     0.116    18.58

---

## 3. Numerics

### 3.1 Grid and transforms
* Arrays are indexed **`[y, x]`**: axis 0 is y, axis 1 is x. Grid points `x_j = 2πj/N`.
* Use `jnp.fft.rfft2` / `irfft2` with **`norm="forward"` everywhere**. With this normalisation
  the coefficients are Fourier amplitudes, so `f̂[0,0]` is the mean, and moving between
  resolutions only means re-indexing coefficients (no rescaling).
* Spectral shape `(N, N/2+1)`. Wavenumbers (integers, since the box is 2π):
  `ky = fftfreq(N)·N` along axis 0, `kx = rfftfreq(N)·N = 0..N/2` along axis 1.
  `|k|² = kx² + ky²`.

### 3.2 Dealiasing mask
Strict square 2/3 rule:

    M(kx, ky) = 1  if |kx| < N/3  and  |ky| < N/3  and  (kx, ky) ≠ (0, 0),   else 0

The mean mode is always excluded; `ω̂[0,0] = 0` at all times. The prognostic state `ω̂` always
satisfies `ω̂ = M·ω̂`.

### 3.3 Operators
    ψ̂ = ω̂ / |k|²        (with ψ̂[0,0] = 0)
    û = i ky ψ̂,   v̂ = −i kx ψ̂
    ω̂_x = i kx ω̂,  ω̂_y = i ky ω̂

**Nonlinear term** (advection, sign as written):

    N̂ = M · rfft2( u ω_x + v ω_y )

with `u, v, ω_x, ω_y` obtained by `irfft2` of the masked coefficients above.

**RHS split:** `∂ω̂/∂t = L ω̂ + G(ω̂)` with

    L = −ν|k|² − α                (diagonal, integrated exactly)
    G(ω̂) = −N̂(ω̂) + M·f̂          (explicit; the forcing lives HERE, not in L)

Keeping the forcing in `G` is deliberate: the laminar test G9 then measures the RK4 order.

### 3.4 Time stepping: Lawson integrating-factor RK4
With `E(h) = exp(L h)` (elementwise) and step `h`:

    a   = G(ω̂_n)
    ω̂_a = E(h/2) (ω̂_n + (h/2) a)
    b   = G(ω̂_a)
    ω̂_b = E(h/2) ω̂_n + (h/2) b
    c   = G(ω̂_b)
    ω̂_c = E(h) ω̂_n + h E(h/2) c
    d   = G(ω̂_c)
    ω̂_{n+1} = E(h) ω̂_n + (h/6) [ E(h) a + 2 E(h/2)(b + c) + d ]

(This is classical RK4 applied to `v = e^{−Lt} ω̂`.) Apply `M` to the result.

### 3.5 Time step and CFL
Saves happen every **0.1** time units, so `dt = 0.1/m` for integer `m`:

    U  = 1.5 A / λ                         (1.5 × laminar peak speed)
    m  = smallest integer with  dt = 0.1/m ≤ 0.5 · (2π/N) / U

Log, at every save,

    CFL = dt · max(|u| + |v|) · N / (2π)

**A run with CFL > 0.7 at any save is void.** Report it; do not tune `dt` by hand.

### 3.6 Resolution change
`up(N→N')` and `down(N'→N)` by copying Fourier coefficients (zero-padding / truncation), with
correct placement of negative-`ky` rows. Because every stored state is masked, Nyquist modes are
zero and no special handling is needed. Snapshots saved "at 64²" are `down(N→64)`: keep
`|kx|, |ky| < 32`, then `irfft2` on the 64 grid, cast to float32.

---

## 4. Diagnostics

Use `⟨·⟩` for the domain mean. Spectral sums use the **Parseval weight**
`w = 1` on the `kx = 0` and `kx = N/2` columns, `w = 2` elsewhere, so `⟨f g⟩ = Σ w Re(f̂ ĝ*)`.

    E = ½⟨u² + v²⟩ = ½ Σ w |k|² |ψ̂|²       energy
    Z = ½⟨ω²⟩                               enstrophy
    Π = ½⟨|∇ω|²⟩                            palinstrophy
    P = A ⟨u sin(k_f y)⟩                    energy injection
    Q = ⟨ω f⟩                               enstrophy injection

Budgets (exact for the continuous equations):

    dE/dt = P − 2νZ − 2αE
    dZ/dt = Q − 2νΠ − 2αZ

**Budget residual** over a window `[t0, t1]`:

    R_E = E(t1) − E(t0) − ∫ (P − 2νZ − 2αE) dt
    R_Z = Z(t1) − Z(t0) − ∫ (Q − 2νΠ − 2αZ) dt

Evaluate the integrands at **every time step**, and accumulate the integrals with the trapezoid
rule in the scan carry. Do not integrate from the 0.1-spaced saves; the quadrature error there
would swamp what is being measured. Report `|R_E| / ∫P dt` and `|R_Z| / ∫Q dt`.

**Shell spectrum:** `E(k) = Σ_{k−½ ≤ |k| < k+½} w · ½|k|²|ψ̂|²` for integer `k ≥ 0`.
Must satisfy `Σ_k E(k) = E`.
Let `K` be the largest integer `< N/3`, the last shell that is complete along the axes.

**Pressure** (diagnostic only): solve `∇²p = 2(u_x v_y − u_y v_x)` spectrally, `p̂[0,0] = 0`,
product computed as in `N̂` (masked).

**Dissipation wavenumber** (Kraichnan): `k_d = (2ν⟨Π⟩)^{1/6} / ν^{1/2}`.

**Standard error** of any time mean over `[50, 250]`: batch means, 10 non-overlapping 20-unit
batches, `SE = std(batch means, ddof=1) / √10`.

---

## 5. What to build and run

### 5.1 Gates — `tests/test_stage1_gates.py` (pytest, float64)
Fields for G1–G3, G5, G6 are random, seeded (`seed=0`), normalised to rms 1. "Band-limited"
means the field is projected with `M` first. Errors are max-abs unless marked relative.

| Gate | Test | Pass if |
|---|---|---|
| G1 | `irfft2(rfft2(x))` vs `x`, N = 64 | ≤ 1e-14 |
| G2 | band-limited ω → u, v. Spectral `u_x + v_y`; spectral `v_x − u_y − ω` | both ≤ 1e-12 |
| G3 | Parseval `⟨ω²⟩` physical vs weighted spectral sum; `Σ E(k)` vs `E` | relative ≤ 1e-12 |
| G4 | `N` vs sympy's exact `u ω_x + v ω_y` for `ψ = sin x cos 2y + 0.7 cos 3x sin(y+0.4) + 0.3 sin(2x+y)`, N = 32 (every product mode is retained) | ≤ 1e-12 |
| G5 | masked N-grid product vs exact product (zero-pad inputs to 2N, multiply, transform, truncate to N, apply M), band-limited inputs | ≤ 1e-12 |
| G5c | *control:* same, inputs NOT band-limited (content up to N/2), unmasked | > 1e-3 (aliasing must show) |
| G6 | `|⟨ψN⟩| / (rms ψ · rms N)` and `|⟨ωN⟩| / (rms ω · rms N)`, band-limited ω | ≤ 1e-12 |
| G6c | *control:* non-band-limited ω, no mask | > 1e-8 |
| G7 | Taylor–Green `ψ = sin x sin y` (u = sin x cos y, v = −cos x sin y), N = 32: pressure vs `p = ¼(cos 2x + cos 2y)` | ≤ 1e-12 |
| G8 | Free decay, N = 32, A = 0, ν = 0.05, dt = 0.01, T = 5: random combination of the modes with `|k|² = 25` (so the nonlinear term vanishes analytically) vs `ω₀·e^{−(25ν+α)t}` | relative ≤ 1e-11 |
| G9 | Laminar spin-up from rest, ν = 0.05, N = 32, T = 10, vs `ω(t) = −(A k_f/λ) cos(k_f y)(1 − e^{−λt})`. At dt = 0.1: `‖err‖/‖ω_L‖`. Over dt = 0.1, 0.05, 0.025, 0.0125: the three observed orders `log2(e_j/e_{j+1})` | ≤ 1e-6; each order in [3.8, 4.2] |
| G10 | `down(up(ω))` vs ω (N → 2N → N); `up` of a band-limited analytic field vs the same field sampled on the 2N grid | ≤ 1e-13 |
| G11 | 100 steps of the forced solver from a random band-limited state: every coefficient with `M = 0` | exactly 0.0 |

Run all gates before any long run. **No long run starts until G1–G11 all pass.**

### 5.2 Long runs (the core of Stage 1)

| Name | Re | N | T |
|---|---|---|---|
| L100  | 100  | 128 | 250 |
| L300  | 300  | 256 | 250 |
| L1000 | 1000 | 256 | 250 |

* **IC:** real white noise on the grid (seed 0), `rfft2`, keep `1 ≤ |k| ≤ 8` and apply M, then
  rescale so `E = 1`. The same recipe (with N) for every run.
* Discard `[0, 50]`. Average over `[50, 250]`.
* Every 0.1: save `t, E, Z, Π, P, Q, CFL`, the shell spectrum `E(k)` (f64), and the 64² float32 ω
  snapshot. Save the full-resolution state at `t = 50` (needed below) and at every chunk end.
* Per run, compute and save to `experiments/stage1/results/<name>.json`:
  `dt, m, max CFL, wall time, ⟨E⟩±SE, ⟨Z⟩±SE, ⟨P⟩, ⟨Q⟩, |R_E|/∫P, |R_Z|/∫Q` over [50, 250],
  `2ν⟨Z⟩/⟨P⟩`, `k_d`, time-mean spectrum, `E(K)/max E(k)`, the spectral slopes below, spin-up
  time, `τ_e`.
* **Spectral slopes:** least-squares slope of `log E(k)` vs `log k` on the time-mean spectrum
  (a) over integer `k ∈ [8, 16]`, (b) over the top 20% of shells, `k ∈ [0.8K, K]`.
  Also report `E(K)/max_k E(k)`.
* **Spin-up time:** first `t` at which `E(t)` enters the [5th, 95th] percentile band of `E` over
  [150, 250]. Also plot the 10-unit block means of `E` over the whole run.
* **Decorrelation time τ_e:** from the 64² snapshots over [50, 250]. Let `ω'` be ω minus its
  time-mean field; `C(τ) = ⟨ω'(t)ω'(t+τ)⟩_t,x / ⟨ω'²⟩_t,x`. `τ_e` is the first τ with
  `C < 1/e`, linearly interpolated.

**Note on "expected inertial-range scaling"** (plan acceptance test). With `k_f = 4` the inverse
cascade range is `k = 1–3`: three shells, arrested by drag. A −5/3 fit there is not meaningful and
is **not** part of this stage. The testable scaling is the forward enstrophy cascade, `k > k_f`,
expected at −3 or steeper with drag. Those are P6/P7 below.

### 5.3 Supporting measurements
1. **Temporal order.** For each long run: start from its `t = 50` state and run to T = 1 at
   `dt/2^j`, j = 0..3, with reference j = 5. `e_j = ‖ω_j − ω_ref‖/‖ω_ref‖`, and the orders are
   `log2(e_j/e_{j+1})` for j = 0..2. If an `e_j < 1e-12`, mark that pair roundoff-limited, report
   it, and leave it out of the order.
2. **Resolution map.** A common truncated IC, same dt, T = 2:

   | Re | grids |
   |---|---|
   | 100  | 32, 64, 128 |
   | 300  | 64, 128, 256 |
   | 1000 | 128, 256, 512 |

   IC is the `t = 50` state of that Re's long run, truncated to the smallest grid's retained set
   (`|kx|, |ky| < N_min/3`) and embedded identically in each grid. dt is that of the **finest**
   grid (§3.5). Grid N is **resolved** iff `‖up(ω_N) − ω_2N‖ / ‖ω_2N‖ ≤ 1e-3` at `t = 1`,
   compared on the 2N grid. Also report the value at t = 2. (The finest grid has no 2N partner
   and is not classified.)
3. **Lyapunov exponent.** For each Re at its long-run N, from the `t = 50` state: 4 twins, each
   perturbed by a random band-limited field (seeds 1–4) scaled to `‖δω‖/‖ω‖ = 1e-8`. Track
   `d(t) = ‖ω_twin − ω‖/‖ω‖` every 0.1 for up to 100 time units. Fit the slope of `ln d` vs t on
   the first contiguous stretch with `1e-6 ≤ d ≤ 1e-2`. Report the mean ± spread over the 4 twins.
4. **float32 rerun.** Repeat L1000 in float32, same IC. Compare `⟨E⟩, ⟨Z⟩` and slope (a) with the
   f64 run, in units of the f64 SE. Report `|R_E|/∫P` and wall-time ratio.
5. **Throughput.** f64 steps/s at N = 64, 128, 256, 512 (Re = 1000 parameters, 1000 scanned
   steps, best of 3 after one warm-up/compile call). Record whether the machine was otherwise idle.

### 5.4 Lower priority, then optional
* *Lower priority:* `k_f ∈ {2, 8}` at Re = 300, N = 256, the same protocol as §5.2 (the U and E_lam
  formulas use the new `k_f`). No predictions registered. Report only.
* *Optional:* Re = 5000 at N = 512, for slope (a) with `k ∈ [8, 32]` (P7). Only if L1000 took
  under 2 h.
* *Optional:* jax-cfd cross-check in **its own venv**, time-boxed to 2 h: the same Re = 1000
  configuration, comparing `⟨E⟩`, `⟨Z⟩` and slope (a). If it does not run inside the box, drop it
  and say so.

---

## 6. Pre-registered predictions

These were written before any code exists. Report each as **HIT** or **MISS** with the measured
value. A MISS is information, not a failure to hide. Do not rerun anything to turn a MISS into a
HIT.

| # | Quantity | Prediction |
|---|---|---|
| P1 | `|R_E| / ∫P`, each long run, [50, 250] | < 1e-4 (the plan's bar is 1e-2) |
| P2 | `|R_Z| / ∫Q`, each long run | < 1e-3 |
| P3 | temporal orders (§5.3.1) | all in [3.5, 4.5] |
| P4 | `E(K) / max E(k)`, each long run | < 1e-6 |
| P5 | top-20% shell slope (b), each long run | < −3 |
| P6 | Re = 1000 slope (a), `k ∈ [8, 16]` | in [−6, −2.8]; point guess −4.3 |
| P7 | Re = 5000 slope, `k ∈ [8, 32]` (optional run) | in [−4.5, −3] |
| P8 | `k_d` at Re = 1000; scaling with Re | in [15, 50]; fitted exponent of `k_d` vs Re (3 points) in [0.4, 0.6] |
| P9 | `⟨E⟩` vs laminar | `⟨E⟩ < E_lam` at every Re (3.698 / 10.63 / 18.58) |
| P10 | `2ν⟨Z⟩ / ⟨P⟩` at Re = 1000 | in [0.02, 0.4] |
| P11 | resolution map | resolved: Re100 at 64, Re300 at 128, Re1000 at 256; NOT resolved at half those |
| P12 | spin-up time, each long run | ≤ 50 |
| P13 | τ_e | τ_e(1000) in [0.5, 5]; τ_e(100) > τ_e(1000) |
| P14 | Lyapunov λ | λ(1000) in [0.2, 3]; λ(100) < λ(300) < λ(1000); λ(100) > 0 |
| P15 | f32 vs f64 (L1000) | ⟨E⟩, ⟨Z⟩, slope within 2 SE; `|R_E|/∫P` < 1e-3; f32 1.3–2.5× faster |
| P16 | f64 steps/s | 64: 1e3–6e3 · 128: 250–1600 · 256: 60–400 · 512: 15–90 |

---

## 7. Suggested layout (layout is yours to choose; the rules in §0 are not)

    src/kolmogorov/
        grid.py         wavenumbers, mask, Parseval weights, up/down
        ops.py          ψ, u, v, derivatives, N̂, pressure
        solver.py       Lawson IF-RK4 step, dt/m selection, chunked scan driver, checkpoints
        diagnostics.py  E, Z, Π, P, Q, shell spectrum, budget accumulators
    tests/
        conftest.py     adds src/ to sys.path, enables x64
        test_stage1_gates.py
    experiments/stage1/
        run_long.py, run_order.py, run_resolution.py, run_lyapunov.py, run_throughput.py
        analyse.py      → results/*.json, figs/*.png
        results/        small JSONs (committed)
        figs/           small PNGs (committed)
        REPORT.md

Each script takes its parameters on the command line and writes one JSON. The report quotes the
exact command lines.

---

## 8. Report — `experiments/stage1/REPORT.md`

1. **Gates:** G1–G11 (and G5c, G6c) with the measured value next to each tolerance, plus the
   `pytest` command and summary line.
2. **Predictions:** P1–P16 table: prediction · measured · HIT/MISS.
3. **Per-run table:** name, Re, N, dt, m, max CFL, wall time, ⟨E⟩±SE, ⟨Z⟩±SE, ⟨P⟩, residuals,
   slopes, `E(K)/maxE`, `k_d`, spin-up, τ_e, λ.
4. **Resolution table:** Re × N → relative difference at t = 1 and t = 2, resolved yes/no.
5. **Figures:** time series of E, Z, P, CFL per run; time-mean spectra (log-log, with the fit
   ranges and a k^−3 reference line); resolution-map spectra; Lyapunov `ln d(t)` with fit windows;
   one 64² vorticity snapshot per Re.
6. **Bugs fixed:** each bug, the gate or run that caught it, and the fix.
7. **Deviations from this spec:** every one, with the reason. "None" is a valid answer.
8. **Spec concerns:** anything in §§2–6 that you believe is wrong. Do not act on it; state it.
9. **Wall time:** total, and per task.
10. `pip freeze` and the machine state (idle or not, on power or battery).

Then stop.
