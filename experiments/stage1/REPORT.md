# Stage 1 Report: Pseudo-spectral Kolmogorov solver

Spec: `handouts/stage01_solver.md`. Code: `src/kolmogorov/`. Tests: `tests/test_stage1_gates.py`.
Every number below is copied from a file in `experiments/stage1/results/`, produced by the command
listed in §10. All commands run from the repo root with the venv at `.venv/`.

## Verdict

**Stage 1 meets the plan's acceptance test.**

| Criterion (PROJECT_PLAN.md §5) | Required | Measured |
|---|---|---|
| Energy budget balance, [50, 250] | < 1% | ≤ 1.7e-9 relative on all three long runs |
| Inertial-range scaling (spec: forward-cascade slope at Re1000, k ∈ [8, 16]) | in [−6, −2.8] | −4.380 ± 0.025 |

All 11 gates pass (13 tests including the two controls), on the first attempt. Of the 16
predictions, 13 HIT, 2 MISS (P11 resolution map, P15 float32 agreement), and 1 was not run (P7,
optional). Neither MISS is a solver defect. Both are explained in §7 and §8. **P11 matters for
Stage 2:** by the spec's own resolution metric, Re1000 at N=256 is converged to about 1% in
‖ω‖, not 0.1%.

---

## 1. Gates (float64)

Command: `.venv/bin/python -m pytest tests -q -s` → `13 passed in 5.32s`
(full output: `results/gates.txt`)

| Gate | Measured | Tolerance |
|---|---|---|
| G1 FFT round trip, N=64 | 1.33e-15 | ≤ 1e-14 |
| G2 div u | 1.19e-16 | ≤ 1e-12 |
| G2 curl u − ω | 4.02e-16 | ≤ 1e-12 |
| G3 Parseval ⟨ω²⟩ (rel) | 7.77e-16 | ≤ 1e-12 |
| G3 Parseval E vs physical ½⟨u²+v²⟩ (rel) | 4.27e-16 | ≤ 1e-12 |
| G3 shell sum Σ E(k) vs E (rel) | 5.70e-16 | ≤ 1e-12 |
| G4 N̂ vs sympy, N=32 | 1.05e-13 | ≤ 1e-12 |
| G5 masked product vs 2N-grid exact product | 1.89e-15 | ≤ 1e-12 |
| G5c control, non-band-limited, unmasked | 3.77e+00 | > 1e-3 |
| G6 ⟨ψN⟩ normalised | 3.72e-18 | ≤ 1e-12 |
| G6 ⟨ωN⟩ normalised | 1.52e-17 | ≤ 1e-12 |
| G6c control ⟨ψN⟩ / ⟨ωN⟩, unmasked | 7.49e-04 / 1.71e-02 | > 1e-8 |
| G7 Taylor–Green pressure | 4.44e-16 | ≤ 1e-12 |
| G8 single-shell (|k|²=25) free decay (rel) | 1.65e-13 | ≤ 1e-11 |
| G9 laminar spin-up error at dt=0.1 (rel) | 2.28e-08 | ≤ 1e-6 |
| G9 orders, dt 0.1 → 0.0125 | 3.9997, 3.9999, 3.9984 | each in [3.8, 4.2] |
| G10 down(up(ω)) round trip | 0.0 | ≤ 1e-13 |
| G10 up(analytic) vs analytic on 2N grid | 3.07e-15 | ≤ 1e-13 |
| G11 max coefficient outside the mask after 100 steps | 0.0 (exact) | == 0 |

## 2. Predictions

Produced by `summarise.py` → `results/predictions.json`.

| # | Quantity | Prediction | Measured | Verdict |
|---|---|---|---|---|
| P1 | \|R_E\| / ∫P over [50,250] | < 1e-4 each | Re100: 1.72e-09, Re300: 1.90e-10, Re1000: 5.44e-10 | **HIT** |
| P2 | \|R_Z\| / ∫Q over [50,250] | < 1e-3 each | Re100: 3.15e-11, Re300: 2.63e-10, Re1000: 1.11e-09 | **HIT** |
| P3 | temporal orders | all in [3.5, 4.5] | L100: 4.004, 4.002, 3.999; L300: 4.000, 3.997, 3.522; L1000: 3.999, 4.001, 4.015 | **HIT** |
| P4 | E(K)/max E(k) | < 1e-6 each | Re100: 5.82e-09, Re300: 1.35e-10, Re1000: 6.28e-08 | **HIT** |
| P5 | top-20% shell slope | < −3 each | Re100: −13.6, Re300: −14.7, Re1000: −7.43 | **HIT** |
| P6 | Re1000 slope, k ∈ [8,16] | in [−6, −2.8], guess −4.3 | −4.380 ± 0.025 (SE) | **HIT** |
| P7 | Re5000 slope, k ∈ [8,32] | in [−4.5, −3] | optional run not done | **NOT RUN** |
| P8 | k_d(Re1000); exponent of k_d vs Re | in [15, 50]; exponent in [0.4, 0.6] | k_d 12.0 / 19.5 / 33.7; exponent 0.449 | **HIT** |
| P9 | ⟨E⟩ < E_lam | each Re | 0.670 vs 3.70; 0.882 vs 10.6; 1.02 vs 18.6 | **HIT** |
| P10 | 2ν⟨Z⟩/⟨P⟩ at Re1000 | in [0.02, 0.4] | 0.0943 | **HIT** |
| P11 | resolution map, rel diff at t=1, bar 1e-3 | resolved at Re100@64, Re300@128, Re1000@256; not at half | Re100: N32 1.54e-1, N64 1.69e-2; Re300: N64 1.06e-1, N128 1.20e-2; Re1000: N128 9.79e-2, N256 1.15e-2 | **MISS** |
| P12 | spin-up time | ≤ 50 each | 0.3 / 0.1 / 0.1 (see §8.3) | **HIT** |
| P13 | τ_e | τ_e(1000) in [0.5, 5]; τ_e(100) > τ_e(1000) | 1.38 / 1.01 / 0.824 | **HIT** |
| P14 | Lyapunov λ | λ(1000) in [0.2, 3]; λ100 < λ300 < λ1000; λ100 > 0 | 0.394 / 0.472 / 0.565 | **HIT** |
| P15 | f32 vs f64 (L1000) | ⟨E⟩, ⟨Z⟩, slope within 2 SE; \|R_E\|/∫P < 1e-3; f32 1.3–2.5× faster | \|Δ⟨E⟩\| 3.17 SE, \|Δ⟨Z⟩\| 2.22 SE, \|Δslope\| 1.17 SE; R_E 3.49e-05; speed-up 1.34× | **MISS** |
| P16 | f64 steps/s | 64: 1e3–6e3 · 128: 250–1600 · 256: 60–400 · 512: 15–90 | 3206 / 576 / 144 / 34 | **HIT** |

## 3. Per-run table

Window [50, 250]. SE from 10 batch means of 20 time units each. Wall times were measured with four
runs sharing four cores (see §7), so they are not throughput numbers; §2 P16 gives those.

| Run | Re | k_f | N | m | dt | max CFL | wall (s) | ⟨E⟩ ± SE | ⟨Z⟩ ± SE | ⟨P⟩ | \|R_E\|/∫P | \|R_Z\|/∫Q | slope [8,16] | slope top-20% | E(K)/maxE | k_d | 2ν⟨Z⟩/⟨P⟩ | τ_e |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L100 | 100 | 4 | 128 | 24 | 4.17e-3 | 0.486 | 136 | 0.670 ± 0.016 | 6.81 ± 0.18 | 0.270 | 1.7e-9 | 3.2e-11 | −6.53 | −13.6 | 5.8e-9 | 12.0 | 0.505 | 1.38 |
| L300 | 300 | 4 | 256 | 80 | 1.25e-3 | 0.360 | 1938 | 0.882 ± 0.018 | 8.68 ± 0.21 | 0.233 | 1.9e-10 | 2.6e-10 | −5.06 | −14.7 | 1.4e-10 | 19.5 | 0.248 | 1.01 |
| L1000 | 1000 | 4 | 256 | 106 | 9.43e-4 | 0.280 | 2544 | 1.015 ± 0.013 | 10.59 ± 0.22 | 0.224 | 5.4e-10 | 1.1e-9 | −4.38 | −7.43 | 6.3e-8 | 33.7 | 0.094 | 0.824 |
| L1000_f32 | 1000 | 4 | 256 | 106 | 9.43e-4 | 0.279 | 1665 | 0.974 ± 0.016 | 10.09 ± 0.24 | 0.213 | 3.5e-5 | 2.6e-6 | −4.41 | −7.46 | 5.7e-8 | 33.5 | 0.095 | 0.844 |
| KF2_300 | 300 | 2 | 256 | 108 | 9.26e-4 | 0.325 | 2457 | 1.49 ± 0.08 | 4.46 | — | 2.4e-10 | 2.5e-10 | −4.98 | −17.2 | 7.8e-13 | 15.0 | — | 2.44 |
| KF8_300 | 300 | 8 | 256 | 40 | 2.50e-3 | 0.442 | 864 | 0.378 ± 0.002 | 13.44 | — | 3.3e-10 | 1.6e-9 | −6.17 | −13.0 | 5.4e-9 | 24.4 | — | 0.908 |

No run came near the CFL void threshold of 0.7. Lyapunov exponents (4 twins, mean [min, max]):
Re100 0.394 [0.385, 0.400], Re300 0.472 [0.423, 0.505], Re1000 0.565 [0.555, 0.575].

## 4. Resolution table

Common IC, same dt, T = 2 (`results/resolution_Re*.json`). "rel diff" is the spec's criterion:
‖up(ω_N) − ω_2N‖/‖ω_2N‖ on the 2N grid. The last two columns are supplementary diagnostics added
in this stage (not pre-registered): the same difference restricted to the modes grid N keeps, and
the share of ‖ω_2N‖ lying outside those modes, which is a lower bound on the criterion.

| Re | N vs 2N | rel diff t=1 | rel diff t=2 | resolved (≤1e-3) | common-mode diff t=1 | tail share t=1 |
|---|---|---|---|---|---|---|
| 100 | 32 vs 64 | 1.54e-1 | 1.96e-1 | no | 1.17e-1 | 1.01e-1 |
| 100 | 64 vs 128 | 1.69e-2 | 1.83e-2 | no | 1.17e-2 | 1.23e-2 |
| 300 | 64 vs 128 | 1.06e-1 | 1.39e-1 | no | 8.36e-2 | 6.48e-2 |
| 300 | 128 vs 256 | 1.20e-2 | 1.56e-2 | no | 8.89e-3 | 8.01e-3 |
| 1000 | 128 vs 256 | 9.79e-2 | 1.09e-1 | no | 8.07e-2 | 5.55e-2 |
| 1000 | 256 vs 512 | 1.15e-2 | 1.21e-2 | no | 8.90e-3 | 7.27e-3 |

Each doubling of N cuts the difference by about 8–9×, which fits spectral convergence on a steep
spectrum. The difference is always about the size of the tail share. The error therefore sits in
the modes near the cutoff, which carry a large share of ‖ω‖. In `figs/resolution.png` the coarse
spectra lie on the fine ones and turn up only in the last few shells before their cutoff.

## 5. Figures (`figs/`)

* `timeseries.png`: E, Z, P and CFL for the three long runs (grey: discarded [0, 50]).
* `spectra.png`: time-mean spectra, fit range k ∈ [8, 16], k^−3 reference.
* `resolution.png`: resolution-map spectra at t = 1.
* `lyapunov.png`: twin separation with the fit band 1e-6 ≤ d ≤ 1e-2.
* `snapshots.png`: 64² vorticity at t = 250 for each Re.

## 6. Bugs fixed

No solver bugs: all gates passed on the first run. Two harness mistakes were caught and fixed:

1. **f32 initial condition.** The first f32 launch built its IC under float32, which does not
   bit-match the f64 IC. It was stopped after ~1 min, deleted, and restarted from the f64 run's
   saved `state_t0.npy` cast to complex64 (new `--ic` option in `run_long.py`).
2. **Throughput file overwrite.** A single-N f64 check overwrote `results/throughput.json`. It was
   re-measured in full on the idle machine. The numbers in P16 come from that re-measurement.

## 7. Deviations from the spec

1. **Machine.** Runs used a 4-core Intel Xeon @ 2.10 GHz Linux x86_64 cloud container, 15 GB RAM,
   CPU only, not the Mac M2. No `caffeinate` (not applicable). Python 3.11 venv as specified.
   P16 throughput is therefore for this machine.
2. **Parallel runs.** L100, L300, L1000 and L1000_f32 ran concurrently, one per core, followed by
   the k_f runs and the supporting measurements. Throughput (P16, and the P15 speed-up) was
   measured separately on the idle machine, best of 3.
3. **Budget normalisers.** R_E and R_Z use per-step trapezoid integrals as specified. The
   normalisers ∫P and ∫Q use the trapezoid rule on the 0.1-spaced saves (they only set the scale).
4. **Resolution map.** Two diagnostic columns were added (§4). The pre-registered criterion and
   verdict are unchanged.
5. **Optional items not run.** Re5000 at N=512 (P7) would need about 5.9e5 steps at 34 steps/s,
   roughly 5 h on this machine, so it was skipped for time. The spec allowed it, since L1000 took
   under 2 h. The jax-cfd cross-check was not attempted.

## 8. Spec concerns

1. **P11: the resolution criterion cannot pass at these grids.** ‖up(ω_N) − ω_2N‖/‖ω_2N‖ is
   bounded below by the share of ‖ω‖ beyond N/3, and in the vorticity norm that share is about 1%
   at every "resolved" grid (§4). A 1e-3 bar would need that share below 1e-3, which means roughly
   twice these grids. Either the bar or the norm should change, e.g. an energy norm, or comparison
   restricted to k ≤ N/6. **Stage 2 needs a decision here:** generate Re1000 at 256 (1% in ‖ω‖
   vs 512) or at 512 (about 4× the cost per unit time). Kochkov et al. generate at higher resolution
   and downsample.
2. **P15: the 2-SE test compares two independent chaotic samples against one run's SE.** f32 and
   f64 separate after a few Lyapunov times, so their time means are independent draws. Against the
   combined SE √(SE₆₄² + SE₃₂²) the differences are ⟨E⟩ 2.02, ⟨Z⟩ 1.50 and slope 0.75, so ⟨E⟩ sits
   at the 2-SE edge. The budget (3.5e-5) and speed-up (1.34×) parts of P15 pass. P15 stays a MISS
   as registered, but it does not show an f32 accuracy problem.
3. **P12: the spin-up definition is not informative here.** The IC has E = 1, already inside the
   [5, 95] percentile band of the statistically steady state. The criterion therefore fires at
   t ≈ 0.1. The 10-unit block means of E (`E_block_means` in the run JSONs) show the real transient:
   Re300 and Re1000 run low (0.65–0.89) for the first 20–40 time units, well inside the discarded 50.
4. **P3: roundoff floor.** L300's third order (3.52) uses e₃ = 2.2e-12, which is near the roundoff
   floor of a 2560-step run. The spec's 1e-12 cutoff for "roundoff-limited" is slightly too low. The
   other eight orders are 3.997–4.015.

## 9. Wall time

Long runs (concurrent, 4 cores): L100 136 s, L300 1938 s, L1000 2544 s, L1000_f32 1665 s,
KF2_300 2457 s, KF8_300 864 s. Supporting runs took minutes each (resolution map at 512: about 3 min).
From the first long-run launch (06:28 UTC) to the final throughput measurement (07:47 UTC) took
1 h 19 min, plus about 3 min for the first throughput pass before that.

## 10. Commands and environment

Packages: `results/pip_freeze.txt` (jax 0.10.2, jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1,
sympy 1.14.0, pytest 9.1.1, matplotlib 3.11.2).

```
.venv/bin/python -m pytest tests -q -s
.venv/bin/python experiments/stage1/run_throughput.py
.venv/bin/python experiments/stage1/run_throughput.py --f32 --n 256
.venv/bin/python experiments/stage1/run_long.py --name L100  --re 100  --n 128
.venv/bin/python experiments/stage1/run_long.py --name L300  --re 300  --n 256
.venv/bin/python experiments/stage1/run_long.py --name L1000 --re 1000 --n 256
.venv/bin/python experiments/stage1/run_long.py --name L1000_f32 --re 1000 --n 256 --f32 --ic data/stage1/L1000/state_t0.npy
.venv/bin/python experiments/stage1/run_long.py --name KF2_300 --re 300 --n 256 --kf 2
.venv/bin/python experiments/stage1/run_long.py --name KF8_300 --re 300 --n 256 --kf 8
.venv/bin/python experiments/stage1/analyse_long.py --name <run>        # for each run above
.venv/bin/python experiments/stage1/run_order.py --name {L100,L300,L1000}
.venv/bin/python experiments/stage1/run_resolution.py --src L100  --grids 32 64 128
.venv/bin/python experiments/stage1/run_resolution.py --src L300  --grids 64 128 256
.venv/bin/python experiments/stage1/run_resolution.py --src L1000 --grids 128 256 512
.venv/bin/python experiments/stage1/run_lyapunov.py --name {L100,L300,L1000}
.venv/bin/python experiments/stage1/summarise.py
.venv/bin/python experiments/stage1/make_figs.py
```

Raw outputs (chunks, states, snapshots) are in `data/stage1/`, which is git-ignored. Each result
JSON records the command that produced it.
