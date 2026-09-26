# UGP-II — Aurora-from-Scratch in JAX · Decision Record

**Owner:** Aryamann Srivastava, 4th year, Dept. of Aerospace Engineering, IIT Kanpur
**Advisors:** Prof. Rajesh Ranjan (ACAL) · Prashant Kumar (PhD)
**Project:** `UGP-II`. This file is the decision record for the NEW project.
**Predecessor:** `~/Desktop/sem7/UGP/project_memory.md` (12,797 lines, sections 1–188).
Do NOT append UGP-II work to that file; it is closed. Read it only for the carry-over rules in §2.

---

## 0. Status of UGP-I (the competition) — CLOSED

NeurIPS 2026 RealPDE Competition, Track 1 "Sim2Real", Codabench #17363, team `aryamannsr`.
Development phase closed **27 Sep 2026, 24:00 UTC**.

    FINAL BANKED SCORE   80.081266   (submission_STACK.zip)
    rel_l2 93.976748 · tke 78.622695 · mvpe 93.244289 · time 90.771403 · sps 39.557734
    Leader (roysegal) 82.10103 · top-10 cutoff 81.76452 · top-50 cutoff 80.43545 · 181 teams

**What was achieved:** ~77 -> 80.08 over roughly seven weeks. The scoring weights, never published,
were recovered by least squares on 40 leaderboard rows to within 0.005:
`final = 0.46743*rel_l2 + 0.10027*tke + 0.09420*mvpe + 0.09689*time + 0.24737*sps + 0.9119`.

**What was proven impossible, with measurements:** top 10 is unreachable by better uncertainty
alone — even a mathematically perfect interval model scores 81.727, below the cutoff. Closing it
requires a better underlying model, and the model is converged.

**Directions closed with numbers** (each has a section in the old record):
POD/SVD truncation · spectral and temporal denoising · denoised inputs · denoised targets (on both a
held-out and the production recipe) · divergence-free projection · spectral losses (BSP, AMSE) ·
momentum/vorticity residual loss · U-Net and MWT from scratch · joint sim+real training · interval
width calibration · interval asymmetry/HDR · error-predictor ranking · position-stratified LUTs ·
batch size · model-load path · CUDA graphs · asset decompression.

---

## 1. Why UGP-II exists — the question UGP-I could not answer

UGP-I ended on a specific, evidenced claim: **the remaining gap is a better PRETRAINED backbone.**
Evidence: the supplied FNO, which never saw a real frame, beat every architecture trainable from
scratch on 81 trajectories (U-Net −16%, MWT −60%); six weeks of fine-tuning moved the residual only
3.7%; and the RealPDEBench paper's own best performer on the Foil dataset is DPOT-L-FT, a large
pretrained foundation model.

That claim was **untestable inside the competition**: outside pretrained weights were banned
(disqualification), only 81 real trajectories existed, and — decisively — the offline validation had
**no predictive power for the main accuracy channel** (fitted local→live slope −3.2, sec186).

Aurora (Bodnar et al., *Nature* 2025, [arXiv 2405.13063](https://arxiv.org/html/2405.13063)) reports
that **pretraining beats from-scratch by 25%**, but cannot say WHICH of its eight heterogeneous
datasets earned that, because the ablation costs 32×A100 for 150k steps.

**On Kolmogorov flow, that question is answerable.** You generate the data, so volume, diversity,
resolution and distribution shift are independent knobs. Prashant proposed exactly this on
26 Sep 2026: reimplement the Aurora architecture from primitives in JAX, decompose into sub-parts,
and "check which data is giving how much accuracy" on simple data like Kolmogorov flow or 2-D
turbulence.

The full plan is in **`PROJECT_PLAN.md`** beside this file: five research questions, the testbed, the
scaled-down architecture, eleven staged deliverables each with an acceptance test, and five
experiments.

---

## 2. ★★★ CARRY-OVER RULES — these were paid for in UGP-I, do not relearn them

**2.1 Any in-sample fit is inflated until scored on DISJOINT data.**
The most expensive lesson of the project. A position-stratified interval table measured **+0.041
in-sample** and **−0.059 out-of-sample**, degrading monotonically as strata were added — pure
overfitting. The old record's rule of thumb is that in-sample free-fit over-predicts by ~2×.

**2.2 Hold out by REGIME, never by sample.**
Frames inside one trajectory are strongly correlated; a random split measures memorisation. In
UGP-II this means holding out whole flow regimes (a Reynolds number, a forcing wavenumber).

**2.3 Pre-register every prediction in writing BEFORE the run.**
With one measurement per day, a failed submission is only wasted if there was no prediction to
reconcile against. This turned the 22 Sep failure into the measurement that solved a calibration
constant exactly (one equation, one unknown). Keep the habit even when measurements are cheap.

**2.4 Know which of your metrics actually transfer.**
UGP-I's calibrated proxy (sec186) measured local→live transfer per channel:
`tke` ×1.3389 and `sps` ×0.6237 were readable; **`rel_l2` was not predictable at all** (slope −3.2),
and `time` was later found to be noise-dominated (0.66 s spread across submissions with identical
compute). Six weeks of accuracy work were steered by an instrument with no resolution.
**Before trusting any validation metric in UGP-II, measure its correlation with the thing you care
about.** On noise-free synthetic data this should be far better — verify, do not assume.

**2.5 A forecast record was kept, and it over-predicted by 0.671.**
Apply a similar haircut to optimistic projections until UGP-II has its own calibration.

**2.6 Physics constraints can be WRONG for measured data.**
Divergence-free projection lost 9 points live. Reason: PIV is a 2-D slice of a 3-D flow, so
`du/dx+dv/dy = -dw/dz != 0`, and 45% of real fluctuation energy is non-solenoidal. Measured mean
|div|: truth 0.00677 vs our prediction 0.00599 — the forecast was ALREADY more divergence-free than
the data. Likewise the 2-D vorticity-transport equation explains only **4–12%** of the measured
`d(omega)/dt` (R² 0.044–0.115), so a momentum loss would penalise a residual the measurement itself
violates by ~95%.
**On Kolmogorov flow this reverses** — the data IS 2-D and noise-free, so physics losses should
behave as the literature expects. That contrast is itself worth a paragraph in the write-up.

---

## 3. Environment

**GPU box** (shared lab machine, SSH host `vm`, 4× RTX PRO 6000 Blackwell, 97 GB each):
* ⛔ Never reboot. Never kill another user's process. Kill by explicit PID only — never
  `pkill -f <pattern>` where the pattern also matches your own ssh command.
* ⛔ Do not launch at ≥90 °C or with SW thermal slowdown active. The box runs hot; idle cards have
  been observed at 88–93 °C. Check `nvidia-smi` before every launch.
* Set `CUDA_VISIBLE_DEVICES` explicitly on every command.
* ⛔ Never scratch to `/tmp` or the VM root partition. Work under the project directory.
* Timings are worthless on a contended card — check utilisation, and use best-of-N.

**Repo.** UGP-II: `ARYAMANN7279/aurora-from-scratch-jax`.
(`ARYAMANN7279/RealPDE-Track1` is the closed UGP-I competition repo, last push 27 Aug 2026.)

**Where things run.** Specs, solver development and CPU runs happen on a CPU machine (the Mac or a
cloud container; Stage 1 ran on the latter). GPU runs go on the lab VM. Results come back as small result JSONs committed under `experiments/`. Raw data stays off git.

---

## 4. Working agreement

* Give the number, not the narrative. Prefer 2-line answers.
* Report failures plainly and immediately. "This is closed, here is the measurement" is better than
  handing over something unverified.
* When something that measurement says will not work is requested, say so once with the number, then
  do what is decided.
* **Split of work:** specs, pre-registered predictions, and result review are done before coding.
  Stage specs live in `handouts/stageNN_*.md`. Implementation commits locally; the spec author
  reviews the report before a stage is accepted.

---

## 5. Log

### 2026-09-26 — project defined
Prashant proposed the Aurora-in-JAX project. Plan drafted (`PROJECT_PLAN.md`): 5 research questions,
11 staged deliverables with acceptance tests, 5 experiments. Agreed to start after the Track 1
development phase closes on 27 Sep.

**First action when UGP-II opens:** Stage 1 only — the pseudo-spectral Kolmogorov solver. Do not
proceed past it until the energy spectrum shows the expected inertial-range scaling and the energy
budget balances to <1%. If the data generator is wrong, every measurement downstream is meaningless.

### 2026-09-26 — Stage 1 designed; spec written
`handouts/stage01_solver.md` is the Stage 1 spec. Fixed design: vorticity form, A=1, α=0.1,
k_f=4, ν=1/Re (Re1000 = Kochkov); rfft2 norm="forward"; strict square 2/3 mask; Lawson IF-RK4 with
forcing in the explicit part; dt=0.1/m from a laminar-speed CFL bound, run void if logged CFL>0.7;
f64. There are 11 pytest gates (G1–G11), three long runs (Re100@128, Re300@256, Re1000@256,
T=250, average over 50–250), supporting runs (temporal order, resolution map, Lyapunov, f32,
throughput) and predictions P1–P16, all pre-registered in the spec.
Changes from the plan's wording:
* The spectral acceptance test is the forward-cascade slope for k>k_f (P6: Re1000, k∈[8,16], in
  [−6,−2.8]). A −5/3 fit over k=1–3 would rest on 3 shells, which is not meaningful.
* The budget is accumulated per step, with P1 predicting <1e-4 against the plan's 1e-2 bar.
Environment: Mac M2, 8 GB, CPU only, Python 3.11 venv (3.14 may lack JAX wheels).

### 2026-09-26 — Stage 1 DONE: solver built, gates pass, acceptance met
Report: `experiments/stage1/REPORT.md` (every number there is traced to a result JSON + command).
* **Acceptance (plan §5):** energy budget |R_E|/∫P ≤ 1.7e-9 on all three long runs (bar 1e-2);
  Re1000 slope over k∈[8,16] = −4.380 ± 0.025 (band [−6,−2.8], guess −4.3). **Stage 1 accepted.**
* Gates G1–G11: all pass, first attempt. Temporal order 4.00 (G9 and all long runs).
* Predictions: 13 HIT, 2 MISS (P11, P15), P7 not run (optional Re5000, ~5 h here).
* Long runs: ⟨E⟩ 0.670 / 0.882 / 1.015 at Re 100/300/1000 (far below E_lam, i.e. turbulent);
  λ 0.39 / 0.47 / 0.57; τ_e 1.38 / 1.01 / 0.82; k_d 12 / 19.5 / 33.7 (∝Re^0.45).
* **P11 MISS, a real finding:** the resolution metric ‖ω_N−ω_2N‖/‖ω_2N‖ bottoms out at the
  vorticity-norm tail share beyond N/3, which is ~1% at Re100@64, Re300@128, Re1000@256, and drops
  ~8–9× per doubling. Not a solver defect, but it bounds data fidelity.
* **P15 MISS:** f32 ⟨E⟩ differs from f64 by 3.2 SE of one run (2.0 combined SE). The two runs are
  independent chaotic samples, so this is sampling, not precision. f32 is only 1.34× faster here.
  Keep f64 for data generation.
* Ran on a 4-core x86 CPU container (not the M2). f64 steps/s: 3206 / 576 / 144 / 34 at
  N = 64 / 128 / 256 / 512. Re1000@256 for 250 time units takes ~40 min on one core.

### 2026-09-26 — Design decisions (closes all four open issues)
0. **DECIDED — generation grid (Stage 2 M1, measured).** The fidelity bar is now set in the saved
   output band: F_out = ‖B(ω_N) − B(ω_2N)‖/‖B(ω_2N)‖ ≤ 1e-3 at t=1, with B the truncation to
   |k| < N_out/2. For the 128² band: **N_gen = 128 / 256 / 512 for Re 100 / 300 / 1000** (k_f=4).
   Re1000 at 256 fails even the 64² band (1.5e-3); at 512 it gives 1.6e-5. Other regimes need
   their own measurement before generation.
1. **DECIDED — levels = spectral bands.** A "level" is a square-shell octave band of the field
   (‖k‖∞ ∈ [0,2), [2,4), [4,8), …). The bands sum exactly to the field. A 64² field has 5 bands and
   a 128² field has 6, with the first 5 identical, so level-count heterogeneity follows from
   resolution heterogeneity. That ties the Perceiver (Stage 4) to RQ2, and E4's L ablation
   compresses scales.
2. **DECIDED — E1 holds out an interior regime.** E1 measures what diversity buys for
   interpolation. Extrapolation beyond the pretraining range is E3's job only.
3. **DECIDED — pure JAX + Optax.** Parameters are plain pytrees and layers are functions. No Flax.
   Optax provides the optimisers only.

### 2026-09-26 — Stages 2–9: specs, code and gates (results in progress)
Every stage follows the same pattern: a spec with pre-registered predictions is committed before
any code (`handouts/stageNN_*.md`), then the gates are written as pytest, then the measurements are
run. Where a pre-registered bar turned out to be ill-posed, it was **not** silently changed. Each
case has a dated amendment in the spec, and the original FAIL is kept in the report.
* **Stage 2 (data):** `src/kdata`, gates D1–D8 pass. The acceptance dataset `acc_v1` has 3
  regimes × 8 seeds × 501 frames at 64² and 128². It is still generating on CPU: the Re1000
  trajectories at 512² take ~3 h each on one core.
* **Stage 3 (levels + embedding):** accepted. Float64 round trip 1.9e-14. The float32 bar missed
  (1.7e-5 vs 1e-5; a conditioning artefact of the random test matrix, amendment A1). The learned
  autoencoders missed their step budgets. **Lesson:** plain Adam at 1e-3 is slow here; budget more
  steps.
* **Stage 4 (Perceiver):** gates K1–K8 pass. The S1–S6 learning measurements are too heavy for
  the CPU and wait on the GPU box (`experiments/stage4/GPU_RUNBOOK.md`).
* **Stage 5 (Swin3D block):** accepted. It matches Aurora's own `Swin3DTransformerBlock`
  (microsoft-aurora 2.0.1) to 2.8e-7 (f32) and 5e-16 (f64). The shifted-window mask is generalised
  to periodic H and W (Aurora's is periodic in W only).
* **Stage 6 (U-Net backbone):** gates pass. It matches Aurora's `Swin3DTransformerBackbone` to
  3e-16 (f64) and 1.9e-7 (f32). Gradient-check amendment A1: the finite-difference roundoff floor.
* **Stage 7 (full model):** gates pass. It is persistence at initialisation (zero heads) and has
  5.1M parameters at D = 64. **Finding:** the model is deliberately *not* translation-equivariant,
  because it carries an absolute position encoding. That is correct, since the Kolmogorov forcing
  breaks y symmetry (amendment A1).
* **Stage 8 (rollout + MAE):** gates pass. **Stage 9 (pushforward):** spec and code are written,
  and the matched-compute runs are queued.
* **Compute lesson:** the 4-core CPU runs Stage 3–8 unit work fine, but anything that trains at
  128² or above, or generates at 512², belongs on the GPU box.
