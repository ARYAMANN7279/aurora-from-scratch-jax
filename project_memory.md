# UGP-II — Aurora from Scratch in JAX · Decision Record

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
Evidence: the supplied FNO, pretrained on simulations and then fine-tuned by the organisers on real
data, beat every architecture trainable from scratch on 81 trajectories (U-Net −16%, MWT −60%); six
weeks of fine-tuning moved the residual only 3.7%; and the RealPDEBench paper's own best performer on
the Foil dataset is DPOT-L-FT, a large pretrained foundation model.
*(Corrected 1 Oct 2026: UGP-I took that FNO for one that never saw a real frame. The
simulation-only checkpoint `sim_fno.pth` scores rel_l2 1.23 on the same windows (E6 baselines), so
the strong one was `sim_real_fno.pth`, fine-tuned on real data.)*

That claim was **untestable inside the competition**: outside pretrained weights were banned
(disqualification), only 81 real trajectories existed, and — decisively — the offline validation had
**no predictive power for the main accuracy channel** (fitted local→live slope −3.2, sec186).

Aurora (Bodnar et al., *Nature* 2025, [arXiv 2405.13063](https://arxiv.org/html/2405.13063)) reports
that **pretraining beats from-scratch by 25%**, but cannot say WHICH of its heterogeneous
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
behave as the literature expects.

---

## 3. Environment

**GPU box** (shared lab machine, SSH host `vm`, 4× RTX PRO 6000 Blackwell, 97 GB each):
* ⛔ Never reboot. Never kill another user's process. Kill by explicit PID only — never
  `pkill -f <pattern>` where the pattern also matches your own ssh command.
* ⛔ **Idle cards only** (the professor's rule, 29 Sep: an overheated card cannot be restarted
  without sudo). Start a job only on a card with nobody else's process on it, below 85 °C and
  without thermal slowdown. Card 0 stays excluded. The box runs hot; cards have been observed at
  88–93 °C. Check `nvidia-smi` before every launch.
* ⛔ **Launch GPU jobs only through `glane3.sh`** (VM scratch area). The account's `~/.bashrc`
  exports `CUDA_VISIBLE_DEVICES=0`, and a `bash -c` started under ssh re-reads it, so setting the
  variable outside the command is not enough (incident, 29 Sep). `glane3.sh` starts jobs in a bash
  that skips `~/.bashrc`, sets the card inside the command, checks where the job really runs, and
  pauses it on heat or when anyone else's process appears on its card.
* ⛔ **Heavy GPU jobs do not fit these limits; run them on the CPU.** Stage 4 at 512² took an idle
  card from below 85 °C to 92 °C in 21 s (1 Oct). Pausing does not help: a paused process keeps its
  CUDA context, so the card stays at full power (133 W at 0% use, 84 °C) and never reaches the
  80 °C resume bar. Killed, the card dropped to 49 W at once.
* ⛔ Never scratch to `/tmp` or the VM root partition. Work under the project directory.
* Timings are worthless on a contended card — check utilisation, and use best-of-N.

**VM facts (measured 28 Sep 2026):**
* **Clone and access:** `/SML_DISK_24TB/rajeshr/Aryamann/aurora-from-scratch-jax`, branch
  `vm-results`. The shared `rajeshr` account holds no GitHub credentials, so every push goes
  through the Mac.
* **Software:**
  * Python 3.11 from `Aryamann/env`; JAX 0.10.2; driver 570, CUDA 12.8;
  * a persistent JAX compile cache (the first run in a process otherwise compiles for about
    8 min);
  * scratch (TMPDIR, pip, matplotlib, XDG) under `Aryamann/ugp2_scratch`;
  * `LD_LIBRARY_PATH` unset for our jobs (the account's OpenFOAM / PETSc / anaconda libraries sit
    on it);
  * `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  * a separate PyTorch environment for comparisons (1 Oct), `ugp2_scratch/torchenv`: torch
    2.11.0+cu128 (the newest build for the driver's CUDA 12.8), torchvision 0.26.0+cu128,
    microsoft-aurora 2.0.1, timm 1.0.30. Point `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
    the scratch area.
* **Cards:** all 4 are shared with other users' jobs, at 74–93 °C. Our lanes take a card only when
  nobody else is computing on it and it is below 85 °C. They SIGSTOP our own process group at
  ≥ 87 °C or on thermal slowdown, and resume at ≤ 80 °C. Failed jobs get 3 retries.
* **CPU:** Xeon w3-2525, 8 cores / 16 threads, typically loaded ~6. Single-core generation takes
  787 s per E1 trajectory at N_gen 256.
* **Precision:**
  * On the GPU, float32 matrix multiplies default to TF32. At that default, 3 float32 gates fail
    (Stage 3 H7, Stage 5 W3/W4, Stage 6 U2/U3), and they pass with
    `JAX_DEFAULT_MATMUL_PRECISION=highest`.
  * **Every GPU run uses `highest`**, which costs 1.07× (128.8 s vs 120.5 s per 1000 steps, both
    measured on a shared card before the 29 Sep fix).
* **Determinism:**
  * Gate D9 fails on the GPU: the frames are bitwise identical, `diag.npz` is not.
  * Likely cause: the scatter-add in `shell_spectrum`, which uses atomics on the GPU (Stage 2
    report, addendum).

**Repo.** UGP-II: `ARYAMANN7279/aurora-from-scratch-jax`.
(`ARYAMANN7279/RealPDE-Track1` is the closed UGP-I competition repo, last push 27 Aug 2026.)

**Where things run.** Specs and analysis happen on the Mac; CPU and GPU runs go on the lab VM. The
cloud container used for the earlier CPU runs is gone (30 Sep), and so is everything it held
outside git, including the Stage 7 parameters. Results come back as small result JSONs committed
under `experiments/`. Raw data stays off git.

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

### 2026-09-26 — Stage 7 and 8 results; restart lesson
* **Stage 7 accepted.** On unseen trajectories, one-step error is 0.070 / 0.146 at Re100 / Re300,
  i.e. 0.32× / 0.46× persistence (N3 < 0.5 HIT). N4 (< 0.1 absolute) missed at Re300.
  5,106,688 parameters, 3000 steps, 1.0 s/step on the CPU.
* **Stage 8: MSE vs MAE is the first real finding.** Both losses give the same one-step error
  (MAE/MSE = 0.96 / 1.01). But over 50-step rollouts the **MSE model stays physical in all 8**
  (E 0.54–1.34× climatology), while the **MAE model blows up in 3 of 4 Re300 rollouts** after step
  25–38 (E up to 19×). One seed only: a lead for E4 (MAE vs MSE, 3 seeds) and for Stage 9 (does
  pushforward fix it?). Aurora trains with MAE, so this matters.
* **Compute lesson:** this cloud container restarts whenever it goes idle, and every background job
  dies. Everything long-running therefore
  checkpoints and resumes bitwise (gate D9; training checkpoints every 250 steps), and is relaunched
  with `data/stage2/lanes.sh`. Long jobs (Re1000 generation at 512², Stage 4 training) belong on the
  GPU box.

### 2026-09-26 — Stage 6, 9, 11 results
* **Stage 6 accepted on the plan's test.** A 16k-step diagnostic overfits one trajectory to 0.070×
  persistence. The pre-registered 4000-step budget missed (0.19×); the cause was the budget, not
  capacity.
* **Stage 9: acceptance NOT met.** At matched compute, pushforward did not reduce rollout drift
  (−13.7% / −5.6%, i.e. worse). Exploratory: at matched steps (2× compute) it cuts drift by 14% /
  11%, winning 4/4 rollouts per regime at ~6–9 SE, with almost no one-step cost. **E4 must report
  both matched-steps and matched-compute comparisons.**
* **Stage 11 accepted.** The harness reproduces the Stage 7 result exactly (difference 0) from a
  config file. This is the tool for E1–E5.

### 2026-09-26 — Stage 10 result
* **Stage 10: acceptance NOT met.** LoRA r=4 (0.84% trainable) improves Re1000 one-step error by
  only 0.5% over zero-shot; full fine-tuning by 3.0%; LoRA at 10× lr by 0.7%. With two
  trajectories and 1000 steps there is little to adapt with. Zero-shot transfer from Re 100/300 to
  the unseen Re1000 is already 0.58× persistence. This moves to E5 (fine-tuning set size sweep,
  plus an in-distribution Re1000 ceiling).

### 2026-09-27 — Stage 2 result
* `acc_v1` complete: 3 regimes × 8 seeds × 501 frames (64², 128²), all checksums verified. Spectra
  stable across seeds (Q4–Q8 HIT; Re100 right at the bars). Loader 29k / 8.9k windows/s.
* **Q3 MISS: a Re1000 trajectory did not regenerate bitwise** in a fresh process (Re 100 / 300 did).
  The solver is deterministic across processes, threads and concurrent load. The result depends on
  the CPU ISA XLA compiles for (SSE4.2 ≠ AVX2 = default). The exact cause for that trajectory is
  unproven; the leading suspect is VM pause/restore across hosts. **For the experiment datasets:
  store the post-spin-up state with every trajectory and record the host CPU / XLA ISA in meta.**

### 2026-09-27 — E1 designed (draft); tooling built; N_gen measured
* Spec `handouts/e1_volume_diversity.md` (**DRAFT, awaiting advisor review**; frozen on approval).
  * Target: (300, 4), interior.
  * Nested regime sets 2 ⊂ 8 ⊂ 32.
  * Cells: T ∈ {32, 64, 128} × n ∈ {2, 8, 32}.
  * Scale: 326 trajectories at 64² and 57 runs. The metric is adv = 1 − e_ft / e_scratch.
  * Open points: regime grid, budgets (fixed vs scaled with T), distance confound.
* Tooling in `experiments/e1/`:
  * the design module;
  * N_gen measurement;
  * the config writer (configs in `experiments/e1/configs/`);
  * `analyse.py`, which builds the adv table and scores E1-P1 to E1-P6 from `results.jsonl`.
* Other changes:
  * `kdata` supports per-regime seed lists.
  * The harness has `data.norm_from_init`: a fine-tune keeps its pretrain normalisation.
  * `run.py` handles lists, `--skip-done` and sharding.
  * 78 tests.
* **N_gen measured for all 33 E1 regimes:** 10 need 128 (Re ≤ 200 at k_f ≤ 5, and Re 125 at
  k_f 6), and the other 23 need 256; none needs 512. The target (300, 4) is at 256 (F = 9.3e-7),
  which agrees with Stage 2. The spec's cost estimate stands.

### 2026-09-27 — E2–E5 specs drafted; E3–E5 configs; E1 runbook
* **Draft specs** (awaiting review): `handouts/e2_heterogeneous_fusion.md`, `e3_ood_transfer.md`,
  `e4_ablations.md` and `e5_lora_vs_full.md`, each with pre-registered predictions anchored on the
  Stage 8/9/10 results.
* **E3–E5 configs:** `experiments/designs.py` → `experiments/make_configs.py` →
  `experiments/e{3,4,5}/configs/`. Runs: E3 33, E4 27, E5 39. Tested in `tests/test_designs.py`.
  * E3 reuses E1's data.
  * E4 and E5 use `acc_v1`.
* **E4 switch:** `backbone.Config.shift` (default True, so earlier results are unchanged). False
  gives unshifted windows in every block. Gate in `tests/test_stage6_gates.py`.
* **E2** needs new code: per-source formats in the harness, a masked loss, and the pad-and-mask
  and per-dataset-encoder baselines. It will be built after the design is reviewed.
* **E1 runbook** for the VM: `experiments/e1/GPU_RUNBOOK.md`. Each lane writes its own rows file.
* **Remote compute:** this cloud container cannot SSH (no client; port 22 blocked), so the VM is
  unreachable from here. Kaggle would work if `www.kaggle.com` and `storage.googleapis.com` are
  allowed and `KAGGLE_USERNAME` / `KAGGLE_KEY` are set in the environment settings. But its quota
  (~30 GPU-h/week, P100 / T4) is small next to E1–E5's total.

### 2026-09-28 — E2 code built; E2–E5 analysis; VM hand-off
* **E2 trainer** in `src/aurora/fusion.py`:
  * sources with their own N and variables, and a loss and metrics over present variables only;
  * the methods perceiver / pad (spectral zero-pad, zero-filled variables, presence fields as extra
    embedded inputs) / per_source (encoder per source, shared backbone).
  * `model.apply` accepts extra non-physical inputs (no residual).
  * 7 gates in `tests/test_e2_fusion.py`, including bitwise equality with the base model in the
    degenerate cases.
* **E2 predictions** pinned to specific sources (spec §3) before any run.
* **Configs:** `experiments/e2/configs/runs.json`, 30 runs; `run.py` routes configs with `method`
  to the fusion trainer.
* **Analysis:** `experiments/analyse_e345.py --exp e2|e3|e4|e5` gives the verdicts per prediction.
  Tested on planted rows.
* **VM hand-off (28 Sep):** the GPU work goes to the VM:
  * data generation;
  * Stage 4 S1–S6;
  * E1–E5 runs, reports, README.
  * The VM side works on branch `vm-results`. It touches only results, reports, spec status lines,
    README and memory, never `src/` or configs (a config change breaks `init_from` hashes).

### 2026-09-28 — Specs frozen; GPU runs started on the VM
* **E1–E5 specs frozen** in git before any run (PR #13): only the status lines changed. The open
  points stay open and are to be addressed in the reports.
* **VM run:**
  * clone at `/SML_DISK_24TB/rajeshr/Aryamann/aurora-from-scratch-jax`, on branch `vm-results`
    (pushes go through the Mac);
  * cards 1–3, since card 0 is another user's;
  * thermal guard: no start ≥ 85 °C; pause ≥ 87 °C; resume ≤ 80 °C;
  * shard count fixed per experiment.
* **Precision rule** (set before measuring): `JAX_DEFAULT_MATMUL_PRECISION=highest` for every run,
  unless it costs > 1.5× the TF32 step time. The choice is recorded in every report.
* **Benchmarks:**
  * E1 trajectory: 132 s at N_gen 128 and 304 s at N_gen 256 (GPU);
  * `acc_v1` Re 1000 at 512²: ~16 min per trajectory (GPU), against ~3 h on CPU;
  * E1 generation is therefore ≈ 21 GPU-h against the spec's 8 (a deviation for the E1 report,
    not a prediction).

### 2026-09-28 — GPU checks, measured costs, run plan
* **GPU tests:** 87/91 pass at the default precision.
  * With `highest`, 90/91 pass. The three that recover were TF32 artefacts; their tolerances are
    unchanged.
  * D9 fails for a diagnostics-only reason (§3).
* **Training cost:** ~120 ms per step for the E1 pretrain model (batch 16, shared card), about 5×
  the specs' 20–30 ms.
  * Likely launch-bound on a tiny model; running several processes per card is the numerics-neutral
    remedy.
  * Projected E1–E5 training: ~62–77 GPU-h. E2's 128² sources cost 4× per step.
* **Data route (owner's decision, since the cards are shared):**
  * `e1_v1` is generated **entirely on CPU**: 4 lanes, one pinned core each, single-threaded XLA,
    about 13 h. It is bitwise reproducible.
  * The two GPU-made benchmark trajectories are kept aside in `data/stage2/e1_v1_gpubench`, for a
    GPU-vs-CPU comparison.
  * `acc_v1` is **mixed-device**: Re 100 / 300 on CPU, Re 1000 on GPU (~16 min per trajectory).
    E2, E4 and E5 compare within this one copy. Only absolute comparisons with the cloud-CPU Stage
    7–10 numbers carry a device caveat.
* **Run plan (all on code 28c09e5, the frozen specs):**
  * Stage 4 lane: `make_512`, then S1–S6.
  * Lanes a0–a2 (`--n-shards 3` fixed; `rows_{i}.jsonl` / `rows_ft_{i}.jsonl`): E4, then E2,
    then E5 pretrain + scratch, a barrier, then the E5 fine-tunes. They start once `acc_v1` is
    finalised.
  * E1 and E3 follow the same layout once `e1_v1` is done.
* **Decision: no `src/` change before the first E-run.** Nothing found affects training numerics,
  so every E-result comes from one code version.
* **Real-data smoke test** (cloud container, 28 Sep): 28 committed configs from all five
  experiments, cut to 2 steps, were run end to end. E2/E4/E5 used the real `acc_v1`; E1/E3 used a
  tiny stand-in with E1's exact tags and seed lists.
  * The runs covered: the pretrain → fine-tune chains with `norm_from_init`, LoRA, all 9 E4
    variants, 128² and (u, v) sources, and the 20- and 50-step rollouts.
  * **It found one real bug.** `swin.shift_mask` failed under `jit` whenever a non-periodic axis is
    shifted, i.e. E4's **L = 6** variant, which would have crashed on the VM.
  * Fixed in PR #17 (`171b239`): the mask is built with numpy only, with identical values. Every
    other config returns before that line and is unaffected.
  * New gate W10 (Stage 5 amendment A1); the E4 design test now runs jitted. After the fix, all 28
    smoke runs pass.

### 2026-09-28 — VM notes: hardware, costs, the L6 crash
* Hardware: RTX PRO 6000 ×4, shared with other users' jobs at 74–93 °C; Xeon w3-2525 (8 cores);
  JAX 0.10.2 (CUDA 12); Python 3.11 venv built from `Aryamann/env`.
* `highest` costs 1.07× TF32 (128.8 s vs 120.5 s for 1000 steps), so the precision rule keeps it.
  One CPU core generates an N_gen-256 E1 trajectory in 787 s. The first run in a training process
  pays ~8 min of compile; a persistent JAX compile cache is enabled.
* D9 on the GPU: after a restart the frames are bitwise identical; only `diag.npz` differs.
* GPU jobs start only on a card where nobody else computes, below 85 °C. *(Correction, 29 Sep: this
  was not in effect until 29 Sep 18:20 VM time. Every job ran on card 0; see the incident entry.)*
* E4's three L6 runs had already crashed on the VM before the fix reached it (E4 amendment A1).
  They are held and run on the fixed code; the other 24 E4 runs continue on `28c09e5`.

### 2026-09-29 — Stage 4 S1–S6 measured: acceptance NOT met
* Shared L = 3 model, relative L2 error 0.167 / 0.085 / 0.063 / 0.060 at C = 3 / 5 / 6 / 8 (bars
  0.02 / 0.02 / 0.03 / 0.05): S1–S4 MISS. Still falling at 6000 steps (16²: 0.187 → 0.167 over the
  last 1000).
* S5 HIT: single-resolution L = C controls reach 0.043 / 0.041 / 0.040 / 0.049. S6 MISS: with level
  encodings ablated the error is 0.049 (bar > 0.171).
* Spec amendment A1: S5 and S6 are confounded with the budget (the controls get 4× the
  per-resolution steps); verdicts kept. Matched 64² pair: encodings on 0.041 vs off 0.049, with
  curves identical to step 3000.

### 2026-09-29 — E4 done: 1 of 5 HIT
Report: `experiments/e4/REPORT.md`. 27 runs, 3 seeds; change vs base, mean ± SE.
* E4-P1 MISS: MAE Re 300 drift −11.1 ± 6.5% (predicted > +50%). Stage 8's MAE blow-up does not
  reproduce at 10k steps with 3 seeds; MAE drifts less than MSE in both regimes (Re 100 −19.2 ± 0.8%).
* E4-P2 MISS: pushforward drift at matched steps −49.8 ± 3.8% / −31.3 ± 8.1%, **and at matched
  compute −44.6 ± 4.6% / −47.1 ± 6.9%** (~10 SE), which reverses Stage 9 (3000 vs 1500 steps, 1 seed,
  20-step drift). Matched compute costs +21.5 ± 3.8% one-step error.
* E4-P3 HIT: noshift +6.0 ± 4.5% one-step.
* E4-P4 MISS: depth6 +9.8 ± 1.6%, depth24 −6.7 ± 3.1% (it helps).
* E4-P5 MISS: L1 +2.0 ± 2.8%, L6 −3.9 ± 4.4% (both within noise).
* Base: one-step ratio to persistence 0.192 / 0.330 (Re 100 / 300), drift 1.52 / 1.94. A base run
  took about 8 min (10k steps plus evaluation) on card 0, which may have been shared.

### 2026-09-29 — E2 done: 1 of 3 HIT
Report: `experiments/e2/REPORT.md`. 30 runs, 3 seeds; relative change in one-step error, mean ± SE.
* E2-P1 MISS: the Perceiver vs the homogeneous reference on unchanged formats is +12.8 ± 2.5% (B,
  set a) and +38.8 ± 7.5% (A, set b). Fusion costs accuracy even where a source's format did not
  change.
* E2-P2 MISS, reversed: pad-and-mask beats the Perceiver on the upsampled source B by
  23.1 ± 1.9% (a) and 22.5 ± 2.8% (c); on A in (a) as well (0.039 vs 0.048).
* E2-P3 HIT: per-source encoders are worse than the Perceiver on B by 7.4 / 22.1 / 8.3% (a / b / c).
* Absolute errors are in each source's own format, so compare within a set. All runs used
  `b940e74`; median 16.5–21.6 min per run.

### 2026-09-29 — E5 done: 2 of 4 HIT
Report: `experiments/e5/REPORT.md`. 39 runs, 3 seeds; Re 1000 one-step error; zero-shot 0.166.
* E5-P1 HIT: gain(full, k) −3.0 / +2.3 / +4.7 / +5.3% for k = 1 / 2 / 4 / 6. With one trajectory,
  full fine-tuning is worse than zero-shot.
* E5-P2 MISS: gain(full, 6) = 5.3% (predicted ≥ 8%); the k = 2 half holds (2.30%).
* E5-P3 MISS: LoRA/full ratio −0.19 / 0.56 / 0.34 / 0.31. Ill-posed at k = 1, where full's gain is
  negative (spec amendment A1); it falls over k = 2–6. LoRA gains 0.6–1.7%.
* E5-P4 HIT: adv(k) 40.9 / 30.8 / 16.2 / 4.9%. Scratch 0.290 → 0.165 catches up with zero-shot by
  k = 6.

### 2026-09-29 — Incident: every GPU job ran on card 0 (fixed)
* **What happened.** The lab account's `~/.bashrc` (line 25) exports `CUDA_VISIBLE_DEVICES=0`.
  Every job the runner launched started through a shell that read that file, which overrode the
  card the runner had chosen. So every GPU job since the launch on 28 Sep ran on card 0 (P0): the
  benchmarks, the GPU tests, `acc_v1` Re 1000, `make_512`, Stage 4, E2, E4, E5 and the first E1
  runs. That was up to 3–4 of our jobs at once, alongside other users' jobs. The runner believed it
  was on idle cards 1–3, so its thermal guard read those cards and never saw card 0's heat.
* **Found** on 29 Sep after the professor's notice that P0 had 4 jobs and was thermal-throttling. At
  18:16 VM time card 0 carried only our two E1 jobs, at 93 °C with slowdown active. They were stopped
  at once.
* **Effect on results: none on the numbers** (identical cards, same code; a pause does not change
  the computation). All GPU timings before the fix (benchmarks, wall times) were measured on a
  shared card. The reports for Stage 4, E2, E4 and E5 carry a dated correction.
* **Fix** (`glane3.sh` in the VM scratch area):
  * jobs start in a bash that skips `~/.bashrc`, with `CUDA_VISIBLE_DEVICES` set inside the command;
  * an interlock kills a job whose processes appear on any card other than the claimed one;
  * our job pauses whenever any other process is on its card, or at ≥ 87 °C, and resumes only when
    the card is idle and ≤ 80 °C;
  * card 0 is excluded for now.
  * Verified with a test job: it claimed card 2 and ran on card 2. E1/E3 restarted on cards 1–3 at
    18:22; the two interrupted E1 runs restart from step 0.

### 2026-09-30 — E1 done: 2 of 6 HIT
Report: `experiments/e1/REPORT.md`. 57 runs, 3 seeds; target (300, 4); scratch 0.150 one-step, 0.70
drift.
* E1-P1 HIT: one-step adv 41.8–49.8% in every cell (≥ 5.6 SE); fine-tuned error 0.076–0.087.
* E1-P2 / P3 MISS: diversity gain −3.4 points (n 2 → 32), volume gain +1.4 (T 32 → 128). Neither
  moves the fine-tuned result at a fixed 20k-step budget.
* E1-P4 MISS: best 49.8%, (2, 32) 47.4%; both above the predicted bands.
* E1-P5 HIT: zero-shot n = 32 0.080–0.089 vs n = 2 0.184–0.186. **Distance, not diversity:** n = 8
  has the same nearest neighbours as n = 2 (Δ ln Re 0.41, Δ k_f 1) and the same zero-shot error;
  only n = 32 has k_f = 4 neighbours.
* E1-P6 MISS: Spearman(one-step, drift) = −0.38. Fine-tuned models drift 51–123% more than
  scratch; the penalty shrinks with n (−102 / −65 / −60%).
* 16 runs ran on card 0 before the 29 Sep fix; the other 41 on idle cards. ~10 ms/step on an idle
  card.

### 2026-09-30 — E3 done: 3 of 4 HIT
Report: `experiments/e3/REPORT.md`. 33 runs, 3 seeds; band Re 125–250 × k_f 3–6; tests at k_f = 4.
* E3-P1 HIT: zero-shot ratio 0.301 / 0.326 / 0.356 / 0.383 / 0.412 at Re 300 / 350 / 450 / 560 / 700
  (ρ = 1.0).
* E3-P2 HIT: one-step adv 35.8–41.4% at every distance (≥ 6.2 SE).
* E3-P3 MISS: adv(700) 36.4 ± 3.5% vs adv(300) 41.4 ± 4.7%; the gap is below 2 SE (flat from Re 350).
* E3-P4 HIT: z(700) = 0.41. Fine-tuned models drift 64–80% more than scratch, as in E1.
* Against E1 at (300, 4): the E3 band (extrapolating, but with k_f = 4) gets 0.30× zero-shot vs E1
  n = 2/8 (enclosing, no k_f = 4) 0.61× and n = 32 0.27×. Zero-shot follows the distance in k_f. The
  fine-tuned adv is similar (41% vs 42–50%).
* All 33 runs ran on idle cards after the fix; ~10 ms/step (a 20k-step pretrain takes a median
  3.5 min).

### 2026-09-30 — Results checked; follow-ups prepared
* Independent check of E1–E5 from the committed files (PR #26): 186 of 186 rows, all five
  `analysis.json` files recompute identically, and every report's verdicts and numbers match.
* Follow-ups for the Stage 4 and 10 misses pre-registered before any run (PR #27); the README ⚠️
  rows resolved or explained (PR #28).
* The cloud container is gone, with everything it held outside git. That includes the Stage 7
  parameters (`data/stage7/onestep_mse_params.npz`), which are not on the VM or the Mac either.
* Stage 10 F1 amended before any run (F1-A1): it runs on a retrained base model and re-runs its
  zero-shot, r = 4 and full baselines, because the base model and the VM's Re 1000 data both differ
  from Stage 10's. `train_onestep.py` and `finetune.py` gained `--out`, so no follow-up overwrites a
  committed result.
* Where they run: Stage 4 F1 and Stage 10 F1 on the VM's CPU, as their specs say; Stage 4 F2 on the
  first idle GPU card.

### 2026-10-01 — Follow-ups: Stage 4 F1 and Stage 10 F1 done
* **Stage 4 F1:** the 16² control (C = 3) at 48,000 steps reaches 0.016 (F1-P1 HIT, F1-P2 MISS), so
  S1's 0.02 bar is reachable with more training. Every check after step 36,000 is below 0.02, but a
  single end-point is noisy at a constant learning rate (0.012–0.020 over the last 12,000 steps).
  The spec's "stretched cosine schedule" was wrong (the script uses a constant lr); a note went into
  the spec at launch, before the result. 6.8 min on one CPU core.
* **Stage 10 F1** (on the retrain, F1-A1): the retrain reproduces Stage 7 within 2.5% (0.0720 /
  0.1450 vs 0.0702 / 0.1457). LoRA gains 0.21 / 0.37 / 0.61 / 0.98% at r = 1 / 4 / 16 / 64, full
  fine-tuning 2.69%: F1-P1 and F1-P2 HIT. Rank is not the limit; r = 64 gets 37% of full's gain.
  About 1 h in all on 4 CPU cores.
* **Stage 4 F2** waits in its `glane3.sh` lane (`f4g`) for an idle card. None has been free since
  30 Sep 23:38: three cards at 92–94 °C with other users' jobs, the fourth busy too.

### 2026-10-01 — Stage 4 F2 moved to the CPU
* The F2 lane claimed idle card 2 at 01:26 (below 85 °C). 21 s later the card was at 92 °C and the
  job was paused. Paused, it held the card at full power (133 W, 0% use, 84 °C) and 20 GB for 1.5 h
  without reaching the 80 °C resume bar. Stopped by PID at 02:58; the card fell to 49 W at once, and
  another user took it soon after.
* F2 now runs on 4 cores of the VM's CPU (lane `f4c2`, cores 4–7, 14 GB), started 03:02: 6.9 s per
  512² step, about 13 h. The device change went into the spec before the run (F2 note).
* At step 500 it matches the original GPU run to 2e-6 at every resolution, but the difference grows
  (1e-4 at step 1000, 2e-3 at 1500), and from step 3000 the two runs are effectively independent:
  up to 27% apart at 512² at the same step (5% at 16²). So one shared-model end-point carries roughly
  ±10–25% noise at 512², S1–S4 included. F1's single-resolution control did not drift.

### 2026-10-01 — Advisor's notes: speed (Stage 12) and RealPDE (E6) opened
* **Advisor's notes (1 Oct):**
  * test the model on RealPDE data;
  * JAX speedups, with PyTorch as the reference, including the same run under `torch.compile`;
  * decorators and parallelism;
  * ideas for the multi-head attention;
  * understand the architecture deeply.
* **Owner's decisions:** the RealPDE test is sim → real pretraining (pretrain on the simulated foil
  data, fine-tune on the real PIV data, compare with scratch and with the organisers' FNO); all
  implementation in-house.
* **Stage 12 spec** (`handouts/stage12_speed.md`): ten predictions, committed before any timing.
  Tooling is in `experiments/stage12/`:
  * the JAX arms J0–J6;
  * an exact PyTorch port of the model;
  * the PyTorch arms T0–T3;
  * Aurora's own backbone as a reference point.
* **The port is exact:** in float64 it matches the JAX model to 2.8e-13 (forward) and 8.4e-10
  (gradients). The spec's float32 bars were below the rounding floor: JAX's own float32 is 1.1e-4
  from float64. Amendment A1, made before any timing, moves the gate to float64.
* **`docs/architecture.md`:** a walkthrough of the model, with shapes, the comparison with Aurora,
  parameters and FLOPs.
* **RealPDE foil data** (the UGP-I competition copy):
  * **real:** 82 trajectories, u and v, 607 frames, 64 × 128 grid;
  * **simulated:** 100 trajectories, u, v and p, 1000 frames;
  * not periodic, so E6 needs rectangular bands and masked shifts.
* **E6 set up** (spec `handouts/e6_realpde_sim2real.md`, five predictions, committed before any run):
  * **Data** (`data/e6/`):
    * simulated frames strided to 32 × 64 as UGP-I strided the real ones, 100,000 frames;
    * UGP-I's real frames copied, with their sha256;
    * the simulated fluctuations are about 4× the real ones, so each source is normalised by its
      own statistics.
  * **Model:** handles rectangular, non-periodic fields.
    * Bands are octave shells of max(|k_y|, |k_x|·H/W).
    * Positions use half a period.
    * Shifted windows are masked.
    * The square path is bitwise unchanged; gates R1–R6 cover the new case and 98 tests pass.
  * **Code:**
    * `run_e6.py`, training and evaluation, scored with the kit's own `scoring.py`;
    * `fno_baseline.py`, which runs the kit's FNOs through UGP-I's `r12_eval`;
    * one shared split module, whose windows must match UGP-I's 900.
  * **Amendment A1, before any run:** p is dropped in a random half of the simulated pretraining
    chunks, as the kit's `mask_prob` did. The zero-shot arms are the pretraining runs' own
    evaluations.
  * **FNO baselines** on the 900 `re_lohi` windows (CPU, through UGP-I's `r12_eval`; the windows are
    identical to UGP-I's):
    * `sim_real_fno.pth`: rel_l2 error 0.0948 (subscore 95.47), TKE 75.90, MVPE 96.09. UGP-I's
      selftest reproduces to 1e-3.
    * `sim_fno.pth`, simulations only: rel_l2 error 1.23 (subscore 61.95). It does not transfer
      zero-shot.
    * So UGP-I's "the FNO never saw a real frame" was wrong (corrected in §1), and E6-P3's anchor
      with it (amendment A2, before any run).

### 2026-10-01 — Stage 4 F2 done; Stage 12 re-timed in one lane; E6 queued
* **Stage 4 F2** (CPU, 03:02–15:50): 0.058 / 0.026 / 0.018 / 0.021 at 16² / 64² / 128² / 512² after
  24,000 steps. It beats the dedicated 6000-step controls at 64² and above (0.42–0.64×) but is
  1.35× worse at 16², so F2-P1 MISSES on 16² alone (4% over the bar, inside the ±10–25% end-point
  noise). F2-P2 (S4's bar) HITS. Every error was still falling. Report updated; the original
  verdicts stand.
* **Stage 12 first timings** (card 3, 12:34–12:43): J0 19.2 ms, J1 19.7 ms (batch 32) and 22.3 ms (batch
  64), J2 12.2 ms, J3 4.9 ms. The card sat at 26% utilisation under J0. S12-P1 and P2 HIT; P4 MISSES
  (donation 1.58×).
* **Host load:** J0's 19.2 ms against the E-runs' ≤ 10.5 ms for the same loop. The host was at load
  11–14 on 16 cores (CPU pressure 27–40%), and a host-bound step slows with it.
  * **Amendment A2**, before any further timing: every arm re-timed in one sequential lane (`s12`),
    with J0 first and last and the host load logged (`logs/hostload.log`).
  * P1, P2 and P4 stay scored on the first pass (`results/first_pass/`).
* **J4 was paused** when another user started a job on card 3. The owner had it killed and requeued.
* **`glane4.sh`** (VM scratch) = `glane3.sh` plus preemption:
  * a paused job is killed (its own process group) and requeued on the next idle card;
  * at once in the timing lane (`PREEMPT_S=0`, which enforces the spec's discard rule), after
    10 min in the E6 lane;
  * tested with a fake `nvidia-smi` before use.
* **E6:**
  * the CPU smoke test passed: S, PD pretrain and PD fine-tune all train, save, reload and score;
  * `score.py` was committed before any run;
  * the lane `e6g` (15 runs, seed 0 of every arm first, pinned to cores 8–11) waits for an idle card.
* **The VM clone** is at the latest `vm-results` commit, updated by bundle.

### 2026-10-02 — E6 done on Kaggle: 2 of 5 HIT
* **Where:** the lab machine's cards stayed busy, so the owner chose their Kaggle account (E6
  amendment A3, before the runs). All 15 runs ran on two T4s in about 2.5 h, with JAX pinned to the
  lab machine's version. The lab's seed-0 S run reproduces to 0.5% (0.1099 against 0.1104).
* **Results**, rollout rel_l2 on the 900 `re_lohi` windows:
  * S (scratch, 65 real trajectories): 0.112, subscore 94.71;
  * PD and PA (simulation pretraining, then a 2000-step fine-tune): 0.137 and 0.135;
  * ZD and ZA (zero-shot): 0.27 and 0.29.
  * The organisers' fine-tuned FNO: 0.0948 (95.47).
* **Verdicts:**
  * E6-P1 MISS: adv(PD) = −22.9% ± 1.5%;
  * E6-P2 MISS: PA = PD;
  * E6-P3 MISS: ZA ≫ S;
  * E6-P4 HIT: fine-tuning cuts the zero-shot error by 49%;
  * E6-P5 HIT: no arm reaches the FNO.
* **Readings:**
  * The simulations do not transfer to the PIV measurements. Zero-shot one-step forecasts are only
    4% better than persistence, and the TKE subscore falls from 76 to 43–48.
  * Not tested: a fine-tune given the same real-data budget as scratch (22,000 steps against 2000).
    That needs its own spec.
* **Stage 12 supplement on Kaggle's T4 (spec §9, S12-K1 to K13, J7 included):**
  * version 1 ran JAX on the CPU, because a fresh venv's JAX fell back silently (J0 at 1.4 s per
    step); its CPU timings are not results;
  * the runner now installs JAX as the E6 job did, and refuses to run without a GPU (commit 215f5e1).

### 2026-10-02 — Stage 12 done: acceptance met; 6 of 10 HIT, T4 supplement 12 of 13
* **Lab card 3** (amendment A3), sequential lane at 13:53–14:19, approved by the owner. Host load
  was 4–9, with one spike to 16 during J6.
  * **Acceptance met:** `lax.scan` (J3) is 2.07× faster than J0 (4.94 against 10.19 ms).
  * HIT: P1, P2, P5, P7, P8, P9. MISS: P3 (J4 1.80× a J0 step), P4 (donation 1.13×, 1.58× on the
    first pass), P6 (bfloat16 1.12×, timed during the load spike), P10 (`torch.compile` forward
    1.81× slower than JAX).
  * The step is host-bound: the card is 49% busy, and J0 runs at 10.2 ms on a quiet host against
    19.2 ms on a loaded one.
  * PyTorch: eager 2.42× slower than J0, `torch.compile` +11%, CUDA graphs 1.36× faster than J0 but
    1.52× slower than JAX scan.
  * bfloat16 changes the forward error by 0.002%.
* **T4 supplement** (Kaggle, version 2): 12 of 13 HIT; K11 (SDPA 35% slower on a T4) MISSES.
  * On a T4 the step is bound by the card, so scan gives only 1.16× and vmap 0.90× per model.
  * Two T4s (J7) give 1.24×.
  * `torch.compile` is 9% faster than JAX there.
  * bfloat16 is not applied on a T4 (the error matches float32 to 5e-9), so K6's HIT is trivial.
* **Practical upshot:** train with scan chunks, as `run_e6.py` does. Donation and vmap over seeds
  stack on top. bfloat16 waits until the step is no longer host-bound.

### 2026-10-02 — E6 follow-up F1: the same real-data budget, 3 of 3 HIT
* **Design** (spec §10, committed before any run): the six E6 pretrained models were trained on real
  data with S's exact recipe (22,000 steps, lr 1e-3, cosine, S's batches by seed). Kaggle job
  `ugp2-e6-f1`, two T4s.
* **Results:**
  * PDm 0.1112 and PAm 0.1094, against S's 0.1117;
  * adv(PDm) = +0.4% ± 1.8% (P1 HIT);
  * PAm not better than PDm beyond 2 SE (P2 HIT, though PAm beats S in all 3 seeds by 2.0% ± 0.9%);
  * PDm is 19% better than the 2000-step fine-tune (P3 HIT).
* **Reading:** E6's −23% was the short fine-tune. Simulation pretraining is a slower route to the
  same accuracy on the real measurements, not a better one.

### 2026-10-02 — E7 done on Kaggle: relative-position bias in attention, 2 of 3 HIT
* **Design** (spec `handouts/e7_attention_rpb.md`, committed before any run): the Stage 12 report's
  top attention idea. Each Swin block learns a bias per head for each of the 245 offsets inside its
  3 × 4 × 4 window (`backbone.Config(rel_pos_bias=True)`, off by default, zero-initialised, no extra
  random draws). Gates G1–G6 pass; 104 tests in all. Arms B (E1 scratch config) and R (with the
  bias), seeds 0–2, same starting weights and batches. Kaggle job `ugp2-e7-runs` (version 2; version
  1 stopped at its smoke run for a missing manifest), two T4s, about 65 min.
* **Results:**
  * one-step error 0.1499 → 0.1396: adv(R) = 6.9% ± 0.7%, every seed 5.6–8.1% (P1 HIT);
  * 20-step drift −0.9% ± 1.6% (P2 MISS);
  * wall time R / B = 1.012 (P3 HIT).
* **Reading:** the bias buys about what doubling the depth bought in E4, for 0.27% more parameters,
  but one-step and rollout gains part ways again, as in E1. Trained weights were not kept (they stay
  in the job's scratch folder).

### 2026-10-03 — Overnight on Kaggle: Stage 4 F3 (3 of 4 HIT) and E8 (5 of 10 HIT)
* **The owner's ask:** the Stage 4 longer run, then the smaller follow-ups, on Kaggle overnight. Both
  specs were committed and pushed before their runs: F3 in e51540d, E8 in 4e507e4 (code d69d575).
* **Stage 4 F3,** the shared Perceiver trained for 192,000 steps (32×), keys 0 and 1, one per T4.
  * Job `ugp2-s4f3-runs` v2, 7.8 h. In v1 Kaggle gave one card, and the two-card guard stopped it
    before any training. The 512² step needs one 7.8 GB XLA block, so the runs reserve 90% of the
    card.
  * 16² falls below S1's 0.02 from step 136,000 (run 0) and 151,500 (run 1), but at the constant
    learning rate spikes push it back up. A spike at step 188,000 left run 0's final window at
    0.038; run 1 ends at 0.017 (P1 MISS).
  * 64² meets S2 (P2 HIT, run 0 at 0.01996), 128² and 512² keep theirs (P3 HIT), and 16² is still
    0.026 at 96,000 steps (P4 HIT).
  * The Stage 4 verdict stays "not met".
* **E8,** six arms × 3 seeds on E1's scratch task (job `ugp2-e8-runs`, 4.1 h; the trained weights are
  in `data/e8_params/` on the VM).
  * The bias replicates (one-step 6.9% ± 0.7%, P1 HIT). Measured from 84 rollouts it does lower
    the 20-step drift, by 3.3% ± 0.6% (P2 MISS: the prediction said under 3%). The gain fades from
    7% at step 1 to 1.4% at step 20 (P3 HIT).
  * Pushforward costs 12.9% one-step error at matched steps (P5 MISS) and lowers the 50-step drift
    by only 6.2% (P4 MISS). The bias still helps under it (P6 HIT).
  * q/k normalisation lowers one-step error by 7.3% ± 0.7%, as much as the bias (P7 MISS), at 10.1%
    more time (P8 MISS).
  * Scan chunks give the same models (P9 HIT), 1.21× faster on a T4 (P10 HIT).
  * Exploratory (`amplitude.py`, CPU): every model decorrelates from the flow by step 30–50. About
    two-thirds of pushforward's 50-step gain is damped amplitude (0.93 of the truth's).
* **Kaggle login:** the CLI's login lapsed around 01:20; the owner renewed it in the morning. The
  jobs themselves were unaffected.

### 2026-10-03 — E9: the bias and q/k normalisation together, 4 of 6 HIT
* **Design** (spec `handouts/e9_rpb_plus_qknorm.md`, commit 0a953e3, before any run): arm RQ (both
  options) and a rerun of B, seeds 0–2, as in E8. Compared with E8's B, R and Q. Kaggle job
  `ugp2-e9-runs`, 70 min; the trained weights are in `data/e9_params/` on the VM.
* **Results:**
  * one-step error 8.9% ± 0.4% below B: P1 MISS against its 9% bar, P2 HIT (below the
    independent 13.7%);
  * RQ beats E8's R and Q in every seed (P3 HIT);
  * 20-step drift 4.2% ± 1.7% lower (P4 HIT); training time 1.11× (P5 HIT);
  * B rerun is within 0.8% of E8's B, against a 0.5% bar (P6 MISS: run-to-run noise on the GPU).
* **Reading:** the gains overlap. RQ goes a quarter of the way from the better single change to
  independent gains, and is the best one-step model so far (0.136).
* **Exploratory** (`attention.py`): without q/k normalisation, attention is near-uniform at the fine
  stages and nearly one-hot at the coarsest (logit SD 8–11). q/k normalisation evens this out; the
  bias does not. So the spec's overlap hypothesis does not fit.

## 6. Status (3 Oct 2026)

**Done:**
* Stages 1–11: built, gates pass, results reported.
  * Acceptance met: 1, 3, 5, 6, 7, 8, 11.
  * Not met, with the causes measured:
    * Stage 4: the shared Perceiver is under-trained at 6000 steps (0.06–0.17 vs bars 0.02–0.05).
    * Stage 9: pushforward does not pay at matched compute.
    * Stage 10: LoRA adapts little from 2 trajectories.
  * Stage 2: spectra stable; bitwise regeneration is machine-dependent at Re 1000.
* E1–E5: specs frozen (28 Sep); all 186 runs done and analysed. The real-data smoke test passed
  after the W10 fix.
* E4: done, 1 of 5 HIT (pushforward wins even at matched compute; MAE is stable; depth 24 helps).
* E2: done, 1 of 3 HIT (pad-and-mask beats the Perceiver on the upsampled source).
* E5: done, 2 of 4 HIT (the pretraining advantage over scratch fades from 41% to 5% as k goes 1 → 6).
* E1: done, 2 of 6 HIT (pretraining halves one-step error in every cell; diversity and volume do not
  move it; zero-shot follows distance; fine-tuned models drift more than scratch).
* E3: done, 3 of 4 HIT (zero-shot degrades slowly with distance; the fine-tuned advantage stays at
  36–41%).
* `e1_v1`: all 326 trajectories generated on CPU (29 Sep, 05:34 VM time).
* 118 tests pass on CPU (3 Oct: the 92 earlier ones, E6's gates R1–R6, E7's G1–G6, E8's Q1–Q6 and H1–H4, and E9's RQ1–RQ3). On the GPU (the 91 tests before W10): 87 pass at default precision and 90
  at `highest`; D9 fails for a diagnostics-only reason.

**Running (lab VM):**
* Nothing. Stage 12 (lab card 3 and the Kaggle T4 supplement), E6–E9 and Stage 4 F3 (Kaggle) are
  done and reported.
* The VM's lanes are idle. The owner approves each launch on the lab machine. Card 0 stays excluded.

**Remaining:** none from the advisor's notes of 1 Oct. Done: E6 follow-up F1 (3 of 3 HIT), the
attention experiment E7 (2 of 3 HIT), and the overnight follow-ups of 3 Oct: Stage 4 F3 (3 of 4 HIT)
and E8 (bias with a finer drift and with pushforward, q/k normalisation, scan chunks; 5 of 10 HIT);
then E9, the two attention changes together (4 of 6 HIT: they mostly overlap).
Possible follow-ups each need their own spec and the owner's go-ahead:
* Stage 4 with a decaying learning rate (F3's 16² spikes);
* a rollout measure that separates tracking from amplitude (E8's exploratory check).
The stage and experiment reports are the record.

**Pre-registered follow-ups (30 Sep; predictions committed in PR #27):**
* **Stage 10 F1**, a LoRA rank sweep: **done** (1 Oct), both predictions HIT. It ran on a retrain of
  the Stage 7 model, because the original parameters are gone (amendment F1-A1 lists the commands).
* **Stage 4 F1**, the C = 3 control at 8× steps: **done** (30 Sep), F1-P1 HIT, F1-P2 MISS (0.016):
  `experiments/stage4/train_perceiver.py --run nocompress --res 16 --steps 48000 --out f1_nocompress_16_48k`.
* **Stage 4 F2**: `train_perceiver.py --run main --steps 24000 --out f2_main_24k`, **done** (1 Oct, on
  the CPU): F2-P1 MISS (1.35× at 16²; 0.42–0.64× at 64² and above), F2-P2 HIT (0.021 at 512²).
* Score each against its table in `handouts/stage10_lora.md` or `handouts/stage04_perceiver.md`,
  add the result to the stage report (the original verdicts stand), and update the README row.

**README ⚠️ rows (30 Sep):**
* Stages 8 and 9 are resolved by E4 (3 seeds).
* Stage 2 is a documented limit (bitwise regeneration across machines is impossible for these
  chaotic flows; the post-spin-up state helps only on the same XLA ISA).
* Stage 10 is explained by F1 (rank is not the limit) and stays not met. Stage 4: F1 shows that
  S1's bar is reachable at C = 3; F2 shows that the shared model beats dedicated controls at 64² and
  above when given the same steps per resolution. F3 (32×) meets S2–S4. S1 is reached intermittently
  but not held at the constant rate, so the row stays ⚠️.

**Possible follow-ups** (each needs a new spec and the owner's go-ahead):
* E1: a distance-matched 32-regime control; pretraining steps scaled with T.
* E3: a downward-in-Re arm and a k_f-distance arm (E1 suggests k_f distance matters most).
* E2: a wider resolution gap, and an unseen format.
* E4: longer budgets (the pushforward and depth effects may change at convergence).
* E5: a LoRA rank sweep.

**Closed, not done:** a deterministic `shell_spectrum`, which would make D9 hold on the GPU. It
would change the CPU `diag.npz` checksums, so the existing datasets would no longer regenerate
bitwise with the current code, and it only buys a diagnostic.

**Verified independently (30 Sep, cloud container):**
* every experiment has exactly one row per config (186 of 186), with no strays, duplicates or
  non-finite metrics;
* all five `analysis.json` files recompute identically from the committed rows;
* the verdicts and headline numbers in every report match.

**Before any further GPU run:** use `glane3.sh` (idle cards only, card-placement interlock). Card 0
stays excluded unless the lab agrees otherwise.
