# UGP-II — Aurora-from-Scratch in JAX · Decision Record

**Owner:** Aryamann Srivastava, 4th year, Dept. of Aerospace Engineering, IIT Kanpur
**Advisors:** Prof. Rajesh Ranjan (ACAL) · Prashant Kumar (PhD)
**Session name:** `UGP-II`. This file is the decision record for the NEW project.
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

**Credits and pacing.** Aryamann is on a Pro plan and hit 82% of the weekly limit during UGP-I.
Remote subagents were observed to bill against the SAME session limit, not a separate pool — four of
them died simultaneously on one rate limit. Do not assume `isolation: "remote"` shifts cost.
**Gemini credits are available and are the intended vehicle for UGP-II implementation work.**
Batch SSH round-trips, keep outputs small, stop redundant sweeps immediately.

**Repo.** `ARYAMANN7279/RealPDE-Track1` (private) exists but is stale (last push 27 Aug 2026) and is
not a working copy of anything. Decide early whether UGP-II gets its own repo — it should.

---

## 4. Working agreement with Aryamann

* Give the number, not the narrative. He asks for 2-line answers often; respect that.
* Report failures plainly and immediately. He would rather hear "this is closed, here is the
  measurement" than be handed something unverified.
* He does the submitting and any logging-in himself. Claude never authenticates.
* When he asks for something that measurement says will not work, say so once with the number, then
  do what he decides.

---

## 5. Log

### 2026-09-26 — project defined
Prashant proposed the Aurora-in-JAX project. Plan drafted (`PROJECT_PLAN.md`): 5 research questions,
11 staged deliverables with acceptance tests, 5 experiments. Agreed to start after the Track 1
development phase closes on 27 Sep. Gemini credits to be used for implementation.

**First action when UGP-II opens:** Stage 1 only — the pseudo-spectral Kolmogorov solver. Do not
proceed past it until the energy spectrum shows the expected inertial-range scaling and the energy
budget balances to <1%. If the data generator is wrong, every measurement downstream is meaningless.
