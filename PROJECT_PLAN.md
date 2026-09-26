# Aurora-from-Scratch in JAX: What Does Pretraining Data Actually Buy?

**Aryamann Srivastava · Dept. of Aerospace Engineering**
Follow-on to the NeurIPS 2026 RealPDE Track 1 UGP.

---

## 0. One-line goal

Reimplement the Aurora architecture from primitives in JAX, at small scale, and use it to
**isolate which properties of pretraining data produce the pretraining advantage** — on
Kolmogorov flow, where data is free and ground truth is exact.

---

## 1. Why this project, and why now

Aurora (Bodnar et al., *Nature* 2025) reports that its **pretrained model beats training from
scratch by 25%**. That single number motivates the whole foundation-model programme for physical
systems. But the paper cannot say *which part* of its million hours earned the 25%, because
pretraining costs 32×A100 for 150k steps and the eight source datasets differ in variables,
resolution, physics and volume all at once.

**That question is answerable on Kolmogorov flow.** You generate the data, so volume, diversity,
resolution and distribution shift are independent knobs you can turn one at a time.

It is also the exact question the UGP ended on. The competition evidence said the remaining gap was
a better *pretrained* backbone, and it could not be tested there: outside weights were banned, only
81 real trajectories existed, and — critically — the offline validation had **no predictive power**
for the main accuracy channel (fitted slope was negative). On synthetic data with exact ground
truth, none of those three obstacles exists.

---

## 2. Research questions

**RQ1 — Volume vs diversity.** At fixed total training samples, is it better to have many
trajectories from few flow regimes, or few trajectories from many? (Aurora conflates these.)

**RQ2 — Does the Perceiver encoder actually earn its place?** Its stated purpose is fusing datasets
with *different variables, resolutions and level counts*. Test it against the naive baselines
(pad-and-mask, or separate per-dataset encoders) on deliberately heterogeneous training sets.
*Direct relevance: in the UGP, joint simulation+real training returned exactly zero across three
variants, and one candidate explanation is that the simulation set carries (u,v,p) while the real
set carries (u,v) with p identically zero — precisely the mismatch this module exists to solve.*

**RQ3 — How far does pretraining transfer out of distribution?** Pretrain inside a Reynolds band,
fine-tune and test outside it. Measure the advantage as a function of distance beyond the band.

**RQ4 — Which architectural components matter?** Ablate shifted windows, U-Net depth, latent level
count L, and the pushforward trick, each against a matched baseline.

**RQ5 — Parameter-efficient adaptation.** Does LoRA on the backbone attention match full
fine-tuning in the small-data regime, and at what fraction of parameters?

---

## 3. Testbed: 2-D Kolmogorov flow

2-D Navier–Stokes with sinusoidal forcing, vorticity form, doubly periodic:

    d(omega)/dt + u . grad(omega) = nu * lap(omega) - alpha*omega + f,   f = -k_f*cos(k_f*y)

Pseudo-spectral in space, RK4 or ETDRK4 in time, 2/3-rule dealiasing.

* Grid 64×64 (development) and 128×128 (final runs).
* Control knobs: Reynolds number via `nu`, forcing wavenumber `k_f`, drag `alpha`.
* Discard an initial transient; store snapshots at a fixed stride.
* Store `(u, v, omega)`; optionally `p` so a variable-mismatch condition can be constructed for RQ2.

**Why this flow:** it is the standard 2-D turbulence benchmark, genuinely chaotic, cheap, and — the
decisive advantage over the UGP data — **noise-free**, so validation measures what it claims to.

---

## 4. Architecture (scaled-down Aurora)

Target **2–10M parameters**, not 1.3B. The architecture and data questions are separable from scale.

**Encoder (Perceiver).** Fields → P×P patches (P=4 at 64×64) → per-variable linear embedding to
dimension D. Sum embeddings across variables at a level; add level encoding. A Perceiver
cross-attention module maps a variable number of physical levels `C` to a **fixed L=3 latent
levels**. Add Fourier positional, patch-area and time encodings. Patch-area encoding is what permits
multi-resolution — keep it, RQ2 depends on it.

**Backbone (3-D Swin Transformer U-Net).** 3 stages down / 3 up with skips. Windowed self-attention,
**shifted windows on alternate blocks**, res-post-norm for stability. Start ~12 layers (Aurora uses
48); depth is an RQ4 knob.

**Decoder.** Perceiver expands L=3 back to `C` levels; per-variable linear heads map patches back to
the grid.

**Rollout.** One-step simulator `Phi(X^{t-1}, X^t) -> X^{t+1}`, applied autoregressively.
**Loss: mean absolute error**, per-variable weighted. (Aurora uses MAE, not MSE — worth carrying
over, and worth an ablation, since MAE is markedly more robust to noisy targets.)

---

## 5. Staged plan — each stage has an acceptance test

| # | Deliverable | Acceptance test |
|---|---|---|
| 1 | Pseudo-spectral Kolmogorov solver | energy spectrum shows the expected inertial-range scaling; energy budget balances to <1% |
| 2 | Dataset generator + loader | reproducible from seed; spectra stable across trajectories |
| 3 | Patch embed / unembed | round-trip reconstruction error ~machine precision |
| 4 | Perceiver level aggregator | inputs with C=3 and C=8 levels both map to L=3; decoder recovers either |
| 5 | Windowed + shifted-window attention | matches a reference Swin block on random input to 1e-5 |
| 6 | U-Net backbone with skips | overfits a single trajectory to near-zero loss |
| 7 | Full encoder→backbone→decoder | identity-ish at init; trains one-step prediction |
| 8 | Autoregressive rollout + MAE loss | stable 50-step rollout without blow-up |
| 9 | Pushforward trick | rollout drift measurably lower than without, at matched compute |
| 10 | LoRA fine-tuning | adapts to unseen `nu` with <1% of parameters trainable |
| 11 | **Ablation harness** | one config file → one run → one row of results |

Stages 1–8 are implementation. **Stages 9–11 are the contribution.**

---

## 6. The experiments

**E1 (RQ1) — volume vs diversity, at fixed total samples.**
A 3×3 grid: `n_regimes` ∈ {2, 8, 32} crossed with `traj_per_regime` chosen to hold
`n_regimes × traj_per_regime` constant. Fine-tune and test on a held-out regime.
*Report the pretraining advantage over from-scratch for each cell.*

**E2 (RQ2) — heterogeneous fusion.**
Build three training sets with deliberate mismatch: (a) mixed resolution 64 and 128; (b) mixed
variables, some trajectories `(u,v,p)`, some `(u,v)`; (c) both. Compare the Perceiver encoder
against pad-and-mask and against per-dataset encoders.

**E3 (RQ3) — out-of-distribution transfer.**
Pretrain on a Reynolds band; evaluate at increasing distance beyond it. Plot advantage vs distance.

**E4 (RQ4) — component ablations.**
Shifted windows on/off; depth 6/12/24; L ∈ {1, 3, 6}; pushforward on/off; MAE vs MSE.

**E5 (RQ5) — LoRA vs full fine-tune** across a sweep of fine-tuning-set sizes.

**Protocol, learned the hard way in the UGP:**
* Every run gets a **pre-registered prediction** written down before it is launched.
* **Hold out by regime, never by sample.** Correlated samples inside a trajectory make random splits
  meaningless.
* Three seeds minimum per cell; report spread, not just the mean.
* **Any in-sample fit is assumed inflated until scored on disjoint data.** In the UGP a stratified
  calibration looked like +0.041 in-sample and measured **−0.059** out-of-sample.

---

## 7. Deliverables

1. A clean JAX implementation, one module per stage, each independently tested.
2. The dataset generator, reproducible from seeds.
3. The ablation harness and results tables.
4. A write-up answering RQ1–RQ5.

## 8. Scope and risks

* **Semester-scale**, not weeks. Stages 1–8 are the bulk of the coding; 9–11 the bulk of the compute.
* **Main risk: compute.** The E1 grid is 9 cells × 3 seeds × (pretrain + fine-tune). Keep the model
  small and the grid at 64×64 until the harness is proven.
* **Second risk: the Swin block.** Windowed + shifted attention is the fiddliest piece; budget time,
  and test it against a reference implementation rather than by eye.
* **Explicit non-goal:** matching Aurora's absolute accuracy. The question is *what the data buys*,
  which is answerable at small scale and is not answerable at Aurora's.
