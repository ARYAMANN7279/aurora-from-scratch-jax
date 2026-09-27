# Experiment E3 Spec: How far does pretraining transfer out of distribution? (RQ3)

**Status: DRAFT for advisor review.** Frozen on approval; later changes go in dated amendments.
**Question:** pretrain on a band of Reynolds numbers. How do the zero-shot error and the
pretraining advantage change with distance beyond the band?
**Tools:** the E1 dataset `e1_v1` (no new data), the Stage 7 model, and the Stage 11 harness.
**Written:** 27 Sep 2026.

---

## 1. Design
* **Pretraining band:** Re ∈ {125, 160, 200, 250} × k_f ∈ {3, 4, 5, 6}, seeds 0–3, giving 64
  trajectories.
* **Test regimes (k_f = 4, beyond the band):** Re ∈ {300, 350, 450, 560, 700}. The distance to the
  band edge is `d = ln(Re / 250)` = 0.18, 0.34, 0.59, 0.81, 1.03, so the far end is 2.8× the band's
  top Reynolds number. All five are listed as `test` in the pretrain configs, so the harness
  refuses any leak.
* **Runs**, all with the Stage 7 default model, MSE loss, batch 16 and cosine decay:

| Run | Data | Steps | lr |
|---|---|---|---|
| pretrain(s) | the band | 20,000 | 1e-3 |
| finetune(Re, s) | pretrain(s) weights and normalisation; seeds 0–1 of the test regime | 2,000 | 1e-4 |
| scratch(Re, s) | seeds 0–1 of the test regime | 22,000 | 1e-3 |

  With seeds s ∈ {0, 1, 2}: 3 pretrain + 15 fine-tune + 15 scratch = **33 runs**, about 3 GPU-h.
  Every evaluation uses seeds 2–3 of the test regime, and measures the one-step relative error and
  the 20-step rollout drift from t₀ = 50.
* **Metrics:**
  * zero-shot ratio z(d): pretrain error / persistence error;
  * `adv(d) = 1 − e_ft / e_scratch`, paired by seed, mean ± SE over the 3 seeds.

## 2. Pre-registered predictions

| # | Prediction |
|---|---|
| E3-P1 | z(d) rises with d: Spearman ρ ≥ 0.9 over the 5 regimes (seed means) |
| E3-P2 | adv(d) > 2 SE at every distance: pretraining still helps at 2.8× the band's Re |
| E3-P3 | adv falls with distance: adv(700) < adv(300), with the gap > 2 SE |
| E3-P4 | z(700) ≤ 0.8: zero-shot still beats persistence. Anchor: Stage 10 found 0.58× at Re 1000 from Re 100/300 |

## 3. Report
* Plots of z(d) and adv(d) with error bars.
* A comparison with E1's (300, 4) cell, which was pretrained on sets that enclose the target, to
  separate interpolation from extrapolation.

## 4. Open points for review
1. Extrapolation only goes upward in Re. A downward arm (a band at Re 450–700, tested down to 125)
   would show whether transfer is symmetric in ln Re.
2. k_f is held at 4 for the test regimes. A k_f-distance arm (k_f 7, 8 at Re 200) would need new
   data.
