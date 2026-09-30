# Stage 10 Spec: LoRA fine-tuning

**Plan acceptance test (PROJECT_PLAN.md §5):** adapts to an unseen ν with < 1% of parameters
trainable.
**Written:** 26 Sep 2026, before any Stage 10 code and before any Stage 7 result.
§§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Design (following Aurora's LoRA on the backbone attention)

* For each Swin3D block's `qkv` (D×3D) and `proj` (D×D) linear maps, the frozen weight W gets a
  low-rank update: `W' = W + (α/r) · A B`, with `A ∈ R^{D_in×r}` (normal, std 1/√D_in) and
  `B ∈ R^{r×D_out}` initialised to **zero**, so the adapted model equals the base model at the
  start. Default r = 4, α = r (scale 1).
* Only the LoRA factors train. Every base parameter is frozen with an Optax `multi_transform`
  (`set_to_zero` for the base).
* **Trainable fraction** for the Stage 7 default model at r = 4:
  `Σ_blocks 4·(w + 3w) + 4·(w + w) = 24 · Σ_blocks w`. The block widths are
  64, 64, 128, 128, 256, 256 (encoder) and 256, 256, 128, 128, 64, 64 (decoder), so
  Σw = 1792, which gives **43,008 parameters ≈ 0.84%** of 5,106,688.

## 2. Gates — `tests/test_stage10_gates.py` (float64)

| Gate | Test | Pass if |
|---|---|---|
| L1 | with B = 0, the LoRA model equals the base model | exact |
| L2 | merged weights `W + (α/r)AB` give the same output as the unmerged forward | ≤ 1e-12 |
| L3 | after one optimiser step, every base parameter is bitwise unchanged and the LoRA B factors changed | exact |
| L4 | trainable-parameter count = 43,008 for the default model at r = 4 | exact |
| L5 | gradient with respect to the LoRA factors vs central finite differences (h = 1e-4, mixed tolerance) | ≤ 1e-6 |

## 3. Measurement (small-data adaptation to an unseen regime)

* **Base model:** the Stage 7 MSE model (trained on Re 100 and 300 only).
* **Target regime:** re1000_kf4, never seen in training. Fine-tuning data: `acc_v1` seeds 0–1
  (the small-data regime, ~1000 windows). Evaluation: seeds 6–7. 64². The base model's
  normalisation is reused.
* **Arms (same key, batch 8, MSE, cosine decay, 1000 steps):**
  * zero-shot (no fine-tuning);
  * LoRA r = 4, lr 1e-3;
  * full fine-tuning, lr 1e-4.
* **Metric:** one-step relative error on the evaluation windows, and its ratio to persistence.

| # | Measurement | Prediction |
|---|---|---|
| A1 | trainable fraction at r = 4 | 0.84% (< 1%) |
| A2 | LoRA error / zero-shot error | ≤ 0.8 (adapts) |
| A3 | LoRA error / full-fine-tune error | ≤ 1.2 (close to full fine-tuning) |
| A4 | zero-shot error / persistence on Re1000 | > the Stage 7 in-distribution ratio at Re300 (the regime shift costs accuracy) |

**Stage 10 acceptance:** L1–L5 pass, and A1 and A2 HIT.

---

## Amendments (after results; the original verdicts stand in the report)

**A1 (26 Sep, after the first gate run): L5 step size.** With h = 1e-4, L5 **failed** at 1.8e-6.
Diagnostic (5 entries, h ∈ {1e-3, 1e-4, 1e-5, 1e-6}): the error grows as h shrinks, so it is
finite-difference roundoff, not a gradient error. The LoRA gradients are tiny (median |g| = 1.5e-6),
so the roundoff floor at h = 1e-4 (≈ ε·loss/h ≈ 2e-12 absolute) exceeds the tolerance. At
h = 1e-3 all 5 entries agree to ≤ 7.1e-8. From now on L5 uses h = 1e-3 (mixed tolerance
unchanged).

**F1 (30 Sep 2026, pre-registered before running): is the LoRA miss a rank limit?**
Stage 10's LoRA (r = 4) improved one-step error by 0.5% over zero-shot, against 3.0% for full
fine-tuning. F1 repeats the exact Stage 10 LoRA arm (same Stage 7 model, two Re 1000
trajectories, 1000 steps, lr 1e-3, seed 0) at ranks r ∈ {1, 16, 64}, which train 0.21% / 3.4% /
13.5% of the parameters. Command: `finetune.py --arm lora --rank r`. It runs on CPU, like Stage 10.

| # | Prediction |
|---|---|
| F1-P1 | the gain over zero-shot does not fall with rank: gain(1) ≤ gain(4) ≤ gain(16) ≤ gain(64), within 0.1 points |
| F1-P2 | even r = 64 stays below half of full fine-tuning's gain: gain(64) < 1.5%. If so, the limit is *which* layers LoRA adapts (attention only), not its rank |

The Stage 10 verdict (A2 MISS) stands whatever F1 finds. F1 only explains it.
