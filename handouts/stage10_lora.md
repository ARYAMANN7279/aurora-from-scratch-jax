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

**F1-A1 (30 Sep 2026, before any F1 run): the base model is a retrain.** The Stage 7 parameter file
existed only on the machine that ran Stages 7–10 (parameters stay out of git), and that machine is
gone, so "the same Stage 7 model" is not available. F1 runs on a retrain instead:
* **Base:** the Stage 7 command unchanged (`train_onestep.py --loss mse`, 3000 steps), on the lab
  machine's CPU with the lab machine's copy of `acc_v1`. That copy was generated separately (Re 100
  and 300 on the lab machine's CPU, Re 1000 on its GPU; same generator, regimes and seeds). Whether
  it is bitwise identical to the original copy cannot be checked, because the original checksums
  were not kept. The retrain's one-step errors at Re 100 / 300 are reported next to Stage 7's
  (0.070 / 0.146) as a check.
* **Baselines re-run:** the base model and the Re 1000 data both differ from Stage 10's, so Stage
  10's numbers (zero-shot 0.2098; gain 0.5% for LoRA r = 4 and 3.0% for full fine-tuning) are not a
  valid baseline for F1. F1 re-runs zero-shot, LoRA r = 4 and full fine-tuning on the retrain, with
  the Stage 10 settings unchanged.
* **Scoring, fixed now:** gain(arm) = 1 − e(arm) / e(zero-shot), all on the retrain. F1-P1 uses
  the re-run gain(4). F1-P2 keeps its registered bar, gain(64) < 1.5%. The ratio
  gain(64) / gain(full), which that bar was meant to keep below one half, is reported beside it and
  does not change the verdict.
* **Commands** (all on CPU; the base's result goes to `experiments/stage7/results/onestep_mse_retrain.json`,
  F1's to `experiments/stage10/results/f1_*.json`, so no committed result is overwritten):
  ```
  train_onestep.py --loss mse --out onestep_mse_retrain
  finetune.py --arm zeroshot --out f1_zeroshot
  finetune.py --arm lora --rank r --out f1_lora_r<r>      # r = 4, 1, 16, 64
  finetune.py --arm full --out f1_full
  ```

**F2 (4 Oct 2026, pre-registered before any F2 fine-tuning run): adaptation to viscosities farther from
the training data.**
* **Why.** Stage 10's target, Re 1000 (ν 3.3× below the training band's lowest), left little to adapt:
  zero-shot was already 0.58× persistence, and full fine-tuning gained only 2.7–3.0% (Stage 10, F1;
  E5 agrees). So the acceptance bar (LoRA ≤ 0.8× zero-shot) was out of reach for any method.
* **Targets:**
  * **Re 3000 (primary):** ν 10× below the band's lowest (Re 300). In log ν that is three times as far
    as Re 1000.
  * **Re 30 (second):** ν 3.3× above the band's highest (Re 100). That is as far as Re 1000, but on the
    viscous side, and the flow there is different:
    * A pilot run of the solver alone (`pilot_viscosity.py` → `results/f2_pilot.json`) found the flow
      steady at Re 10, 15 and 20, weakly unsteady at Re 30 (the vorticity changes 2.9% per frame,
      against 26% at Re 100), and chaotic at Re 100.
    * Farther out on the viscous side there is nothing to forecast.
  * **Re 1000:** Stage 10's target, rerun with the same protocol for comparison.
* **Data:** a new dataset, `s10f2_v1` (`experiments/stage10/configs/s10f2_v1.json`).
  * Re 30 on a 128² grid and Re 3000 on 1024², k_f 4, seeds 0, 1, 6 and 7.
  * Otherwise `acc_v1`'s settings: spin-up 50, 100 time units recorded, a frame every 0.2, at 64² and
    128².
  * Grids are checked with Stage 2's rule (`resolution_f2.py`): F_64(1024 vs 2048) ≤ 1e-3 at t = 1
    for Re 3000, and 128 vs 256 for Re 30. If a check fails, the trajectories are regenerated on the
    finer grid before any fine-tuning.
  * Re 3000 is generated on idle lab GPUs (measured: 0.44 h per trajectory at 1024²), Re 30 on the
    CPU.
* **Protocol:** Stage 10's exactly.
  * The F1 base model (the Stage 7 retrain) and its normalisation; fine-tuning on the target's seeds
    0–1, evaluation on 6–7.
  * Batch 8, MSE, cosine decay, 1000 steps. LoRA r = 4 (0.84% trainable) at lr 1e-3; full fine-tuning
    at lr 1e-4.
  * Three seeds (0–2) per fine-tuning arm. `--seed` sets the LoRA initialisation and the batch order,
    and seed 0 is Stage 10's. Zero-shot needs no seed.
  * Every run is on an idle lab GPU, with matmul precision `highest`. F1's CPU numbers at Re 1000 are
    reported beside the rerun.
  * `finetune.py` gains `--dataset`, `--target` and `--seed`. With the defaults it reproduces F1's
    zero-shot result bitwise (checked on the lab machine's CPU).
* **Measure:** gain(arm) = 1 − e(arm) / e(zero-shot), the one-step error on the target's evaluation
  seeds, as mean ± SE over the seeds.
* **Acceptance, fixed now.** Stage 10's test (A1 and A2) is evaluated at Re 3000, the farther
  viscosity. It is met if LoRA r = 4's mean error there is ≤ 0.8× zero-shot. Re 30 and Re 1000 are
  reported beside it but do not decide it. The Stage 10 verdict at Re 1000 stands either way.

| # | Prediction |
|---|---|
| F2-P1 | At Re 3000 the base transfers worse than at Re 1000: zero-shot / persistence is higher at Re 3000 |
| F2-P2 | There is more to gain at Re 3000: full fine-tuning's mean gain is larger at Re 3000 than at Re 1000 |
| F2-P3 | Even so, LoRA does not reach the acceptance bar at Re 3000: its mean gain is below 20% |
| F2-P4 | On the viscous side the base fails: at Re 30, zero-shot is no better than persistence (ratio ≥ 1) |
| F2-P5 | There, LoRA adapts at the acceptance level: mean gain ≥ 20% at Re 30 |
| F2-P6 | And adapting attention alone is not the limit there: LoRA's mean gain at Re 30 is at least half of full fine-tuning's |

Scored by `score_f2.py`, written before any F2 fine-tuning run.

**F3 (4 Oct 2026, pre-registered before any F3 measuring run): can anything reach Stage 10's bar?**
* **Why.** F2 left two questions open:
  * Can any model of this architecture reach the bar (error ≤ 0.8× zero-shot) at Re 3000 or Re 1000
    when given more target data and more steps? If so, can LoRA?
  * How much of the error is out of reach for any model that sees only the 64² band?

  The base matters here. It was trained for 3000 steps at batch 8, about two passes over its 12
  trajectories, so ample training on the target may beat its zero-shot error by more than
  adaptation alone would. (Correction, 4 Oct, after the runs: about four passes, not two: 24,000
  samples over 5,988 one-step pairs. No prediction depends on the figure.)
* **Data: `s10f3_v1`** (`experiments/stage10/configs/s10f3_v1.json`): Re 3000, k_f 4, 1024², seeds 0–7.
  * Seeds 0, 1, 6 and 7 are copies of `s10f2_v1`'s.
  * Seeds 2–5 are new, generated on lab card 3. Generation started before this commit; it is solver
    output and no prediction depends on it.
  * Re 1000 uses `acc_v1`'s seeds 0–7, as before.
* **Arms.** Each trains on the target's seeds 0–5, three times Stage 10's data, and is evaluated on
  seeds 6–7. Batch 8, MSE, cosine decay to 1% of the learning rate:
  * **scratch:** every parameter from a random initialisation (key = seed), lr 1e-3, 20,000 steps.
    This is Stage 7's recipe, with about as many samples as E5's scratch arm. It uses the base's
    normalisation, so its error is measured exactly as the other arms'.
  * **full:** the F1 base, every parameter, lr 1e-4, 10,000 steps (10× Stage 10).
  * **lora:** r = 4 on attention, lr 1e-3, 10,000 steps.
  * Seeds 0–2 for each arm, with an evaluation every 1000 steps.
  * Zero-shot is F2's: the same base and the same evaluation files. F3 reruns it on `s10f3_v1` as a
    check, which must agree.
  * All runs on lab card 3, with matmul precision `highest`.
  * `finetune.py` gains `--arm scratch`, `--train-seeds` and `--eval-every`. With the defaults it
    gives the same results as before (checked on the lab machine's CPU against F1's zero-shot and a
    short LoRA run).
* **Measure:** gain = 1 − e / e(zero-shot), on the final error, as mean ± SE over seeds.
  * The **ceiling** is the better of scratch and full, by mean final error.
  * The best error along each curve is chosen on the evaluation seeds, so it is reported only as an
    optimistic figure.
* **Floor (`closure_f3.py`):** how much of one frame's change in the 64² band depends on the scales
  finer than the band.
  * For Re 100, 300, 1000 and 3000, the exact solver runs from the post-spin-up state of evaluation
    seeds 6–7. Every time unit (20 samples per seed) it advances one frame twice: from the full
    state, and from the state's 64² band.
  * The measure is the error over persistence in the four normalised channels, as `finetune.py`
    measures it.
  * It is the error of a predictor that knows the equations exactly but sees only the current band.
    A learned model also sees the previous frame, so this is a reference, not a strict bound.
* **Reading, fixed now:**
  * ceiling < 20% at a target: no method reaches Stage 10's bar there with up to six trajectories at
    these budgets;
  * ceiling ≥ 20% but LoRA < 20%: more data makes the bar reachable, but not for attention-only LoRA
    at r = 4;
  * LoRA ≥ 20%: LoRA meets the bar with three times the data and ten times the steps, so the
    Stage 10 miss is the registered data budget.
  * The Stage 10 verdict stands in every case. Its registered test is two trajectories and 1000
    steps at Re 1000, and F2 added Re 3000.

| # | Prediction |
|---|---|
| F3-P1 | At Re 3000 the ceiling reaches the bar: its gain is ≥ 20% |
| F3-P2 | At Re 3000 LoRA stays below the bar: its gain is < 20% |
| F3-P3 | At Re 3000 pretraining still pays with six trajectories: full's mean error is below scratch's |
| F3-P4 | At Re 1000 the ceiling reaches the bar: its gain is ≥ 20% |
| F3-P5 | At Re 1000 LoRA stays below the bar: its gain is < 20% |
| F3-P6 | The base is far from the floor at Re 3000: the closure ratio is below half the base's zero-shot ratio (0.64) |
| F3-P7 | The floor rises with Re: the closure ratio grows from Re 100 to 300, 1000 and 3000 |

Scored by `score_f3.py`, written before any F3 measuring run.
* **Smoke tests on the lab machine's CPU** ran after the predictions above were written and before
  this commit. None changed them:
  * the default zero-shot reproduces F1's result exactly, and a 20-step LoRA run matches the old
    code bit for bit;
  * a 20-step scratch run;
  * `closure_f3.py` at Re 100 with 2 samples per seed (ratio 4.5e-4).

Commands:
```
generate.py --config experiments/stage10/configs/s10f3_v1.json --only re3000_kf4:2 re3000_kf4:3 re3000_kf4:4 re3000_kf4:5
generate.py --config experiments/stage10/configs/s10f3_v1.json --finalize
closure_f3.py
finetune.py --arm zeroshot --dataset s10f3_v1 --target re3000_kf4 --out f3_re3000_zeroshot
finetune.py --arm scratch --steps 20000 --seed <s> --dataset s10f3_v1 --target re3000_kf4 --train-seeds 0 1 2 3 4 5 --eval-every 1000 --out f3_re3000_scratch_s<s>
finetune.py --arm full    --steps 10000 ... --out f3_re3000_full_s<s>      # same flags otherwise
finetune.py --arm lora    --steps 10000 ... --out f3_re3000_lora_s<s>
# Re 1000: --dataset acc_v1 --target re1000_kf4 and f3_re1000_* names, otherwise the same
```

**F4 (5 Oct 2026, pre-registered before any F4 measuring run): the base or the learning rate?**
* **Why.** In F3, training from scratch (lr 1e-3, 20,000 steps) beat full fine-tuning of the F1 base
  (lr 1e-4, 10,000 steps) by 16–21% (F3-P3 MISS). Two causes were not separated:
  * the base saw only about four passes over its data;
  * fine-tuning used a tenth of scratch's learning rate, and half its steps.
* **Bases:**
  * **B3k:** the F1 base (Stage 7's recipe, 3000 steps, about four passes).
  * **B30k:** the same recipe for 30,000 steps (about 40 passes), trained on lab card 3:
    `train_onestep.py --loss mse --steps 30000 --params-name onestep_mse_long --out onestep_mse_long`.
    `train_onestep.py` gains `--params-name`, whose default leaves its behaviour unchanged, so the
    F1 base's files are not touched. `finetune.py` gains `--base`; its default is the F1 base.
* **Arms.** Full fine-tuning on the target's seeds 0–5 for 20,000 steps (scratch's budget), batch 8,
  cosine decay, seeds 0–2, at Re 3000 (`s10f3_v1`) and Re 1000 (`acc_v1`), all on lab card 3:
  * **b3k_lo** and **b3k_hi:** B3k at lr 1e-4 and at lr 1e-3;
  * **b30k_lo** and **b30k_hi:** B30k at lr 1e-4 and at lr 1e-3;
  * **B30k zero-shot**, and **b30k_lora:** LoRA r = 4 on B30k in Stage 10's registered setting
    (seeds 0–1, 1000 steps, lr 1e-3), seeds 0–2;
  * **reference:** F3's scratch arm (same targets, budget and card).

| # | Prediction (each must hold at both targets) |
|---|---|
| F4-P1 | With B3k, the higher learning rate does better: e(b3k_hi) < e(b3k_lo) |
| F4-P2 | With the learning rate and the steps matched, B3k is no worse a start than random: e(b3k_hi) ≤ e(scratch) |
| F4-P3 | At Stage 10's learning rate, the longer-trained base does better: e(b30k_lo) < e(b3k_lo) |
| F4-P4 | At Stage 10's learning rate, the longer-trained base beats scratch: e(b30k_lo) < e(scratch) |
| F4-P5 | The longer-trained base transfers better zero-shot than B3k |
| F4-P6 | LoRA on B30k in Stage 10's setting still misses the bar: its gain over B30k's zero-shot is < 20% |

Scored by `score_f4.py`, written before any F4 measuring run.
* **Smoke tests before this commit** (lab CPU), after the predictions were written. None changed them:
  * the default zero-shot reproduces F1's result exactly;
  * a 2-step `train_onestep.py --params-name` run wrote only its own files, and the F1 base's files are
    unchanged.
* **Reading, fixed now:**
  * if P1 and P2 hold and P4 fails, F3's result came from the learning rate, not the base;
  * if P3 and P4 hold, the base's training matters: a well-trained base, fine-tuned at Stage 10's
    rate, beats scratch;
  * both can hold.
  * The Stage 10 verdict stands in every case.
* **Commands:**
  ```
  train_onestep.py --loss mse --steps 30000 --params-name onestep_mse_long --out onestep_mse_long
  finetune.py --arm zeroshot --base onestep_mse_long --dataset s10f3_v1 --target re3000_kf4 --out f4_re3000_b30k_zeroshot
  finetune.py --arm full [--lr 1e-3] [--base onestep_mse_long] --steps 20000 --seed <s> --dataset s10f3_v1 --target re3000_kf4 --train-seeds 0 1 2 3 4 5 --eval-every 1000 --out f4_re3000_<arm>_s<s>
  finetune.py --arm lora --base onestep_mse_long --seed <s> --dataset s10f3_v1 --target re3000_kf4 --out f4_re3000_b30k_lora_s<s>
  # Re 1000: --dataset acc_v1 --target re1000_kf4 and f4_re1000_* names, otherwise the same
  ```
