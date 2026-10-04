# Stage 10 Report: LoRA fine-tuning

Spec: `handouts/stage10_lora.md` (including amendment A1). Code: `src/aurora/lora.py`.
Tests: `tests/test_stage10_gates.py`. Measurement: `finetune.py` → `results/finetune_*.json`.

## Verdict

**The plan's acceptance test is not met.** LoRA is implemented correctly: all gates pass, and it
trains 43,008 parameters, 0.84% of the model. But in the pre-registered small-data setting (two
Re1000 trajectories, 1000 steps) **neither LoRA nor full fine-tuning adapts the Stage 7 model
much to the unseen regime**. LoRA improves one-step error by 0.5%, full fine-tuning by 3.0%. A2
(LoRA ≤ 0.8× zero-shot) is a MISS. An exploratory LoRA run at 10× the learning rate also improves
only 0.7%, so the learning rate is not the bottleneck.

What the stage does show is that **zero-shot transfer is already reasonable**. The Stage 7 model,
trained only on Re 100 and 300, predicts the never-seen Re1000 at 0.58× persistence, against 0.46×
in-distribution at Re300.

**Follow-up F1 (pre-registered 30 Sep 2026; the verdicts above stand).** A LoRA rank sweep on a
retrained base model. The gain grows with rank but stays small: 0.2% at r = 1 and 1.0% at r = 64,
against 2.7% for full fine-tuning. Even r = 64, with 13.5% of the parameters trainable, gets only 37%
of full fine-tuning's gain, so rank is not what holds LoRA back here (F1-P1 and F1-P2 HIT). See
"Follow-up F1" below.

**Follow-up F2 (pre-registered 4 Oct 2026; the verdicts above stand).** The same test at viscosities
farther from the training data, 3 seeds. All 6 predictions HIT, and **the acceptance test is still
not met at Re 3000**, ν 10× below the training band: LoRA gains 0.46% and full fine-tuning 3.2%. At
Re 30, ν 3.3× above it, the base model is worse than persistence (1.92×). There LoRA cuts the error
by 71% and full fine-tuning by 82%. An exploratory check of the base model shows the difference:
* at Re 30 its error is largely systematic, a predicted change 2.2× too large;
* at Re 1000 and Re 3000 it is the in-distribution kind, a direction error that grows with Re.
Fine-tuning from two trajectories fixes the first kind and barely touches the second. See
"Follow-up F2" below.

## Gates

| Gate | Measured | Bar |
|---|---|---|
| L1 with B = 0, the LoRA model equals the base model | exact | exact |
| L2 merged vs unmerged forward | 7.2e-16 | ≤ 1e-12 |
| L3 only the LoRA factors change; every base leaf bitwise unchanged | exact | exact |
| L4 trainable count | 43,008 (0.842% of 5,106,688) | exact |
| L5 LoRA gradient vs finite differences (A1: h = 1e-3, mixed) | 1.4e-7 | ≤ 1e-6 |

**L5 history:** at the pre-registered h = 1e-4 the check **failed** (1.8e-6) at the
finite-difference roundoff floor; the LoRA gradients are tiny. See amendment A1.

## Predictions

Target regime re1000_kf4 (never seen in training). Fine-tuning data: seeds 0–1. Evaluation:
seeds 6–7. One-step relative error.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| A1 | trainable fraction 0.84% (< 1%) | 0.842% | **HIT** |
| A2 | LoRA / zero-shot ≤ 0.8 | 0.2087 / 0.2098 = 0.995 | **MISS** |
| A3 | LoRA / full fine-tune ≤ 1.2 | 0.2087 / 0.2035 = 1.03 | **HIT** (not informative: full fine-tuning barely improves either) |
| A4 | zero-shot / persistence at Re1000 > Stage 7 in-distribution Re300 ratio (0.463) | 0.581 | **HIT** |

| Arm | Trainable | lr | Error (ratio to persistence) | Curve (every 250 steps) |
|---|---|---|---|---|
| zero-shot | 0 | — | 0.2098 (0.581) | — |
| LoRA r = 4 | 43,008 | 1e-3 | 0.2087 (0.578) | 0.2090, 0.2087, 0.2087, 0.2087 |
| full | 5,106,688 | 1e-4 | 0.2035 (0.564) | 0.2067, 0.2052, 0.2038, 0.2035 |
| LoRA r = 4 (exploratory) | 43,008 | 1e-2 | 0.2084 (0.577) | — |

## Reading

The adaptation gap is not specific to LoRA. With two trajectories, 1000 steps, and a
one-step-MSE objective, even full fine-tuning moves the error only 3%. Either the Re1000 error is
dominated by what cannot be learned from so little data, or the model's Re1000 behaviour is already
close to what this fine-tuning signal can reach. The next steps belong to experiment E5 (LoRA vs
full, swept over fine-tuning set size), not to this stage:
* a larger fine-tuning set (the E5 sweep);
* an in-distribution Re1000 model as the ceiling;
* LoRA on the MLPs and embeddings as well as attention.

## Follow-up F1: LoRA rank sweep (pre-registered 30 Sep 2026)

Spec: F1 and amendment F1-A1 in `handouts/stage10_lora.md`, both committed before any run.
Results: `results/f1_*.json`; scored by `score_f1.py` → `results/f1_analysis.json`. One run per arm
(seed 0), as in Stage 10, so there are no standard errors.

**Base model: a retrain (F1-A1).** The original Stage 7 parameters are gone, so the Stage 7 command
was rerun unchanged on the lab machine's CPU, with that machine's copy of `acc_v1`. That copy is not
bitwise identical to the original: persistence on the Re 300 evaluation seeds is 0.309 against
0.315. The retrain reproduces Stage 7 closely:

| | Re 100 error (ratio) | Re 300 error (ratio) |
|---|---|---|
| Stage 7 | 0.0702 (0.322) | 0.1457 (0.463) |
| retrain | 0.0720 (0.330) | 0.1450 (0.470) |

Every F1 arm, including the re-run baselines, uses this retrain; Stage 10's own numbers are not
mixed in. gain = 1 − error / zero-shot error, on the Re 1000 evaluation seeds 6–7.

| Arm | Trainable | Error (ratio) | Gain |
|---|---|---|---|
| zero-shot | 0 | 0.2128 (0.584) | — |
| LoRA r = 1 | 10,752 (0.21%) | 0.2123 | 0.21% |
| LoRA r = 4 | 43,008 (0.84%) | 0.2120 | 0.37% |
| LoRA r = 16 | 172,032 (3.4%) | 0.2115 | 0.61% |
| LoRA r = 64 | 688,128 (13.5%) | 0.2107 | 0.98% |
| full | 5,106,688 (100%) | 0.2071 | 2.69% |

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| F1-P1 | the gain does not fall with rank: gain(1) ≤ gain(4) ≤ gain(16) ≤ gain(64), within 0.1 points | 0.21 → 0.37 → 0.61 → 0.98% | **HIT** |
| F1-P2 | gain(64) < 1.5% | 0.98% | **HIT** |

* **Rank helps, but little.** Each 4× in rank adds 0.16–0.37 points. At r = 64 LoRA trains 13.5% of
  the model and gets 37% of full fine-tuning's gain, below the half that F1-P2 was written to test.
* **So the Stage 10 miss is not a rank limit.** What remains is where LoRA adapts (attention only)
  and how little any arm can learn from two trajectories. Even full fine-tuning gains only 2.7%
  here (3.0% in Stage 10), in line with E5, where it gains at most 5.3% from six. LoRA on the MLPs
  was not tested.
* **The re-run baselines agree with Stage 10:** zero-shot 0.584× persistence (Stage 10: 0.581×),
  LoRA r = 4 gains 0.37% (0.52%), full fine-tuning 2.69% (3.02%).
* Ran on 4 cores of the lab machine's CPU: the retrain took 0.41 s per step (26 min; Stage 7 took
  1.03 s per step on the original 4-core machine), each LoRA run about 6 min, full fine-tuning 7.5 min.

## Follow-up F2: farther viscosities (pre-registered 4 Oct 2026)

Spec: F2 in `handouts/stage10_lora.md`, committed before any F2 fine-tuning run (0196e65). Results:
`results/f2_*.json`, scored by `score_f2.py` → `results/f2_analysis.json`. Where each step ran, and
the software: `results/f2_lab_env.json`. Exploratory check: `diagnose_f2.py` →
`results/f2_diagnosis.json`.

**Data: `s10f2_v1`.** Re 30 on a 128² grid and Re 3000 on 1024², k_f 4, seeds 0, 1, 6 and 7, each with
501 frames at 64² and 128². The manifest verified all 8 trajectories' checksums.
* **Both grids pass Stage 2's rule** (F_64 ≤ 1e-3 at t = 1, `results/f2_resolution_*.json`):

  | Regime | Registered check | One grid coarser |
  |---|---|---|
  | Re 3000 | 1024 vs 2048: 5.6e-9 | 512 vs 1024: 1.9e-4 |
  | Re 30 | 128 vs 256: 2.3e-12 | 64 vs 128: 1.2e-4 |

* **The two flows are far apart.** On the evaluation seeds, persistence's error is 0.037 at Re 30
  (weakly unsteady, as the pilot found) and 0.40 at Re 3000, against 0.36 at Re 1000.
* **Generation:**
  * Re 3000 took 53 min for two trajectories on lab card 3, as measured for the spec. Card 2
    needed 6.6 h for the other two: it kept reaching 87 °C, so the lane paused it 185 times and
    restarted it 25 times from its checkpoint.
  * Re 30 took 25 s per trajectory on 4 CPU cores.

**Protocol as registered.** All 21 fine-tuning runs ran on lab card 3 (an RTX PRO 6000 Blackwell)
with matmul precision `highest`. A LoRA run took 8 s and a full one 12 s. The first of each, which
compiled the programs, took 2.3–2.4 min.

gain = 1 − error / zero-shot error, on each target's evaluation seeds 6–7, as mean ± SE over seeds 0–2.

| Target | ν against the training band | Persistence | Zero-shot (ratio) | LoRA r = 4: error, gain | Full: error, gain |
|---|---|---|---|---|---|
| Re 30 | 3.3× above | 0.0369 | 0.0707 (1.92) | 0.0202, **71.4% ± 0.1%** | 0.0124, 82.5% ± 0.02% |
| Re 1000 (rerun) | 3.3× below | 0.3645 | 0.2128 (0.58) | 0.2120, 0.35% ± 0.01% | 0.2071, 2.70% ± 0.01% |
| Re 3000 | 10× below | 0.4041 | 0.2588 (0.64) | 0.2576, **0.46% ± 0.01%** | 0.2506, 3.20% ± 0.01% |

* **The seeds barely matter.** Gains by seed (0, 1, 2):
  * Re 3000: LoRA 0.47, 0.45 and 0.47%; full 3.21, 3.18 and 3.20%;
  * Re 30: LoRA 71.5, 71.2 and 71.5%; full 82.5% for every seed.
* **The rerun reproduces F1.** The lab GPU's Re 1000 zero-shot error matches F1's CPU value to 2e-8
  (relative). With seed 0, which is F1's, LoRA gains 0.3655% on both machines, and full fine-tuning
  2.687% on the GPU against 2.686% on the CPU.
* **The curves have flattened by step 1000.** At Re 30, LoRA's error falls from 0.0203 to 0.0202
  over the last 250 steps.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| F2-P1 | zero-shot / persistence higher at Re 3000 than at Re 1000 | 0.641 vs 0.584 | **HIT** |
| F2-P2 | full fine-tuning's mean gain larger at Re 3000 than at Re 1000 | 3.20% vs 2.70% | **HIT** |
| F2-P3 | LoRA's mean gain at Re 3000 below 20% | 0.46% | **HIT** |
| F2-P4 | zero-shot no better than persistence at Re 30 (ratio ≥ 1) | 1.92 | **HIT** |
| F2-P5 | LoRA's mean gain at Re 30 ≥ 20% | 71.4% | **HIT** |
| F2-P6 | LoRA's mean gain at Re 30 ≥ half of full fine-tuning's | 71.4% vs 82.5% | **HIT** |
| Acceptance | at Re 3000, LoRA's mean error ≤ 0.8× zero-shot (0.84% trainable) | 0.995× | **not met** |

**Readings**
* **Farther into turbulence there is little more to adapt.**
  * At Re 3000 the base transfers a little worse than at Re 1000: 0.64× persistence, against 0.58×.
  * Full fine-tuning gains half a point more (P2 holds by that margin). LoRA gains under half a
    percent, as at Re 1000.
* **The viscous side is a different case.**
  * At Re 30 the base model is worse than persistence.
  * LoRA cuts its error by 71%, to 0.55× persistence. Full fine-tuning cuts 82%, to 0.34×, about the
    base's in-distribution level at Re 100 (0.33×).
  * By gain, LoRA gets 87% of full fine-tuning's with 0.84% of the parameters. By error, it stays
    1.6× above full fine-tuning (0.0202 against 0.0124).
  * So attention-only LoRA at r = 4 can adapt this model when there is something for it to fix.
* **Exploratory: what kind of error the base makes.** Not a pre-registered prediction.
  * `diagnose_f2.py` splits the base's one-step error on each regime's evaluation seeds. It compares
    the predicted change d = x̂ − x₀ with the true change d* = x₁ − x₀, in normalised variables over
    all four channels:
    * the size, amp = ‖d‖ / ‖d*‖;
    * the direction, corr = ⟨d, d*⟩ / (‖d‖ ‖d*‖).
  * Error / persistence then follows exactly from the identity ratio² = 1 + amp² − 2·amp·corr.
  * The best single factor on d sets amp to corr. It leaves the ratio √(1 − corr²).
  * Re 100 and Re 300, the base's training regimes, are measured on their held-out seeds 6–7.

  | Regime | Error / persistence | Size (amp) | Direction (corr) | After the best single factor | Best factor |
  |---|---|---|---|---|---|
  | Re 30 | 1.92 | 2.17 | 0.47 | 0.88 | 0.22 |
  | Re 100 (trained on) | 0.33 | 0.97 | 0.94 | 0.33 | 0.98 |
  | Re 300 (trained on) | 0.47 | 0.87 | 0.88 | 0.47 | 1.01 |
  | Re 1000 | 0.58 | 0.79 | 0.81 | 0.58 | 1.02 |
  | Re 3000 | 0.64 | 0.75 | 0.77 | 0.64 | 1.03 |

  * **From Re 100 to Re 3000 the base makes one kind of error: the direction of its predicted
    change.**
    * The correlation falls steadily with Re: 0.94, 0.88, 0.81, 0.77.
    * The size is already the best one for that direction (best factor 0.98–1.03), as expected of
      a model trained on MSE. Rescaling gains nothing.
    * The direction error is largest in vorticity, the channel that carries the small scales. Its
      correlation at Re 3000 is 0.68, against 0.88 for the velocities.
    * Re 1000 and Re 3000 continue the training regimes' trend.
  * **At Re 30 the base's predicted change is 2.2× too large,** and its direction is poor (0.47).
    * One number, shrinking the change 4.6-fold, would remove 54% of the error (1.92 → 0.88).
    * LoRA removes 71% and full fine-tuning 82%, so they also fix part of the direction.
  * **So two trajectories and 1000 steps correct a systematic error but barely improve a direction
    error.** That also fits Stage 10 and F1 at Re 1000.
  * Not measured here:
    * whether more data would help at Re 3000 (in E5, full fine-tuning gains at most 5.3% from six
      trajectories at Re 1000);
    * how the fine-tuned models' errors split. Their weights were not kept, so only the base model
      is diagnosed.
* **The Stage 10 verdict stands:** the acceptance test fails at the registered Re 1000 and at the
  farther Re 3000. LoRA would pass the bar at Re 30 (0.29× zero-shot). But the spec named Re 30 a
  second target that does not decide acceptance.

## Commands
```
.venv/bin/python -m pytest tests/test_stage10_gates.py -q -s
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot
.venv/bin/python experiments/stage10/finetune.py --arm lora
.venv/bin/python experiments/stage10/finetune.py --arm full
.venv/bin/python experiments/stage10/finetune.py --arm lora --lr 1e-2   # exploratory
# follow-up F1 (amendment F1-A1), on CPU: retrain the base, then the baselines and the rank sweep
.venv/bin/python experiments/stage7/train_onestep.py --loss mse --out onestep_mse_retrain
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --out f1_zeroshot
for r in 1 4 16 64; do .venv/bin/python experiments/stage10/finetune.py --arm lora --rank $r --out f1_lora_r$r; done
.venv/bin/python experiments/stage10/finetune.py --arm full --out f1_full
python experiments/stage10/score_f1.py
# follow-up F2: Re 3000 on idle lab GPUs, Re 30 on the CPU; then the 21 runs on one idle lab GPU
.venv/bin/python experiments/stage10/pilot_viscosity.py
.venv/bin/python experiments/stage2/generate.py --config experiments/stage10/configs/s10f2_v1.json --only re3000_kf4:0 re3000_kf4:1 re3000_kf4:6 re3000_kf4:7 re30_kf4:0 re30_kf4:1 re30_kf4:6 re30_kf4:7
.venv/bin/python experiments/stage10/resolution_f2.py --tag re3000_kf4 --grids 512 1024 2048
.venv/bin/python experiments/stage10/resolution_f2.py --tag re30_kf4 --grids 64 128 256
.venv/bin/python experiments/stage2/generate.py --config experiments/stage10/configs/s10f2_v1.json --finalize
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --out f2_re1000_zeroshot
for a in lora full; do for s in 0 1 2; do .venv/bin/python experiments/stage10/finetune.py --arm $a --seed $s --out f2_re1000_${a}_s$s; done; done
for t in re3000 re30; do
  .venv/bin/python experiments/stage10/finetune.py --arm zeroshot --dataset s10f2_v1 --target ${t}_kf4 --out f2_${t}_zeroshot
  for a in lora full; do for s in 0 1 2; do
    .venv/bin/python experiments/stage10/finetune.py --arm $a --seed $s --dataset s10f2_v1 --target ${t}_kf4 --out f2_${t}_${a}_s$s
  done; done
done
python experiments/stage10/score_f2.py
.venv/bin/python experiments/stage10/diagnose_f2.py   # exploratory
```
