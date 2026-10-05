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

**Follow-up F3 (pre-registered 4 Oct 2026; the verdicts above stand).** Can anything reach the bar?
With six target trajectories, 3 seeds per arm:
* **A model trained from scratch beats the bar at both targets:** 26% below zero-shot at Re 3000 and
  30% at Re 1000.
* **Fine-tuning the base does not.** Full fine-tuning gains 12% and LoRA 1%, even with ten times
  Stage 10's steps.
* **The physics allows far more.** An exact solver that sees only the 64² band predicts the next frame
  at 0.05× persistence at Re 3000, against 0.47× for the best model.

6 of 7 predictions HIT. F3-P3, that pretraining still pays with six trajectories, is a MISS. So the
Stage 10 miss lies in adapting this base, not in the data or the physics. See "Follow-up F3" below.

**Follow-up F4 (pre-registered 5 Oct 2026; the verdicts above stand).** The base or the learning rate?
5 of 6 predictions HIT.
* **The base's training is what mattered.** Trained ten times longer, the base halves its
  in-distribution error. Fine-tuned at Stage 10's own learning rate, it beats scratch at both
  targets.
* **The higher learning rate lifts the F1 base only to scratch's level** (F4-P2 MISS).
* **A well-trained base leaves no room for the bar.** Its zero-shot error is already close to that of
  a model trained on six target trajectories. From it, full fine-tuning on six trajectories gains
  7–9.5% and LoRA in Stage 10's setting 2%.
* So the 20% bar mostly measured how under-trained the Stage 7 base was. See "Follow-up F4" below.

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

## Follow-up F3: can anything reach the bar? (pre-registered 4 Oct 2026)

Spec: F3 in `handouts/stage10_lora.md`, committed before any F3 measuring run (a6a5510). Results:
`results/f3_*.json`, scored by `score_f3.py` → `results/f3_analysis.json`. Floor: `closure_f3.py` →
`results/f3_closure.json`.

**Set-up as registered.**
* **Data:** Re 3000 seeds 2–5 were generated on lab card 3 in 1.8 h, completing `s10f3_v1`. Its
  manifest verified all 8 trajectories. Re 1000 uses `acc_v1`.
* **Arms:** each trained on the target's seeds 0–5 and evaluated on seeds 6–7, with 3 seeds per arm.
  * scratch: 20,000 steps at lr 1e-3;
  * full fine-tuning of the F1 base: 10,000 steps at lr 1e-4;
  * LoRA r = 4: 10,000 steps at lr 1e-3.
* **Where:** all 20 runs on lab card 3. A scratch run took about 5 min, a full one 2.5–2.9 min and a
  LoRA run 0.8–1.6 min.
* **Check:** the zero-shot reruns equal F2's results exactly at both targets.

Mean ± SE over seeds 0–2. Error, with its ratio to persistence in brackets, and the gain over
zero-shot:

| Target | Zero-shot | Scratch | Full | LoRA r = 4 | Floor |
|---|---|---|---|---|---|
| Re 3000 | 0.2588 (0.64) | 0.1917 (0.47): **25.9% ± 0.4%** | 0.2292 (0.57): 11.5% ± 0.01% | 0.2561 (0.63): 1.1% ± 0.01% | 0.054 |
| Re 1000 | 0.2128 (0.58) | 0.1485 (0.41): **30.2% ± 0.8%** | 0.1871 (0.51): 12.1% ± 0.01% | 0.2110 (0.58): 0.9% ± 0.01% | 0.029 |

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| F3-P1 | the ceiling reaches the bar at Re 3000 (gain ≥ 20%) | scratch, 25.9% | **HIT** |
| F3-P2 | LoRA stays below the bar at Re 3000 | 1.1% | **HIT** |
| F3-P3 | full fine-tuning's mean error is below scratch's at Re 3000 | 0.2292 vs 0.1917 | **MISS** |
| F3-P4 | the ceiling reaches the bar at Re 1000 | scratch, 30.2% | **HIT** |
| F3-P5 | LoRA stays below the bar at Re 1000 | 0.9% | **HIT** |
| F3-P6 | the floor at Re 3000 is below half the base's zero-shot ratio (0.32) | 0.054 | **HIT** |
| F3-P7 | the floor rises with Re | 0.0003, 0.0066, 0.029, 0.054 | **HIT** |

The floor, as error over persistence, next to the models:

| Regime | Floor, all channels | Floor, vorticity | Base zero-shot | Best F3 model |
|---|---|---|---|---|
| Re 100 (trained on) | 0.0003 | 0.0005 | 0.33 | — |
| Re 300 (trained on) | 0.0066 | 0.0095 | 0.47 | — |
| Re 1000 | 0.029 | 0.039 | 0.58 | 0.41 (scratch) |
| Re 3000 | 0.054 | 0.073 | 0.64 | 0.47 (scratch) |

**Readings**
* **With six trajectories the bar is reachable at both targets, but only by training from
  scratch.**
  * Fine-tuning the base falls short: full fine-tuning gains 11.5–12.1% and LoRA 0.9–1.1%.
  * That is about four times F2's full fine-tuning gain and twice its LoRA gain, from three times
    the data and ten times the steps.
  * The reading fixed in the spec applies at both targets: more data makes the bar reachable, but
    not for attention-only LoRA at r = 4. Full fine-tuning at Stage 10's learning rate does not
    reach it either.
* **Here the base is a worse start than a random one (F3-P3 MISS).**
  * Scratch is ahead even at equal steps. At 10,000 steps (seed 0) it is at 0.196 against full
    fine-tuning's 0.229 at Re 3000, and 0.152 against 0.187 at Re 1000.
  * In seed 0 it passes full fine-tuning's final error by step 4000 at Re 3000 and step 3000 at
    Re 1000.
  * Full fine-tuning's curve has flattened: 0.2296 → 0.2292 over its last 2000 steps at Re 3000.
  * F3 does not separate two explanations:
    * the base saw only about four passes over its 12 trajectories (Stage 7's 3000 steps at batch 8:
      24,000 samples over 5,988 one-step pairs);
    * fine-tuning's learning rate, a tenth of scratch's, may keep the model near the base.
  * E5's base saw about seven times as many training samples, and there full fine-tuning beat
    scratch by 4.9% at six trajectories.
* **The physics floor is far below every model.**
  * An exact solver that sees only the current 64² band predicts the next frame at 0.054×
    persistence at Re 3000, and at 0.029× at Re 1000.
  * The floor rises 160-fold from Re 100 to Re 3000, as the scales finer than the band gain weight.
    It is highest in vorticity.
  * The base's error is 12 times the floor at Re 3000, and the best scratch model's 9 times.
  * So F2's direction error is not missing information. It is modelling error, and training on
    the target removes part of it.
  * The floor is a reference, not a bound: a learned model does not know the equations, but it also
    sees the previous frame.
* **The Stage 10 verdict stands.** F3 locates the miss: it lies in adapting this base at Stage 10's
  learning rates, not in the amount of information in the data or in the physics. LoRA with three
  times the data and ten times the steps still gains only about 1%.

## Follow-up F4: the base or the learning rate? (pre-registered 5 Oct 2026)

Spec: F4 in `handouts/stage10_lora.md`, committed before any F4 measuring run (3ca04eb). Results:
`results/f4_*.json` and `experiments/stage7/results/onestep_mse_long.json`, scored by `score_f4.py` →
`results/f4_analysis.json`.

**Set-up as registered.**
* **B30k:** Stage 7's recipe for 30,000 steps, trained on lab card 3 in 12.6 min (18 ms per step). It
  halves the F1 base's (B3k's) in-distribution error:
  * Re 100: 0.032 (0.15× persistence), against 0.072 (0.33×);
  * Re 300: 0.083 (0.27×), against 0.145 (0.47×).
* **Arms:** full fine-tuning on the target's seeds 0–5 for 20,000 steps, from B3k and from B30k, each
  at lr 1e-4 and at lr 1e-3. Also B30k's zero-shot error, and LoRA r = 4 on B30k in Stage 10's
  registered setting.
* **Where:** 3 seeds per arm, all on lab card 3. A 20,000-step run took 5.8–6.8 min.

Mean error over seeds 0–2, with its ratio to persistence in brackets (SE ≤ 0.0005, scratch's ≤ 0.0017):

| Arm | Re 3000 | Re 1000 |
|---|---|---|
| zero-shot, B3k | 0.2588 (0.64) | 0.2128 (0.58) |
| zero-shot, B30k | 0.2008 (0.50) | 0.1498 (0.41) |
| scratch (F3) | 0.1917 (0.47) | 0.1485 (0.41) |
| b3k_lo: B3k, lr 1e-4 | 0.2221 (0.55) | 0.1794 (0.49) |
| b3k_hi: B3k, lr 1e-3 | 0.1940 (0.48) | 0.1485 (0.41) |
| b30k_lo: B30k, lr 1e-4 | 0.1847 (0.46) | 0.1395 (0.38) |
| b30k_hi: B30k, lr 1e-3 | **0.1818 (0.45)** | **0.1361 (0.37)** |
| LoRA on B30k, Stage 10's setting | 0.1963 (0.49) | 0.1468 (0.40) |

Each fine-tuned arm against scratch (1 − e / e(scratch)) and against its own base's zero-shot:

| Arm | Re 3000: vs scratch | Re 3000: vs own zero-shot | Re 1000: vs scratch | Re 1000: vs own zero-shot |
|---|---|---|---|---|
| b3k_lo | −15.8% | 14.2% | −20.8% | 15.7% |
| b3k_hi | −1.2% | 25.1% | 0.0% | 30.2% |
| b30k_lo | +3.7% | 8.0% | +6.1% | 6.9% |
| b30k_hi | +5.2% | 9.5% | +8.4% | 9.2% |
| LoRA on B30k | −2.4% | 2.2% | +1.1% | 2.0% |

| # | Prediction (at both targets) | Measured (Re 3000; Re 1000) | Verdict |
|---|---|---|---|
| F4-P1 | with B3k, lr 1e-3 beats lr 1e-4 | 0.194 vs 0.222; 0.149 vs 0.179 | **HIT** |
| F4-P2 | with the learning rate and steps matched, B3k is no worse a start than random | 0.1940 vs 0.1917; 0.14848 vs 0.14850 | **MISS** (at Re 3000) |
| F4-P3 | at Stage 10's learning rate, B30k beats B3k | 0.185 vs 0.222; 0.139 vs 0.179 | **HIT** |
| F4-P4 | at Stage 10's learning rate, B30k beats scratch | 0.185 vs 0.192; 0.139 vs 0.149 | **HIT** |
| F4-P5 | B30k transfers better zero-shot than B3k | 0.201 vs 0.259; 0.150 vs 0.213 | **HIT** |
| F4-P6 | LoRA on B30k in Stage 10's setting misses the bar | 2.2%; 2.0% | **HIT** |

**Readings**
* **The base's training was the main cause (F4-P3 and P4).** The reading fixed in the spec applies.
  * Fine-tuned at Stage 10's own learning rate, the base trained ten times longer beats scratch by
    3.7% at Re 3000 and 6.1% at Re 1000. At lr 1e-3 the margins are 5.2% and 8.4%.
  * With this base, pretraining pays with six trajectories, as it did in E5.
* **The learning rate mattered too, but only up to scratch's level (F4-P1 HIT, F4-P2 MISS).**
  * At lr 1e-3 the F1 base ends 12.7% (Re 3000) and 17.2% (Re 1000) below its lr 1e-4 runs.
  * It still only reaches scratch: 1.2% worse at Re 3000, level at Re 1000.
  * With about four passes of training, the F1 base gave fine-tuning nothing a random start lacks.
* **What this means for Stage 10's bar.** The bar compares LoRA with its own base's zero-shot error.
  * **With the F1 base,** zero-shot was poor, so any ample training on the target beat it by 25–30%
    (scratch, and b3k_hi).
  * **A well-trained base transfers zero-shot almost as well as a model trained on six target
    trajectories:** 0.201 against scratch's 0.192 at Re 3000, and 0.150 against 0.149 at Re 1000.
  * **From that base nothing tested gains 20%.** Full fine-tuning on six trajectories for 20,000
    steps gains 7–9.5%, and LoRA in Stage 10's setting 2%.
  * So the bar mostly measured how under-trained the Stage 7 base was. For a well-trained base it is
    out of reach with up to six target trajectories, whatever the method.
* **LoRA on the better base gains 2.2% and 2.0%** in Stage 10's setting.
  * That is about five times its gain on the F1 base in F2 (0.46% and 0.35%), and still a tenth of
    the bar.
  * At Re 1000 it beats scratch trained on six trajectories (0.1468 against 0.1485), with a third of
    the data and a twentieth of the steps.
* **At lr 1e-3 the runs overfit a little.** For example, b3k_hi at Re 3000 (seed 0) is at its best
  near step 12,000 (0.193) and ends at 0.195. The best errors along the curves are in the analysis
  file, as optimistic figures only.
* **The Stage 10 verdict stands.**

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
# follow-up F3: Re 3000 seeds 2-5 (seeds 0, 1, 6, 7 copied from s10f2_v1), then 20 runs and the floor on one idle lab GPU
.venv/bin/python experiments/stage2/generate.py --config experiments/stage10/configs/s10f3_v1.json --only re3000_kf4:2 re3000_kf4:3 re3000_kf4:4 re3000_kf4:5
.venv/bin/python experiments/stage2/generate.py --config experiments/stage10/configs/s10f3_v1.json --finalize
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --dataset s10f3_v1 --target re3000_kf4 --out f3_re3000_zeroshot
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --out f3_re1000_zeroshot
for td in re3000:s10f3_v1 re1000:acc_v1; do t=${td%%:*}; d=${td#*:}
  for an in scratch:20000 full:10000 lora:10000; do a=${an%%:*}; n=${an#*:}; for s in 0 1 2; do
    .venv/bin/python experiments/stage10/finetune.py --arm $a --steps $n --seed $s --dataset $d --target ${t}_kf4 --train-seeds 0 1 2 3 4 5 --eval-every 1000 --out f3_${t}_${a}_s$s
  done; done
done
.venv/bin/python experiments/stage10/closure_f3.py
python experiments/stage10/score_f3.py
# follow-up F4, on one idle lab GPU: the longer base, then 32 runs
.venv/bin/python experiments/stage7/train_onestep.py --loss mse --steps 30000 --params-name onestep_mse_long --out onestep_mse_long
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --base onestep_mse_long --dataset s10f3_v1 --target re3000_kf4 --out f4_re3000_b30k_zeroshot
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot --base onestep_mse_long --out f4_re1000_b30k_zeroshot
for td in re3000:s10f3_v1 re1000:acc_v1; do t=${td%%:*}; d=${td#*:}
  for s in 0 1 2; do .venv/bin/python experiments/stage10/finetune.py --arm lora --base onestep_mse_long --seed $s --dataset $d --target ${t}_kf4 --out f4_${t}_b30k_lora_s$s; done
  for ab in "b3k_lo:" "b3k_hi:--lr 1e-3" "b30k_lo:--base onestep_mse_long" "b30k_hi:--base onestep_mse_long --lr 1e-3"; do a=${ab%%:*}; o=${ab#*:}
    for s in 0 1 2; do .venv/bin/python experiments/stage10/finetune.py --arm full $o --steps 20000 --seed $s --dataset $d --target ${t}_kf4 --train-seeds 0 1 2 3 4 5 --eval-every 1000 --out f4_${t}_${a}_s$s; done
  done
done
python experiments/stage10/score_f4.py
```
