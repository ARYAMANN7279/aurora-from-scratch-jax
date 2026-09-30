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
```
