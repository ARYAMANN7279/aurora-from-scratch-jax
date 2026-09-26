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

## Commands
```
.venv/bin/python -m pytest tests/test_stage10_gates.py -q -s
.venv/bin/python experiments/stage10/finetune.py --arm zeroshot
.venv/bin/python experiments/stage10/finetune.py --arm lora
.venv/bin/python experiments/stage10/finetune.py --arm full
.venv/bin/python experiments/stage10/finetune.py --arm lora --lr 1e-2   # exploratory
```
