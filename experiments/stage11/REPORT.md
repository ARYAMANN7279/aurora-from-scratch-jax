# Stage 11 Report: Ablation harness

Spec: `handouts/stage11_harness.md`. Code: `src/aurora/harness.py`, `experiments/run.py`.
Tests: `tests/test_stage11_gates.py`. Result rows: `experiments/results.jsonl`.

## Verdict

**Accepted.** One config file → one run → one row. All gates pass. The harness, driven only by
`experiments/configs/h1_stage7_repro.json`, reproduces the Stage 7 result **exactly**: the one-step
errors match to every printed digit (relative difference 0 in both regimes; predicted ≤ 2%).

## Gates

Command: `.venv/bin/python -m pytest tests/test_stage11_gates.py -q` → 5 passed (X2 and X3 are one
test, as are X4 and X5).

| Gate | Result |
|---|---|
| X1 config hash ignores key order, `name` and `prediction`, and changes with anything else | pass |
| X2 a config without a written prediction is refused | pass |
| X3 a held-out regime that also appears in training is refused | pass |
| X4 a tiny config runs end to end and appends exactly one complete row | pass |
| X5 running the same config twice gives identical metrics and bitwise-identical parameters | pass |
| X6 sweep expansion gives the Cartesian product with unique names and hashes | pass |
| X7 `init_from` starts from the named run's parameters bitwise; `lora_rank` trains only LoRA factors | pass |

## Prediction

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| H1 | reproduce Stage 7 through the harness within 2% per regime | Re100 0.070202 vs 0.070202; Re300 0.145691 vs 0.145691 (difference 0) | **HIT** |

Run: config hash `608f9636ba072379`, 5,106,688 parameters, 3039 s on the shared CPU.

## Notes
* The dataset manifest can be partial (`generate.py --finalize --partial`), so the harness can run
  on the completed regimes while the rest are still generating. The full finalize replaces it.
* Every row stores the full config, its hash, the written prediction and the git commit, so each
  number in the experiments traces back to one run.
