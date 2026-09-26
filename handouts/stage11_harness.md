# Stage 11 Spec: Ablation harness

**Plan acceptance test (PROJECT_PLAN.md §5):** one config file → one run → one row of results.
**Written:** 26 Sep 2026, before any Stage 11 code. §§2–4 are fixed, and the Stage 1 rules apply.

---

## 1. Why this stage matters

Experiments E1–E5 are grids of runs (E1 alone is 9 cells × 3 seeds × pretrain + fine-tune). Every
number in the write-up must trace back to one config, one run and one row. The harness also
enforces the protocol rules of project_memory §2 mechanically: regime hold-outs, pre-registration
and seeds.

## 2. Design

### 2.1 Config (JSON)
```json
{
 "name": "e1_r8_t4_seed0",
 "prediction": "free text, required: the pre-registered expectation for this run",
 "data":  {"dataset": "acc_v1", "N": 64, "stride": 1,
           "train": [{"tag": "re100_kf4", "seeds": [0,1,2,3]}], "test": ["re300_kf4"], "val": []},
 "model": {"D": 64, "P": 4, "L": 3, "heads": 4,
           "backbone": {"enc_depths": [2,2,2], "enc_heads": [2,4,8], "dec_depths": [2,2,2],
                        "dec_heads": [8,4,2], "window": [3,4,4]}},
 "train": {"steps": 3000, "batch": 8, "lr": 1e-3, "schedule": "cosine", "loss": "mse",
           "pushforward": false, "seed": 0, "init_from": null, "lora_rank": null},
 "eval":  {"splits": ["test"], "rollout_steps": 0}
}
```
* `prediction` is **required and non-empty**: a run without a written prediction is refused.
* The splits go through Stage 2's `make_splits`, so a held-out regime that leaks into training is
  refused.
* `init_from` names an earlier run whose final parameters start this one (pretrain → fine-tune).
  `lora_rank` switches fine-tuning to LoRA (Stage 10).

### 2.2 Run
`harness.run(config) -> row`: builds data, model and optimiser from the config alone; trains;
evaluates; saves parameters to `data/runs/<config_hash>/params.npz`; appends one row to
`experiments/results.jsonl`.

### 2.3 Row
`{config_hash, name, prediction, git_commit, started, wall_s, n_params, n_trainable, final_loss,
metrics: {split/tag: {rel_err, persistence, ratio, [rollout_drift]}}, config}`.
`config_hash` is the SHA-256 of the canonical JSON of the config (sorted keys, no whitespace) with
`name` and `prediction` removed, so renaming or re-wording does not change it.

### 2.4 Sweeps
`harness.expand(sweep) -> [config]`: a base config plus a dict of dotted-path lists
(`{"train.seed": [0,1,2], "model.L": [1,3,6]}`) → the Cartesian product, with names derived from
the swept values.

## 3. Gates — `tests/test_stage11_gates.py`

| Gate | Test | Pass if |
|---|---|---|
| X1 | config hash: key order and `name`/`prediction` don't matter; any other change does | exact |
| X2 | a config with an empty `prediction` is refused (ValueError) | raises |
| X3 | a held-out regime also in training is refused | raises |
| X4 | a tiny config (tiny dataset, D = 16, 3 steps) runs end to end and appends exactly one row with every field present | exact |
| X5 | running the same tiny config twice gives identical metrics and parameters (determinism on one machine) | bitwise |
| X6 | `expand` produces the Cartesian product with unique names and hashes | exact |
| X7 | `init_from` loads the named run's parameters (the fine-tune starts from them bitwise); `lora_rank` trains only LoRA factors | exact |

## 4. Prediction

| # | Measurement | Prediction |
|---|---|---|
| H1 | reproduce the Stage 7 one-step result through the harness (same data, model, schedule, seed): relative-error difference per regime | ≤ 2% relative (identical recipe; small differences come only from batch-sampling order) |

**Stage 11 acceptance:** X1–X7 pass, and H1 HIT.
