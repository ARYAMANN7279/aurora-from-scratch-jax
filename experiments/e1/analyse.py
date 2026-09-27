"""E1 analysis (handouts/e1_volume_diversity.md, sections 3–5): results.jsonl -> adv table + verdicts.

usage: analyse.py [--results experiments/results.jsonl] [--out experiments/e1/results/analysis.json]
Rows are matched to the design by config hash (make_configs.py output), never by name.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import e1_design as E  # noqa: E402


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def analyse(rows, pre, ft, sc):
    """rows: config_hash -> result row. Returns the analysis dict (cells, verdicts)."""
    from aurora.harness import config_hash

    tgt = E.tag(E.TARGET)
    get = lambda c, split: rows[config_hash(c)]["metrics"][f"{split}/{tgt}"]
    metrics = ("rel_err", "rollout_drift")
    scratch = {c["train"]["seed"]: get(c, "val") for c in sc}
    cells = {}
    for p, f in zip(pre, ft):
        n, T = len(p["data"]["train"]), sum(len(x["seeds"]) for x in p["data"]["train"])
        s = f["train"]["seed"]
        cell = cells.setdefault(f"n{n}_T{T}", dict(n=n, T=T, seeds=[], zero_shot=[],
                                                    **{f"adv_{m}": [] for m in metrics}))
        cell["seeds"].append(s)
        cell["zero_shot"].append(get(p, "test")["rel_err"])
        for m in metrics:
            cell[f"adv_{m}"].append(1 - get(f, "val")[m] / scratch[s][m])
    for c in cells.values():
        for k in ["zero_shot"] + [f"adv_{m}" for m in metrics]:
            v = np.array(c[k])
            c[k + "_mean"], c[k + "_se"] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    ns, Ts = sorted(E.SETS), list(E.TOTALS)
    A = lambda n, T, k="adv_rel_err_mean": cells[f"n{n}_T{T}"][k]
    v = {}
    v["E1-P1"] = all(c["adv_rel_err_mean"] > 2 * c["adv_rel_err_se"] for c in cells.values())
    v["E1-P2"] = all(A(ns[0], T) < A(ns[1], T) < A(ns[2], T) for T in Ts)
    div = np.mean([A(ns[-1], T) - A(ns[0], T) for T in Ts])
    vol = np.mean([A(n, Ts[-1]) - A(n, Ts[0]) for n in ns])
    v["E1-P3"] = bool(div > vol)
    best = max(c["adv_rel_err_mean"] for c in cells.values())
    v["E1-P4"] = bool(0.10 <= best <= 0.40 and 0.0 <= A(ns[0], Ts[0]) <= 0.15)
    v["E1-P5"] = all(A(ns[-1], T, "zero_shot_mean") < A(ns[0], T, "zero_shot_mean") for T in Ts)
    keys = sorted(cells)
    a1 = np.array([cells[k]["adv_rel_err_mean"] for k in keys])
    a2 = np.array([cells[k]["adv_rollout_drift_mean"] for k in keys])
    rho = float(np.corrcoef(rank(a1), rank(a2))[0, 1])
    v["E1-P6"] = bool(rho >= 0.7)
    return dict(cells=cells, scratch=scratch, diversity_gain=float(div), volume_gain=float(vol),
                best_adv=float(best), spearman_onestep_drift=rho,
                verdicts={k: "HIT" if bool(x) else "MISS" for k, x in v.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "experiments", "results.jsonl"))
    ap.add_argument("--configs", default=os.path.join(HERE, "configs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis.json"))
    args = ap.parse_args()
    rows = {}
    for line in open(args.results):
        if line.strip():
            r = json.loads(line)
            rows[r["config_hash"]] = r
    load = lambda name: json.load(open(os.path.join(args.configs, name)))
    res = analyse(rows, load("pretrain.json"), load("finetune.json"), load("scratch.json"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("cell      adv one-step        adv drift           zero-shot")
    for k, c in sorted(res["cells"].items(), key=lambda kv: (kv[1]["n"], kv[1]["T"])):
        print(f"{k:9s} {c['adv_rel_err_mean']:+.3f} ± {c['adv_rel_err_se']:.3f}   "
              f"{c['adv_rollout_drift_mean']:+.3f} ± {c['adv_rollout_drift_se']:.3f}   {c['zero_shot_mean']:.3f}")
    print(json.dumps(res["verdicts"]))
