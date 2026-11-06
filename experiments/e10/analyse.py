"""Score E10's predictions P1-P6 (handouts/e10_diagnostics.md) from results/diag.json.

usage: python experiments/e10/analyse.py   (writes results/analysis.json and prints the verdicts)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results", "diag.json")))
FIELDS = ("u", "v", "p", "omega")


def band(r, lo, hi):
    e, t = np.asarray(r["shell_err"]), np.asarray(r["shell_truth"])
    return e[lo:hi].sum() / t[lo:hi].sum()


def seed(name):
    return name.rsplit("_s", 1)[1]


scratch = {seed(n): D[n] for n in D if n.startswith("e1_scratch")}
ft = {n: D[n] for n in D if n.startswith("e1_ft")}
assert len(scratch) == 3 and len(ft) == 27, (len(scratch), len(ft))
out = {"models": {}}
for n, r in sorted(D.items()):
    t, f = r["trained"], r["refit"]
    out["models"][n] = dict(
        rel_err=t["rel_err"], g1=abs(t["rel_err"] / r["stored_rel_err"] - 1),
        share_eval=t["share_eval"], share_train=t["share_train"], field_rel_err=t["field_rel_err"],
        R_low=band(t, 1, 5), R_mid=band(t, 8, 20), R_high=band(t, 12, 20), gap=t["gap"],
        refit_train=1 - f["train_mse"] / t["train_mse"], refit_eval=1 - f["rel_err"] / t["rel_err"],
        div_ratio=t["div_ratio"], curl_mismatch=t["curl_mismatch"])
M = out["models"]

v = {}
v["G1"] = max(m["g1"] for m in M.values()) < 1e-4
v["P1"] = all(max(m["share_eval"], key=m["share_eval"].get) == "omega" and min(m["share_eval"], key=m["share_eval"].get) == "p"
              and m["share_eval"]["omega"] > 0.35 for m in M.values())
v["P2"] = all(m["R_high"] >= 3 * m["R_low"] for m in M.values())
adv = {}
for n in ft:
    if "_n32_" in n:
        s = scratch[seed(n)]["trained"]
        a = lambda lo, hi: 1 - band(D[n]["trained"], lo, hi) / band(s, lo, hi)
        adv[n] = {"low": a(1, 5), "mid": a(8, 20)}
out["adv_by_scale_n32"] = adv
v["P3"] = all(x["mid"] > x["low"] for x in adv.values())
v["P4"] = all(M[f"e1_scratch_s{s}"]["gap"] > 2 for s in scratch) and all(
    M[n]["gap"] < M[f"e1_scratch_s{seed(n)}"]["gap"] for n in ft)
v["P5"] = all(m["refit_train"] < 0.03 and abs(m["refit_eval"]) < 0.01 for m in M.values())
v["P6"] = (all(1e-3 < m["div_ratio"] < 1e-1 for m in M.values())
           and all(M[n]["div_ratio"] < M[f"e1_scratch_s{seed(n)}"]["div_ratio"] for n in ft if "_n32_" in n)
           and all(m["curl_mismatch"] < m["field_rel_err"]["omega"] for m in M.values()))
out["verdicts"] = {k: "HIT" if x else "MISS" for k, x in v.items()}


def summ(names, key, sub=None):
    x = np.array([M[n][key] if sub is None else M[n][key][sub] for n in names])
    return f"{x.mean():.4g} (range {x.min():.4g} to {x.max():.4g})"


groups = {"scratch": sorted(n for n in M if "scratch" in n), "ft n32": sorted(n for n in ft if "_n32_" in n),
          "ft all": sorted(ft)}
out["summary"] = {}
for g, names in groups.items():
    s = {k: summ(names, k) for k in ("rel_err", "R_low", "R_mid", "R_high", "gap", "refit_train", "refit_eval",
                                     "div_ratio", "curl_mismatch")}
    s.update({f"share_{f}": summ(names, "share_eval", f) for f in FIELDS})
    s.update({f"share_train_{f}": summ(names, "share_train", f) for f in FIELDS})
    s.update({f"rel_err_{f}": summ(names, "field_rel_err", f) for f in FIELDS})
    out["summary"][g] = s
# error by shell, averaged per group, for the figure
out["shell_ratio"] = {g: np.mean([np.asarray(D[n]["trained"]["shell_err"]) / np.maximum(np.asarray(D[n]["trained"]["shell_truth"]), 1e-30)
                                  for n in names], 0).tolist() for g, names in groups.items()}
json.dump(out, open(os.path.join(HERE, "results", "analysis.json"), "w"), indent=1)
print(json.dumps(out["verdicts"]))
for g, s in out["summary"].items():
    print(g)
    for k, x in s.items():
        print(f"  {k:18s} {x}")
for n, x in sorted(adv.items()):
    print(n, {k: round(y, 4) for k, y in x.items()})
