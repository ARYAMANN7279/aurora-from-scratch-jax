"""Analysis of E3, E4, E5 (handouts/e3_ood_transfer.md, e4_ablations.md, e5_lora_vs_full.md).

usage: analyse_e345.py --exp e3|e4|e5 --results <rows.jsonl> [more.jsonl ...]
Rows are matched to the design (experiments/designs.py) by config hash, never by name. Writes
experiments/<exp>/results/analysis.json and prints the table and verdicts (HIT / MISS).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import designs as D  # noqa: E402
from aurora.harness import config_hash  # noqa: E402


def _ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"))


def _rho(x, y):
    r = lambda a: np.argsort(np.argsort(a)).astype(float)
    return float(np.corrcoef(r(np.asarray(x)), r(np.asarray(y)))[0, 1])


def _verdicts(v):
    return {k: "HIT" if bool(x) else "MISS" for k, x in v.items()}


def e3(rows):
    pre, ft, sc = D.e3(config_hash)
    m = lambda c, key: rows[config_hash(c)]["metrics"][key]
    out = {}
    for r in D.E3_TESTS:
        t = D.tag(r)
        z = [m(p, f"test/{t}")["ratio"] for p in pre]
        f = {c["train"]["seed"]: m(c, f"val/{t}") for c in ft if c["data"]["train"][0]["tag"] == t}
        s = {c["train"]["seed"]: m(c, f"val/{t}") for c in sc if c["data"]["train"][0]["tag"] == t}
        adv = [1 - f[k]["rel_err"] / s[k]["rel_err"] for k in D.SEEDS]
        adv_d = [1 - f[k]["rollout_drift"] / s[k]["rollout_drift"] for k in D.SEEDS]
        out[t] = dict(d=float(np.log(r[0] / 250)), zero_shot=_ms(z), adv=_ms(adv), adv_drift=_ms(adv_d))
    ts = [D.tag(r) for r in D.E3_TESTS]
    near, far = out[ts[0]]["adv"], out[ts[-1]]["adv"]
    v = {"E3-P1": _rho([out[t]["d"] for t in ts], [out[t]["zero_shot"][0] for t in ts]) >= 0.9,
         "E3-P2": all(out[t]["adv"][0] > 2 * out[t]["adv"][1] for t in ts),
         "E3-P3": far[0] < near[0] and (near[0] - far[0]) > 2 * np.hypot(near[1], far[1]),
         "E3-P4": out[ts[-1]]["zero_shot"][0] <= 0.8}
    return dict(regimes=out, verdicts=_verdicts(v))


def e4(rows):
    runs = D.e4()
    by = {}
    for c in runs:
        by.setdefault(c["name"].rsplit("_s", 1)[0][3:], {})[c["train"]["seed"]] = rows[config_hash(c)]["metrics"]
    regs = ("re100_kf4", "re300_kf4")
    out = {}
    for name in D.E4_VARIANTS:
        row = {}
        for t in regs:
            for key in ("rel_err", "rollout_drift"):
                ch = [by[name][s][f"val/{t}"][key] / by["base"][s][f"val/{t}"][key] - 1 for s in D.SEEDS]
                row[f"{t}/{key}"] = _ms(ch)
        row["onestep"] = _ms([np.mean([by[name][s][f"val/{t}"]["rel_err"] / by["base"][s][f"val/{t}"]["rel_err"] - 1
                                       for t in regs]) for s in D.SEEDS])
        out[name] = row
    one = lambda n: out[n]["onestep"][0]
    pf_c = [out["pf_compute"][f"{t}/rollout_drift"] for t in regs]
    v = {"E4-P1": out["mae"]["re300_kf4/rollout_drift"][0] > 0.5,
         "E4-P2": all(out["pf_steps"][f"{t}/rollout_drift"][0] <= -0.05 for t in regs)
                  and not any(mu < -2 * se for mu, se in pf_c),
         "E4-P3": 0.03 <= one("noshift") <= 0.20,
         "E4-P4": one("depth6") > 0.03 and one("depth24") > -0.03,
         "E4-P5": one("L1") > 0.03 and abs(one("L6")) <= 0.03}
    return dict(change_vs_base=out, verdicts=_verdicts(v))


def e5(rows):
    pre, ft, sc = D.e5(config_hash)
    key = "val/re1000_kf4"
    m = lambda c, split: rows[config_hash(c)]["metrics"][split]["rel_err"]
    zs = {p["train"]["seed"]: m(p, "test/re1000_kf4") for p in pre}
    get = {}
    for c in ft + sc:
        kind = "scratch" if c in sc else ("lora" if c["train"]["lora_rank"] else "full")
        get[(kind, len(c["data"]["train"][0]["seeds"]), c["train"]["seed"])] = m(c, key)
    out = {}
    for k in D.E5_SIZES:
        g = {meth: [1 - get[(meth, k, s)] / zs[s] for s in D.SEEDS] for meth in ("full", "lora")}
        out[k] = dict(gain_full=_ms(g["full"]), gain_lora=_ms(g["lora"]),
                      lora_over_full=_ms([gl / gf if gf else np.nan for gl, gf in zip(g["lora"], g["full"])]),
                      adv=_ms([1 - get[("full", k, s)] / get[("scratch", k, s)] for s in D.SEEDS]))
    ks = list(D.E5_SIZES)
    gf = [out[k]["gain_full"][0] for k in ks]
    v = {"E5-P1": all(a < b for a, b in zip(gf, gf[1:])),
         "E5-P2": 0.02 <= out[2]["gain_full"][0] <= 0.10 and out[6]["gain_full"][0] >= 0.08,
         "E5-P3": _rho(ks, [out[k]["lora_over_full"][0] for k in ks]) <= -0.8,
         "E5-P4": _rho(ks, [out[k]["adv"][0] for k in ks]) <= -0.8 and out[1]["adv"][0] > 2 * out[1]["adv"][1]}
    return dict(zero_shot=zs, sizes={str(k): x for k, x in out.items()}, verdicts=_verdicts(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["e3", "e4", "e5"])
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()
    rows = {}
    for path in args.results:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[r["config_hash"]] = r
    res = {"e3": e3, "e4": e4, "e5": e5}[args.exp](rows)
    out = os.path.join(HERE, args.exp, "results", "analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
