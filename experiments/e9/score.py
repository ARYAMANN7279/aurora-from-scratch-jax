"""Score E9 (handouts/e9_rpb_plus_qknorm.md, section 4) from results/rows.jsonl and E8's rows.

err = one-step relative error, d20 = the 20-step drift from 84 rollouts (metrics["val/re300_kf4"]); train_s = the
training loop's time. adv_X(Y) = 1 - m(Y) / m(X), paired by seed; mean and SE over seeds (ddof 1). B is this job's
B unless written B8; R8 and Q8 are E8's R and Q. A prediction is scored only when the arms it uses have all 3 seeds.
E9-P1  adv_B(RQ) on err >= 0.09 and > 2 SE
E9-P2  adv_B(RQ) on err < 0.137 (the independent combination of E8's two gains)
E9-P3  err(RQ) < err(R8) and err(RQ) < err(Q8) in every seed
E9-P4  adv_B(RQ) on d20 >= 0.03 and > 2 SE
E9-P5  train_s(RQ) / train_s(B) < 1.15, the ratio taken within each card and averaged over the two cards
E9-P6  |err(B) / err(B8) - 1| < 0.005 in every seed
usage: score.py   (writes results/analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
KEY = "val/re300_kf4"
CARD = {"RQ_s0": 0, "B_s1": 0, "RQ_s2": 0, "B_s0": 1, "RQ_s1": 1, "B_s2": 1}  # kaggle_e9.py's queues


def load(path):
    rows = {}
    for line in (open(path) if os.path.exists(path) else []):
        r = json.loads(line)
        arm, seed = r["name"].split("_")[1], int(r["name"].split("_s")[-1])
        rows.setdefault(arm, {})[seed] = r
    return rows


rows = load(os.path.join(RES, "rows.jsonl"))
e8 = load(os.path.join(HERE, "..", "e8", "results", "rows.jsonl"))
rows.update({a + "8": e8[a] for a in ("B", "R", "Q") if a in e8})
PICK = {"err": lambda r: r["metrics"][KEY]["rel_err"], "d20": lambda r: r["metrics"][KEY]["rollout_drift_many"],
        "d50": lambda r: r["metrics"][KEY]["rollout_drift_many_long"], "train_s": lambda r: r["train_s"]}


def get(arm, field):
    return {s: PICK[field](r) for s, r in sorted(rows.get(arm, {}).items())}


def mean_se(x):
    x = np.asarray(list(x), float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)


def adv(base, arm, field):
    X, Y = get(base, field), get(arm, field)
    return mean_se(1 - Y[s] / X[s] for s in SEEDS)


def full(*arms):
    return all(set(SEEDS) <= set(rows.get(a, {})) for a in arms)


summary = {a: {f: dict(per_seed=get(a, f), mean_se=mean_se(get(a, f).values())) for f in PICK}
           for a in ("B", "RQ", "B8", "R8", "Q8") if rows.get(a)}
curves = {a: [float(x) for x in np.mean([r["metrics"][KEY]["rollout_err_by_step"] for r in rows[a].values()], axis=0)]
          for a in ("B", "RQ", "B8", "R8", "Q8") if rows.get(a)}
P, detail = {f"E9-P{i}": None for i in range(1, 7)}, {}
if full("B", "RQ"):
    m, se = adv("B", "RQ", "err")
    detail["E9-P1"] = dict(adv=m, se=se, per_seed={s: 1 - get("RQ", "err")[s] / get("B", "err")[s] for s in SEEDS})
    P["E9-P1"] = m >= 0.09 and m > 2 * se
    P["E9-P2"] = m < 0.137
    detail["E9-P2"] = dict(adv=m, bound=0.137)
    m, se = adv("B", "RQ", "d20")
    detail["E9-P4"] = dict(adv_d20=m, se=se, adv_d50=adv("B", "RQ", "d50"))
    P["E9-P4"] = m >= 0.03 and m > 2 * se
    ts = {n: rows[n.split("_")[0]][int(n[-1])]["train_s"] for n in CARD}
    per_card = []
    for c in (0, 1):
        rq = [ts[n] for n in CARD if CARD[n] == c and n.startswith("RQ")]
        b = [ts[n] for n in CARD if CARD[n] == c and n.startswith("B")]
        per_card.append(float(np.mean(rq) / np.mean(b)))
    detail["E9-P5"] = dict(within_card=per_card, mean=float(np.mean(per_card)))
    P["E9-P5"] = float(np.mean(per_card)) < 1.15
if full("RQ", "R8", "Q8"):
    eRQ, eR, eQ = get("RQ", "err"), get("R8", "err"), get("Q8", "err")
    detail["E9-P3"] = dict(adv_over_R8=adv("R8", "RQ", "err"), adv_over_Q8=adv("Q8", "RQ", "err"),
                           per_seed={s: dict(RQ=eRQ[s], R8=eR[s], Q8=eQ[s]) for s in SEEDS})
    P["E9-P3"] = all(eRQ[s] < eR[s] and eRQ[s] < eQ[s] for s in SEEDS)
if full("B", "B8"):
    ch = {s: get("B", "err")[s] / get("B8", "err")[s] - 1 for s in SEEDS}
    detail["E9-P6"] = dict(err_change=ch, d20_change={s: get("B", "d20")[s] / get("B8", "d20")[s] - 1 for s in SEEDS})
    P["E9-P6"] = all(abs(v) < 0.005 for v in ch.values())
out = dict(arms=summary, predictions=P, detail=detail, curves=curves)
os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, detail=detail), indent=1))
