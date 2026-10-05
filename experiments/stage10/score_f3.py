"""Score Stage 10 F3 (handouts/stage10_lora.md, F3) from results/f3_<target>_<arm>_s<seed>.json and results/f3_closure.json.

For each target (re3000_kf4, re1000_kf4): the zero-shot error (F2's run; F3's own zero-shot run must equal it), and per
seed gain(arm) = 1 - e(arm) / e(zero-shot) for scratch, full and lora trained on 6 trajectories, mean and SE over
seeds 0-2 (ddof 1). The ceiling is whichever of scratch and full has the lower mean error.
F3-P1  ceiling gain at Re 3000 >= 0.20            F3-P4  ceiling gain at Re 1000 >= 0.20
F3-P2  lora gain at Re 3000 < 0.20                F3-P5  lora gain at Re 1000 < 0.20
F3-P3  e(full) < e(scratch) at Re 3000 (means)
F3-P6  closure ratio at Re 3000 < 0.5 x the base's zero-shot ratio there
F3-P7  closure ratio rises with Re: Re 100 < 300 < 1000 < 3000
The best error along each curve (every 1000 steps, chosen on the evaluation seeds) is reported as an optimistic
figure; the predictions use the final errors.
usage: score_f3.py   (writes results/f3_analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TARGETS, ARMS, SEEDS = ("re3000_kf4", "re1000_kf4"), ("scratch", "full", "lora"), (0, 1, 2)


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


out = dict(targets={}, predictions={f"F3-P{i}": None for i in range(1, 8)})
T = out["targets"]
for t in TARGETS:
    short = t.split("_")[0]
    zs, zs3 = load(f"f2_{short}_zeroshot"), load(f"f3_{short}_zeroshot")
    if not zs:
        continue
    row = dict(zero_shot=zs["eval"]["rel_err"], ratio_zs=zs["eval"]["ratio"],
               f3_zero_shot=zs3["eval"]["rel_err"] if zs3 else None)
    for arm in ARMS:
        runs = [load(f"f3_{short}_{arm}_s{s}") for s in SEEDS]
        if all(runs):
            e = [r["eval"]["rel_err"] for r in runs]
            best = [min(c["eval"]["rel_err"] for c in r["curve"]) for r in runs]
            row[arm] = dict(err=e, err_mean=float(np.mean(e)), gain=[1 - x / row["zero_shot"] for x in e],
                            gain_mean_se=mean_se([1 - x / row["zero_shot"] for x in e]),
                            ratio_mean=float(np.mean([r["eval"]["ratio"] for r in runs])),
                            best_along_curve=best, best_gain_mean=float(np.mean([1 - x / row["zero_shot"] for x in best])),
                            trainable_frac=runs[0]["trainable_frac"], steps=runs[0]["steps"], wall_s=[r["wall_s"] for r in runs])
    if "scratch" in row and "full" in row:
        c = min(("scratch", "full"), key=lambda a: row[a]["err_mean"])
        row["ceiling"] = dict(arm=c, gain=1 - row[c]["err_mean"] / row["zero_shot"])
    T[t] = row
P = out["predictions"]
done = lambda t: t in T and all(a in T[t] for a in ARMS)
if done("re3000_kf4"):
    P["F3-P1"] = T["re3000_kf4"]["ceiling"]["gain"] >= 0.20
    P["F3-P2"] = T["re3000_kf4"]["lora"]["gain_mean_se"][0] < 0.20
    P["F3-P3"] = T["re3000_kf4"]["full"]["err_mean"] < T["re3000_kf4"]["scratch"]["err_mean"]
if done("re1000_kf4"):
    P["F3-P4"] = T["re1000_kf4"]["ceiling"]["gain"] >= 0.20
    P["F3-P5"] = T["re1000_kf4"]["lora"]["gain_mean_se"][0] < 0.20
cl = load("f3_closure")
if cl:
    out["closure"] = {t: dict(ratio=r["ratio"], by_channel=r["by_channel"], by_seed=r["by_seed"]) for t, r in cl.items()}
    if "re3000_kf4" in T:
        P["F3-P6"] = cl["re3000_kf4"]["ratio"] < 0.5 * T["re3000_kf4"]["ratio_zs"]
    r = [cl[t]["ratio"] for t in ("re100_kf4", "re300_kf4", "re1000_kf4", "re3000_kf4")]
    P["F3-P7"] = bool(all(a < b for a, b in zip(r, r[1:])))
with open(os.path.join(RES, "f3_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=P, ceilings={t: r.get("ceiling") for t, r in T.items()},
                      gains={t: {a: r[a]["gain_mean_se"] for a in ARMS if a in r} for t, r in T.items()},
                      closure={t: r["ratio"] for t, r in (cl or {}).items()}), indent=1))
