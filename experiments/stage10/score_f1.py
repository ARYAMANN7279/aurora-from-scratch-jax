"""Score Stage 10 follow-up F1 (handouts/stage10_lora.md: F1 and amendment F1-A1) from its result JSONs.

gain(arm) = 1 - e(arm) / e(zero-shot), in percent, every arm on the retrained base model (F1-A1).
F1-P1: gain(1) <= gain(4) <= gain(16) <= gain(64) within 0.1 points, read as: each gain is at least
       the previous one minus 0.1 points.
F1-P2: gain(64) < 1.5 (the registered bar); gain(64) / gain(full) is reported beside it.
usage: score_f1.py   (writes results/f1_analysis.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES, S7 = os.path.join(HERE, "results"), os.path.join(HERE, "..", "stage7", "results")
load = lambda d, n: json.load(open(os.path.join(d, f"{n}.json")))

zs = load(RES, "f1_zeroshot")
lora = {r: load(RES, f"f1_lora_r{r}") for r in (1, 4, 16, 64)}
full = load(RES, "f1_full")
gain = lambda d: 100 * (1 - d["eval"]["rel_err"] / zs["eval"]["rel_err"])
g = {r: gain(d) for r, d in lora.items()}
chain = (1, 4, 16, 64)
retrain, orig = load(S7, "onestep_mse_retrain")["eval"], load(S7, "onestep_mse")["eval"]
out = dict(
    base_check={t: dict(retrain=retrain[t]["rel_err"], original=orig[t]["rel_err"],
                        retrain_ratio=retrain[t]["ratio"], original_ratio=orig[t]["ratio"]) for t in orig},
    zeroshot=zs["eval"],
    lora={r: dict(rel_err=d["eval"]["rel_err"], gain_pct=g[r], trainable_frac=d["trainable_frac"]) for r, d in lora.items()},
    full=dict(rel_err=full["eval"]["rel_err"], gain_pct=gain(full)),
    F1_P1=dict(bar="each gain >= the previous one - 0.1 points, over r = 1, 4, 16, 64",
               hit=all(g[b] >= g[a] - 0.1 for a, b in zip(chain, chain[1:]))),
    F1_P2=dict(bar="gain(64) < 1.5", hit=g[64] < 1.5, gain64_over_full=g[64] / gain(full)),
)
with open(os.path.join(RES, "f1_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
