"""Score Stage 4 F4 (handouts/stage04_perceiver.md, section 5, F4) from results/f4_main_192k_cos_s{0,1}.json.

For each run and resolution, the mean of the 10 evaluations ending at 192,000 steps (187,500-192,000), as in F3. A
prediction holds only if it holds in both runs.
F4-P1  16² < 0.02          F4-P2  64² < 0.02, 128² < 0.03, 512² < 0.05
F4-P3  16² below F3's run with the same key (results/f3_main_192k_s{0,1}.json, the same window)
F4-P4  no evaluation after step 144,000 exceeds twice the median of the 10 before it, at any resolution
Also: each run's end-points, the windows at 96,000 and 144,000 steps, and F3's numbers alongside.
usage: score_f4.py   (writes results/f4_analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS, RESOLUTIONS = (0, 1), ("16", "64", "128", "512")


def window(curve, end, res):
    steps = [r["step"] for r in curve]
    i = steps.index(end)
    return float(np.mean([r[f"eval_{res}"] for r in curve[i - 9:i + 1]]))


def spikes(curve, after):
    out = []
    for n in RESOLUTIONS:
        e = [r[f"eval_{n}"] for r in curve]
        out += [(curve[i]["step"], n, e[i]) for i in range(10, len(e))
                if curve[i]["step"] > after and e[i] > 2 * np.median(e[i - 10:i])]
    return out


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


out = dict(runs={}, f3={}, predictions={k: None for k in ("F4-P1", "F4-P2", "F4-P3", "F4-P4")})
for s in SEEDS:
    r, f3 = load(f"f4_main_192k_cos_s{s}"), load(f"f3_main_192k_s{s}")
    out["f3"][s] = {n: window(f3["curve"], 192000, n) for n in RESOLUTIONS}
    if r:
        c = r["curve"]
        out["runs"][s] = dict(at_192k={n: window(c, 192000, n) for n in RESOLUTIONS},
                              at_144k={n: window(c, 144000, n) for n in RESOLUTIONS},
                              at_96k={n: window(c, 96000, n) for n in RESOLUTIONS},
                              end_point=r["eval_rel_err"], late_spikes=spikes(c, 144000),
                              all_spikes=len(spikes(c, 0)), wall_h=r["wall_s"] / 3600)
if len(out["runs"]) == len(SEEDS):
    R = out["runs"]
    out["predictions"] = {
        "F4-P1": all(R[s]["at_192k"]["16"] < 0.02 for s in SEEDS),
        "F4-P2": all(R[s]["at_192k"]["64"] < 0.02 and R[s]["at_192k"]["128"] < 0.03 and R[s]["at_192k"]["512"] < 0.05
                     for s in SEEDS),
        "F4-P3": all(R[s]["at_192k"]["16"] < out["f3"][s]["16"] for s in SEEDS),
        "F4-P4": all(not R[s]["late_spikes"] for s in SEEDS),
    }
with open(os.path.join(RES, "f4_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
