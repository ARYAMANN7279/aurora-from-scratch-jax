"""Score Stage 4 F3 (handouts/stage04_perceiver.md, section 5, F3) from results/f3_main_192k_s{0,1}.json.

For each run and resolution, the mean of the 10 evaluations ending at 192,000 steps (187,500-192,000) and at
96,000 (91,500-96,000); a prediction holds only if it holds in both runs.
F3-P1  16² < 0.02 at 192,000          F3-P2  64² < 0.02 at 192,000
F3-P3  128² < 0.03 and 512² < 0.05 at 192,000
F3-P4  16² >= 0.02 at 96,000
Also: each run's end-points, the first step whose 10-evaluation window at 16² is below 0.02, and F1's control
(the same window measure over its last 10 evaluations) for comparison.
usage: score_f3.py   (writes results/f3_analysis.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS, RESOLUTIONS = (0, 1), ("16", "64", "128", "512")


def window(curve, end, n, res):
    steps = [r["step"] for r in curve]
    i = steps.index(end)
    return float(np.mean([r[f"eval_{res}"] for r in curve[i - 9:i + 1]])) if i >= 9 else None


runs = {}
for s in SEEDS:
    p = os.path.join(RES, f"f3_main_192k_s{s}.json")
    if os.path.exists(p):
        runs[s] = json.load(open(p))
out = dict(runs={}, predictions={k: None for k in ("F3-P1", "F3-P2", "F3-P3", "F3-P4")})
for s, r in runs.items():
    c = r["curve"]
    first = next((row["step"] for i, row in enumerate(c) if i >= 9 and window(c, row["step"], 10, "16") < 0.02), None)
    out["runs"][s] = dict(at_192k={n: window(c, 192000, 10, n) for n in RESOLUTIONS},
                          at_96k={n: window(c, 96000, 10, n) for n in RESOLUTIONS},
                          end_point=r["eval_rel_err"], first_16_window_below_0_02=first,
                          wall_h=r["wall_s"] / 3600, step_s=r["step_s"])
if len(runs) == len(SEEDS):
    R = out["runs"].values()
    out["predictions"] = {
        "F3-P1": all(x["at_192k"]["16"] < 0.02 for x in R),
        "F3-P2": all(x["at_192k"]["64"] < 0.02 for x in R),
        "F3-P3": all(x["at_192k"]["128"] < 0.03 and x["at_192k"]["512"] < 0.05 for x in R),
        "F3-P4": all(x["at_96k"]["16"] >= 0.02 for x in R),
    }
f1 = json.load(open(os.path.join(RES, "f1_nocompress_16_48k.json")))
out["f1_control_16_last10"] = float(np.mean([r["eval_16"] for r in f1["curve"][-10:]]))
with open(os.path.join(RES, "f3_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(dict(predictions=out["predictions"], runs={s: {k: v for k, v in x.items() if k != "step_s"}
                                                            for s, x in out["runs"].items()}), indent=1))
