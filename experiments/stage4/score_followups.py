"""Score the Stage 4 follow-ups F1 and F2 (handouts/stage04_perceiver.md, section 5) from their result JSONs.

F1-P1: error < 0.029; F1-P2: error > 0.02 (the 16² control trained for 48,000 steps).
F2: every error within 1.3x of the single-resolution control at the same C, read as
    error / control <= 1.3 (a shared model better than its control also counts as within);
    and S4, error at 512² < 0.05.
usage: score_followups.py   (uses whichever of the two runs has a result; writes results/followups_analysis.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


out = {}
f1 = load("f1_nocompress_16_48k")
if f1:
    e = f1["eval_rel_err"]["16"]
    curve = {r["step"]: r["eval_16"] for r in f1["curve"]}
    out["F1"] = dict(error=e, control_6000=load("nocompress_16")["eval_rel_err"]["16"],
                     curve_at={s: curve.get(s) for s in (6000, 12000, 24000, 36000, 48000)},
                     F1_P1=dict(bar="error < 0.029", hit=e < 0.029), F1_P2=dict(bar="error > 0.02", hit=e > 0.02))
f2 = load("f2_main_24k")
if f2:
    rows = {}
    for n, c in {"16": 3, "64": 5, "128": 6, "512": 8}.items():
        e, ctrl = f2["eval_rel_err"][n], load(f"nocompress_{n}")["eval_rel_err"][n]
        rows[n] = dict(C=c, error=e, control=ctrl, ratio_to_control=e / ctrl, main_6000=load("main")["eval_rel_err"][n])
    out["F2"] = dict(rows=rows,
                     within_1_3x=dict(bar="error / control <= 1.3 at every C",
                                      hit=all(r["ratio_to_control"] <= 1.3 for r in rows.values())),
                     S4=dict(bar="error at 512² < 0.05", hit=rows["512"]["error"] < 0.05))
with open(os.path.join(RES, "followups_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
