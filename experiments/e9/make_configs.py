"""E9's 6 configs (handouts/e9_rpb_plus_qknorm.md, section 2), from E8's arm-B configs (the same training and
evaluation, E8's 84 rollouts included).

usage: make_configs.py   (writes configs.json)
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E8 = {c["name"]: c for c in json.load(open(os.path.join(HERE, "..", "e8", "configs.json")))}
out = []
for arm, what in (("RQ", "relative-position bias and q/k normalisation"), ("B", "none (the control, rerun as the session check)")):
    for seed in (0, 1, 2):
        c = copy.deepcopy(E8[f"e8_B_s{seed}"])
        c["name"] = f"e9_{arm}_s{seed}"
        c["prediction"] = f"E9 arm {arm} ({what}), seed {seed}: E9-P1 to E9-P6; handouts/e9_rpb_plus_qknorm.md"
        if arm == "RQ":
            c["model"]["backbone"].update(rel_pos_bias=True, qk_norm=True)
        out.append(c)
with open(os.path.join(HERE, "configs.json"), "w") as f:
    json.dump(out, f, indent=1)
print(len(out), "configs")
