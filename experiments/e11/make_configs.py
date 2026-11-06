"""E11's 9 configs (handouts/e11_omega_only.md, section 4), from E8's arm-B configs (E1's scratch task).

usage: make_configs.py   (writes configs.json)
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E8 = {c["name"]: c for c in json.load(open(os.path.join(HERE, "..", "e8", "configs.json")))}
ARMS = {  # arm: (what changes, how)
    "B": ("none (the control)", {}),
    "W": ("ω only, u, v, p derived; loss over all four", {"model.derive_from_omega": True}),
    "WO": ("ω only, u, v, p derived; loss on ω", {"model.derive_from_omega": True, "train.loss_fields": ["omega"]}),
}


def set_(d, dotted, v):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = v


out = []
for arm, (what, changes) in ARMS.items():
    for seed in (0, 1, 2):
        c = copy.deepcopy(E8[f"e8_B_s{seed}"])
        c["name"] = f"e11_{arm}_s{seed}"
        c["prediction"] = f"E11 arm {arm} ({what}), seed {seed}: E11-P1 to E11-P6; handouts/e11_omega_only.md"
        for k, v in changes.items():
            set_(c, k, v)
        out.append(c)
with open(os.path.join(HERE, "configs.json"), "w") as f:
    json.dump(out, f, indent=1)
print(len(out), "configs")
