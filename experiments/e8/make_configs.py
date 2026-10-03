"""E8's 18 configs (handouts/e8_followups.md, section 4), from E7's arm-B configs (E1's scratch task).

usage: make_configs.py   (writes configs.json)
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E7 = {c["name"]: c for c in json.load(open(os.path.join(HERE, "..", "e7", "configs.json")))}
ARMS = {  # arm: (what changes, how)
    "B": ("none (the control)", {}),
    "R": ("relative-position bias", {"model.backbone.rel_pos_bias": True}),
    "BP": ("pushforward", {"train.pushforward": True}),
    "RP": ("relative-position bias and pushforward", {"model.backbone.rel_pos_bias": True, "train.pushforward": True}),
    "Q": ("q/k normalisation", {"model.backbone.qk_norm": True}),
    "BS": ("scan chunks of 50 steps", {"train.scan_chunk": 50}),
}
EVAL = {"drift_every": 20, "drift_steps": 50}  # 21 starts per trajectory, 84 rollouts per run


def set_(d, dotted, v):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = v


out = []
for arm, (what, changes) in ARMS.items():
    for seed in (0, 1, 2):
        c = copy.deepcopy(E7[f"e7_B_s{seed}"])
        c["name"] = f"e8_{arm}_s{seed}"
        c["prediction"] = f"E8 arm {arm} ({what}), seed {seed}: E8-P1 to E8-P10; handouts/e8_followups.md"
        for k, v in changes.items():
            set_(c, k, v)
        c["eval"].update(EVAL)
        out.append(c)
with open(os.path.join(HERE, "configs.json"), "w") as f:
    json.dump(out, f, indent=1)
print(len(out), "configs")
