"""E1 follow-up F1 (handouts/e1_volume_diversity.md, section F1): reruns of two E1 cells with training curves.

usage: make_curves.py   ->  experiments/e1/configs/curves.json
The cells (n = 2, T = 32) and (n = 32, T = 32), seeds 0-2, are E1's own configs with `train.curve_every` added
(pretraining and scratch every 500 steps, fine-tuning every 100). Each fine-tune points at its rerun pretrain.
The list is in run order (pretrains before the fine-tunes that start from them), for one lane.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from aurora.harness import config_hash  # noqa: E402
from e1_design import run_configs  # noqa: E402

CELLS, EVERY = ("n2_T32", "n32_T32"), {"pre": 500, "ft": 100, "scratch": 500}
PRED = ("E1 F1 rerun of {orig} with a training curve: its final one-step error is within 2% of {orig}'s (F1-P1)"
        "{extra}; handouts/e1_volume_diversity.md F1")


def curves():
    pre, ft, sc = run_configs(config_hash)
    by = {c["name"]: c for c in pre + ft + sc}
    out = []
    for s in (0, 1, 2):
        new_pre = {}
        for cell in CELLS:
            c = copy.deepcopy(by[f"e1_pre_{cell}_s{s}"])
            orig = c["name"]
            c["name"], c["prediction"] = f"e1c_pre_{cell}_s{s}", PRED.format(orig=orig, extra="")
            c["train"]["curve_every"] = EVERY["pre"]
            new_pre[cell] = c
            out.append(c)
        for cell in CELLS:
            c = copy.deepcopy(by[f"e1_ft_{cell}_s{s}"])
            orig = c["name"]
            c["name"], c["prediction"] = f"e1c_ft_{cell}_s{s}", PRED.format(orig=orig, extra="")
            c["train"].update(curve_every=EVERY["ft"], init_from=config_hash(new_pre[cell]))
            out.append(c)
        c = copy.deepcopy(by[f"e1_scratch_s{s}"])
        orig = c["name"]
        c["name"] = f"e1c_scratch_s{s}"
        c["prediction"] = PRED.format(orig=orig, extra=", and its final error is within 2% of its curve's minimum (F1-P2)")
        c["train"]["curve_every"] = EVERY["scratch"]
        out.append(c)
    return out


if __name__ == "__main__":
    cfgs = curves()
    path = os.path.join(HERE, "configs", "curves.json")
    with open(path, "w") as f:
        json.dump(cfgs, f, indent=1)
    for seed in (0, 1, 2):
        with open(os.path.join(HERE, "configs", f"curves_s{seed}.json"), "w") as f:
            json.dump([c for c in cfgs if c["train"]["seed"] == seed], f, indent=1)
    print(f"wrote {path}: {len(cfgs)} runs")
