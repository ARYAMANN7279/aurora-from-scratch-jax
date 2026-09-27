"""Write the E3, E4, E5 run configs from experiments/designs.py.

usage: make_configs.py   ->  experiments/e3/configs/{pretrain,finetune,scratch}.json,
                             experiments/e4/configs/runs.json, experiments/e5/configs/{pretrain,finetune,scratch}.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from aurora.harness import config_hash  # noqa: E402
import designs  # noqa: E402


def write(exp, files):
    d = os.path.join(HERE, exp, "configs")
    os.makedirs(d, exist_ok=True)
    for name, cfgs in files.items():
        with open(os.path.join(d, name), "w") as f:
            json.dump(cfgs, f, indent=1)
    print(exp, {k: len(v) for k, v in files.items()})


pre, ft, sc = designs.e3(config_hash)
write("e3", {"pretrain.json": pre, "finetune.json": ft, "scratch.json": sc})
write("e4", {"runs.json": designs.e4()})
pre, ft, sc = designs.e5(config_hash)
write("e5", {"pretrain.json": pre, "finetune.json": ft, "scratch.json": sc})
