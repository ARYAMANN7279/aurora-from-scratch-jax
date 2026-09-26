"""Generate a dataset from a config (handouts/stage02_dataset.md, section 2). Resumable.

usage: generate.py --config experiments/stage2/configs/acc_v1.json [--shard 0 --n-shards 4]
       generate.py --config ... --finalize
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from kdata import finalize, generate_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--root", help="default: data/stage2/<config name>")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--finalize", action="store_true")
args = ap.parse_args()

with open(args.config) as f:
    cfg = json.load(f)
root = args.root or os.path.join(ROOT, "data", "stage2", cfg["name"])
if args.finalize:
    man = finalize(root)
    print(f"manifest: {len(man['trajectories'])} trajectories verified")
else:
    generate_dataset(cfg, root, shard=args.shard, n_shards=args.n_shards)
