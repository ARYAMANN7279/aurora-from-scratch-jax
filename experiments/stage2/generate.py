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
ap.add_argument("--partial", action="store_true", help="with --finalize: manifest of completed trajectories only")
ap.add_argument("--only", nargs="*", help="restrict to trajectories tag:seed, generated in this order")
args = ap.parse_args()

with open(args.config) as f:
    cfg = json.load(f)
root = args.root or os.path.join(ROOT, "data", "stage2", cfg["name"])
if args.finalize:
    man = finalize(root, partial=args.partial)
    print(f"manifest: {len(man['trajectories'])} trajectories verified")
elif args.only:
    from kdata import generate_trajectory, regime_tag, traj_dir  # noqa: E402

    regs = {regime_tag(r["Re"], r["kf"]): r for r in cfg["regimes"]}
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(os.path.join(root, "dataset.json")):
        json.dump(cfg, open(os.path.join(root, "dataset.json"), "w"), indent=1)
    for item in args.only:
        tag, seed = item.split(":")
        r, seed = regs[tag], int(seed)
        if seed not in cfg["seeds"] or os.path.exists(os.path.join(traj_dir(root, tag, seed), "meta.json")):
            continue
        meta, _ = generate_trajectory(root, r["Re"], r["kf"], seed, r["N_gen"], cfg["T_spin"], cfg["T_rec"],
                                      cfg["save_dt"], tuple(cfg["N_out"]))
        print(f"done {tag} seed {seed} in {meta['wall_s']:.0f}s", flush=True)
else:
    generate_dataset(cfg, root, shard=args.shard, n_shards=args.n_shards)
