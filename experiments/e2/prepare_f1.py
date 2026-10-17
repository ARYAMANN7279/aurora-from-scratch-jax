"""Assemble E2 F1 without copying or changing the earlier datasets.

Run on CPU: prepare_f1.py [--data-root data/stage2]. Reuses 12 complete trajectories,
generates only Re 30 seeds 2–5, then verifies all 16 trajectories' checksums.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from kdata import finalize, generate_dataset, traj_dir  # noqa: E402


def prepare(data_root):
    data_root = Path(data_root).resolve()
    dest = data_root / "e2f1_v1"
    cfg = dict(name="e2f1_v1", regimes=[dict(Re=30, kf=4, N_gen=128),
                                      dict(Re=3000, kf=4, N_gen=1024)],
               seeds=list(range(8)), T_spin=50.0, T_rec=100.0, save_dt=0.2, N_out=[64, 128])
    reused = []
    for re, source, seeds in ((30, "s10f2_v1", (0, 1, 6, 7)), (3000, "s10f3_v1", range(8))):
        for seed in seeds:
            tag = f"re{re}_kf4"
            src = Path(traj_dir(str(data_root / source), tag, seed))
            dst = Path(traj_dir(str(dest), tag, seed))
            if not (src / "meta.json").is_file():
                raise FileNotFoundError(src / "meta.json")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(os.path.relpath(src, dst.parent), target_is_directory=True)
            if dst.resolve() != src.resolve():
                raise ValueError(f"refusing to replace existing trajectory: {dst}")
            reused.append(dict(tag=tag, seed=seed, source=source))
    generate_dataset(cfg, str(dest))
    man = finalize(str(dest))
    provenance = dict(reused=reused, generated=[dict(tag="re30_kf4", seed=s) for s in range(2, 6)],
                      verified_trajectories=len(man["trajectories"]))
    (ROOT / "experiments/e2/results/f1_data.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(f"Verified {len(man['trajectories'])} trajectories; reused 12, generated 4", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=ROOT / "data/stage2")
    prepare(ap.parse_args().data_root)
