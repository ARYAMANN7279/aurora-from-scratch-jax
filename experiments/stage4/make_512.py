"""512^2 frames for the C = 8 measurements (handouts/stage04_perceiver.md, section 4).

Re = 1000, k_f = 4, seed 0, generation grid 512, T_spin = 50 (as acc_v1), 40 frames at 0.2.
usage: make_512.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from kdata import generate_trajectory  # noqa: E402

meta, _ = generate_trajectory(os.path.join(ROOT, "data", "stage4", "f512"), 1000, 4, 0, 512,
                              T_spin=50.0, T_rec=7.8, save_dt=0.2, out_res=(512,), chunk_frames=10)
print(meta["n_frames"], "frames,", f"{meta['wall_s']:.0f}s")
