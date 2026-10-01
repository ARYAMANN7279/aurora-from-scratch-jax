"""E6 held-out split and test windows (handouts/e6_realpde_sim2real.md, section 2); numpy only, so
that both the JAX runs and the PyTorch FNO baselines use the same definition."""
import numpy as np

HOLD = (3750, 5025, 25425, 26700)  # UGP-I's re_lohi split: the 2 lowest and 2 highest Re


def val_starts(meta, n=900):
    """UGP-I's r12_eval.val_starts(900, "re_lohi"), reproduced line for line (seed 0)."""
    off, lens, names = meta["off"], meta["lens"], meta["names"]
    RE = np.array([int(x.split("_")[0]) for x in names])
    VT = set(np.where(np.isin(RE, list(HOLD)))[0].tolist())
    st = [t0 for i in range(len(lens)) if i in VT for t0 in range(off[i], off[i] + lens[i] - 39)]
    rng = np.random.default_rng(0)
    return rng.choice(np.array(st), size=min(n, len(st)), replace=False)
