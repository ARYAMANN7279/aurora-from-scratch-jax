"""Window loader and regime splits (handouts/stage02_dataset.md, section 2.5). Pure numpy."""
import json
import os

import numpy as np

from .generate import traj_dir


def read_manifest(root):
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def _entries(man, spec):
    """spec items: a regime tag ("re300_kf4") or {"tag": ..., "seeds": [...]} -> list of (tag, seed)."""
    have = {(t["tag"], t["seed"]) for t in man["trajectories"]}
    out = []
    for item in spec:
        tag, seeds = (item, None) if isinstance(item, str) else (item["tag"], item.get("seeds"))
        found = sorted(s for (t, s) in have if t == tag and (seeds is None or s in seeds))
        if not found or (seeds is not None and len(found) != len(seeds)):
            raise ValueError(f"split entry {item!r} does not match the manifest")
        out += [(tag, s) for s in found]
    return out


def make_splits(root, train, test=(), val=()):
    """Regime-level splits. A regime used in `test` may not appear in `train` or `val`.
    `train` and `val` may share a regime only with disjoint seeds."""
    man = read_manifest(root)
    sp = {"train": _entries(man, train), "val": _entries(man, val), "test": _entries(man, test)}
    test_regimes = {t for t, _ in sp["test"]}
    for name in ("train", "val"):
        clash = test_regimes & {t for t, _ in sp[name]}
        if clash:
            raise ValueError(f"held-out regime(s) {sorted(clash)} also in {name}")
    overlap = set(sp["train"]) & set(sp["val"])
    if overlap:
        raise ValueError(f"trajectories in both train and val: {sorted(overlap)}")
    return sp


class WindowDataset:
    """Windows of n_in + n_out frames spaced by `stride`, never crossing a trajectory."""

    def __init__(self, root, n_res, members, stride=1, n_in=2, n_out=1):
        self.n_in, self.n_out, self.stride = n_in, n_out, stride
        self.members = list(members)
        self.arrays = [np.load(os.path.join(traj_dir(root, t, s), f"frames_{n_res}.npy"), mmap_mode="r")
                       for t, s in self.members]
        self.span = (n_in + n_out - 1) * stride
        idx = [(i, t) for i, a in enumerate(self.arrays) for t in range(a.shape[0] - self.span)]
        self.index = np.array(idx, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def window(self, j):
        i, t = self.index[j]
        w = np.asarray(self.arrays[i][t:t + self.span + 1:self.stride])
        return w[: self.n_in], w[self.n_in:]

    def batches(self, batch_size, seed, shuffle=True, drop_last=True):
        order = np.random.default_rng(seed).permutation(len(self)) if shuffle else np.arange(len(self))
        stop = len(order) - (len(order) % batch_size if drop_last else 0)
        for b in range(0, stop, batch_size):
            ws = [self.window(j) for j in order[b:b + batch_size]]
            yield np.stack([w[0] for w in ws]), np.stack([w[1] for w in ws])

    def channel_stats(self):
        """Per-channel mean and std over every frame of this dataset's trajectories (float64)."""
        s1 = s2 = 0.0
        n = 0
        for a in self.arrays:
            x = np.asarray(a, dtype=np.float64)
            s1 = s1 + x.sum(axis=(0, 2, 3))
            s2 = s2 + (x**2).sum(axis=(0, 2, 3))
            n += x.shape[0] * x.shape[2] * x.shape[3]
        mean = s1 / n
        return mean, np.sqrt(s2 / n - mean**2)
