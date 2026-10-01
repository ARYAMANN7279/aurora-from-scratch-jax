"""E6 data (handouts/e6_realpde_sim2real.md, section 2).

Simulated foil frames at the competition's 32 x 64, prepared exactly as UGP-I prepared the real
ones (UGP/local_harness/prep_train2.py): each trajectory strided [::2, ::2] and cropped to 32 x 64,
then all trajectories concatenated along time, with offsets so that no window crosses a boundary.
The channels are u, v, p. UGP-I's real frames and their metadata are copied unchanged, with the
sources' sha256 recorded.

usage (needs h5py; the lab machine's Aryamann/env has it): python experiments/e6/prep_sim.py
Writes data/e6/{sim_frames.npy, sim_meta.json, real_frames.npy, real_meta.json, prep.json}.
"""
import glob
import hashlib
import json
import os
import shutil

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
UGP = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
OUT = os.path.join(ROOT, "data", "e6")
os.makedirs(OUT, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


parts, names, lens, res, aoas, dts = [], [], [], [], [], []
for fp in sorted(glob.glob(f"{UGP}/data/comp_sim/train_sim/*.h5")):
    with h5py.File(fp, "r") as f:
        arr = np.stack([np.asarray(f[k], dtype=np.float32)[:, ::2, ::2][:, :32, :64] for k in ("u", "v", "p")], -1)
        res.append(int(np.asarray(f["re"]))), aoas.append(int(np.asarray(f["aoa"])))
        dts.append(float(np.mean(np.diff(np.asarray(f["t"], dtype=np.float64)))))
    parts.append(arr)
    names.append(os.path.basename(fp))
    lens.append(arr.shape[0])
X = np.concatenate(parts, 0)
off = np.cumsum([0] + lens).tolist()
np.save(os.path.join(OUT, "sim_frames.npy"), X)
json.dump(dict(names=names, lens=lens, off=off, re=res, aoa=aoas, dt=dts, channels=["u", "v", "p"]),
          open(os.path.join(OUT, "sim_meta.json"), "w"))

src = {"real_frames.npy": f"{UGP}/local_harness/tr_frames.npy", "real_meta.json": f"{UGP}/local_harness/tr_meta.json"}
for dst, s in src.items():
    shutil.copyfile(s, os.path.join(OUT, dst))
real_meta = json.load(open(os.path.join(OUT, "real_meta.json")))
summary = dict(
    sim=dict(trajectories=len(names), frames=int(X.shape[0]), shape=list(X.shape[1:]), dt_mean=float(np.mean(dts)),
             mean=X.reshape(-1, 3).mean(0).tolist(), std=X.reshape(-1, 3).std(0).tolist(), zeros=float((X[..., 0] == 0).mean())),
    real=dict(trajectories=len(real_meta["names"]), frames=int(sum(real_meta["lens"]))),
    sources={k: dict(path=v, sha256=sha256(v)) for k, v in src.items()},
    command="python experiments/e6/prep_sim.py")
json.dump(summary, open(os.path.join(OUT, "prep.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
