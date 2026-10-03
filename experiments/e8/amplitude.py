"""E8, exploratory (not a pre-registered prediction): what the drift measures late in a rollout.

For the kept weights of arms B, R, BP, RP and Q, re-runs E8's 84 rollouts (50 steps from every 20th frame from frame
50 on, evaluation trajectories 2-5) on the CPU and records at each step k, averaged over the rollouts:
  err(k)   the relative error, as rollout_err_by_step (checked against the GPU's numbers);
  amp(k)   RMS(prediction) / RMS(truth): below 1 means the predicted fields are damped;
  corr(k)  the correlation between prediction and truth over variables and grid points.
For fields that no longer correlate, err = sqrt(1 + amp^2): an error below sqrt(2) late in a rollout can come from
damping rather than tracking.
usage: amplitude.py <params dir>   (the job's params/<name>.npz; writes results/amplitude.json)
"""
import json
import os
import sys
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aurora import harness, model  # noqa: E402
from aurora.rollout import rollout  # noqa: E402
from kdata import traj_dir  # noqa: E402

rows = {json.loads(l)["name"]: json.loads(l) for l in open(os.path.join(HERE, "results", "rows.jsonl"))}
ds = os.path.join(ROOT, "data", "stage2", "e1_v1")
AX = (2, 3, 4)
out, fns = {}, {}
for name in sorted(rows):
    arm = name.split("_")[1]
    if arm not in ("B", "R", "BP", "RP", "Q"):
        continue
    mc = harness.model_config(rows[name]["config"]["model"])
    if mc not in fns:  # one compilation per architecture
        fns[mc] = jax.jit(jax.vmap(partial(rollout, n=50, dt=0.2, mc=mc), in_axes=(None, 0, 0)))
    z = np.load(os.path.join(sys.argv[1], f"{name}.npz"))
    template = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32)
    p = jax.flatten_util.ravel_pytree(template)[1](jnp.asarray(z["flat"]))
    E, A, C = [], [], []
    for s in (2, 3, 4, 5):
        v = np.load(os.path.join(traj_dir(ds, "re300_kf4", s), "frames_64.npy"))
        v = ((v - z["mean"][:, None, None]) / z["std"][:, None, None]).astype(np.float32)
        st0 = np.arange(50, v.shape[0] - 50, 20)
        pred = np.asarray(fns[mc](p, jnp.asarray(v[st0 - 1]), jnp.asarray(v[st0])))
        tr = v[st0[:, None] + np.arange(1, 51)]
        E.append(np.sqrt(((pred - tr) ** 2).sum(AX) / (tr**2).sum(AX)))
        A.append(np.sqrt((pred**2).sum(AX) / (tr**2).sum(AX)))
        pc, tc = pred - pred.mean(AX, keepdims=True), tr - tr.mean(AX, keepdims=True)
        C.append((pc * tc).sum(AX) / np.sqrt((pc**2).sum(AX) * (tc**2).sum(AX)))
    E, A, C = (np.concatenate(x) for x in (E, A, C))
    gpu = rows[name]["metrics"]["val/re300_kf4"]["rollout_err_by_step"]
    out[name] = dict(n_rollouts=int(E.shape[0]), err=E.mean(0).tolist(), amp=A.mean(0).tolist(), corr=C.mean(0).tolist(),
                     err_gpu=gpu)
    print(name, "err(1) cpu/gpu %.4f/%.4f" % (E.mean(0)[0], gpu[0]), "| k=20: err %.3f amp %.3f corr %.3f" %
          (E.mean(0)[19], A.mean(0)[19], C.mean(0)[19]), "| k=50: err %.3f amp %.3f corr %.3f" %
          (E.mean(0)[49], A.mean(0)[49], C.mean(0)[49]), flush=True)
with open(os.path.join(HERE, "results", "amplitude.json"), "w") as f:
    json.dump(out, f, indent=1)
