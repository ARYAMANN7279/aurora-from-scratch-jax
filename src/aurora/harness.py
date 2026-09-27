"""Ablation harness: one config -> one run -> one row (handouts/stage11_harness.md).

Enforces the protocol mechanically: a run needs a written prediction, and held-out regimes may not
leak into training (Stage 2 make_splits).
"""
import copy
import hashlib
import itertools
import json
import os
import subprocess
import time
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from kdata import make_splits, traj_dir

from . import backbone as bb
from . import lora as lora_mod
from . import model
from .rollout import mae, rollout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def canonical(cfg):
    c = copy.deepcopy(cfg)
    c.pop("name", None)
    c.pop("prediction", None)
    return json.dumps(c, sort_keys=True, separators=(",", ":"))


def config_hash(cfg):
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()[:16]


def validate(cfg):
    if not isinstance(cfg.get("prediction"), str) or not cfg["prediction"].strip():
        raise ValueError("config has no written prediction: pre-register before running")


def model_config(m):
    b = m["backbone"]
    return model.ModelConfig(m["D"], m["P"], m["L"], m["heads"],
                             bb.Config(m["D"], tuple(b["enc_depths"]), tuple(b["enc_heads"]),
                                       tuple(b["dec_depths"]), tuple(b["dec_heads"]), tuple(b["window"])))


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand(sweep):
    """{"base": config, "grid": {"train.seed": [0, 1], ...}} -> list of configs (Cartesian product)."""
    base, grid = sweep["base"], sweep.get("grid", {})
    keys = sorted(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        c = copy.deepcopy(base)
        for k, v in zip(keys, values):
            _set(c, k, v)
        c["name"] = base["name"] + "".join(f"__{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
        out.append(c)
    return out


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run(cfg, data_root=None, runs_root=None, results_path=None, verbose=False):
    """Train and evaluate one config; append its row to results_path; return the row."""
    validate(cfg)
    h = config_hash(cfg)
    data_root = data_root or os.path.join(REPO, "data", "stage2")
    runs_root = runs_root or os.path.join(REPO, "data", "runs")
    results_path = results_path or os.path.join(REPO, "experiments", "results.jsonl")
    d, m, t, e = cfg["data"], cfg["model"], cfg["train"], cfg["eval"]
    ds = os.path.join(data_root, d["dataset"])
    sp = make_splits(ds, train=d["train"], test=d.get("test", []), val=d.get("val", []))
    N, stride = d["N"], d.get("stride", 1)
    load = lambda tag, s: np.load(os.path.join(traj_dir(ds, tag, s), f"frames_{N}.npy"))[::stride]
    raw_train = [(tag, s, load(tag, s)) for tag, s in sp["train"]]
    if d.get("norm_from_init"):  # keep the normalisation the initial weights were trained with
        z = np.load(os.path.join(runs_root, t["init_from"], "params.npz"))
        mu, sd = z["mean"], z["std"]
    else:
        allf = np.concatenate([a for _, _, a in raw_train])
        mu, sd = allf.mean(axis=(0, 2, 3)), allf.std(axis=(0, 2, 3))
    norm = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
    tags = list(dict.fromkeys(tag for tag, _, _ in raw_train))
    by_tag = {tag: jnp.stack([norm(a) for tt, _, a in raw_train if tt == tag]) for tag in tags}
    dt = 0.2 * stride

    mc = model_config(m)
    key = jax.random.PRNGKey(t["seed"])
    template = model.init(key, mc, dtype=jnp.float32)
    _, unravel = jax.flatten_util.ravel_pytree(template)
    base = template
    if t.get("init_from"):
        z = np.load(os.path.join(runs_root, t["init_from"], "params.npz"))
        base = unravel(jnp.asarray(z["flat"]))
    if t.get("lora_rank"):
        r = t["lora_rank"]
        trainable, full = lora_mod.init(key, base, r=r), (lambda q: lora_mod.merge(base, q, r=r))
    else:
        trainable, full = base, (lambda q: q)

    steps = t["steps"]
    lr = optax.cosine_decay_schedule(t["lr"], max(steps, 1), alpha=1e-2) if t.get("schedule") == "cosine" else t["lr"]
    opt = optax.adam(lr)
    state = opt.init(trainable)
    fwd = jax.vmap(partial(model.apply, variables=model.VARS, dt=dt, mc=mc), in_axes=(None, 0, 0))
    lossf = (lambda a, b: jnp.mean((a - b) ** 2)) if t.get("loss", "mse") == "mse" else mae
    pf = bool(t.get("pushforward"))

    def objective(q, xm, x0, xp, xp2):
        p = full(q)
        y1 = fwd(p, xm, x0)
        if not pf:
            return lossf(y1, xp)
        return lossf(y1, xp) + lossf(fwd(p, x0, jax.lax.stop_gradient(y1)), xp2)

    @jax.jit
    def step(q, st, xm, x0, xp, xp2):
        loss, g = jax.value_and_grad(objective)(q, xm, x0, xp, xp2)
        upd, st = opt.update(g, st, q)
        return optax.apply_updates(q, upd), st, loss

    rng = np.random.default_rng(t["seed"])
    B = t["batch"]
    t0 = time.time()
    loss = jnp.nan
    for _ in range(steps):  # same sampling scheme as experiments/stage7/train_onestep.py
        v = by_tag[tags[rng.integers(len(tags))]]
        F = v.shape[1]
        s = rng.integers(0, v.shape[0], B)
        k = rng.integers(1, F - 2 if pf else F - 1, B)
        trainable, state, loss = step(trainable, state, v[s, k - 1], v[s, k], v[s, k + 1], v[s, np.minimum(k + 2, F - 1)])
    p = full(trainable)

    @jax.jit
    def sq(prm, xm, x0, xp):
        return jnp.sum((fwd(prm, xm, x0) - xp) ** 2), jnp.sum((x0 - xp) ** 2), jnp.sum(xp**2)

    roll = jax.jit(partial(rollout, n=e.get("rollout_steps", 0), dt=dt, mc=mc)) if e.get("rollout_steps") else None
    metrics = {}
    for split in e["splits"]:
        groups = {}
        for tag, s in sp[split]:
            groups.setdefault(tag, []).append(norm(load(tag, s)))
        for tag, arrs in groups.items():
            a = b = c = 0.0
            drift = []
            for v in arrs:
                F = v.shape[0]
                for j in range(1, F - 1, 16):
                    k = np.arange(j, min(j + 16, F - 1))
                    x, y, zz = sq(p, v[k - 1], v[k], v[k + 1])
                    a, b, c = a + float(x), b + float(y), c + float(zz)
                if roll is not None:
                    n = e["rollout_steps"]
                    pred = np.asarray(roll(p, v[49], v[50]))
                    tr = np.asarray(v[51:51 + n])
                    drift.append(float(np.mean(np.sqrt(((pred - tr) ** 2).sum(axis=(1, 2, 3)) / (tr**2).sum(axis=(1, 2, 3))))))
            row = dict(rel_err=float(np.sqrt(a / c)), persistence=float(np.sqrt(b / c)), ratio=float(np.sqrt(a / b)))
            if drift:
                row["rollout_drift"] = float(np.mean(drift))
            metrics[f"{split}/{tag}"] = row

    run_dir = os.path.join(runs_root, h)
    os.makedirs(run_dir, exist_ok=True)
    flat, _ = jax.flatten_util.ravel_pytree(p)
    np.savez(os.path.join(run_dir, "params.npz"), flat=np.asarray(flat), mean=mu, std=sd)
    n_tr = sum(x.size for x in jax.tree_util.tree_leaves(trainable)) if steps else 0
    out = dict(config_hash=h, name=cfg.get("name"), prediction=cfg["prediction"], git_commit=_git_commit(),
               started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), wall_s=time.time() - t0,
               n_params=int(flat.size), n_trainable=int(n_tr), final_loss=float(loss), metrics=metrics, config=cfg)
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(out) + "\n")
    if verbose:
        print(json.dumps({k: out[k] for k in ("config_hash", "name", "metrics", "wall_s")}))
    return out
