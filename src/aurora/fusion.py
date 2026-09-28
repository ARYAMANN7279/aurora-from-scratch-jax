"""E2: training on heterogeneous sources (handouts/e2_heterogeneous_fusion.md).

A source is one regime in one format: its own resolution N and its own variable subset. Three
methods fuse the sources:
  perceiver   one model; each source enters at its native N with its own variables (levels and
              per-variable embeddings absorb the differences)
  pad         pad-and-mask: every source is brought to the largest N (spectral zero-padding) and to
              all variables (zero-filled), with a per-variable presence field as an extra embedded
              input; predictions are truncated back to the source's N
  per_source  a separate embedding + Perceiver per source, one shared backbone
The loss and every metric cover only the variables a source provides, at its native N.
Configs go through experiments/run.py like harness configs (they carry "method").
"""
import copy
import json
import os
import time
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from kdata import make_splits, traj_dir
from kolmogorov.grid import irfft2, resample, rfft2

from . import model
from .embed import VARS
from .embed import init as init_embed
from .harness import REPO, _git_commit, config_hash, model_config, validate
from . import perceiver

METHODS = ("perceiver", "pad", "per_source")
MASKS = tuple(f"mask_{v}" for v in VARS)


def spectral_resample(x, n_out):
    """(..., N, N) real fields -> (..., n_out, n_out) by zero-padding / truncating Fourier modes."""
    if x.shape[-1] == n_out:
        return x
    flat = x.reshape(-1, *x.shape[-2:])
    out = jax.vmap(lambda f: irfft2(resample(rfft2(f), n_out), n_out))(flat)
    return out.reshape(*x.shape[:-2], n_out, n_out).astype(x.dtype)


def init(key, mc, method, sources, dtype=jnp.float32):
    p = model.init(key, mc, dtype=dtype)
    if method == "pad":
        extra = init_embed(jax.random.fold_in(key, 1), mc.D, mc.P, MASKS, dtype=dtype)["embed"]
        p["emb"]["embed"].update(extra)
    elif method == "per_source":
        emb, per = {}, {}
        for i, s in enumerate(sources):
            k1, k2 = jax.random.split(jax.random.fold_in(key, 100 + i))
            e = init_embed(k1, mc.D, mc.P, VARS, dtype=dtype)
            e["head"] = jax.tree_util.tree_map(jnp.zeros_like, e["head"])  # persistence at init
            emb[s["name"]], per[s["name"]] = e, perceiver.init(k2, mc.D, mc.L, dtype=dtype)
        p["emb"], p["per"] = emb, per
    return p


def predict(p, x_prev, x_cur, src, method, n_max, dt, mc):
    """x_prev, x_cur: (V_src, N, N) in src["vars"] order -> prediction (V_src, N, N), same order."""
    return _predict(p, x_prev, x_cur, src, method, n_max, dt, mc).astype(x_cur.dtype)


def _predict(p, x_prev, x_cur, src, method, n_max, dt, mc):
    vs, N = tuple(src["vars"]), x_cur.shape[-1]
    idx = jnp.array([VARS.index(v) for v in vs])
    if method == "perceiver":
        return model.apply(p, x_prev, x_cur, vs, dt, mc)[idx]
    if method == "per_source":
        q = dict(p, emb=p["emb"][src["name"]], per=p["per"][src["name"]])
        return model.apply(q, x_prev, x_cur, vs, dt, mc)[idx]
    # pad-and-mask
    def full(x):
        z = jnp.zeros((len(VARS), n_max, n_max), x.dtype).at[idx].set(spectral_resample(x, n_max))
        m = jnp.zeros((len(VARS), n_max, n_max), x.dtype).at[idx].set(1.0)
        return jnp.concatenate([z, m])
    y = model.apply(p, full(x_prev), full(x_cur), VARS + MASKS, dt, mc)
    return spectral_resample(y[idx], N)


def run(cfg, data_root=None, runs_root=None, results_path=None, verbose=False):
    """Train and evaluate one E2 config; append its row; return the row."""
    validate(cfg)
    h = config_hash(cfg)
    data_root = data_root or os.path.join(REPO, "data", "stage2")
    runs_root = runs_root or os.path.join(REPO, "data", "runs")
    results_path = results_path or os.path.join(REPO, "experiments", "results.jsonl")
    d, t, e, method = cfg["data"], cfg["train"], cfg["eval"], cfg["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    ds = os.path.join(data_root, d["dataset"])
    sources = d["sources"]
    for s in sources:  # regime-level protocol: train and val seeds of a source must be disjoint
        make_splits(ds, train=[{"tag": s["tag"], "seeds": s["train_seeds"]}],
                    val=[{"tag": s["tag"], "seeds": s["val_seeds"]}])
    n_max = max(s["N"] for s in sources)
    load = lambda s, seed: np.load(os.path.join(traj_dir(ds, s["tag"], seed), f"frames_{s['N']}.npy"))[
        :, [VARS.index(v) for v in s["vars"]]]

    # per-variable normalisation over every training frame that provides the variable
    acc = {v: [] for v in VARS}
    raw = {}
    for s in sources:
        raw[s["name"]] = [load(s, k) for k in s["train_seeds"]]
        for i, v in enumerate(s["vars"]):
            acc[v].append(np.concatenate([a[:, i] for a in raw[s["name"]]]).ravel())
    mu = {v: float(np.concatenate(acc[v]).mean()) if acc[v] else 0.0 for v in VARS}
    sd = {v: float(np.concatenate(acc[v]).std()) if acc[v] else 1.0 for v in VARS}

    def norm(s, a):
        m = np.array([mu[v] for v in s["vars"]])[:, None, None]
        q = np.array([sd[v] for v in s["vars"]])[:, None, None]
        return jnp.asarray(((a - m) / q).astype(np.float32))

    train = {s["name"]: jnp.stack([norm(s, a) for a in raw[s["name"]]]) for s in sources}
    dt = 0.2
    mc = model_config(cfg["model"])
    key = jax.random.PRNGKey(t["seed"])
    params = init(key, mc, method, sources)
    steps = t["steps"]
    lr = optax.cosine_decay_schedule(t["lr"], max(steps, 1), alpha=1e-2) if t.get("schedule") == "cosine" else t["lr"]
    opt = optax.adam(lr)
    state = opt.init(params)

    def make_step(s):
        fwd = jax.vmap(partial(predict, src=s, method=method, n_max=n_max, dt=dt, mc=mc), in_axes=(None, 0, 0))

        @jax.jit
        def step(q, st, xm, x0, xp):
            loss, g = jax.value_and_grad(lambda q: jnp.mean((fwd(q, xm, x0) - xp) ** 2))(q)
            upd, st = opt.update(g, st, q)
            return optax.apply_updates(q, upd), st, loss
        return step

    steppers = {s["name"]: make_step(s) for s in sources}
    rng = np.random.default_rng(t["seed"])
    B = t["batch"]
    t0 = time.time()
    loss = jnp.nan
    for _ in range(steps):
        s = sources[rng.integers(len(sources))]
        v = train[s["name"]]
        i = rng.integers(0, v.shape[0], B)
        k = rng.integers(1, v.shape[1] - 1, B)
        params, state, loss = steppers[s["name"]](params, state, v[i, k - 1], v[i, k], v[i, k + 1])

    metrics = {}
    n_roll = e.get("rollout_steps", 0)
    for s in sources:
        f = partial(predict, src=s, method=method, n_max=n_max, dt=dt, mc=mc)
        sq = jax.jit(lambda q, xm, x0, xp: (jnp.sum((jax.vmap(f, in_axes=(None, 0, 0))(q, xm, x0) - xp) ** 2),
                                            jnp.sum((x0 - xp) ** 2), jnp.sum(xp ** 2)))
        def rollout(q, a, b):
            def body(c, _):
                y = f(q, *c)
                return (c[1], y), y
            return jax.lax.scan(body, (a, b), None, length=n_roll)[1]
        roll = jax.jit(rollout) if n_roll else None
        a = b = c = 0.0
        drift = []
        for seed in s["val_seeds"]:
            v = norm(s, load(s, seed))
            F = v.shape[0]
            for j in range(1, F - 1, 16):
                k = np.arange(j, min(j + 16, F - 1))
                x, y, z = sq(params, v[k - 1], v[k], v[k + 1])
                a, b, c = a + float(x), b + float(y), c + float(z)
            if roll is not None:
                pred, tr = np.asarray(roll(params, v[49], v[50])), np.asarray(v[51:51 + n_roll])
                drift.append(float(np.mean(np.sqrt(((pred - tr) ** 2).sum(axis=(1, 2, 3)) / (tr ** 2).sum(axis=(1, 2, 3))))))
        row = dict(rel_err=float(np.sqrt(a / c)), persistence=float(np.sqrt(b / c)), ratio=float(np.sqrt(a / b)))
        if drift:
            row["rollout_drift"] = float(np.mean(drift))
        metrics[f"val/{s['name']}"] = row

    run_dir = os.path.join(runs_root, h)
    os.makedirs(run_dir, exist_ok=True)
    flat, _ = jax.flatten_util.ravel_pytree(params)
    np.savez(os.path.join(run_dir, "params.npz"), flat=np.asarray(flat),
             mean=np.array([mu[v] for v in VARS]), std=np.array([sd[v] for v in VARS]))
    out = dict(config_hash=h, name=cfg.get("name"), prediction=cfg["prediction"], git_commit=_git_commit(),
               started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), wall_s=time.time() - t0,
               n_params=int(flat.size), n_trainable=int(flat.size), final_loss=float(loss), metrics=metrics,
               config=copy.deepcopy(cfg))
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    if verbose:
        print({k: out[k] for k in ("config_hash", "name", "metrics", "wall_s")})
    return out
