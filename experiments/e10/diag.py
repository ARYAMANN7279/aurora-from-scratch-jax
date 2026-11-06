"""E10: diagnostics of E1's trained models (handouts/e10_diagnostics.md).

usage: python experiments/e10/diag.py [--only NAME ...]   (writes experiments/e10/results/diag.json)
Needs data/runs/<hash>/params.npz for E1's scratch and fine-tune rows and data/stage2/e1_v1/re300_kf4.
Every measure comes from one pass per model: the decoded tokens that feed the four linear heads.
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
jax.config.update("jax_default_matmul_precision", "highest")

from aurora import backbone as bb  # noqa: E402
from aurora import encodings as enc  # noqa: E402
from aurora import harness, model, perceiver  # noqa: E402
from aurora.embed import VARS, embed  # noqa: E402
from aurora.levels import n_levels  # noqa: E402
from aurora.patches import patchify, unpatchify  # noqa: E402

DATA = os.path.join(ROOT, "data", "stage2", "e1_v1", "re300_kf4")
RUNS = os.path.join(ROOT, "data", "runs")
TRAIN, EVAL = (0, 1), (2, 3, 4, 5)
DT = 0.2


def tokens(p, x_prev, x_cur, mc):
    """Head inputs: the decoded tokens summed over output levels (T, D), and the level count C.
    model.apply's prediction is x_cur + unpatchify(S @ W_v + C b_v) for each field v."""
    H, W = x_cur.shape[-2:]
    C, T = n_levels(H), (H // mc.P) * (W // mc.P)
    tok = (embed(p["emb"], x_prev, VARS, mc.P) + p["frame"][0] + embed(p["emb"], x_cur, VARS, mc.P) + p["frame"][1])
    z = perceiver.aggregate(p["per"], tok, mc.heads)
    t_emb = enc.lead_time(jnp.asarray(DT, x_cur.dtype), mc.D).astype(x_cur.dtype)
    y = bb.apply(p["bb"], z.reshape(1, mc.L * T, mc.D), t_emb[None], (mc.L, H // mc.P, W // mc.P), mc.backbone)
    y = y[0].reshape(mc.L, T, 2 * mc.D) @ p["mid"]["W"] + p["mid"]["b"]
    return perceiver.decode(p["per"], y, C, mc.heads).sum(axis=0), C


def predict(S, C, heads, x_cur, mc):
    """(n, T, D) tokens -> (n, 4, H, W) predictions with heads {v: (W, b)}."""
    H = x_cur.shape[-1]
    return x_cur + jnp.stack([unpatchify(S @ heads[v][0] + C * heads[v][1], mc.P, H) for v in VARS], axis=1)


def refit(S, C, x_cur, x_next, mc):
    """Exact least-squares heads for the MSE of x_next given tokens S (float64 normal equations)."""
    F = np.concatenate([np.asarray(S, np.float64).reshape(-1, S.shape[-1]), np.ones((S.shape[0] * S.shape[1], 1))], 1)
    G = F.T @ F
    out = {}
    for i, v in enumerate(VARS):
        Y = np.asarray(patchify(x_next[:, i] - x_cur[:, i], mc.P), np.float64).reshape(-1, mc.P * mc.P)
        coef = np.linalg.solve(G, F.T @ Y)
        out[v] = (jnp.asarray(coef[:-1], jnp.float32), jnp.asarray(coef[-1] / C, jnp.float32))
    return out


def shells(N):
    k = np.fft.fftfreq(N, 1.0 / N)
    return np.rint(np.hypot(*np.meshgrid(k, k, indexing="ij"))).astype(int)


def shell_power(x, sh):
    """Power of (..., N, N) fields per integer shell |k|, summed over leading axes (Parseval-normalised)."""
    N = x.shape[-1]
    P = (np.abs(np.fft.fft2(np.asarray(x, np.float64))) ** 2 / N**2).reshape(-1, N * N).sum(0)
    return np.bincount(sh.ravel(), weights=P, minlength=sh.max() + 1)


def ddx(f, axis):
    N = f.shape[-1]
    k = np.fft.fftfreq(N, 1.0 / N)
    shape = [1] * f.ndim
    shape[axis] = N
    return np.real(np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(f, axis=axis), axis=axis))


def physics(phys):
    """(n, 4, N, N) physical fields on the 2π grid -> (divergence ratio D, curl mismatch C)."""
    u, v, w = (np.asarray(phys[:, i], np.float64) for i in (0, 1, 3))
    ux, uy, vx, vy = ddx(u, -1), ddx(u, -2), ddx(v, -1), ddx(v, -2)
    grad = np.sqrt(np.sum(ux**2 + uy**2 + vx**2 + vy**2))
    return float(np.linalg.norm(ux + vy) / grad), float(np.linalg.norm(w - (vx - uy)) / np.linalg.norm(w))


def pairs(seeds, norm):
    a = [norm(np.load(os.path.join(DATA, f"seed_{s:04d}", "frames_64.npy"))) for s in seeds]
    return tuple(jnp.concatenate([x[sl] for x in a]) for sl in (slice(0, -2), slice(1, -1), slice(2, None)))


def load_model(row):
    mc = harness.model_config(row["config"]["model"])
    _, unravel = jax.flatten_util.ravel_pytree(model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32))
    z = np.load(os.path.join(RUNS, row["config_hash"], "params.npz"))
    return mc, unravel(jnp.asarray(z["flat"])), z["mean"], z["std"]


def head_inputs(p, xm, x0, mc, batch=16):
    f = jax.jit(jax.vmap(lambda a, b: tokens(p, a, b, mc)[0]))
    return jnp.concatenate([f(xm[i:i + batch], x0[i:i + batch]) for i in range(0, xm.shape[0], batch)]), n_levels(x0.shape[-1])


def measure(row):
    mc, p, mu, sd = load_model(row)
    norm = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
    heads = {v: (p["emb"]["head"][v]["W"], p["emb"]["head"][v]["b"]) for v in VARS}
    tr, ev = pairs(TRAIN, norm), pairs(EVAL, norm)
    S_tr, C = head_inputs(p, tr[0], tr[1], mc)
    S_ev, _ = head_inputs(p, ev[0], ev[1], mc)
    new = refit(S_tr, C, tr[1], tr[2], mc)
    sh = shells(ev[1].shape[-1])
    out = {"name": row["name"], "config_hash": row["config_hash"], "stored_rel_err": row["metrics"]["val/re300_kf4"]["rel_err"]}
    for tag, hd in (("trained", heads), ("refit", new)):
        y_tr, y_ev = predict(S_tr, C, hd, tr[1], mc), predict(S_ev, C, hd, ev[1], mc)
        e_tr, e_ev = np.asarray(y_tr - tr[2], np.float64), np.asarray(y_ev - ev[2], np.float64)
        sq_ev = (e_ev**2).sum(axis=(0, 2, 3))
        r = {"train_mse": float((e_tr**2).mean()), "eval_mse": float((e_ev**2).mean()),
             "rel_err": float(np.sqrt(sq_ev.sum() / (np.asarray(ev[2], np.float64) ** 2).sum()))}
        if tag == "trained":
            sq_tr = (e_tr**2).sum(axis=(0, 2, 3))
            truth_ev = (np.asarray(ev[2], np.float64) ** 2).sum(axis=(0, 2, 3))
            phys = np.asarray(y_ev) * sd[None, :, None, None] + mu[None, :, None, None]
            r.update(share_train=dict(zip(VARS, (sq_tr / sq_tr.sum()).tolist())),
                     share_eval=dict(zip(VARS, (sq_ev / sq_ev.sum()).tolist())),
                     field_rel_err=dict(zip(VARS, np.sqrt(sq_ev / truth_ev).tolist())),
                     gap=r["eval_mse"] / r["train_mse"],
                     shell_err=shell_power(e_ev, sh).tolist(), shell_truth=shell_power(np.asarray(ev[2]), sh).tolist())
            r["div_ratio"], r["curl_mismatch"] = physics(phys)
        out[tag] = r
    return out


def e1_rows():
    rows = {}
    for f in ("rows_0", "rows_1", "rows_2", "rows_ft_0", "rows_ft_1", "rows_ft_2"):
        for line in open(os.path.join(ROOT, "experiments", "e1", "results", f + ".jsonl")):
            r = json.loads(line)
            if r["name"].startswith(("e1_scratch_s", "e1_ft_")):
                rows[r["name"]] = r
    return [rows[k] for k in sorted(rows)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "e10", "results", "diag.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for row in e1_rows():
        if (a.only and row["name"] not in a.only) or row["name"] in done:
            continue
        done[row["name"]] = measure(row)
        print(row["name"], round(done[row["name"]]["trained"]["rel_err"], 5), round(done[row["name"]]["stored_rel_err"], 5), flush=True)
        json.dump(done, open(a.out, "w"), indent=1)
