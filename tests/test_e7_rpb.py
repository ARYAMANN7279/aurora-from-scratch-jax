"""E7 gates G1-G6 (handouts/e7_attention_rpb.md, section 3): a relative-position bias inside the attention
windows. Without the flag nothing changes (G1); with it, the other weights are those of the same seed and a
zero table computes the same backbone (G2); the offset index is exact (G3); the bias enters the logits as
specified (G4); a trained table changes the backbone (G5) and every table gets a gradient under jit (G6).
At initialisation Aurora's AdaLN modulation is zero, so every block is the identity and attention cannot
matter; G2, G5 and G6 therefore also run with a random modulation, the same for both models, as after
training starts. float64 unless stated."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from aurora import backbone as bb
from aurora import swin

WINDOW = (3, 4, 4)


def _cfg(rpb):  # the E1 backbone
    return bb.Config(64, (2, 2, 2), (2, 4, 8), (2, 2, 2), (8, 4, 2), WINDOW, rel_pos_bias=rpb)


def _inputs(dtype=jnp.float64, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((2, 3 * 16 * 16, 64)), dtype)
    t = jnp.asarray(rng.standard_normal((2, 64)), dtype)
    return x, t


def _modulate(p, seed):
    """Give every block a random AdaLN modulation (same draws for any model of this config), in place."""
    rng = np.random.default_rng(seed)
    for b in (b for l in p["enc"] + p["dec"] for b in l["blocks"]):
        for n in ("norm1", "norm2"):
            b[n] = {k: jnp.asarray(rng.standard_normal(v.shape) * 0.3, v.dtype) for k, v in b[n].items()}
    return p


def _strip(p):  # the params without the bias tables
    drop = lambda layers: [{**l, "blocks": [{k: v for k, v in b.items() if k != "rpb"} for b in l["blocks"]]}
                           for l in layers]
    return {**p, "enc": drop(p["enc"]), "dec": drop(p["dec"])}


def test_G1_default_config_has_no_bias():
    assert bb.Config(64, (2,), (2,), (2,), (2,), WINDOW).rel_pos_bias is False
    p = bb.init(jax.random.PRNGKey(0), _cfg(False), dtype=jnp.float64)
    assert not any("rpb" in b for l in p["enc"] + p["dec"] for b in l["blocks"])


def test_G2_zero_table_is_the_same_backbone():
    p0 = bb.init(jax.random.PRNGKey(0), _cfg(False), dtype=jnp.float64)
    p1 = bb.init(jax.random.PRNGKey(0), _cfg(True), dtype=jnp.float64)
    same = jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), p0, _strip(p1))
    assert all(jax.tree_util.tree_leaves(same)), "the flag changed a weight other than the tables"
    heads = [2, 2, 4, 4, 8, 8, 8, 8, 4, 4, 2, 2]
    tables = [b["rpb"] for l in p1["enc"] + p1["dec"] for b in l["blocks"]]
    assert [t.shape for t in tables] == [(245, h) for h in heads]
    assert all(float(jnp.abs(t).max()) == 0.0 for t in tables)
    assert sum(t.size for t in tables) == 245 * 56 == 13720
    x, t = _inputs()
    for state in ("init", "modulated"):
        if state == "modulated":
            p0, p1 = _modulate(p0, 7), _modulate(p1, 7)
        f0 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(False)))(p0)
        f1 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(True)))(p1)
        err = float(jnp.max(jnp.abs(f1 - f0)) / jnp.max(jnp.abs(f0)))
        print(f"\nG2 zero table vs no table, {state}: {err:.1e} (pass if <= 1e-12)")
        assert err <= 1e-12


def test_G3_offset_index_exact():
    for ws in ((3, 4, 4), (1, 4, 4), (3, 2, 2), (2, 4, 1)):
        idx = swin.rel_index(ws, WINDOW)
        coords = list(itertools.product(*[range(w) for w in ws]))  # the window's token order (C-major)
        ref = np.empty((len(coords), len(coords)), np.int64)
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                d = [a[k] - b[k] + WINDOW[k] - 1 for k in range(3)]
                ref[i, j] = (d[0] * (2 * WINDOW[1] - 1) + d[1]) * (2 * WINDOW[2] - 1) + d[2]
        assert np.array_equal(idx, ref), ws
        assert idx.min() >= 0 and idx.max() < swin.n_rel(WINDOW)
        centre = (swin.n_rel(WINDOW) - 1) // 2  # the zero offset
        assert (np.diag(idx) == centre).all() and np.array_equal(idx + idx.T, np.full_like(idx, 2 * centre))


def test_G4_bias_enters_the_logits():
    rng = np.random.default_rng(1)
    Bw, N, D, H = 6, 48, 32, 4
    r = lambda *shape: jnp.asarray(rng.standard_normal(shape) / 8)
    p = {"qkv": {"W": r(D, 3 * D), "b": r(3 * D)}, "proj": {"W": r(D, D), "b": r(D)}}
    xw = jnp.asarray(rng.standard_normal((Bw, N, D)))
    bias = jnp.asarray(rng.standard_normal((H, N, N)))
    mask = jnp.asarray(rng.random((3, N, N)) < 0.7) | jnp.eye(N, dtype=bool)[None]
    for m in (None, mask):
        got = swin.window_attention(p, xw, H, m, bias)
        qkv = (np.asarray(xw) @ np.asarray(p["qkv"]["W"]) + np.asarray(p["qkv"]["b"])).reshape(Bw, N, 3, H, D // H)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        lg = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D // H) + np.asarray(bias)[None]
        if m is not None:
            lg = np.where(np.tile(np.asarray(m), (Bw // 3, 1, 1))[:, None], lg, -np.inf)
        a = np.exp(lg - lg.max(-1, keepdims=True))
        o = (a / a.sum(-1, keepdims=True)) @ v
        ref = o.transpose(0, 2, 1, 3).reshape(Bw, N, D) @ np.asarray(p["proj"]["W"]) + np.asarray(p["proj"]["b"])
        err = float(np.max(np.abs(np.asarray(got) - ref)))
        assert err <= 1e-12, (m is None, err)


def test_G5_a_nonzero_table_changes_the_backbone():
    p0 = _modulate(bb.init(jax.random.PRNGKey(0), _cfg(False), dtype=jnp.float64), 7)
    p1 = _modulate(bb.init(jax.random.PRNGKey(0), _cfg(True), dtype=jnp.float64), 7)
    rng = np.random.default_rng(2)
    for b in (b for l in p1["enc"] + p1["dec"] for b in l["blocks"]):
        b["rpb"] = jnp.asarray(rng.standard_normal(b["rpb"].shape))
    x, t = _inputs()
    f0 = bb.apply(p0, x, t, (3, 16, 16), _cfg(False))
    f1 = bb.apply(p1, x, t, (3, 16, 16), _cfg(True))
    assert float(jnp.max(jnp.abs(f1 - f0))) > 1e-3


def test_G6_every_table_gets_a_gradient_under_jit():
    cfg = _cfg(True)
    p = _modulate(bb.init(jax.random.PRNGKey(3), cfg, dtype=jnp.float32), 8)
    x, t = _inputs(jnp.float32, seed=4)
    y = jnp.asarray(np.random.default_rng(5).standard_normal((2, 3 * 16 * 16, 128)), jnp.float32)
    loss = lambda q: jnp.mean((bb.apply(q, x, t, (3, 16, 16), cfg) - y) ** 2)
    g = jax.jit(jax.grad(loss))(p)
    norms = [float(jnp.linalg.norm(b["rpb"])) for l in g["enc"] + g["dec"] for b in l["blocks"]]
    assert len(norms) == 12 and all(np.isfinite(norms)) and min(norms) > 0, norms
