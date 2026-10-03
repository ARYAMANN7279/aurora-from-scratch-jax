"""E8 gates Q1-Q6 (handouts/e8_followups.md, section 3): q/k normalisation in the attention windows. Without the flag
nothing changes (Q1); with it, the other weights are those of the same seed (Q2); the normalisation enters the
attention as specified (Q3) and makes it blind to the scale of q and k (Q4); at initialisation the model computes the
same backbone (Q5), and every gain and bias gets a gradient under jit (Q6). As in E7's gates, Aurora's AdaLN
modulation is zero at initialisation, so Q6 runs with a random modulation. float64 unless stated."""
import jax
import jax.numpy as jnp
import numpy as np

from aurora import backbone as bb
from aurora import swin

WINDOW = (3, 4, 4)


def _cfg(qkn, rpb=False):  # the E1 backbone
    return bb.Config(64, (2, 2, 2), (2, 4, 8), (2, 2, 2), (8, 4, 2), WINDOW, rel_pos_bias=rpb, qk_norm=qkn)


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


def _strip(p):  # the params without the q/k normalisation
    drop = lambda layers: [{**l, "blocks": [{k: v for k, v in b.items() if k not in ("qn", "kn")} for b in l["blocks"]]}
                           for l in layers]
    return {**p, "enc": drop(p["enc"]), "dec": drop(p["dec"])}


def _attn_params(rng, D, qkn):
    r = lambda *shape: jnp.asarray(rng.standard_normal(shape) / 8)
    p = {"qkv": {"W": r(D, 3 * D), "b": r(3 * D)}, "proj": {"W": r(D, D), "b": r(D)}}
    if qkn:
        dh = D // 4
        p["qn"] = {"g": jnp.asarray(1 + rng.standard_normal(dh) / 4), "b": r(dh)}
        p["kn"] = {"g": jnp.asarray(1 + rng.standard_normal(dh) / 4), "b": r(dh)}
    return p


def test_Q1_default_config_has_no_qk_norm():
    assert bb.Config(64, (2,), (2,), (2,), (2,), WINDOW).qk_norm is False
    p = bb.init(jax.random.PRNGKey(0), _cfg(False), dtype=jnp.float64)
    assert not any(k in b for l in p["enc"] + p["dec"] for b in l["blocks"] for k in ("qn", "kn"))


def test_Q2_the_flag_changes_no_other_weight():
    p0 = bb.init(jax.random.PRNGKey(0), _cfg(False), dtype=jnp.float64)
    p1 = bb.init(jax.random.PRNGKey(0), _cfg(True), dtype=jnp.float64)
    same = jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), p0, _strip(p1))
    assert all(jax.tree_util.tree_leaves(same)), "the flag changed a weight other than the q/k norms"
    blocks = [b for l in p1["enc"] + p1["dec"] for b in l["blocks"]]
    assert len(blocks) == 12
    for b in blocks:  # head dimension 32 at every stage of the E1 backbone
        for n in ("qn", "kn"):
            assert b[n]["g"].shape == b[n]["b"].shape == (32,)
            assert float(jnp.abs(b[n]["g"] - 1).max()) == 0.0 and float(jnp.abs(b[n]["b"]).max()) == 0.0
    assert sum(b[n][k].size for b in blocks for n in ("qn", "kn") for k in ("g", "b")) == 12 * 4 * 32 == 1536


def test_Q3_normalisation_enters_the_attention():
    rng = np.random.default_rng(1)
    Bw, N, D, H = 6, 48, 32, 4
    p = _attn_params(rng, D, True)
    xw = jnp.asarray(rng.standard_normal((Bw, N, D)))
    bias = jnp.asarray(rng.standard_normal((H, N, N)))
    mask = jnp.asarray(rng.random((3, N, N)) < 0.7) | jnp.eye(N, dtype=bool)[None]

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * np.asarray(g) + np.asarray(b)

    for m, bi in ((None, None), (mask, None), (mask, bias)):
        got = swin.window_attention(p, xw, H, m, bi)
        qkv = (np.asarray(xw) @ np.asarray(p["qkv"]["W"]) + np.asarray(p["qkv"]["b"])).reshape(Bw, N, 3, H, D // H)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q, k = ln(q, p["qn"]["g"], p["qn"]["b"]), ln(k, p["kn"]["g"], p["kn"]["b"])
        lg = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D // H)
        if bi is not None:
            lg = lg + np.asarray(bi)[None]
        if m is not None:
            lg = np.where(np.tile(np.asarray(m), (Bw // 3, 1, 1))[:, None], lg, -np.inf)
        a = np.exp(lg - lg.max(-1, keepdims=True))
        o = (a / a.sum(-1, keepdims=True)) @ v
        ref = o.transpose(0, 2, 1, 3).reshape(Bw, N, D) @ np.asarray(p["proj"]["W"]) + np.asarray(p["proj"]["b"])
        err = float(np.max(np.abs(np.asarray(got) - ref)))
        assert err <= 1e-12, (m is None, bi is None, err)


def test_Q4_blind_to_the_scale_of_q_and_k():
    """Scaling the q and k projections by 10 leaves the normalised attention unchanged (up to LayerNorm's eps) and
    changes the plain attention a lot."""
    rng = np.random.default_rng(2)
    Bw, N, D, H = 6, 48, 32, 4
    xw = jnp.asarray(rng.standard_normal((Bw, N, D)))
    for qkn, check in ((True, lambda d: d < 1e-3), (False, lambda d: d > 0.1)):
        p = _attn_params(np.random.default_rng(3), D, qkn)
        s = jnp.concatenate([jnp.full(2 * D, 10.0), jnp.ones(D)])  # q and k columns x10, v unchanged
        p10 = {**p, "qkv": {"W": p["qkv"]["W"] * s, "b": p["qkv"]["b"] * s}}
        o, o10 = swin.window_attention(p, xw, H), swin.window_attention(p10, xw, H)
        d = float(jnp.max(jnp.abs(o10 - o)) / jnp.max(jnp.abs(o)))
        print(f"\nQ4 q/k norm {qkn}: relative change {d:.1e}")
        assert check(d), (qkn, d)


def test_Q5_same_backbone_at_initialisation():
    x, t = _inputs()
    for rpb in (False, True):
        p0 = bb.init(jax.random.PRNGKey(0), _cfg(False, rpb), dtype=jnp.float64)
        p1 = bb.init(jax.random.PRNGKey(0), _cfg(True, rpb), dtype=jnp.float64)
        f0 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(False, rpb)))(p0)
        f1 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(True, rpb)))(p1)
        assert float(jnp.max(jnp.abs(f1 - f0))) == 0.0
        q0, q1 = _modulate(p0, 7), _modulate(p1, 7)  # once attention contributes, the normalisation matters
        g0 = bb.apply(q0, x, t, (3, 16, 16), _cfg(False, rpb))
        g1 = bb.apply(q1, x, t, (3, 16, 16), _cfg(True, rpb))
        assert float(jnp.max(jnp.abs(g1 - g0))) > 1e-3


def test_Q6_every_gain_and_bias_gets_a_gradient_under_jit():
    cfg = _cfg(True)
    p = _modulate(bb.init(jax.random.PRNGKey(3), cfg, dtype=jnp.float32), 8)
    x, t = _inputs(jnp.float32, seed=4)
    y = jnp.asarray(np.random.default_rng(5).standard_normal((2, 3 * 16 * 16, 128)), jnp.float32)
    loss = lambda q: jnp.mean((bb.apply(q, x, t, (3, 16, 16), cfg) - y) ** 2)
    g = jax.jit(jax.grad(loss))(p)
    norms = [float(jnp.linalg.norm(b[n][k])) for l in g["enc"] + g["dec"] for b in l["blocks"]
             for n in ("qn", "kn") for k in ("g", "b")]
    assert len(norms) == 48 and all(np.isfinite(norms)) and min(norms) > 0, norms
