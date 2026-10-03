"""E9 gates (handouts/e9_rpb_plus_qknorm.md, section 2): the relative-position bias and q/k normalisation together.
With both flags the other weights are those of the same seed and, at initialisation, the backbone is exactly the
plain one (RQ1); once attention contributes (a random AdaLN modulation), the combination differs from either part
alone (RQ2); every table, gain and bias gets a gradient under jit (RQ3). float64 unless stated."""
import jax
import jax.numpy as jnp
import numpy as np

from aurora import backbone as bb

WINDOW = (3, 4, 4)
NEW = ("rpb", "qn", "kn")


def _cfg(rpb, qkn):  # the E1 backbone
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


def _strip(p):
    drop = lambda layers: [{**l, "blocks": [{k: v for k, v in b.items() if k not in NEW} for b in l["blocks"]]}
                           for l in layers]
    return {**p, "enc": drop(p["enc"]), "dec": drop(p["dec"])}


def test_RQ1_same_weights_and_same_backbone_at_initialisation():
    p0 = bb.init(jax.random.PRNGKey(0), _cfg(False, False), dtype=jnp.float64)
    p1 = bb.init(jax.random.PRNGKey(0), _cfg(True, True), dtype=jnp.float64)
    same = jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), p0, _strip(p1))
    assert all(jax.tree_util.tree_leaves(same)), "the flags changed a weight other than the new ones"
    for b in (b for l in p1["enc"] + p1["dec"] for b in l["blocks"]):
        assert set(NEW) <= set(b) and float(jnp.abs(b["rpb"]).max()) == 0.0
        assert all(float(jnp.abs(b[n]["g"] - 1).max()) == 0.0 and float(jnp.abs(b[n]["b"]).max()) == 0.0 for n in ("qn", "kn"))
    x, t = _inputs()
    f0 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(False, False)))(p0)
    f1 = jax.jit(lambda p: bb.apply(p, x, t, (3, 16, 16), _cfg(True, True)))(p1)
    assert float(jnp.max(jnp.abs(f1 - f0))) == 0.0


def test_RQ2_the_combination_differs_from_either_part_once_attention_contributes():
    x, t = _inputs()
    out = {}
    for rpb, qkn in ((True, False), (False, True), (True, True)):
        p = _modulate(bb.init(jax.random.PRNGKey(0), _cfg(rpb, qkn), dtype=jnp.float64), 7)
        rng = np.random.default_rng(2)  # a trained-looking table: the same draws in every model that has one
        for b in (b for l in p["enc"] + p["dec"] for b in l["blocks"]):
            if rpb:
                b["rpb"] = jnp.asarray(rng.standard_normal(b["rpb"].shape))
        out[(rpb, qkn)] = bb.apply(p, x, t, (3, 16, 16), _cfg(rpb, qkn))
    both = out[(True, True)]
    assert float(jnp.max(jnp.abs(both - out[(True, False)]))) > 1e-3
    assert float(jnp.max(jnp.abs(both - out[(False, True)]))) > 1e-3


def test_RQ3_every_new_parameter_gets_a_gradient_under_jit():
    cfg = _cfg(True, True)
    p = _modulate(bb.init(jax.random.PRNGKey(3), cfg, dtype=jnp.float32), 8)
    x, t = _inputs(jnp.float32, seed=4)
    y = jnp.asarray(np.random.default_rng(5).standard_normal((2, 3 * 16 * 16, 128)), jnp.float32)
    g = jax.jit(jax.grad(lambda q: jnp.mean((bb.apply(q, x, t, (3, 16, 16), cfg) - y) ** 2)))(p)
    blocks = [b for l in g["enc"] + g["dec"] for b in l["blocks"]]
    norms = [float(jnp.linalg.norm(b["rpb"])) for b in blocks]
    norms += [float(jnp.linalg.norm(b[n][k])) for b in blocks for n in ("qn", "kn") for k in ("g", "b")]
    assert len(norms) == 12 + 48 and all(np.isfinite(norms)) and min(norms) > 0, norms
