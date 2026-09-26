"""Stage 10 gates L1-L5 (handouts/stage10_lora.md, section 2). float64."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from aurora import lora, model, n_params

VARS = model.VARS
MC = model.ModelConfig()


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def setup(seed=0):
    base = model.init(jax.random.PRNGKey(seed), MC)
    leaves, tree = jax.tree_util.tree_flatten(base)
    ks = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves) + 3)
    base = jax.tree_util.tree_unflatten(tree, [a + 0.02 * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])
    lo = lora.init(ks[-3], base, r=4)
    return base, lo, jax.random.normal(ks[-2], (4, 64, 64)), jax.random.normal(ks[-1], (4, 64, 64))


def nonzero_B(lo, key):
    leaves, tree = jax.tree_util.tree_flatten(lo)
    ks = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(tree, [a + 0.05 * jax.random.normal(k, a.shape) if a.shape[0] == 4 else a
                                               for a, k in zip(leaves, ks)])


def test_L1_zero_B_is_base():
    base, lo, a, b = setup()
    assert jnp.array_equal(model.apply(lora.merge(base, lo), a, b, VARS, 0.2, MC), model.apply(base, a, b, VARS, 0.2, MC))


def test_L2_merged_equals_unmerged():
    base, lo, _, _ = setup()
    lo = nonzero_B(lo, jax.random.PRNGKey(9))
    blk, lb = base["bb"]["enc"][1]["blocks"][1], lo["enc"][1][1]
    x = jax.random.normal(jax.random.PRNGKey(3), (10, blk["qkv"]["W"].shape[0]))
    merged = lora.merge(base, lo)["bb"]["enc"][1]["blocks"][1]
    err = 0.0
    for name in ("qkv", "proj"):
        unmerged = x @ blk[name]["W"] + (x @ lb[name]["A"]) @ lb[name]["B"]
        err = max(err, float(jnp.max(jnp.abs(x @ merged[name]["W"] - unmerged))))
    report("L2 merged vs unmerged", err, 1e-12)
    assert err <= 1e-12


def test_L3_only_lora_changes():
    base, lo, a, b = setup()
    tgt = jax.random.normal(jax.random.PRNGKey(7), (4, 64, 64))
    loss = lambda l: jnp.mean((model.apply(lora.merge(base, l), a, b, VARS, 0.2, MC) - tgt) ** 2)
    opt = optax.adam(1e-3)
    st = opt.init(lo)
    g = jax.grad(loss)(lo)
    upd, st = opt.update(g, st, lo)
    lo2 = optax.apply_updates(lo, upd)
    # the base is a constant of the loss: the update touches only the LoRA pytree, and merging the
    # updated factors leaves every non-LoRA leaf of the merged model bitwise equal to the base
    merged = lora.merge(base, lo2)
    for side in ("enc", "dec"):
        for layer_b, layer_m in zip(base["bb"][side], merged["bb"][side]):
            for bb_, bm in zip(layer_b["blocks"], layer_m["blocks"]):
                for key in ("norm1", "norm2", "mlp"):
                    assert all(jnp.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(bb_[key]),
                                                                     jax.tree_util.tree_leaves(bm[key])))
                assert jnp.array_equal(bb_["qkv"]["b"], bm["qkv"]["b"])
    for key in ("emb", "per", "frame", "mid"):
        assert all(jnp.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(base[key]),
                                                         jax.tree_util.tree_leaves(merged[key])))
    changed = [not jnp.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(lo), jax.tree_util.tree_leaves(lo2))]
    assert any(changed)
    Bs_changed = all(not jnp.array_equal(l1[n]["B"], l2[n]["B"]) for s in ("enc", "dec")
                     for L1, L2 in zip(lo[s], lo2[s]) for l1, l2 in zip(L1, L2) for n in ("qkv", "proj"))
    assert Bs_changed


def test_L4_trainable_count():
    base, lo, _, _ = setup()
    assert lora.n_params(lo) == 43008
    frac = lora.n_params(lo) / n_params(base)
    print(f"\nL4 trainable fraction: {100 * frac:.3f}% of {n_params(base)}")
    assert n_params(base) == 5106688 and frac < 0.01


def test_L5_gradient():
    base, lo, a, b = setup(2)
    lo = nonzero_B(lo, jax.random.PRNGKey(11))
    tgt = jax.random.normal(jax.random.PRNGKey(8), (4, 64, 64))
    loss = lambda l: jnp.mean((model.apply(lora.merge(base, l), a, b, VARS, 0.2, MC) - tgt) ** 2)
    flat, unravel = jax.flatten_util.ravel_pytree(lo)
    g, _ = jax.flatten_util.ravel_pytree(jax.grad(loss)(lo))
    gmed = float(jnp.median(jnp.abs(g)))
    worst = 0.0
    for i in np.random.default_rng(0).choice(flat.size, 5, replace=False):
        h = 1e-3
        e = jnp.zeros_like(flat).at[i].set(h)
        fd = (loss(unravel(flat + e)) - loss(unravel(flat - e))) / (2 * h)
        worst = max(worst, abs(float(fd - g[i])) / max(abs(float(g[i])), gmed))
    report("L5 LoRA grad vs finite differences (mixed, h = 1e-3 per amendment A1)", worst, 1e-6)
    assert worst <= 1e-6
