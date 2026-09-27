"""E3–E5 designs (experiments/designs.py) checked against their specs: run counts, no held-out
regime in pretraining, every fine-tune pointing at its own pretrain run, and every E4 variant
building a model that runs."""
import os
import sys

import jax
import jax.numpy as jnp

from aurora import harness, model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import designs as D  # noqa: E402


def _tags(split):
    return {x["tag"] for x in split}


def test_e3():
    pre, ft, sc = D.e3(harness.config_hash)
    assert (len(pre), len(ft), len(sc)) == (3, 15, 15)
    for p in pre:
        assert not _tags(p["data"]["train"]) & _tags(p["data"]["test"])
        assert len(p["data"]["train"]) == 16 and len(p["data"]["test"]) == 5
    hashes = {harness.config_hash(p): p["train"]["seed"] for p in pre}
    for f in ft:
        assert hashes[f["train"]["init_from"]] == f["train"]["seed"] and f["data"]["norm_from_init"]
    assert len({harness.config_hash(c) for c in pre + ft + sc}) == 33


def test_e4_variants_build_and_run():
    runs = D.e4()
    assert len(runs) == 27 and len({harness.config_hash(c) for c in runs}) == 27
    by = {c["name"].rsplit("_s", 1)[0]: c for c in runs if c["train"]["seed"] == 0}
    assert by["e4_pf_compute"]["train"]["steps"] * 2 == by["e4_pf_steps"]["train"]["steps"]
    assert by["e4_noshift"]["model"]["backbone"]["shift"] is False and by["e4_mae"]["train"]["loss"] == "mae"
    x = jnp.zeros((4, 32, 32), jnp.float32)
    for name in ("e4_base", "e4_noshift", "e4_depth6", "e4_L1", "e4_L6"):
        m = dict(by[name]["model"], D=16, heads=2)
        m["backbone"] = dict(m["backbone"], enc_heads=[1, 2, 2], dec_heads=[2, 2, 1])
        mc = harness.model_config(m)
        p = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32)
        y = model.apply(p, x, x, model.VARS, 0.2, mc)
        assert y.shape == (4, 32, 32)


def test_e5():
    pre, ft, sc = D.e5(harness.config_hash)
    assert (len(pre), len(ft), len(sc)) == (3, 24, 12)
    for p in pre:
        assert "re1000_kf4" not in _tags(p["data"]["train"])
    for f in ft:
        assert f["data"]["train"][0]["tag"] == "re1000_kf4"
        assert not set(f["data"]["train"][0]["seeds"]) & set(D.E5_EVAL)
        assert (f["train"]["lora_rank"] == 4) == ("_lora_" in f["name"])
    assert len({harness.config_hash(c) for c in pre + ft + sc}) == 39
