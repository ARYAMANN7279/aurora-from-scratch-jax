"""The report follow-ups retain original configs and isolate the requested changes."""
import copy
import importlib.util
import json
from pathlib import Path

from aurora.harness import config_hash

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_e2_followup_preserves_original_and_changes_only_regimes():
    d = module("designs", "experiments/designs.py")
    original = json.loads((ROOT / "experiments/e2/configs/runs.json").read_text())
    assert [config_hash(c) for c in d.e2()] == [config_hash(c) for c in original]
    new = d.e2f1()
    assert len(new) == len({config_hash(c) for c in new}) == 30
    for old, cfg in zip(original, new):
        old, cfg = copy.deepcopy(old), copy.deepcopy(cfg)
        assert cfg["data"]["dataset"] == "e2f1_v1"
        assert [s["tag"] for s in cfg["data"]["sources"]] == ["re30_kf4", "re3000_kf4"]
        for src in cfg["data"]["sources"]:
            assert not set(src["train_seeds"]) & set(src["val_seeds"])
        cfg["data"]["dataset"] = old["data"]["dataset"]
        for a, b in zip(cfg["data"]["sources"], old["data"]["sources"]):
            a["tag"] = b["tag"]
        assert config_hash(cfg) == config_hash(old)


def test_curve_configs_point_to_reruns_without_changing_training():
    import sys

    sys.path.insert(0, str(ROOT / "experiments/e1"))
    from e1_design import run_configs
    from make_curves import curves

    p, f, s = run_configs(config_hash)
    originals = {c["name"]: c for c in p + f + s}
    runs = curves()
    assert len(runs) == len({config_hash(c) for c in runs}) == 15
    seen = set()
    for c in runs:
        old = originals[c["name"].replace("e1c_", "e1_")]
        q = copy.deepcopy(c)
        assert q["train"].pop("curve_every") > 0
        if q["train"]["init_from"]:
            assert q["train"]["init_from"] in seen
        q["train"]["init_from"] = old["train"]["init_from"]
        assert config_hash(q) == config_hash(old)
        seen.add(config_hash(c))


def test_e2_followup_scores_paired_predictions():
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import designs as d
    import analyse_e345 as a

    rows = {}
    for cfg in d.e2f1():
        st = cfg["name"].split("_")[1]
        method = cfg["method"]
        metrics = {}
        for src in ("A", "B"):
            base = {"A": .1, "B": .2}[src]
            cost = .5 if (st, src) == ("b", "A") else .2
            perceiver = base * (1 if st == "h" else 1 + cost)
            factor = {"perceiver": 1, "pad": .8, "per_source": 1.2}[method]
            metrics[f"val/{src}"] = dict(rel_err=perceiver * factor)
        rows[config_hash(cfg)] = dict(metrics=metrics)
    assert a.e2f1(rows)["verdicts"] == {f"F1-P{i}": "HIT" for i in range(1, 5)}
    # A single reversed comparison must produce a MISS, never get averaged across sources.
    for cfg in d.e2f1():
        if cfg["name"].startswith("e2f1_c_pad_"):
            rows[config_hash(cfg)]["metrics"]["val/B"]["rel_err"] = 1
    assert a.e2f1(rows)["verdicts"]["F1-P1"] == "MISS"
