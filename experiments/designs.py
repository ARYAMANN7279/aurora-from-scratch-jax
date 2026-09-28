"""Run configs for E2–E5 (handouts/e2_heterogeneous_fusion.md, e3_ood_transfer.md, e4_ablations.md,
e5_lora_vs_full.md).

The design of each experiment lives here, in one place. make_configs.py writes the files, and
tests/test_designs.py checks them against the specs.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e1"))

from e1_design import MODEL, _run, tag  # noqa: E402

SEEDS = (0, 1, 2)

# ---------------- E3: out-of-distribution transfer (dataset e1_v1) ----------------
E3_BAND = [(re, kf) for re in (125, 160, 200, 250) for kf in (3, 4, 5, 6)]
E3_TESTS = [(300, 4), (350, 4), (450, 4), (560, 4), (700, 4)]
E3_BAND_SEEDS, E3_FT, E3_EVAL = [0, 1, 2, 3], [0, 1], [2, 3]


def e3(hash_fn):
    ev = [{"tag": tag(r), "seeds": E3_EVAL} for r in E3_TESTS]
    pre, ft, sc = [], [], []
    for s in SEEDS:
        p = _run(f"e3_pre_s{s}", f"E3 pretrain on the band Re 125-250, seed {s}: zero-shot errors at the 5 "
                 f"out-of-band regimes feed E3-P1, E3-P4; handouts/e3_ood_transfer.md",
                 [{"tag": tag(r), "seeds": E3_BAND_SEEDS} for r in E3_BAND], ev, [], ["test"], 20_000, 1e-3, s)
        pre.append(p)
        for r in E3_TESTS:
            data = [{"tag": tag(r), "seeds": E3_FT}]
            val = [{"tag": tag(r), "seeds": E3_EVAL}]
            ft.append(_run(f"e3_ft_{tag(r)}_s{s}", f"E3 fine-tune at {tag(r)}, seed {s}: adv(d) for E3-P2, E3-P3",
                           data, [], val, ["val"], 2_000, 1e-4, s, init_from=hash_fn(p)))
            sc.append(_run(f"e3_scratch_{tag(r)}_s{s}", f"E3 scratch at {tag(r)}, seed {s}: e_scratch of adv(d)",
                           data, [], val, ["val"], 22_000, 1e-3, s))
    return pre, ft, sc


# ---------------- E4: component ablations (dataset acc_v1) ----------------
E4_STEPS = 10_000
E4_VARIANTS = {
    "base": {},
    "noshift": {"model.backbone.shift": False},
    "depth6": {"model.backbone.enc_depths": [1, 1, 1], "model.backbone.dec_depths": [1, 1, 1]},
    "depth24": {"model.backbone.enc_depths": [4, 4, 4], "model.backbone.dec_depths": [4, 4, 4]},
    "L1": {"model.L": 1},
    "L6": {"model.L": 6},
    "pf_steps": {"train.pushforward": True},
    "pf_compute": {"train.pushforward": True, "train.steps": E4_STEPS // 2},
    "mae": {"train.loss": "mae"},
}


def _set(d, dotted, v):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = v


def e4():
    out = []
    train = [{"tag": t, "seeds": [0, 1, 2, 3, 4, 5]} for t in ("re100_kf4", "re300_kf4")]
    val = [{"tag": t, "seeds": [6, 7]} for t in ("re100_kf4", "re300_kf4")]
    for name, change in E4_VARIANTS.items():
        for s in SEEDS:
            c = _run(f"e4_{name}_s{s}", f"E4 ablation '{name}', seed {s}: handouts/e4_ablations.md E4-P1..P5",
                     train, [], val, ["val"], E4_STEPS, 1e-3, s, dataset="acc_v1")
            c["eval"]["rollout_steps"] = 50
            for k, v in change.items():
                _set(c, k, copy.deepcopy(v))
            out.append(c)
    return out


# ---------------- E5: LoRA vs full fine-tuning (dataset acc_v1) ----------------
E5_SIZES = (1, 2, 4, 6)
E5_EVAL = [6, 7]


def e5(hash_fn):
    tgt = "re1000_kf4"
    val = [{"tag": tgt, "seeds": E5_EVAL}]
    pre, ft, sc = [], [], []
    for s in SEEDS:
        p = _run(f"e5_pre_s{s}", f"E5 pretrain on Re 100/300, seed {s}: zero-shot on Re 1000 is the reference "
                 f"for the fine-tuning gains (E5-P1..P4); handouts/e5_lora_vs_full.md",
                 [{"tag": t, "seeds": [0, 1, 2, 3, 4, 5, 6, 7]} for t in ("re100_kf4", "re300_kf4")],
                 val, [], ["test"], 10_000, 1e-3, s, dataset="acc_v1")
        pre.append(p)
        for k in E5_SIZES:
            data = [{"tag": tgt, "seeds": list(range(k))}]
            for method, lr, rank in (("full", 1e-4, None), ("lora", 1e-3, 4)):
                c = _run(f"e5_{method}_k{k}_s{s}", f"E5 {method} fine-tune on {k} Re 1000 trajectories, seed {s}",
                         data, [], val, ["val"], 2_000, lr, s, init_from=hash_fn(p), dataset="acc_v1")
                c["train"]["lora_rank"] = rank
                ft.append(c)
            sc.append(_run(f"e5_scratch_k{k}_s{s}", f"E5 scratch on {k} Re 1000 trajectories, seed {s}",
                           data, [], val, ["val"], 12_000, 1e-3, s, dataset="acc_v1"))
    return pre, ft, sc


# ---------------- E2: heterogeneous fusion (dataset acc_v1) ----------------
E2_ALL, E2_UV = ["u", "v", "p", "omega"], ["u", "v"]
E2_SETS = {  # set -> (A = re100_kf4 format, B = re300_kf4 format) as (N, variables)
    "a": ((128, E2_ALL), (64, E2_ALL)),
    "b": ((64, E2_ALL), (64, E2_UV)),
    "c": ((128, E2_ALL), (64, E2_UV)),
    "h": ((64, E2_ALL), (64, E2_ALL)),
}
E2_METHODS = ("perceiver", "pad", "per_source")
E2_STEPS = 10_000


def e2():
    out = []
    for name, (fa, fb) in E2_SETS.items():
        sources = [{"name": "A", "tag": "re100_kf4", "N": fa[0], "vars": fa[1], "train_seeds": [0, 1, 2, 3, 4, 5],
                    "val_seeds": [6, 7]},
                   {"name": "B", "tag": "re300_kf4", "N": fb[0], "vars": fb[1], "train_seeds": [0, 1, 2, 3, 4, 5],
                    "val_seeds": [6, 7]}]
        for method in (E2_METHODS if name != "h" else ("perceiver",)):
            for s in SEEDS:
                out.append({"name": f"e2_{name}_{method}_s{s}",
                            "prediction": f"E2 set ({name}), method {method}, seed {s}: "
                                          f"handouts/e2_heterogeneous_fusion.md E2-P1..P3",
                            "method": method, "data": {"dataset": "acc_v1", "sources": copy.deepcopy(sources)},
                            "model": copy.deepcopy(MODEL),
                            "train": {"steps": E2_STEPS, "batch": 16, "lr": 1e-3, "schedule": "cosine", "seed": s},
                            "eval": {"rollout_steps": 20}})
    return out


__all__ = ["e2", "e3", "e4", "e5", "MODEL"]
