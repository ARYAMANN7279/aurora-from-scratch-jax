"""E1 design (handouts/e1_volume_diversity.md): regimes, cells and the run configs, in one place."""
import copy

TARGET = (300, 4)
SET2 = [(200, 3), (450, 5)]
SET8 = [(re, kf) for re in (160, 200, 450, 560) for kf in (3, 5)]
SET32 = [(re, kf) for re in (125, 160, 200, 250, 350, 450, 560, 700) for kf in (3, 4, 5, 6)]
SETS = {2: SET2, 8: SET8, 32: SET32}
TOTALS = (32, 64, 128)
TRAIN_SEEDS = (0, 1, 2)
FT_SEEDS, EVAL_SEEDS = [0, 1], [2, 3, 4, 5]
STEPS_PRE, STEPS_FT, LR_PRE, LR_FT = 20_000, 2_000, 1e-3, 1e-4
DATASET = "e1_v1"

MODEL = {"D": 64, "P": 4, "L": 3, "heads": 4,
         "backbone": {"enc_depths": [2, 2, 2], "enc_heads": [2, 4, 8], "dec_depths": [2, 2, 2],
                      "dec_heads": [8, 4, 2], "window": [3, 4, 4]}}
EVAL = {"rollout_steps": 20}

assert set(SET2) <= set(SET8) <= set(SET32) and TARGET not in SET32


def tag(r):
    return f"re{r[0]}_kf{r[1]}"


def all_regimes():
    return [TARGET] + SET32


def seeds_needed():
    """regime -> number of trajectories: the largest T/n over the cells that use it (seeds nested)."""
    need = {TARGET: len(FT_SEEDS) + len(EVAL_SEEDS)}
    for n, regs in SETS.items():
        for r in regs:
            need[r] = max(need.get(r, 0), max(TOTALS) // n)
    return need


def dataset_config(ngen, name=DATASET, T_spin=50.0, T_rec=100.0, save_dt=0.2, N_out=(64,)):
    """ngen: regime -> N_gen. Per-regime seed lists (kdata supports regime-level "seeds")."""
    need = seeds_needed()
    regs = [{"Re": r[0], "kf": r[1], "N_gen": int(ngen[r]), "seeds": list(range(need[r]))} for r in all_regimes()]
    return {"name": name, "regimes": regs, "T_spin": T_spin, "T_rec": T_rec, "save_dt": save_dt,
            "N_out": list(N_out)}


def _run(name, prediction, train, test, val, splits, steps, lr, seed, init_from=None, batch=16,
         model=MODEL, dataset=DATASET, N=64):
    data = {"dataset": dataset, "N": N, "stride": 1, "train": train, "test": test, "val": val}
    if init_from:
        data["norm_from_init"] = True
    return {"name": name, "prediction": prediction, "data": data, "model": copy.deepcopy(model),
            "train": {"steps": steps, "batch": batch, "lr": lr, "schedule": "cosine", "loss": "mse",
                      "pushforward": False, "seed": seed, "init_from": init_from, "lora_rank": None},
            "eval": dict(EVAL, splits=splits)}


def run_configs(hash_fn, **kw):
    """-> (pretrain, finetune, scratch) config lists. hash_fn = aurora.harness.config_hash, used to
    point every fine-tune at its pretrain run. kw overrides steps/model/dataset (smoke tests)."""
    steps_pre, steps_ft = kw.pop("steps_pre", STEPS_PRE), kw.pop("steps_ft", STEPS_FT)
    tgt = tag(TARGET)
    ft_data = [{"tag": tgt, "seeds": FT_SEEDS}]
    ev = [{"tag": tgt, "seeds": EVAL_SEEDS}]
    pre, ft, sc = [], [], []
    for s in TRAIN_SEEDS:
        for n, regs in SETS.items():
            for T in TOTALS:
                k = T // n
                p = _run(f"e1_pre_n{n}_T{T}_s{s}",
                         f"E1 pretrain cell (n={n}, T={T}), seed {s}: its test/{tgt} error is the zero-shot "
                         f"input to E1-P5 (n=32 beats n=2 at every T); handouts/e1_volume_diversity.md",
                         [{"tag": tag(r), "seeds": list(range(k))} for r in regs], ev, [], ["test"],
                         steps_pre, LR_PRE, s, **kw)
                pre.append(p)
                ft.append(_run(f"e1_ft_n{n}_T{T}_s{s}",
                               f"E1 fine-tune of cell (n={n}, T={T}), seed {s}: with scratch seed {s} gives adv "
                               f"for E1-P1 to E1-P4 and E1-P6; handouts/e1_volume_diversity.md",
                               ft_data, [], ev, ["val"], steps_ft, LR_FT, s, init_from=hash_fn(p), **kw))
        sc.append(_run(f"e1_scratch_s{s}",
                       f"E1 scratch baseline, seed {s}: the e_scratch of adv = 1 - e_ft / e_scratch "
                       f"(E1-P1 to E1-P4, E1-P6); handouts/e1_volume_diversity.md",
                       ft_data, [], ev, ["val"], steps_pre + steps_ft, LR_PRE, s, **kw))
    return pre, ft, sc
