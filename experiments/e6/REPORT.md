# E6 Report: Does pretraining on simulations buy accuracy on real flow measurements? (RealPDE foil)

Spec: `handouts/e6_realpde_sim2real.md` (amendments A1, A2 and A3; follow-up F1 in §10). Code:
`experiments/e6/`. Scored by `score.py` → `results/analysis.json`, from `results/rows.jsonl`; F1 by
`score_f1.py` → `results/analysis_f1.json`, from `results/rows_f1.jsonl`.

## Verdict

**2 of 5 predictions HIT (E6-P4, P5).** At the registered budgets, pretraining on the simulations
does not help on real flow measurements. It hurts.
* **Scratch wins.** The model trained from scratch on the 65 real training trajectories forecasts
  the held-out conditions best: rollout rel_l2 **0.112** (subscore 94.71).
* **Pretraining hurts.** Pretraining on the simulations and then fine-tuning for 2000 steps leaves
  the error **21–23% higher**: 0.137 (disjoint simulations), 0.135 (all simulations).
* **Zero-shot transfer is poor.** Without fine-tuning, the pretrained models reach 0.27–0.29, and
  their one-step forecasts barely beat persistence.
* **Close to the organisers' FNO.** The from-scratch model comes within 0.8 subscore points of the
  organisers' fine-tuned FNO (95.47), which may have seen the held-out conditions.

**Follow-up F1, all 3 predictions HIT.** Given the same 22,000 real-data steps as scratch, the
pretrained models end where scratch ends:
* +0.4% ± 1.8% (disjoint simulations);
* +2.0% ± 0.9% (all simulations).

So the 21–23% deficit came from the short fine-tune, not from the pretraining. Simulation pretraining
buys no accuracy on these measurements: at most a 2% edge when the simulations include the test
conditions.

## Set-up

* **Data:**
  * **Real:** 81 trajectories of PIV measurements past an airfoil: u and v, strided to 32 × 64 as
    the competition evaluated, frame spacing 0.02.
  * **Simulated:** 100 trajectories: u, v and p, 1000 frames each, spacing 0.02002.
  * Each source is normalised with its own statistics, computed on its training trajectories.
* **Held-out test:** UGP-I's `re_lohi` split, the 2 lowest and 2 highest Reynolds numbers (3750,
  5025, 25,425, 26,700) at every angle of attack.
  * That is 16 real trajectories, scored on UGP-I's 900 windows. Our split module selects exactly
    those windows (checked).
  * Training uses the other 65 real trajectories.
* **Model:** the E1 configuration (D = 64, 12 Swin blocks), adapted to rectangular, non-periodic
  fields:
  * patches P = 2, giving a token grid of 3 × 16 × 32;
  * masked shifted windows;
  * position encodings over half a period.
* **Arms** (3 seeds each, batch 16, MSE):
  * **S:** real data from scratch, 22,000 steps;
  * **PD and PA:** 20,000 steps of pretraining on the simulations, then 2000 steps of fine-tuning
    on real data at lr 1e-4. PD's simulations exclude the held-out conditions; PA's include them.
  * **ZD and ZA:** the PD and PA pretrained models on real data, without fine-tuning. These are the
    pretraining runs' own evaluations (A1).
  * In half the pretraining chunks, p is dropped from the inputs (A1).
* **Metrics:**
  * **Rollout (primary):** a 20-step rollout from the last 2 of the 20 input frames. rel_l2, TKE
    and mean-velocity-profile errors come from the competition kit's `scoring.py`, with its 0–100
    subscores.
  * **One-step:** the error of the first frame, and its ratio to persistence.
  * **Advantage:** adv(P) = 1 − err(P) / err(S), paired by seed.
* **Where:**
  * **Kaggle (amendment A3):** two NVIDIA T4 cards, one queue each (`kaggle_run.py`, commit
    3d21de7). Software pinned to the lab machine's: JAX 0.10.2, optax 0.2.8, numpy 2.4.6.
    Float32 with matmul precision `highest`.
  * **Why Kaggle:** other users' jobs held the lab machine's cards. The one card that freed up
    passed 93 °C within seconds under our runs.
  * **Runs:** the 15 runs took about 2.5 h on the two cards. A from-scratch run took 29–30 min, a
    pretraining run 27–28 min, a fine-tune 4 min. The thermal throttle never slept.
  * **Smoke tests:** one on the lab machine's CPU (S, PD pretraining and PD fine-tuning; 200 steps,
    100 windows) and one on Kaggle passed before the runs.

## FNO baselines (measured before any run)

The organisers' FNOs on the same 900 windows, through UGP-I's harness (`fno_baseline.py`, CPU):

| Model | rel_l2 | TKE | MVPE | Subscores (rel_l2 / TKE / MVPE) |
|---|---|---|---|---|
| `sim_real_fno.pth` (fine-tuned by the organisers on real data, which may include the held-out conditions) | 0.0948 | 0.635 | 0.0813 | 95.47 / 75.90 / 96.09 |
| `sim_fno.pth` (simulations only) | 1.228 | 0.980 | 1.558 | 61.95 / 67.12 / 56.21 |

* `sim_real_fno` reproduces UGP-I's selftest to 1e-3. It is an upper reference, not an honest
  baseline.
* `sim_fno` does not transfer to real data zero-shot. That corrected E6-P3's anchor (A2).

## Results

On the 900 `re_lohi` test windows. Mean ± SE over 3 seeds (`results/analysis.json`).

| Arm | Rollout rel_l2 | Subscores (rel_l2 / TKE / MVPE) | One-step error | One-step / persistence |
|---|---|---|---|---|
| **S**, from scratch | **0.1117 ± 0.0008** | **94.71** / 75.58 / 94.24 | 0.0418 | 0.753 |
| **PD**, disjoint simulations, fine-tuned | 0.1373 ± 0.0025 | 93.58 / 73.26 / 93.05 | 0.0438 | 0.789 |
| **PA**, all simulations, fine-tuned | 0.1350 ± 0.0031 | 93.68 / 73.27 / 93.26 | 0.0437 | 0.786 |
| **ZD**, disjoint simulations, zero-shot | 0.2685 ± 0.0080 | 88.17 / 48.28 / 89.25 | 0.0531 | 0.956 |
| **ZA**, all simulations, zero-shot | 0.2891 ± 0.0212 | 87.39 / 43.25 / 88.34 | 0.0531 | 0.956 |
| `sim_real_fno` (organisers; upper reference) | 0.0948 | 95.47 / 75.90 / 96.09 | | |
| `sim_fno` (organisers; simulations only) | 1.228 | 61.95 / 67.12 / 56.21 | | |

Rollout rel_l2 by seed (0 / 1 / 2):

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| S | 0.1104 | 0.1114 | 0.1132 |
| PD | 0.1325 | 0.1388 | 0.1406 |
| PA | 0.1376 | 0.1288 | 0.1386 |
| ZD | 0.2606 | 0.2844 | 0.2604 |
| ZA | 0.2515 | 0.2910 | 0.3247 |

## Predictions

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| E6-P1 | adv(PD) on rollout rel_l2 ≥ 10% and > 2 SE | adv(PD) = **−22.9% ± 1.5%** | **MISS** |
| E6-P2 | err(PA) < err(PD) by more than 2 SE | err(PD) − err(PA) = 0.0023 ± 0.0044 | **MISS** |
| E6-P3 | zero-shot ZA beats S | 0.289 against 0.112 | **MISS** |
| E6-P4 | err(PD) at least 10% below err(ZD) | 48.8% ± 1.5% below | **HIT** |
| E6-P5 | no arm reaches `sim_real_fno`'s rel_l2 subscore (95.47) | best is S at 94.71 | **HIT** |

## Readings

* **Pretraining on simulations costs accuracy at these budgets.**
  * adv(PD) = −22.9% ± 1.5% and adv(PA) = −20.9% ± 2.7%. All six fine-tuned runs are worse than
    their seed's scratch run.
  * In E1 the same protocol (20,000 pretraining steps, a 2000-step fine-tune at lr 1e-4) halved the
    one-step error. There, source and target were the same equations.
* **The simulations do not transfer to the measurements.**
  * Zero-shot, the pretrained models' one-step forecasts are only 4% better than persistence (0.956).
  * The TKE subscore collapses from 76 to 43–48: the simulated flow's fluctuations do not carry
    over.
  * Fine-tuning recovers most of the gap (E6-P4) but stops 21–23% short of training from scratch.
* **Simulating the test conditions does not help.** PA and PD are indistinguishable, both
  fine-tuned (0.0023 ± 0.0044) and zero-shot (0.29 ± 0.02 against 0.27 ± 0.01). The Reynolds-number
  coverage of the pretraining data is not what limits transfer here.
* **The real data alone go a long way.** Trained from scratch on 65 trajectories, the model scores
  94.71. The organisers' FNO scores 95.47, after fine-tuning on real data that may include these
  test conditions.
* **The short fine-tune, not the pretraining, made the difference.** The fine-tuned arms see 2000
  real-data steps, against S's 22,000. Follow-up F1 (below) gives the pretrained starts S's full
  budget, and they match scratch.

## Cross-device check

The lab machine's seed-0 S run (RTX PRO 6000, `results/rows_lab.jsonl`) and Kaggle's (T4) agree:
* rel_l2 0.1099 against 0.1104;
* subscores 94.79 against 94.77.

Run on a different card, the same code and seed land within 0.5%. That is small next to every
difference above.

## Follow-up F1: the same real-data budget (spec §10)

* **The question.** Does the pretrained start help, or still hurt, when it gets the same real-data
  training as scratch?
* **The arms.** The six E6 pretrained models (PD and PA, 3 seeds, the same weights E6 scored; sha256
  in `results/kaggle_env_f1.json`) were trained on the 65 real trajectories with S's exact recipe:
  22,000 steps, lr 1e-3, cosine, batch 16. Each run used S's seed, so it drew the same batches, and
  only the starting weights differ.
* **Where:** Kaggle job `ugp2-e6-f1` on two T4s, 2 Oct, commit 6633f0e, 29 min per run. The rows are
  in `results/rows_f1.jsonl`.

| Arm | Rollout rel_l2 | Subscores (rel_l2 / TKE / MVPE) | One-step / persistence | adv vs S, by seed (0 / 1 / 2) |
|---|---|---|---|---|
| S (E6) | 0.1117 ± 0.0008 | 94.71 / 75.58 / 94.24 | 0.753 | |
| **PDm**, disjoint simulations, full budget | 0.1112 ± 0.0011 | 94.73 / 76.24 / 94.38 | 0.744 | −2.2% / −0.2% / +3.7% |
| **PAm**, all simulations, full budget | **0.1094 ± 0.0006** | **94.81** / 76.10 / 94.42 | 0.744 | +1.7% / +0.7% / +3.6% |
| PD (E6, 2000-step fine-tune) | 0.1373 ± 0.0025 | 93.58 / 73.26 / 93.05 | 0.789 | |

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| E6F-P1 | at the same budget, \|adv(PDm)\| < 5% | adv(PDm) = +0.4% ± 1.8% | **HIT** |
| E6F-P2 | err(PAm) is not below err(PDm) by more than 2 SE | 0.0018 ± 0.0014 (1.3 SE) | **HIT** |
| E6F-P3 | err(PDm) at least 10% below err(PD) | 19.0% ± 2.2% below | **HIT** |

* **At the same budget, the starting point barely matters.**
  * PDm lands on scratch: +0.4% ± 1.8%, with seeds either side.
  * PAm is 2.0% ± 0.9% better, positive in all three seeds but well inside the 5% band of E6F-P1's
    "neither helps nor hurts much".
  * With 22,000 real-data steps at lr 1e-3, the real data dominate where the model ends up.
* **E6's deficit was the fine-tuning budget.**
  * The same pretrained weights reach 0.137 after 2000 steps at lr 1e-4, and 0.111 after S's
    recipe: 19% better.
  * Read with E6, simulation pretraining is a slower route to the same place, not a better one.
* **The simulations of the test conditions are worth at most a few percent.** PA's edge over PD
  (0.0018 ± 0.0014) is not significant on its own, but the advantage over scratch is positive in all
  three PA seeds. More seeds would be needed to call it.
* **The fluctuation subscore recovers fully.** TKE is 76.1–76.2 for both full-budget arms, against
  75.6 for scratch and 73.3 after the short fine-tune.

