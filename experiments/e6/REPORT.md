# E6 Report: Does pretraining on simulations buy accuracy on real flow measurements? (RealPDE foil)

Spec: `handouts/e6_realpde_sim2real.md` (amendments A1 and A2). Code: `experiments/e6/`. Scored by
`score.py` → `results/analysis.json`, from the rows in `results/rows.jsonl`.

**Status: in progress (1 Oct 2026).** The 15 runs wait for an idle GPU. So far this file has the
set-up and the FNO baselines. The results and the verdicts on E6-P1 to P5 will be added when the
runs land.

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
  * The lab machine's idle cards 1–3, in lane `e6g` (`glane4.sh`, pinned to CPU cores 8–11).
  * A job paused for 10 min is killed and rerun from the start on the next idle card.
  * A CPU smoke test (200 steps, 100 windows) ran first and passed for S, PD pretraining and PD
    fine-tuning.

## FNO baselines (measured before any run)

The organisers' FNOs on the same 900 windows, through UGP-I's harness (`fno_baseline.py`, CPU):

| Model | rel_l2 | TKE | MVPE | Subscores (rel_l2 / TKE / MVPE) |
|---|---|---|---|---|
| `sim_real_fno.pth` (fine-tuned by the organisers on real data, which may include the held-out conditions) | 0.0948 | 0.635 | 0.0813 | 95.47 / 75.90 / 96.09 |
| `sim_fno.pth` (simulations only) | 1.228 | 0.980 | 1.558 | 61.95 / 67.12 / 56.21 |

* `sim_real_fno` reproduces UGP-I's selftest to 1e-3. It is an upper reference, not an honest
  baseline.
* `sim_fno` does not transfer to real data zero-shot. That corrected E6-P3's anchor (A2).
