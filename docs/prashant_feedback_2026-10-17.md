# Prashant's report feedback, 17 Oct 2026

Source: the handwritten annotations in `FINAL_REPORT_261017_165949.pdf` (17 pages).
The original scientific verdicts remain the record; new measurements are dated follow-ups.

| Annotated page | Request | Resolution / evidence |
|---|---|---|
| 7, top | Figure of training steps / loops | Add a training-protocol schematic and E1 learning curves (E1 F1, 15 reruns). |
| 7, margin; 16, top | Contours: truth, zero-shot and fine-tuned results | Add fixed-trajectory E1 vorticity contours at steps 1, 5 and 20, shared scales and error maps. |
| 8, E1 drift plot | Why do n = 8 and n = 32 have a smaller rollout penalty? | Diagnose original E1 rollout amplitude and alignment; distinguish evidence from untested explanations. |
| 10, E2 | Repeat on more diverse physics, Re 30 and 3000 | E2 F1: same 30-run comparison with a 100× Re gap; preserve the original E2 results. |
| 11, F1 / FNO rows | Any reason for the remaining FNO gap? | Quantify the error gap and explain the uncontrolled training / architecture differences. |
| 12, E4 table | Make “negative is better” clear | Put the relative-change definition and sign convention in the table headers and caption. |
| 13, speed paragraph | Tick beside CUDA-graph comparison | Retain the comparison; no correction was requested. |
| 14, hypothesis | Underlined proximity interpretation | Keep the interpretation qualified by fixed compute, distance confounding and small scale. |
| 15, bottom | Propose an architecture informed by RealPDE experience | Add a specific, untested proposal and an evaluation plan; respect PIV's non-solenoidal component. |

Execution status and commands will be recorded here after validation and measurements. Training
curves and E2 F1 predictions are in their respective `handouts/` files before any training run.

## Execution record

* Predictions registered in `90517d2`; training / scoring tooling committed in `606e4f6`.
* VM interpreter: the repository's `.venv/bin/python` (Python 3.11.15, JAX 0.10.2). The parent
  `Aryamann/env` does not currently contain JAX. The old scratch GPU wrappers were not found;
  `experiments/lab/glane3.sh` and `gpu_guard.py` restore the placement and thermal protections.
  Scratch, temporary files and the compilation cache live under `data/feedback_scratch/`.
* Focused tests passed: 16 tests in 147.84 seconds, then 6 tooling tests in 0.18 seconds,
  including one newly added scoring test (17 distinct checks). The full CPU suite passed:
  **125 tests in 982.08 seconds** (16 min 22 s, alongside three cold GPU compilations).
  Log: `logs/feedback_full_tests.log` on the VM.
* The original E2, E3, E4 and E5 analyses recompute exactly from their committed rows.
* `e2f1_v1` is assembled and checksum-verified: 12 original trajectories linked, four Re 30
  trajectories generated on CPU; all 16 complete. Provenance: `experiments/e2/results/f1_data.json`.
* Original E1 weights were used for the exploratory diagnosis and contours. Mean rollout drifts
  agree with the original result rows within 9.97e-6; the error / amplitude / alignment identity
  holds within 1.07e-14. Data: `experiments/e1/results/fields.json`. Raw example fields stay off git.
* The three guarded training lanes run one E1 seed (5 runs), then that seed's E2 F1 shard
  (10 runs). Logs: `logs/feedback_lane_{0,1,2}.log`; results:
  `experiments/e1/results/curves_rows_{0,1,2}.jsonl` and
  `experiments/e2/results/f1_rows_{0,1,2}.jsonl`. Each lane uses `--skip-done` for restarts.
  Initial guard PIDs: 1026958, 1026959, 1026960; initial cards 1, 3, 2 respectively. Card 0 is excluded.
* Initial E1 n = 2 pretrains, seeds 0 / 1 / 2, differ from the historical runs by −1.34% /
  +0.72% / +2.15% one-step error. F1-P1's “every run within 2%” prediction therefore misses;
  retain that verdict and the original results. All three curves contain steps 0–20,000.
* E1 F1 is now complete: **15/15 expected unique configs**, all on code `606e4f6`, with finite
  losses / curves and every final curve point matching its final metric. **0/2 predictions HIT**.
  Six historical comparisons exceed 2% (maximum 2.82%). Scratch minima occur at steps
  9,000 / 9,500 / 9,000; final errors are 5.04 / 4.47 / 4.39% higher. The original final-step
  scores are retained; the minima are a diagnostic, not independently validated new baselines.
* Report prose, protocol schematic, contours, rollout diagnosis, sign labels and architecture
  proposal are drafted. `scripts/build_report.mjs` builds the PDF from the Markdown and checks
  that every image loaded. The final PDF awaits the new training measurements.

Remaining: finish and score the 30 E2 F1 runs, add its results / figure, finish the status records,
rebuild and inspect the final PDF, and sync the finished work. E1's learning curves, figures and
report are complete; its two registered misses are retained.
