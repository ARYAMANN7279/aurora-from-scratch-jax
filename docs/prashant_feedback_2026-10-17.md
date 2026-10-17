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
