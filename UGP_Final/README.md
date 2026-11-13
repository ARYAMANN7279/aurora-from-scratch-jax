# UGP Parts I and II: final presentation and report

The whole undergraduate project in one place: Part I (forecasting a real airfoil wake in the NeurIPS 2026
RealPDE competition) and Part II (this repository: Aurora rebuilt in JAX, and what pretraining data buys).

* `UGP_Final_Presentation.pptx`: 38 slides (33 for the talk, 5 backup), each with speaker notes. The notes
  are also in `UGP_Final_speaker_notes.md`.
* `UGP_Final_Report.pdf` (from `UGP_Final_Report.md`): the story in plain words with figures; formulas and
  full tables in the appendices.

Rebuild from this folder:

```bash
../.venv/bin/python make_figs.py                  # report figures (three crops need the UGP-I folder beside this repository)
../.venv/bin/python deck/make_deck_figs.py        # deck charts, drawn at slide size
../.venv/bin/python deck/make_physics_figs.py     # flow fields and physics maps from E11's trained models
(cd deck && npm install && node build.js)         # deck and speaker notes
node build_report.mjs                             # report PDF
```

`make_physics_figs.py` needs E11's trained weights in `experiments/e11/results/params/`, which are kept out of
git; the figures it made are committed in `figs/`.
