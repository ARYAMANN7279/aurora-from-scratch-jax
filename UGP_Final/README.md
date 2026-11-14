# UGP Parts I and II: final presentation and report

The whole undergraduate project in one place: Part I (forecasting a real airfoil wake in the NeurIPS 2026
RealPDE competition) and Part II (Aurora rebuilt in JAX, and what pretraining data buys).

* `UGP_Final_Presentation.pptx` (and `.pdf`): 40 slides. Part I is the UGP-I deck, Part II the UGP-II deck,
  merged into one with joint title, contents, part covers, "Both parts together" and an appendix pointer.
  Speaker notes are in the slides.
* `UGP_Final_Report.pdf` (from `UGP_Final_Report.md`): the complete two-part report (about 42 pages): both parts in detail, with figures, tables, code pointers and appendices.

Rebuild (needs the `UGP` and `UGP-II` folders next to this one, and python-pptx):

```bash
python deck/merge_decks.py                       # the deck, from the two source decks
../UGP-II/.venv/bin/python make_figs.py          # report figures
node build_report.mjs                            # report PDF
```
