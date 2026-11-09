# Project report (IEEE conference format)

`main.tex` uses the course's IEEE conference template (`IEEEtran.cls`). Figures are in `figures/` and are
copied from `../reports/`, which the experiment scripts regenerate.

## Open in Overleaf
1. Overleaf: **New Project → Upload Project** → choose `overleaf_upload.zip`.
2. Set the main document to `main.tex` (Menu → Main document) and compile with pdfLaTeX.

## Rebuild locally
    pdflatex main.tex && pdflatex main.tex

Every number in the report comes from files in the repository:
`reports/logs/*.csv|json` (training curves, ablations), `models/metadata.json` (test-split metrics, SHAP,
mood hold-out), `reports/logs/dataset_stats.json` (dataset counts), and the README tables (cross-validation).
