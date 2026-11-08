#!/usr/bin/env bash
# Re-runs every experiment the report needs and regenerates its figures/tables.
#
#   bash scripts/run_report_experiments.sh          # original protocol (best epoch picked on the test split)
#   bash scripts/run_report_experiments.sh --val    # clean protocol (best epoch picked on a 10% validation split)
#
# Takes roughly 20-40 minutes on an M-series Mac. Output: reports/logs/, reports/*.png, reports/ablation_results.md
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

VAL_ARGS=""; PLOT_ARGS=""
if [[ "${1:-}" == "--val" ]]; then VAL_ARGS="--val-frac 0.1"; PLOT_ARGS="--val"; fi

echo "==> [1/7] Attention-CNN";              python src/train_cnn.py $VAL_ARGS
echo "==> [2/7] Attention-CRNN (full)";      python src/train_crnn.py $VAL_ARGS
echo "==> [3/7] Ablation: no attention";     python src/train_crnn.py --variant no_attention $VAL_ARGS
echo "==> [4/7] Ablation: no BiGRU";         python src/train_crnn.py --variant no_gru $VAL_ARGS
echo "==> [5/7] Ablation: no SpecAugment";   python src/train_crnn.py --variant no_specaugment $VAL_ARGS
echo "==> [6/7] Figures and tables"
python scripts/plot_training_curves.py $PLOT_ARGS
python scripts/summarize_ablations.py $PLOT_ARGS
python scripts/plot_dataset_samples.py
echo "==> [7/7] Refreshing the website models with the new checkpoints"
python scripts/export_models.py --skip-training
python scripts/build_demo_samples.py
echo; echo "All done. Commit with:"
echo "  git add reports/ models/ web/public/data web/public/samples && git commit -m 'Report experiments: logs, loss curves, ablations' && git push"
