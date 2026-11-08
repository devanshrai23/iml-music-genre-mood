"""Draw the (mandatory) training loss vs. epochs figure from the logged CSVs.

    python scripts/plot_training_curves.py          # logs of the original protocol
    python scripts/plot_training_curves.py --val    # logs of the --val-frac runs

Reads reports/logs/{cnn_history,crnn_full_history}[ _val].csv written by
src/train_cnn.py and src/train_crnn.py, and writes reports/training_loss_curve.png/.pdf.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "logs"

ap = argparse.ArgumentParser()
ap.add_argument("--val", action="store_true", help="use the logs of the validation-split runs")
args = ap.parse_args()
sfx = "_val" if args.val else ""
cnn_path, crnn_path = LOGS / f"cnn_history{sfx}.csv", LOGS / f"crnn_full{sfx}_history.csv"
missing = [p.name for p in (cnn_path, crnn_path) if not p.exists()]
if missing:
    raise SystemExit(f"Missing {', '.join(missing)} in reports/logs/. Run the training scripts first.")
cnn, crnn = pd.read_csv(cnn_path), pd.read_csv(crnn_path)
split = crnn["eval_split"].iloc[0]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.7))
C_CNN, C_CRNN, C_SLICE = "#6b7280", "#1d4ed8", "#0f9d8a"

ax1.plot(crnn.epoch, crnn.train_loss, "-o", ms=3, color=C_CRNN, label="Attention-CRNN: train (CE + label smoothing 0.05)")
ax1.plot(crnn.epoch, crnn.eval_loss, "--", color=C_CRNN, label=f"Attention-CRNN: {split} (CE, per 3 s slice)")
ax1.plot(cnn.epoch, cnn.train_loss, "-s", ms=3, color=C_CNN, label="Attention-CNN: train (CE)")
ax1.plot(cnn.epoch, cnn.eval_loss, "--", color=C_CNN, label=f"Attention-CNN: {split} (CE, per 30 s clip)")
ax1.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss vs. epochs")
ax1.grid(alpha=0.3, linestyle=":")
ax1.legend(loc="upper right")

best = crnn.loc[crnn.eval_song_acc.idxmax()]
ax2.plot(crnn.epoch, crnn.eval_song_acc * 100, "-o", ms=3, color=C_CRNN, label="Attention-CRNN: song level (soft vote)")
ax2.plot(crnn.epoch, crnn.eval_slice_acc * 100, "-.", color=C_SLICE, label="Attention-CRNN: single 3 s slice")
ax2.plot(cnn.epoch, cnn.eval_acc * 100, "-s", ms=3, color=C_CNN, label="Attention-CNN: 30 s clip")
ax2.scatter([best.epoch], [best.eval_song_acc * 100], s=80, facecolors="none", edgecolors="k", zorder=5,
            label=f"selected CRNN epoch ({int(best.epoch)})")
ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"Accuracy on the {split} split vs. epochs")
ax2.grid(alpha=0.3, linestyle=":")
ax2.legend(loc="lower right")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / "reports" / f"training_loss_curve.{ext}", dpi=200, bbox_inches="tight")
print(f"Saved -> reports/training_loss_curve.png / .pdf  (evaluation split: {split})")
print(f"CRNN: final train loss {crnn.train_loss.iloc[-1]:.3f}, best {split} song acc {best.eval_song_acc:.3f} at epoch {int(best.epoch)}")
print(f"CNN : final train loss {cnn.train_loss.iloc[-1]:.3f}, best {split} acc {cnn.eval_acc.max():.3f} at epoch {int(cnn.loc[cnn.eval_acc.idxmax()].epoch)}")
