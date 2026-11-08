"""Collect the CRNN ablation runs into one table (reports/ablation_results.md/.csv).

    python scripts/summarize_ablations.py [--val]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "reports" / "logs"
ROWS = [
    ("full", "Full Attention-CRNN (CNN + BiGRU + attention + SpecAugment, soft vote)"),
    ("no_attention", "w/o temporal attention (mean pooling over time)"),
    ("no_gru", "w/o BiGRU (per-step linear projection)"),
    ("no_specaugment", "w/o SpecAugment"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--val", action="store_true")
sfx = "_val" if ap.parse_args().val else ""

lines = ["| Configuration | Params | Song acc. (soft vote) | Slice acc. (no vote) | Macro F1 | Best epoch |",
         "|---|---:|---:|---:|---:|---:|"]
csv = ["config,params,test_song_acc,test_slice_acc,test_macro_f1,best_epoch,selection_split"]
for key, label in ROWS:
    p = LOGS / f"crnn_{key}{sfx}_summary.json"
    if not p.exists():
        lines.append(f"| {label} | – | not run | | | |")
        continue
    s = json.loads(p.read_text())
    lines.append(f"| {label} | {s['params']:,} | {s['test_song_acc']:.3f} | {s['test_slice_acc']:.3f} | "
                 f"{s['test_macro_f1']:.3f} | {s['best_epoch']} |")
    csv.append(f"{key},{s['params']},{s['test_song_acc']},{s['test_slice_acc']},{s['test_macro_f1']},"
               f"{s['best_epoch']},{s['selection_split']}")
cnn = LOGS / f"cnn_summary{sfx}.json"
if cnn.exists():
    s = json.loads(cnn.read_text())
    lines.append(f"| Attention-CNN on full 30 s clip (no slicing, no BiGRU) | {s['params']:,} | {s['test_acc']:.3f} | – | "
                 f"{s['test_macro_f1']:.3f} | {s['best_epoch']} |")
    csv.append(f"cnn_full_clip,{s['params']},{s['test_acc']},,{s['test_macro_f1']},{s['best_epoch']},{s['selection_split']}")

(ROOT / "reports" / "ablation_results.md").write_text("\n".join(lines) + "\n")
(ROOT / "reports" / "ablation_results.csv").write_text("\n".join(csv) + "\n")
print("\n".join(lines))
print("\nSaved -> reports/ablation_results.md / .csv")
