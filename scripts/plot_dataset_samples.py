"""Dataset figures and statistics for the report.

    python scripts/plot_dataset_samples.py

Writes:
  reports/dataset_samples.png     one log-mel spectrogram per GTZAN genre
  reports/mood_distribution.png   Spotify mood-quadrant counts (if data/raw/dataset.csv exists)
  reports/logs/dataset_stats.json counts used in the report's dataset section
  reports/logs/spotify_samples.csv a few example rows for the "sample data" table
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import musicml  # noqa: E402,F401  (Apple Silicon threading guards)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from musicml.config import HOP, MEL_DIR, SR, SPOTIFY_CSV  # noqa: E402

stats = {}
genres = sorted(d.name for d in MEL_DIR.iterdir() if d.is_dir())
counts = {g: len(list((MEL_DIR / g).glob("*.npy"))) for g in genres}
stats["gtzan"] = {"genres": genres, "clips_per_genre": counts, "total_clips": sum(counts.values()),
                  "sample_rate": SR, "n_mels": 128, "hop_length": HOP}

fig, axes = plt.subplots(2, 5, figsize=(12, 4.2), sharex=True, sharey=True)
for ax, g in zip(axes.ravel(), genres):
    f = sorted((MEL_DIR / g).glob("*.npy"))[0]
    mel = np.load(f).astype(np.float32)[:, :1280]
    secs = mel.shape[1] * HOP / SR
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=[0, secs, 0, mel.shape[0]])
    ax.set_title(f"{g} ({f.stem.split('.')[-1]})", fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("Time (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Mel bin")
fig.suptitle("GTZAN: one log-mel spectrogram (128 bands, 22.05 kHz) per genre", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "dataset_samples.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved -> reports/dataset_samples.png")

if SPOTIFY_CSV.exists():
    df = pd.read_csv(SPOTIFY_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
    raw = len(df)
    df = df.drop_duplicates(subset="track_id").dropna(subset=["valence", "energy"])
    df["mood"] = np.where(df.valence >= 0.5, np.where(df.energy >= 0.5, "happy", "calm"),
                          np.where(df.energy >= 0.5, "tense", "sad"))
    mc = df["mood"].value_counts().reindex(["happy", "tense", "sad", "calm"])
    stats["spotify"] = {"raw_rows": raw, "unique_tracks": len(df),
                        "genres": int(df["track_genre"].nunique()) if "track_genre" in df else None,
                        "mood_counts": {k: int(v) for k, v in mc.items()},
                        "mood_share": {k: round(float(v) / len(df), 4) for k, v in mc.items()}}
    fig, ax = plt.subplots(figsize=(4.6, 3))
    colors = {"happy": "#e2a12b", "tense": "#d0493a", "sad": "#4a72c9", "calm": "#3d9b7c"}
    ax.bar(mc.index, mc.values, color=[colors[m] for m in mc.index])
    for i, v in enumerate(mc.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Tracks")
    ax.set_ylim(0, mc.max() * 1.25)
    ax.set_title(f"Spotify mood quadrants (n = {len(df):,})", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "mood_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    cols = ["track_name", "artists", "track_genre", "danceability", "energy", "valence", "loudness", "tempo",
            "acousticness", "mood"]
    pick = pd.concat([df[df.mood == m].sort_values("popularity", ascending=False).head(1) for m in ["happy", "tense", "sad", "calm"]])
    pick[[c for c in cols if c in pick]].to_csv(ROOT / "reports" / "logs" / "spotify_samples.csv", index=False)
    print("Saved -> reports/mood_distribution.png, reports/logs/spotify_samples.csv")

(ROOT / "reports" / "logs").mkdir(parents=True, exist_ok=True)
(ROOT / "reports" / "logs" / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
