"""Pre-compute analyses for a gallery of GTZAN *test* songs.

The JSON files land in web/public/samples/, so the deployed website (e.g. on
Vercel) can show full results even when no backend is running.

    python scripts/build_demo_samples.py            # 2 songs per genre, with audio
    python scripts/build_demo_samples.py --no-audio # JSON only (no GTZAN audio shipped)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from musicml.audio import load_clip  # noqa: E402
from musicml.config import AUDIO_DIR, MODELS_DIR, SR  # noqa: E402
from musicml.predict import GenrePredictor, MoodPredictor, load_metadata  # noqa: E402

OUT = ROOT / "web" / "public" / "samples"


def write_audio(y, dest_stem):
    """Write a compressed copy of the clip; MP3 when libsndfile supports it, else WAV."""
    for ext, fmt, sub in [("mp3", "MP3", "MPEG_LAYER_III"), ("wav", "WAV", "PCM_16")]:
        try:
            path = dest_stem.parent / f"{dest_stem.name}.{ext}"
            sf.write(path, y.astype(np.float32), SR, format=fmt, subtype=sub)
            return path.name
        except Exception:
            continue
    return None


def pick_tracks(meta, per_genre):
    tracks = meta.get("test_tracks", [])
    have = {k for t in tracks for k in t["preds"]}
    main = next((k for k in ("crnn", "cnn", "svm", "rf") if k in have), None)
    chosen = []
    for g in meta["genres"]:
        pool = [t for t in tracks if t["label"] == g]
        # 1) a song where RF fails but the CRNN succeeds (the interesting case), 2) clean hits
        rescue = [t for t in pool if main != "rf" and t["preds"].get(main) == g and t["preds"].get("rf") not in (None, g)]
        hits = [t for t in pool if t["preds"].get(main) == g and t not in rescue]
        misses = [t for t in pool if t["preds"].get(main) not in (None, g)]
        picks = (rescue[:1] + hits + misses)[:per_genre]
        if misses and per_genre >= 3:
            picks = picks[: per_genre - 1] + misses[:1]
        chosen += [(t, "rescue" if t in rescue else ("miss" if t in misses else "hit")) for t in picks]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-genre", type=int, default=2)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    meta = load_metadata()
    if "test_tracks" not in meta:
        sys.exit("models/metadata.json has no test split. Run python scripts/export_models.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    predictor = GenrePredictor(MODELS_DIR)

    index = []
    for t, tag in pick_tracks(meta, args.per_genre):
        path = AUDIO_DIR / t["label"] / t["file"]
        if not path.exists():
            print("missing audio, skipped:", path)
            continue
        sid = Path(t["file"]).stem
        y, info = load_clip(str(path))
        res = predictor.analyze(y, info, title=f"GTZAN {sid}", true_label=t["label"])
        res["id"] = sid
        if not args.no_audio:
            name = write_audio(y, OUT / "audio" / sid)
            res["audio"] = f"samples/audio/{name}" if name else None
        (OUT / f"{sid}.json").write_text(json.dumps(res))
        m = res["models"]
        index.append({"id": sid, "title": res["title"], "label": t["label"], "tag": tag,
                      "crnn": m.get("crnn", {}).get("pred"), "rf": m.get("rf", {}).get("pred"),
                      "audio": res.get("audio")})
        print(f"{sid:18s} true={t['label']:9s} crnn={index[-1]['crnn']}  rf={index[-1]['rf']}  [{tag}]")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))

    if "mood" in meta and (MODELS_DIR / "mood_rf.joblib").exists():
        mp = MoodPredictor(MODELS_DIR)
        presets = [dict(p, result=mp.predict(p["features"])) for p in meta["mood"].get("presets", [])]
        (OUT / "mood.json").write_text(json.dumps(presets))
        print(f"mood presets: {len(presets)}")
    print(f"\nWrote {len(index)} samples to {OUT}")


if __name__ == "__main__":
    main()
