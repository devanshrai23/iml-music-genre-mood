from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

AUDIO_DIR = Path("data/raw/Data/genres_original")
OUT_CSV = Path("data/processed/gtzan_librosa_features.csv")
SR = 22050

def summarize(name, arr, out):
    arr = np.atleast_2d(arr)
    for i, row in enumerate(arr):
        suffix = "" if arr.shape[0] == 1 else f"_{i + 1}"
        out[f"{name}{suffix}_mean"] = float(row.mean())
        out[f"{name}{suffix}_std"] = float(row.std())

def extract(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        f = {"filename": path.name, "label": path.parent.name}
        summarize("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), f)
        summarize("chroma", librosa.feature.chroma_stft(y=y, sr=sr), f)
        summarize("spec_centroid", librosa.feature.spectral_centroid(y=y, sr=sr), f)
        summarize("spec_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr), f)
        summarize("spec_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr), f)
        summarize("spec_contrast", librosa.feature.spectral_contrast(y=y, sr=sr), f)
        summarize("zcr", librosa.feature.zero_crossing_rate(y), f)
        summarize("rms", librosa.feature.rms(y=y), f)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        f["tempo"] = float(np.atleast_1d(tempo)[0])
        return f
    except Exception as e:
        return {"filename": path.name, "error": str(e)}

if __name__ == "__main__":
    files = sorted(AUDIO_DIR.rglob("*.wav"))
    print(f"Found {len(files)} audio files")
    results = Parallel(n_jobs=-1, verbose=5)(delayed(extract)(p) for p in files)

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ok).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(ok)} rows -> {OUT_CSV}")
    for b in bad:
        print("FAILED:", b["filename"], "|", b["error"])
