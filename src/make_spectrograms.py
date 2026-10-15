import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
from joblib import Parallel, delayed

AUDIO_DIR = Path("data/raw/Data/genres_original")
OUT_DIR = Path("data/processed/mel")
SR = 22050

def convert(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        db = librosa.power_to_db(mel, ref=np.max).astype(np.float16)
        out = OUT_DIR / path.parent.name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{path.stem}.npy", db)
    except Exception as e:
        print("FAILED:", path.name, e)

if __name__ == "__main__":
    files = sorted(AUDIO_DIR.rglob("*.wav"))
    Parallel(n_jobs=-1, verbose=5)(delayed(convert)(p) for p in files)
