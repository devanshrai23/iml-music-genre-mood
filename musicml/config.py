"""Constants that must match the training scripts in src/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MEL_DIR = DATA_DIR / "processed" / "mel"
FEATURES_CSV = DATA_DIR / "processed" / "gtzan_librosa_features.csv"
AUDIO_DIR = DATA_DIR / "raw" / "Data" / "genres_original"
SPOTIFY_CSV = DATA_DIR / "raw" / "dataset.csv"

# Audio / spectrogram settings (src/make_spectrograms.py, src/train_crnn.py)
SR = 22050
N_MELS = 128
HOP = 512                 # librosa default hop_length
CLIP_SECONDS = 30.0       # GTZAN clip length
FIXED_FRAMES = 1280       # ~29.7 s at SR / HOP
NUM_SLICES = 10           # CRNN: 10 x 3 s slices
SLICE_LEN = 128           # frames per slice
SEED = 42

# sorted(os.listdir(mel_dir)) in the training scripts
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Mood (src/mood_final.py)
MOODS = ["calm", "happy", "sad", "tense"]          # np.sort(y.unique())
MOOD_BASE_FEATURES = ["danceability", "loudness", "tempo", "acousticness",
                      "instrumentalness", "speechiness", "liveness", "mode", "key"]
MOOD_CALIBRATION = [1.45, 1.0, 1.0, 1.0]           # calm prior boost

FEATURE_GROUPS = {
    "mfcc": "Timbre (MFCC)",
    "chroma": "Harmony (Chroma)",
    "spec": "Spectral shape",
    "zcr": "Noisiness (ZCR)",
    "rms": "Loudness (RMS)",
    "tempo": "Rhythm (Tempo)",
}


def feature_group(name: str) -> str:
    for prefix, label in FEATURE_GROUPS.items():
        if name.startswith(prefix):
            return label
    return "Other"
