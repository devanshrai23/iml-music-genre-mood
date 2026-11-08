"""Audio loading, feature extraction and spectrogram preprocessing.

These functions reproduce src/extract_features.py, src/make_spectrograms.py,
src/train_cnn.py (MelDataset) and src/train_crnn.py (GTZANSliceDataset).
"""
import base64
import io
import warnings
from collections import OrderedDict

import numpy as np

warnings.filterwarnings("ignore")
import librosa  # noqa: E402

from ._numba_safe import install as _install_numba_safe  # noqa: E402

_install_numba_safe()   # avoids numba segfaults on Apple Silicon when torch is loaded

from .config import CLIP_SECONDS, FIXED_FRAMES, HOP, N_MELS, NUM_SLICES, SLICE_LEN, SR

MIN_SECONDS = 5.0


# --------------------------------------------------------------------------- loading
def load_clip(path, offset=None):
    """Load an audio file as mono 22 050 Hz and cut a 30 s analysis window.

    ``offset`` is the window start in seconds. When it is None and the file is
    longer than 30 s, the window is centred in the track (intros and outros are
    usually less typical of a genre than the middle).
    """
    y, _ = librosa.load(path, sr=SR, mono=True)
    total = len(y) / SR
    if total < MIN_SECONDS:
        raise ValueError(f"Clip is {total:.1f} s long; at least {MIN_SECONDS:.0f} s of audio is needed.")
    if total > CLIP_SECONDS + 0.5:
        if offset is None:
            offset = (total - CLIP_SECONDS) / 2.0
        offset = float(np.clip(offset, 0.0, total - CLIP_SECONDS))
        start = int(round(offset * SR))
        y = y[start:start + int(CLIP_SECONDS * SR)]
    else:
        offset = 0.0
    return y, {"total_duration": round(total, 2), "offset": round(offset, 2),
               "duration": round(len(y) / SR, 2)}


# --------------------------------------------------------------------------- handcrafted features
def _summarize(name, arr, out):
    arr = np.atleast_2d(arr)
    for i, row in enumerate(arr):
        suffix = "" if arr.shape[0] == 1 else f"_{i + 1}"
        out[f"{name}{suffix}_mean"] = float(row.mean())
        out[f"{name}{suffix}_std"] = float(row.std())


def _estimate_tempo_safe(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset_env, max_size=4 * sr // 512)
    freqs = librosa.tempo_frequencies(len(ac), sr=sr, hop_length=512)
    mask = (freqs >= 40.0) & (freqs <= 240.0)
    if not np.any(mask):
        return 120.0
    return float(freqs[mask][np.argmax(ac[mask])])


def extract_features(y, sr=SR):
    """The 89 named summary features used by the RF / SVM baselines."""
    f = OrderedDict()
    _summarize("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), f)
    _summarize("chroma", librosa.feature.chroma_stft(y=y, sr=sr), f)
    _summarize("spec_centroid", librosa.feature.spectral_centroid(y=y, sr=sr), f)
    _summarize("spec_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr), f)
    _summarize("spec_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr), f)
    _summarize("spec_contrast", librosa.feature.spectral_contrast(y=y, sr=sr), f)
    _summarize("zcr", librosa.feature.zero_crossing_rate(y), f)
    _summarize("rms", librosa.feature.rms(y=y), f)
    f["tempo"] = _estimate_tempo_safe(y, sr)
    return f


# --------------------------------------------------------------------------- spectrograms
def mel_db(y, sr=SR):
    """Log-mel spectrogram exactly as cached by src/make_spectrograms.py (float16 round trip)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float16).astype(np.float32)


def crnn_input(mel):
    """(10, 1, 128, 128) slices: edge-pad/crop to 1280 frames, then per-track standardise."""
    need = NUM_SLICES * SLICE_LEN
    if mel.shape[1] < need:
        mel = np.pad(mel, ((0, 0), (0, need - mel.shape[1])), mode="edge")
    else:
        mel = mel[:, :need]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    slices = [mel[:, s * SLICE_LEN:(s + 1) * SLICE_LEN] for s in range(NUM_SLICES)]
    return np.stack(slices)[:, None, :, :].astype(np.float32)


def cnn_input(mel):
    """(1, 1, 128, 1280): standardise first, then zero-pad/crop (src/train_cnn.py order)."""
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if mel.shape[1] < FIXED_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, FIXED_FRAMES - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :FIXED_FRAMES]
    return mel[None, None, :, :].astype(np.float32)


# --------------------------------------------------------------------------- visuals for the website
def waveform_peaks(y, points=480):
    """Peak envelope scaled to 0..1, for drawing a waveform."""
    if len(y) == 0:
        return [0.0] * points
    chunks = np.array_split(np.abs(y), points)
    peaks = np.array([c.max() if len(c) else 0.0 for c in chunks])
    peaks = peaks / (peaks.max() + 1e-9)
    return [round(float(p), 3) for p in peaks]


def spectrogram_png(mel, width=640):
    """Render the log-mel spectrogram (first 1280 frames) to a magma PNG data URL."""
    from matplotlib import colormaps
    from PIL import Image

    m = mel[:, :FIXED_FRAMES]
    if m.shape[1] < FIXED_FRAMES:
        m = np.pad(m, ((0, 0), (0, FIXED_FRAMES - m.shape[1])), mode="constant", constant_values=m.min())
    factor = max(1, m.shape[1] // width)
    m = m[:, : (m.shape[1] // factor) * factor].reshape(m.shape[0], -1, factor).mean(axis=2)
    lo, hi = np.percentile(m, 2), m.max()
    norm = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)[::-1]   # low frequencies at the bottom
    rgb = (colormaps["magma"](norm)[:, :, :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


SECONDS_PER_FRAME = HOP / SR
