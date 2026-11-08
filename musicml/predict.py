"""Load exported models and turn one audio clip into everything the website shows."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import (SECONDS_PER_FRAME, cnn_input, crnn_input, extract_features, load_clip,
                    mel_db, spectrogram_png, waveform_peaks)
from .config import (GENRES, MODELS_DIR, MOOD_BASE_FEATURES, MOOD_CALIBRATION, MOODS,
                     NUM_SLICES, SLICE_LEN, feature_group)

TOP_K = 10


def _r(x, n=4):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


def _shap_per_class(explainer, X):
    """Return (shap[F, C], base[C]) for a single row, across shap versions."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                         # old shap: list of (n, F)
        sv = np.stack([s[0] for s in sv], axis=-1)
    else:                                            # shap >= 0.45: (n, F, C)
        sv = sv[0]
    base = np.atleast_1d(explainer.expected_value)
    return sv, base


def load_metadata(models_dir=MODELS_DIR):
    path = Path(models_dir) / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


class GenrePredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.dir = Path(models_dir)
        self.meta = load_metadata(self.dir)
        self.genres = self.meta.get("genres", GENRES)
        self.feature_names = self.meta.get("feature_names")
        self.available = []
        self.crnn = self.cnn = self.rf = self.svm = self.explainer = None
        self._load()

    # ----------------------------------------------------------------- loading
    def _load(self):
        import joblib
        import torch
        from .models import AttentionCRNN, AudioAttentionCNN

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.torch = torch
        self.device = torch.device(os.environ.get("MUSICML_DEVICE", "cpu"))

        for name, cls, attr in [("crnn.pt", AttentionCRNN, "crnn"), ("cnn.pt", AudioAttentionCNN, "cnn")]:
            p = self.dir / name
            if p.exists():
                m = cls(num_classes=len(self.genres))
                m.load_state_dict(torch.load(p, map_location="cpu"))
                setattr(self, attr, m.to(self.device).eval())
                self.available.append(attr)
        if (self.dir / "rf.joblib").exists():
            import shap
            self.rf = joblib.load(self.dir / "rf.joblib")
            self.explainer = shap.TreeExplainer(self.rf)
            self.available.append("rf")
        if (self.dir / "svm.joblib").exists():
            self.svm = joblib.load(self.dir / "svm.joblib")
            self.available.append("svm")
        if not self.available:
            raise FileNotFoundError(
                f"No models found in {self.dir}. Run `python scripts/export_models.py` first.")

    # ----------------------------------------------------------------- inference
    def analyze_file(self, path, offset=None, **extra):
        y, info = load_clip(path, offset=offset)
        return self.analyze(y, info, **extra)

    def analyze(self, y, info=None, title=None, true_label=None):
        t0 = time.perf_counter()
        torch = self.torch
        mel = mel_db(y)
        out = {
            "title": title,
            "true_label": true_label,
            "clip": info or {"duration": round(len(y) / 22050, 2), "offset": 0.0},
            "genres": self.genres,
            "waveform": waveform_peaks(y),
            "spectrogram": spectrogram_png(mel),
            "slice_seconds": round(SLICE_LEN * SECONDS_PER_FRAME, 3),
            "models": {},
        }

        with torch.no_grad():
            if self.crnn is not None:
                logits, alpha = self.crnn(torch.from_numpy(crnn_input(mel)).to(self.device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()            # (10, C)
                mean = probs.mean(axis=0)
                out["models"]["crnn"] = {
                    "probs": _r(mean),
                    "pred": self.genres[int(mean.argmax())],
                    "slices": [{"probs": _r(p, 3), "pred": self.genres[int(p.argmax())]} for p in probs],
                    "attention": [_r(a, 4) for a in alpha.cpu().numpy()],     # (10, 16)
                }
            if self.cnn is not None:
                logits, attn = self.cnn(torch.from_numpy(cnn_input(mel)).to(self.device))
                p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                out["models"]["cnn"] = {"probs": _r(p), "pred": self.genres[int(p.argmax())],
                                        "attention": _r(attn.cpu().numpy()[0])}  # (40,)

        if self.rf is not None or self.svm is not None:
            feats = extract_features(y)
            names = self.feature_names or list(feats.keys())
            X = pd.DataFrame([[feats[n] for n in names]], columns=names)
            for key, model in [("rf", self.rf), ("svm", self.svm)]:
                if model is None:
                    continue
                p = model.predict_proba(X)[0]
                classes = [str(c) for c in model.classes_]
                full = np.array([p[classes.index(g)] if g in classes else 0.0 for g in self.genres])
                out["models"][key] = {"probs": _r(full), "pred": str(model.predict(X)[0])}
            if self.explainer is not None:
                out["shap"] = self._explain(X, names)

        out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return out

    def _explain(self, X, names):
        sv, base = _shap_per_class(self.explainer, X)                # (F, C), (C,)
        classes = [str(c) for c in self.rf.classes_]
        values = X.iloc[0].to_numpy()
        per_class = {}
        for ci, cls in enumerate(classes):
            col = sv[:, ci]
            top = np.argsort(-np.abs(col))[:TOP_K]
            groups = {}
            for fi, name in enumerate(names):
                g = feature_group(name)
                groups[g] = groups.get(g, 0.0) + float(col[fi])
            per_class[cls] = {
                "base": round(float(base[ci] if len(base) > ci else base[0]), 4),
                "top": [{"name": names[i], "value": round(float(values[i]), 4),
                         "shap": round(float(col[i]), 5), "group": feature_group(names[i])} for i in top],
                "groups": {k: round(v, 5) for k, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))},
            }
        return {"model": "rf", "by_class": per_class}


# ===================================================================== mood
def mood_frame(rows):
    """Base Spotify features + the five interaction features from src/mood_final.py."""
    df = pd.DataFrame(rows)[MOOD_BASE_FEATURES].astype(float)
    out = df.copy()
    out["loudness_x_dance"] = df["loudness"] * df["danceability"]
    out["acoustic_x_inst"] = df["acousticness"] * df["instrumentalness"]
    out["acoustic_to_loudness"] = df["acousticness"] / (np.abs(df["loudness"]) + 1.0)
    out["dance_x_tempo"] = df["danceability"] * df["tempo"]
    out["speech_to_acoustic"] = df["speechiness"] / (df["acousticness"] + 0.01)
    return out


class MoodPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        import joblib
        import shap
        path = Path(models_dir) / "mood_rf.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        self.model = joblib.load(path)
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = [str(c) for c in self.model.classes_]
        self.weights = np.array([MOOD_CALIBRATION[MOODS.index(c)] for c in self.classes])

    def predict(self, features: dict):
        missing = [k for k in MOOD_BASE_FEATURES if k not in features]
        if missing:
            raise ValueError(f"Missing mood features: {', '.join(missing)}")
        X = mood_frame([features])
        raw = self.model.predict_proba(X)[0]
        cal = raw * self.weights
        cal_norm = cal / cal.sum()
        pred = self.classes[int(np.argmax(cal))]
        sv, base = _shap_per_class(self.explainer, X)
        ci = self.classes.index(pred)
        col = sv[:, ci]
        order = np.argsort(-np.abs(col))
        return {
            "classes": self.classes,
            "probs_raw": _r(raw),
            "probs_calibrated": _r(cal_norm),
            "pred": pred,
            "pred_uncalibrated": self.classes[int(np.argmax(raw))],
            "shap": {"class": pred, "base": round(float(base[ci]), 4),
                     "top": [{"name": X.columns[i], "value": round(float(X.iloc[0, i]), 4),
                              "shap": round(float(col[i]), 5)} for i in order[:8]]},
        }
