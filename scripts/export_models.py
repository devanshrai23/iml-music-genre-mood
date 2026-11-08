"""Train/collect every model the website needs and save it to models/.

Run from the repository root on the machine that has the datasets:

    python scripts/export_models.py

What it does:
  1. Runs any missing pipeline step (feature CSV, cached mel-spectrograms,
     CNN and CRNN training) using the existing scripts in src/.
  2. Copies the trained CNN / CRNN weights to models/.
  3. Fits the RF and SVM baselines on the same song-level 80/20 split the CRNN
     used, so the website never "predicts" a song the models were trained on.
  4. Fits the calibrated mood RF (if data/raw/dataset.csv exists).
  5. Re-evaluates every model through musicml/ (the code the website uses) and
     writes models/metadata.json + web/public/data/metrics.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from musicml.config import (AUDIO_DIR, FEATURES_CSV, MEL_DIR, MODELS_DIR, MOOD_BASE_FEATURES,  # noqa: E402
                            MOOD_CALIBRATION, MOODS, SEED, SPOTIFY_CSV, feature_group)

WEB_DATA = ROOT / "web" / "public" / "data"
REPORTS = ROOT / "reports"


def step(msg):
    print(f"\n==> {msg}", flush=True)


def run_script(name):
    print(f"    running src/{name} (this can take a while)...", flush=True)
    subprocess.run([sys.executable, f"src/{name}"], cwd=ROOT, check=True)


def ensure_prerequisites(skip_training):
    if not FEATURES_CSV.exists() or not MEL_DIR.exists():
        if not AUDIO_DIR.exists():
            sys.exit(f"GTZAN audio not found at {AUDIO_DIR}. See README > 'Place Datasets'.")
    if not FEATURES_CSV.exists():
        step("Feature CSV missing")
        run_script("extract_features.py")
    if not MEL_DIR.exists() or not any(MEL_DIR.rglob("*.npy")):
        step("Cached mel-spectrograms missing")
        run_script("make_spectrograms.py")
    for pt, script in [("best_cnn_model.pt", "train_cnn.py"), ("best_crnn.pt", "train_crnn.py")]:
        if not (REPORTS / pt).exists():
            if skip_training:
                print(f"    reports/{pt} missing and --skip-training set: that model will be left out.")
                continue
            step(f"reports/{pt} missing, training it now")
            run_script(script)


def metrics_block(y_true, y_pred, labels):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "per_class": {lab: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                            "f1": round(float(f[i]), 3)} for i, lab in enumerate(labels)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def song_split():
    """Identical to src/train_crnn.py / src/compare_explanations.py."""
    genres = sorted(d.name for d in MEL_DIR.iterdir() if d.is_dir())
    paths, labels = [], []
    for gi, g in enumerate(genres):
        for f in sorted((MEL_DIR / g).glob("*.npy")):
            paths.append(f)
            labels.append(gi)
    paths, labels = np.array(paths), np.array(labels)
    tr, te = train_test_split(np.arange(len(paths)), test_size=0.20, stratify=labels, random_state=SEED)
    return genres, paths, labels, tr, te


def export_genre(meta):
    import torch
    from musicml.audio import cnn_input, crnn_input
    from musicml.models import AttentionCRNN, AudioAttentionCNN

    step("Building the song-level 80/20 split")
    genres, paths, labels, tr, te = song_split()
    names = np.array([p.stem + ".wav" for p in paths])
    print(f"    {len(paths)} songs, {len(tr)} train / {len(te)} test")
    meta["genres"] = genres

    df = pd.read_csv(FEATURES_CSV).set_index("filename")
    feature_cols = [c for c in df.columns if c != "label"]
    meta["feature_names"] = feature_cols
    has = np.isin(names, df.index)
    tr_f, te_f = tr[has[tr]], te[has[te]]
    X_tr = df.loc[names[tr_f], feature_cols]
    X_te = df.loc[names[te_f], feature_cols]
    y_tr = np.array(genres)[labels[tr_f]]
    y_te = np.array(genres)[labels[te_f]]

    preds = {}
    step("Fitting Random Forest (300 trees) on the train split")
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    joblib.dump(rf, MODELS_DIR / "rf.joblib", compress=3)
    preds["rf"] = dict(zip(names[te_f], rf.predict(X_te)))

    step("Fitting RBF-SVM (C=10, with probability estimates)")
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=SEED))
    svm.fit(X_tr, y_tr)
    joblib.dump(svm, MODELS_DIR / "svm.joblib", compress=3)
    preds["svm"] = dict(zip(names[te_f], svm.predict(X_te)))

    step("Global TreeSHAP importance on the test split")
    import shap
    sv = shap.TreeExplainer(rf).shap_values(X_te)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv          # (n, F, C)
    mean_abs = np.abs(sv).mean(axis=0)                                  # (F, C)
    overall = mean_abs.mean(axis=1)
    order = np.argsort(-overall)[:20]
    rf_classes = [str(c) for c in rf.classes_]
    meta["shap_global"] = {
        "overall": [{"name": feature_cols[i], "value": round(float(overall[i]), 5),
                     "group": feature_group(feature_cols[i])} for i in order],
        "by_class": {c: [{"name": feature_cols[i], "value": round(float(mean_abs[i, ci]), 5)}
                         for i in np.argsort(-mean_abs[:, ci])[:10]] for ci, c in enumerate(rf_classes)},
    }

    for pt, out, cls, prep, key in [
        ("best_crnn.pt", "crnn.pt", AttentionCRNN, crnn_input, "crnn"),
        ("best_cnn_model.pt", "cnn.pt", AudioAttentionCNN, cnn_input, "cnn"),
    ]:
        src = REPORTS / pt
        if not src.exists():
            continue
        step(f"Copying {pt} -> models/{out} and re-checking it on the test split")
        shutil.copy(src, MODELS_DIR / out)
        model = cls(num_classes=len(genres))
        model.load_state_dict(torch.load(src, map_location="cpu"))
        model.eval()
        p = {}
        with torch.no_grad():
            for i in te:
                mel = np.load(paths[i]).astype(np.float32)
                logits, _ = model(torch.from_numpy(prep(mel)))
                probs = torch.softmax(logits, dim=-1).numpy().mean(axis=0)
                p[names[i]] = genres[int(probs.argmax())]
        preds[key] = p

    meta["models"] = {}
    for key, p in preds.items():
        files = [n for n in names[te] if n in p]
        y_true = [genres[labels[np.where(names == n)[0][0]]] for n in files]
        m = metrics_block(y_true, [p[n] for n in files], genres)
        meta["models"][key] = m
        print(f"    {key:5s} test accuracy {m['accuracy']:.3f} | macro F1 {m['macro_f1']:.3f}")

    meta["test_tracks"] = [
        {"file": n, "label": genres[labels[i]], "preds": {k: str(v[n]) for k, v in preds.items() if n in v}}
        for i, n in zip(te, names[te])
    ]


def export_mood(meta):
    if not SPOTIFY_CSV.exists():
        print(f"\n    {SPOTIFY_CSV} not found: skipping the mood model.")
        return
    from musicml.predict import mood_frame
    step("Mood: deduplicating Spotify tracks and deriving quadrants")
    df = pd.read_csv(SPOTIFY_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates(subset="track_id").dropna(subset=["valence", "energy"] + MOOD_BASE_FEATURES)
    df["mood"] = np.where(df.valence >= 0.5, np.where(df.energy >= 0.5, "happy", "calm"),
                          np.where(df.energy >= 0.5, "tense", "sad"))
    X = mood_frame(df[MOOD_BASE_FEATURES].to_dict("records"))
    X.index = df.index
    y = df["mood"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    step("Mood: fitting RF (200 trees, min_samples_leaf=4 to keep the file small)")
    clf = RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=4,
                                 random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    joblib.dump(clf, MODELS_DIR / "mood_rf.joblib", compress=3)
    classes = [str(c) for c in clf.classes_]
    w = np.array([MOOD_CALIBRATION[MOODS.index(c)] for c in classes])
    probs = clf.predict_proba(X_te)
    raw = np.array(classes)[probs.argmax(1)]
    cal = np.array(classes)[(probs * w).argmax(1)]
    meta["mood"] = {
        "classes": classes,
        "calibration": dict(zip(classes, w.tolist())),
        "rows": int(len(df)),
        "raw": metrics_block(y_te, raw, classes),
        "calibrated": metrics_block(y_te, cal, classes),
        "features": {f: {"min": round(float(df[f].quantile(0.01)), 3), "max": round(float(df[f].quantile(0.99)), 3),
                         "median": round(float(df[f].median()), 3)} for f in MOOD_BASE_FEATURES},
    }
    print(f"    raw acc {meta['mood']['raw']['accuracy']:.3f} F1 {meta['mood']['raw']['macro_f1']:.3f} | "
          f"calibrated acc {meta['mood']['calibrated']['accuracy']:.3f} F1 {meta['mood']['calibrated']['macro_f1']:.3f}")

    import shap
    sub = X_te.sample(n=min(800, len(X_te)), random_state=SEED)
    sv = shap.TreeExplainer(clf).shap_values(sub)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    imp = np.abs(sv).mean(axis=0)
    meta["mood"]["shap_global"] = {c: [{"name": X.columns[i], "value": round(float(imp[i, ci]), 5)}
                                       for i in np.argsort(-imp[:, ci])[:10]] for ci, c in enumerate(classes)}

    # Example tracks for the website: popular, correctly predicted test tracks per mood.
    test = df.loc[X_te.index].assign(pred=cal)
    if "popularity" in test:
        test = test.sort_values("popularity", ascending=False)
    presets = []
    for mood in classes:
        rows = test[(test.mood == mood) & (test.pred == mood)].head(3)
        for _, r in rows.iterrows():
            presets.append({"title": str(r.get("track_name", "")), "artist": str(r.get("artists", "")),
                            "genre": str(r.get("track_genre", "")), "mood": mood,
                            "valence": round(float(r.valence), 3), "energy": round(float(r.energy), 3),
                            "features": {f: round(float(r[f]), 4) for f in MOOD_BASE_FEATURES}})
    meta["mood"]["presets"] = presets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-training", action="store_true", help="never run train_cnn.py / train_crnn.py")
    ap.add_argument("--skip-mood", action="store_true")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    ensure_prerequisites(args.skip_training)

    meta = {"version": 1}
    export_genre(meta)
    if not args.skip_mood:
        export_mood(meta)

    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=1))
    public = {k: v for k, v in meta.items() if k != "feature_names"}
    (WEB_DATA / "metrics.json").write_text(json.dumps(public))
    step("Done")
    for f in sorted(MODELS_DIR.iterdir()):
        print(f"    models/{f.name:16s} {f.stat().st_size / 1e6:7.2f} MB")
    print("    web/public/data/metrics.json written. Next: python scripts/build_demo_samples.py")


if __name__ == "__main__":
    main()
