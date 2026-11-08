import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

print("Loading and deduplicating Spotify Tracks dataset...")
df = pd.read_csv("data/raw/dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates(subset="track_id").dropna(subset=["valence", "energy"])

def derive_mood(r):
    if r.valence >= 0.5:
        return "happy" if r.energy >= 0.5 else "calm"
    return "tense" if r.energy >= 0.5 else "sad"

df["mood"] = df.apply(derive_mood, axis=1)

# Base 9 non-leaky features (strictly excluding valence and energy)
base_cols = [
    "danceability", "loudness", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "mode", "key"
]

# Domain-specific acoustic interaction features (0% valence/energy leakage)
df_feat = df[base_cols].copy()
df_feat["loudness_x_dance"] = df["loudness"] * df["danceability"]
df_feat["acoustic_x_inst"] = df["acousticness"] * df["instrumentalness"]
df_feat["acoustic_to_loudness"] = df["acousticness"] / (np.abs(df["loudness"]) + 1.0)
df_feat["dance_x_tempo"] = df["danceability"] * df["tempo"]
df_feat["speech_to_acoustic"] = df["speechiness"] / (df["acousticness"] + 0.01)

y = df["mood"]
classes = np.sort(y.unique())  # ['calm', 'happy', 'sad', 'tense']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("Running 5-Fold Stratified CV on Feature-Engineered RF...")
probs = cross_val_predict(clf, df_feat, y, cv=cv, method="predict_proba", n_jobs=1)

# 1. Uncalibrated predictions with interaction features
pred_raw = classes[np.argmax(probs, axis=1)]
print(f"\n--- Uncalibrated RF + Interactions | Acc: {accuracy_score(y, pred_raw):.3f} | Macro F1: {f1_score(y, pred_raw, average='macro'):.3f} ---")

# 2. Prior-Calibrated Threshold Tuning for Minority Class ('calm' is index 0)
# Boost minority 'calm' prior multiplier (1.45x) to recover calm tracks misclassified as sad/happy
cal_weights = np.array([1.45, 1.0, 1.0, 1.0])
cal_probs = probs * cal_weights
pred = classes[np.argmax(cal_probs, axis=1)]

acc = accuracy_score(y, pred)
prec = precision_score(y, pred, average="macro")
rec = recall_score(y, pred, average="macro")
macro_f1 = f1_score(y, pred, average="macro")

print(f"\n=== Upgraded Spotify Mood (Interactions + Prior Calibration) | Acc: {acc:.3f} ({acc:.1%}) | Macro Prec: {prec:.3f} | Macro Rec: {rec:.3f} | Macro F1: {macro_f1:.3f} ===")
print(classification_report(y, pred))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay.from_predictions(y, pred, xticks_rotation=45, cmap="viridis", ax=ax)
plt.title(f"Spotify Mood: Calibrated RF + Interactions (Acc: {acc:.1%}, Macro F1: {macro_f1:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_Mood_Final.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_Mood_Final.png")
