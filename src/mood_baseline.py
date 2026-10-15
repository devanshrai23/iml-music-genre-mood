import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

print("Loading and cleaning Spotify Tracks dataset...")
df = pd.read_csv("data/raw/dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
raw_count = len(df)
df = df.drop_duplicates(subset="track_id").dropna(subset=["valence", "energy"])
print(f"Deduplicated tracks by track_id: {raw_count} -> {len(df)} unique tracks")

def derive_mood(r):
    if r.valence >= 0.5:
        return "happy" if r.energy >= 0.5 else "calm"
    return "tense" if r.energy >= 0.5 else "sad"

df["mood"] = df.apply(derive_mood, axis=1)

feature_cols = [
    "danceability", "loudness", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "mode", "key"
]
X = df[feature_cols]
y = df["mood"]

print("\nRunning 5-Fold Cross-Validation on Random Forest (Mood)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=1)

acc = accuracy_score(y, pred)
macro_f1 = f1_score(y, pred, average="macro")
print(f"\n=== Spotify Mood RandomForest | Accuracy = {acc:.3f} | Macro F1 = {macro_f1:.3f} ===")
print(classification_report(y, pred))

ConfusionMatrixDisplay.from_predictions(y, pred, xticks_rotation=45)
plt.title(f"Spotify Mood Classification (RF | Acc: {acc:.1%}, Macro F1: {macro_f1:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_Mood_RF.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_Mood_RF.png")

print("Computing SHAP feature importance for Mood prediction...")
X_tr, X_sub, y_tr, y_sub = train_test_split(X, y, test_size=2000, stratify=y, random_state=42)
clf.fit(X_tr, y_tr)
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_sub)

plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_vals,
    X_sub,
    plot_type="bar",
    class_names=list(clf.classes_),
    show=False,
)
plt.title("SHAP Feature Importance for Spotify Mood (Excluding Valence & Energy)")
plt.tight_layout()
plt.savefig("reports/shap_mood_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> reports/shap_mood_importance.png")
