import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

print("Loading Spotify dataset...")
df = pd.read_csv("data/raw/dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates(subset="track_id").dropna(subset=["valence", "energy"])

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. HistGradBoost Confusion Matrix
print("Generating HistGradBoost Confusion Matrix...")
hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=42)
pred_hgb = cross_val_predict(hgb, X, y, cv=cv, n_jobs=-1)
acc_hgb = accuracy_score(y, pred_hgb)
f1_hgb = f1_score(y, pred_hgb, average="macro")

ConfusionMatrixDisplay.from_predictions(y, pred_hgb, xticks_rotation=45)
plt.title(f"Spotify Mood: HistGradBoost Balanced (Acc: {acc_hgb:.1%}, F1: {f1_hgb:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_Mood_HistGB.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_Mood_HistGB.png")

# 2. Deep MLP Confusion Matrix
print("Generating Deep MLP Confusion Matrix...")
mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, early_stopping=True, random_state=42)
)
pred_mlp = cross_val_predict(mlp, X, y, cv=cv, n_jobs=-1)
acc_mlp = accuracy_score(y, pred_mlp)
f1_mlp = f1_score(y, pred_mlp, average="macro")

ConfusionMatrixDisplay.from_predictions(y, pred_mlp, xticks_rotation=45)
plt.title(f"Spotify Mood: Deep MLP (Acc: {acc_mlp:.1%}, F1: {f1_mlp:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_Mood_MLP.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_Mood_MLP.png")

# 3. Side-by-Side Comparison Bar Chart (RF vs HistGB vs Deep MLP)
print("Generating 3-Model Comparison Bar Chart...")
metrics = ["Overall Accuracy", "Macro Recall", "Macro F1", "Minority ('calm') Recall"]
rf_scores  = [0.705, 0.648, 0.660, 0.360]
hgb_scores = [0.660, 0.664, 0.636, 0.660]
mlp_scores = [0.665, 0.600, 0.608, 0.270]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - width, rf_scores, width, label="Random Forest", color="#2b5c8f")
b2 = ax.bar(x, hgb_scores, width, label="HistGradBoost (Balanced)", color="#d95f02")
b3 = ax.bar(x + width, mlp_scores, width, label="Deep MLP (128-64)", color="#7570b3")

ax.set_ylabel("Score")
ax.set_title("Spotify Mood Classification: Classical Ensembles vs. Deep MLP")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.85)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("reports/mood_model_comparison.png", dpi=150)
plt.close()
print("Saved -> reports/mood_model_comparison.png")
