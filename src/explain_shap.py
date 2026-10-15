import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading GTZAN librosa features...")
df = pd.read_csv("data/processed/gtzan_librosa_features.csv")
X = df.drop(columns=["filename", "label"])
y = df["label"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training Random Forest for SHAP explanation...")
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

print("Computing SHAP values via TreeExplainer...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_te)

# 1. Global Multi-Class Feature Importance Bar Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_te,
    plot_type="bar",
    class_names=list(clf.classes_),
    max_display=20,
    show=False,
)
plt.title("Global Feature Importance Across All 10 Genres (SHAP)")
plt.tight_layout()
plt.savefig("reports/shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> reports/shap_global_importance.png")

# 2. Beeswarm plot for specific contrasting genres (e.g., Classical vs Rock)
classes = list(clf.classes_)
for target_genre in ["classical", "rock", "metal", "hiphop"]:
    if target_genre in classes:
        idx = classes.index(target_genre)
        # Handle both list-of-arrays (older shap) and 3D array (shap >= 0.45)
        sv_genre = shap_values[:, :, idx] if isinstance(shap_values, np.ndarray) else shap_values[idx]
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv_genre, X_te, max_display=15, show=False)
        plt.title(f"SHAP Feature Impact for Genre: {target_genre.capitalize()}")
        plt.tight_layout()
        plt.savefig(f"reports/shap_beeswarm_{target_genre}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> reports/shap_beeswarm_{target_genre}.png")
