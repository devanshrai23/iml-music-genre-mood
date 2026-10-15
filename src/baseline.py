import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("data/processed/gtzan_librosa_features.csv")
X = df.drop(columns=["filename", "label"])
y = df["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
}

for name, model in models.items():
    pred = cross_val_predict(model, X, y, cv=cv)
    print(f"\n=== {name} | accuracy = {accuracy_score(y, pred):.3f} ===")
    print(classification_report(y, pred))
    ConfusionMatrixDisplay.from_predictions(y, pred, xticks_rotation=45)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"reports/confusion_{name}.png", dpi=150)
    plt.close()
