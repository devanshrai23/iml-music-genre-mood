import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 1. Load Handcrafted Features (CSV) & Align with Cached Mel-Spectrograms
df = pd.read_csv("data/processed/gtzan_librosa_features.csv")
mel_dir = "data/processed/mel"
genres = sorted([d for d in os.listdir(mel_dir) if os.path.isdir(os.path.join(mel_dir, d))])
genre_to_idx = {g: i for i, g in enumerate(genres)}

track_paths, track_labels, track_filenames = [], [], []
for g in genres:
    files = sorted(glob.glob(os.path.join(mel_dir, g, "*.npy")))
    for f in files:
        track_paths.append(f)
        track_labels.append(genre_to_idx[g])
        fname = os.path.basename(f).replace(".npy", ".wav")
        track_filenames.append(fname)

track_paths = np.array(track_paths)
track_labels = np.array(track_labels)
track_filenames = np.array(track_filenames)

# Exact same 80/20 track-level split used in train_crnn.py
train_idx, test_idx = train_test_split(
    np.arange(len(track_paths)), test_size=0.20, stratify=track_labels, random_state=SEED
)

# Align tabular features by filename
df_indexed = df.set_index("filename")
feature_cols = [c for c in df.columns if c not in ("filename", "label")]

X_train = df_indexed.loc[track_filenames[train_idx], feature_cols]
y_train = track_labels[train_idx]
X_test = df_indexed.loc[track_filenames[test_idx], feature_cols]
y_test = track_labels[test_idx]
test_paths = track_paths[test_idx]
test_fnames = track_filenames[test_idx]

# 2. Train Random Forest & TreeSHAP on the Exact Same Split
print("Fitting Random Forest & TreeSHAP on aligned 80/20 track split...")
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 3. Load Trained Attention-CRNN
class AttentionCRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((4, 2)))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((8, 1)))
        self.gru = nn.GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.attn_W = nn.Linear(256, 128)
        self.attn_v = nn.Linear(128, 1, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.squeeze(2).permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        u = torch.tanh(self.attn_W(gru_out))
        alpha = F.softmax(self.attn_v(u).squeeze(-1), dim=-1)
        context = torch.sum(gru_out * alpha.unsqueeze(-1), dim=1)
        return self.classifier(context), alpha

model = AttentionCRNN(num_classes=len(genres)).to(device)
model.load_state_dict(torch.load("reports/best_crnn.pt", map_location=device))
model.eval()

def evaluate_track_crnn(mel_path):
    mel = np.load(mel_path).astype(np.float32)[:, :1280]
    if mel.shape[1] < 1280:
        mel = np.pad(mel, ((0, 0), (0, 1280 - mel.shape[1])), mode="edge")
    mel_norm = (mel - mel.mean()) / (mel.std() + 1e-6)

    slices = [mel_norm[:, s*128:(s+1)*128] for s in range(10)]
    batch = torch.tensor(np.array(slices), dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        logits, alphas = model(batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        alphas = alphas.cpu().numpy()  # (10, 16) -> 160 temporal steps across 30s
    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    full_alpha = alphas.flatten()  # 160 steps across 30 seconds
    return pred_idx, mean_probs[pred_idx], mel, full_alpha

crnn_preds, crnn_confs, mel_list, alpha_list = [], [], [], []
for p in test_paths:
    p_idx, conf, mel, full_a = evaluate_track_crnn(p)
    crnn_preds.append(p_idx)
    crnn_confs.append(conf)
    mel_list.append(mel)
    alpha_list.append(full_a)
crnn_preds = np.array(crnn_preds)

# Select Case 1 (Both Correct with high confidence) & Case 2 (RF Wrong, CRNN Correct on Rock/Country/Reggae)
both_right = np.where((rf_preds == y_test) & (crnn_preds == y_test))[0]
crnn_rescued = np.where((rf_preds != y_test) & (crnn_preds == y_test))[0]

# Prefer rock/reggae/country for Case 2
preferred = [i for i in crnn_rescued if genres[y_test[i]] in ("rock", "reggae", "country", "disco")]
idx_agree = both_right[0]
idx_rescue = preferred[0] if len(preferred) > 0 else crnn_rescued[0]

print(f"Case 1 (Agreement): {test_fnames[idx_agree]} | True: {genres[y_test[idx_agree]]} | RF: {genres[rf_preds[idx_agree]]} | CRNN: {genres[crnn_preds[idx_agree]]}")
print(f"Case 2 (CRNN Rescue): {test_fnames[idx_rescue]} | True: {genres[y_test[idx_rescue]]} | RF Pred: {genres[rf_preds[idx_rescue]]} (Wrong) | CRNN Pred: {genres[crnn_preds[idx_rescue]]} (Correct)")

# 4. Plot 2x2 Side-by-Side Comparison Figure (Local SHAP vs. Temporal Attention)
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.0, 1.35]})

for row, idx, tag in [(0, idx_agree, "Case 1 (Both Correct)"), (1, idx_rescue, "Case 2 (RF Misclassified, CRNN Correct)")]:
    true_g = genres[y_test[idx]]
    rf_g = genres[rf_preds[idx]]
    crnn_g = genres[crnn_preds[idx]]
    fname = test_fnames[idx]

    # Extract local SHAP values for RF's predicted class
    target_cls = rf_preds[idx]
    if isinstance(shap_values, list):
        sv = shap_values[target_cls][idx]
    else:
        sv = shap_values[idx, :, target_cls]

    top_k = np.argsort(np.abs(sv))[-8:]
    ax_shap = axes[row, 0]
    colors = ["#d95f02" if v > 0 else "#2b5c8f" for v in sv[top_k]]
    ax_shap.barh([feature_cols[j] for j in top_k], sv[top_k], color=colors)
    ax_shap.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_shap.set_title(f"{tag}: {fname}\nLocal TreeSHAP (RF Pred: '{rf_g}', True: '{true_g}')", fontsize=10, fontweight="bold")
    ax_shap.set_xlabel("SHAP Contribution to RF Prediction", fontsize=9)
    ax_shap.tick_params(axis="both", labelsize=8)

    # Plot Mel-spectrogram with overlaid CRNN Temporal Attention curve
    ax_mel = axes[row, 1]
    mel_data = mel_list[idx]
    alpha_curve = alpha_list[idx]
    im = ax_mel.imshow(mel_data, aspect="auto", origin="lower", cmap="magma", extent=[0, 30, 0, 128], alpha=0.88)

    ax_attn = ax_mel.twinx()
    t_axis = np.linspace(0, 30, len(alpha_curve))
    ax_attn.plot(t_axis, alpha_curve, color="#00ffcc", linewidth=1.8, label="CRNN Attention $\\alpha_t$")
    ax_attn.fill_between(t_axis, 0, alpha_curve, color="#00ffcc", alpha=0.22)
    ax_attn.set_ylabel("Attention Weight $\\alpha_t$", color="#008066", fontsize=9)
    ax_attn.tick_params(axis="y", labelsize=8)
    ax_attn.set_ylim(0, max(alpha_curve) * 1.35)

    ax_mel.set_title(f"CRNN Mel-Spectrogram & Temporal Attention (CRNN Pred: '{crnn_g}' [{crnn_confs[idx]:.0%}])", fontsize=10, fontweight="bold")
    ax_mel.set_xlabel("Time (seconds across 10 x 3s slices)", fontsize=9)
    ax_mel.set_ylabel("Mel Frequency Bin", fontsize=9)

plt.tight_layout()
plt.savefig("reports/shap_vs_attention_comparison.png", dpi=150)
plt.close()
print("Saved -> reports/shap_vs_attention_comparison.png")
