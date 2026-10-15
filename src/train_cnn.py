import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

# 1. Reproducibility & Apple Silicon GPU (MPS) Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using compute device: {device}")

MEL_DIR = Path("data/processed/mel")
FIXED_FRAMES = 1280  # ~29.7 seconds at sr=22050, hop=512

# 2. Load file paths and labels
files = sorted(MEL_DIR.rglob("*.npy"))
genres = sorted({p.parent.name for p in files})
genre_to_idx = {g: i for i, g in enumerate(genres)}
labels = [genre_to_idx[p.parent.name] for p in files]

print(f"Found {len(files)} cached Mel-spectrograms across {len(genres)} genres.")

train_files, test_files, y_tr, y_te = train_test_split(
    files, labels, test_size=0.2, stratify=labels, random_state=SEED
)

class MelDataset(Dataset):
    def __init__(self, file_list, label_list, augment=False):
        self.files = file_list
        self.labels = label_list
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx]).astype(np.float32)
        # Standardize per-clip
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        # Pad or crop time dimension to FIXED_FRAMES
        if mel.shape[1] < FIXED_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, FIXED_FRAMES - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :FIXED_FRAMES]

        if self.augment:
            # Simple SpecAugment: random time and frequency masking
            t0 = random.randint(0, FIXED_FRAMES - 100)
            mel[:, t0:t0 + random.randint(20, 80)] = 0.0
            f0 = random.randint(0, 100)
            mel[f0:f0 + random.randint(5, 16), :] = 0.0

        return torch.tensor(mel).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

train_loader = DataLoader(MelDataset(train_files, y_tr, augment=True), batch_size=32, shuffle=True)
test_loader = DataLoader(MelDataset(test_files, y_te, augment=False), batch_size=32, shuffle=False)

# 3. CNN with Temporal Attention Architecture
class AudioAttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),  # -> (32, 64, 320)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),  # -> (64, 32, 80)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (128, 16, 40)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 40))  # Collapse freq -> (256, 1, 40 time steps)
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x).squeeze(2)        # (B, 256, T=40)
        feat_t = feat.permute(0, 2, 1)            # (B, T=40, 256)
        attn_scores = self.attention(feat_t)      # (B, T=40, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(feat_t * attn_weights, dim=1)  # (B, 256)
        out = self.classifier(context)
        return out, attn_weights.squeeze(-1)

model = AudioAttentionCNN(num_classes=len(genres)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# 4. Training Loop (25 Epochs -> ~30-45 seconds on M5 MPS)
EPOCHS = 25
best_acc = 0.0
best_preds = []

print(f"Training CNN with Temporal Attention for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    if acc >= best_acc:
        best_acc = acc
        best_preds = preds
        torch.save(model.state_dict(), "reports/best_cnn_model.pt")
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Test Accuracy: {acc:.3f} (Best: {best_acc:.3f})")

# 5. Final Evaluation & Plots
macro_f1 = f1_score(y_te, best_preds, average="macro")
print(f"\n=== Preliminary Attention-CNN | Best Test Accuracy = {best_acc:.3f} | Macro F1 = {macro_f1:.3f} ===")
print(classification_report(y_te, best_preds, target_names=genres))

ConfusionMatrixDisplay.from_predictions(y_te, best_preds, display_labels=genres, xticks_rotation=45)
plt.title(f"Attention-CNN (Acc: {best_acc:.1%}, Macro F1: {macro_f1:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_CNN.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_CNN.png")

# 6. Save Sample Mel-Spectrogram + Attention Weights Plot
model.load_state_dict(torch.load("reports/best_cnn_model.pt", map_location=device))
model.eval()
sample_x, sample_y = MelDataset(test_files, y_te, augment=False)[0]
with torch.no_grad():
    logits, attn = model(sample_x.unsqueeze(0).to(device))
    pred_genre = genres[logits.argmax(dim=1).item()]
    true_genre = genres[sample_y.item()]
    attn_np = attn.cpu().numpy()[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=False)
ax1.imshow(sample_x.squeeze(0).numpy(), aspect="auto", origin="lower", cmap="magma")
ax1.set_title(f"Mel-Spectrogram ({test_files[0].name}) | True: {true_genre} | Predicted: {pred_genre}")
ax1.set_ylabel("Mel Frequency Bin")

time_axis = np.linspace(0, 30, len(attn_np))
ax2.plot(time_axis, attn_np, color="crimson", linewidth=2)
ax2.fill_between(time_axis, attn_np, color="crimson", alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Attention Weight")
ax2.set_title("Learned Temporal Attention Weights (Which part of the clip drove the prediction)")
plt.tight_layout()
plt.savefig("reports/cnn_attention_example.png", dpi=150)
plt.close()
print("Saved -> reports/cnn_attention_example.png")
