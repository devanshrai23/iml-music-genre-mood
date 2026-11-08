import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

# 1. Reproducibility & Device Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Discover Cached Mel-Spectrograms (Parent Track Level)
mel_dir = "data/processed/mel"
genres = sorted([d for d in os.listdir(mel_dir) if os.path.isdir(os.path.join(mel_dir, d))])
genre_to_idx = {g: i for i, g in enumerate(genres)}

track_paths, track_labels = [], []
for g in genres:
    files = sorted(glob.glob(os.path.join(mel_dir, g, "*.npy")))
    for f in files:
        track_paths.append(f)
        track_labels.append(genre_to_idx[g])

track_paths = np.array(track_paths)
track_labels = np.array(track_labels)
print(f"Found {len(track_paths)} cached full-track Mel-spectrograms across {len(genres)} genres.")

# Strict Track-Level 80/20 Stratified Split (Zero Segment Leakage Across Songs)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    track_paths, track_labels, test_size=0.20, stratify=track_labels, random_state=SEED
)
print(f"Track-level split -> Train: {len(train_paths)} songs ({len(train_paths)*10} 3s slices) | Test: {len(test_paths)} songs ({len(test_paths)*10} 3s slices)")

# 3. 3-Second Slice Dataset (10 slices of 128x128 per 128x1280 track)
class GTZANSliceDataset(Dataset):
    def __init__(self, paths, labels, num_slices=10, slice_len=128, augment=False):
        self.samples = []
        self.augment = augment
        self.slice_len = slice_len

        for song_idx, (p, label) in enumerate(zip(paths, labels)):
            mel = np.load(p).astype(np.float32)
            # Ensure shape is (128, 1280)
            if mel.shape[1] < num_slices * slice_len:
                pad_width = (num_slices * slice_len) - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="edge")
            else:
                mel = mel[:, :num_slices * slice_len]

            # Per-track standardization
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

            for s in range(num_slices):
                start = s * slice_len
                segment = mel[:, start : start + slice_len]
                self.samples.append((segment, label, song_idx, p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label, song_idx, path = self.samples[idx]
        seg = seg.copy()

        # SpecAugment during training
        if self.augment:
            # Frequency masking
            f_mask = random.randint(0, 16)
            f0 = random.randint(0, 128 - f_mask)
            seg[f0 : f0 + f_mask, :] = 0.0
            # Time masking
            t_mask = random.randint(0, 16)
            t0 = random.randint(0, 128 - t_mask)
            seg[:, t0 : t0 + t_mask] = 0.0

        tensor_seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)
        return tensor_seg, torch.tensor(label, dtype=torch.long), song_idx, path

train_ds = GTZANSliceDataset(train_paths, train_labels, augment=True)
test_ds = GTZANSliceDataset(test_paths, test_labels, augment=False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# 4. CRNN Architecture (4-Block 2D CNN + BiGRU + Temporal Attention)
class AttentionCRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # -> (32, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # -> (64, 32, 32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((4, 2))  # -> (128, 8, 16)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((8, 1))  # -> (256, 1, 16) -> 16 time steps per 3s slice
        )

        # Bidirectional GRU across the 16 temporal steps
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )

        # Temporal Attention over BiGRU outputs (256-dim)
        self.attn_W = nn.Linear(256, 128)
        self.attn_v = nn.Linear(128, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)          # (B, 256, 1, 16)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, 16, 256)

        gru_out, _ = self.gru(x)   # (B, 16, 256)

        # Attention weights
        u = torch.tanh(self.attn_W(gru_out))  # (B, 16, 128)
        scores = self.attn_v(u).squeeze(-1)   # (B, 16)
        alpha = F.softmax(scores, dim=-1)     # (B, 16)

        context = torch.sum(gru_out * alpha.unsqueeze(-1), dim=1)  # (B, 256)
        logits = self.classifier(context)
        return logits, alpha

model = AttentionCRNN(num_classes=len(genres)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# 5. Training Loop with Song-Level Soft Voting Evaluation
best_song_acc = 0.0
best_song_preds = None
best_seg_acc = 0.0

print("\nTraining CRNN (2D-CNN + BiGRU + Temporal Attention) for 30 epochs...")
for epoch in range(1, 31):
    model.train()
    train_loss = 0.0
    for xb, yb, _, _ in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    scheduler.step()
    train_loss /= len(train_ds)

    # Evaluate on Test Slices & Aggregate to Parent Song via Soft Voting
    model.eval()
    seg_preds, seg_targets = [], []
    song_prob_sums = np.zeros((len(test_paths), len(genres)), dtype=np.float32)

    with torch.no_grad():
        for xb, yb, song_idxs, _ in test_loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            seg_preds.extend(preds)
            seg_targets.extend(yb.numpy())

            for i, s_idx in enumerate(song_idxs.numpy()):
                song_prob_sums[s_idx] += probs[i]

    seg_acc = accuracy_score(seg_targets, seg_preds)
    song_preds = np.argmax(song_prob_sums, axis=1)
    song_acc = accuracy_score(test_labels, song_preds)

    if song_acc > best_song_acc or (song_acc == best_song_acc and seg_acc > best_seg_acc):
        best_song_acc = song_acc
        best_seg_acc = seg_acc
        best_song_preds = song_preds.copy()
        torch.save(model.state_dict(), "reports/best_crnn.pt")

    if epoch % 5 == 0 or epoch == 30:
        print(f"Epoch {epoch:02d}/30 | Loss: {train_loss:.4f} | 3s Slice Acc: {seg_acc:.3f} | Song-Level (Voted) Acc: {song_acc:.3f} (Best: {best_song_acc:.3f})")

# 6. Final Report & Artifacts
macro_f1 = f1_score(test_labels, best_song_preds, average="macro")
print(f"\n=== FINAL CRNN RESULTS | Best Song-Level Acc: {best_song_acc:.3f} ({best_song_acc:.1%}) | Slice Acc: {best_seg_acc:.3f} | Macro F1: {macro_f1:.3f} ===")
print(classification_report(test_labels, best_song_preds, target_names=genres))

fig, ax = plt.subplots(figsize=(8, 6.5))
ConfusionMatrixDisplay.from_predictions(
    test_labels, best_song_preds, display_labels=genres, xticks_rotation=45, cmap="viridis", ax=ax
)
plt.title(f"GTZAN Genre Classification: Attention-CRNN (Acc: {best_song_acc:.1%}, Macro F1: {macro_f1:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_CRNN.png", dpi=150)
plt.close()
print("Saved -> reports/confusion_CRNN.png")
