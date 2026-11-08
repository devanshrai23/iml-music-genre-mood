"""Attention-CRNN (2D-CNN + BiGRU + temporal attention) on 3 s slices with song-level soft voting.

Default run (no flags) reproduces the original experiment exactly and additionally
logs every epoch to reports/logs/crnn_full_history.csv.

    python src/train_crnn.py                                # original protocol, best epoch picked on the test split
    python src/train_crnn.py --val-frac 0.1                 # clean protocol: pick best epoch on a validation split
    python src/train_crnn.py --variant no_attention         # ablations: no_attention | no_gru | no_specaugment
"""
import argparse
import csv
import glob
import json
import os
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

VARIANTS = ["full", "no_attention", "no_gru", "no_specaugment"]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", choices=VARIANTS, default="full", help="architecture / training ablation")
parser.add_argument("--val-frac", type=float, default=0.0,
                    help="fraction of TRAIN songs held out to select the best epoch (0 = original: select on test)")
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()

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
print(f"Using device: {device} | variant: {args.variant} | val-frac: {args.val_frac}")

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
# Optional validation split carved out of the TRAIN songs only (the test split is untouched).
if args.val_frac > 0:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=args.val_frac, stratify=train_labels, random_state=SEED
    )
    sel_name = "validation"
else:
    val_paths, val_labels = test_paths, test_labels
    sel_name = "test"
print(f"Track-level split -> Train: {len(train_paths)} songs ({len(train_paths)*10} 3s slices) | "
      f"Test: {len(test_paths)} songs | Epoch selection on: {sel_name} ({len(val_paths)} songs)")

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

train_ds = GTZANSliceDataset(train_paths, train_labels, augment=(args.variant != "no_specaugment"))
test_ds = GTZANSliceDataset(test_paths, test_labels, augment=False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
if args.val_frac > 0:
    val_loader = DataLoader(GTZANSliceDataset(val_paths, val_labels, augment=False), batch_size=64, shuffle=False)
else:
    val_loader = test_loader

# 4. CRNN Architecture (4-Block 2D CNN + BiGRU + Temporal Attention)
class AttentionCRNN(nn.Module):
    """variant="full" is the original model (same layers, same parameter names)."""

    def __init__(self, num_classes=10, variant="full"):
        super().__init__()
        self.variant = variant
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

        if variant == "no_gru":
            # Ablation: replace the recurrent layer by a per-time-step projection (no temporal context)
            self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        else:
            # Bidirectional GRU across the 16 temporal steps
            self.gru = nn.GRU(
                input_size=256,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.25
            )

        if variant != "no_attention":
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
        if self.variant == "no_gru":
            gru_out = self.proj(x)
        else:
            gru_out, _ = self.gru(x)   # (B, 16, 256)

        if self.variant == "no_attention":
            # Ablation: uniform weights = plain mean pooling over time
            alpha = torch.full(gru_out.shape[:2], 1.0 / gru_out.shape[1], device=gru_out.device)
        else:
            # Attention weights
            u = torch.tanh(self.attn_W(gru_out))  # (B, 16, 128)
            scores = self.attn_v(u).squeeze(-1)   # (B, 16)
            alpha = F.softmax(scores, dim=-1)     # (B, 16)

        context = torch.sum(gru_out * alpha.unsqueeze(-1), dim=1)  # (B, 256)
        logits = self.classifier(context)
        return logits, alpha

model = AttentionCRNN(num_classes=len(genres), variant=args.variant).to(device)
n_params = sum(p.numel() for p in model.parameters())
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


def evaluate(loader, n_songs):
    """Slice accuracy, song accuracy (soft voting), song predictions and mean slice CE loss (no smoothing)."""
    model.eval()
    seg_preds, seg_targets, losses = [], [], []
    song_prob_sums = np.zeros((n_songs, len(genres)), dtype=np.float32)
    with torch.no_grad():
        for xb, yb, song_idxs, _ in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            losses.append(F.cross_entropy(logits, yb.to(device), reduction="sum").item())
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            seg_preds.extend(np.argmax(probs, axis=1))
            seg_targets.extend(yb.numpy())
            for i, s_idx in enumerate(song_idxs.numpy()):
                song_prob_sums[s_idx] += probs[i]
    song_preds = np.argmax(song_prob_sums, axis=1)
    return accuracy_score(seg_targets, seg_preds), song_preds, sum(losses) / len(seg_targets)


# 5. Training Loop with Song-Level Soft Voting Evaluation
os.makedirs("reports/logs", exist_ok=True)
os.makedirs("reports/ablations", exist_ok=True)
ckpt = "reports/best_crnn.pt" if args.variant == "full" else f"reports/ablations/crnn_{args.variant}.pt"
tag = args.variant + ("_val" if args.val_frac > 0 else "")
history = []

best_song_acc = 0.0
best_song_preds = None
best_seg_acc = 0.0
best_epoch = 0

print(f"\nTraining CRNN [{args.variant}] ({n_params:,} parameters) for {args.epochs} epochs...")
for epoch in range(1, args.epochs + 1):
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

    # Evaluate on the selection split's slices & aggregate to parent song via soft voting
    seg_acc, song_preds, eval_loss = evaluate(val_loader, len(val_paths))
    song_acc = accuracy_score(val_labels, song_preds)

    if song_acc > best_song_acc or (song_acc == best_song_acc and seg_acc > best_seg_acc):
        best_song_acc = song_acc
        best_seg_acc = seg_acc
        best_song_preds = song_preds.copy()
        best_epoch = epoch
        torch.save(model.state_dict(), ckpt)

    history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "eval_loss": round(eval_loss, 5),
                    "eval_slice_acc": round(seg_acc, 5), "eval_song_acc": round(song_acc, 5), "eval_split": sel_name})
    print(f"Epoch {epoch:02d}/{args.epochs} | Train loss: {train_loss:.4f} | {sel_name} loss: {eval_loss:.4f} | "
          f"3s Slice Acc: {seg_acc:.3f} | Song-Level (Voted) Acc: {song_acc:.3f} (Best: {best_song_acc:.3f})")

with open(f"reports/logs/crnn_{tag}_history.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
    w.writeheader()
    w.writerows(history)

# 6. Final Report & Artifacts (the test split is evaluated with the selected checkpoint)
if args.val_frac > 0:
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_seg_acc, test_preds, _ = evaluate(test_loader, len(test_paths))
else:
    test_seg_acc, test_preds = best_seg_acc, best_song_preds
test_acc = accuracy_score(test_labels, test_preds)
macro_f1 = f1_score(test_labels, test_preds, average="macro")
summary = {"model": "crnn", "variant": args.variant, "selection_split": sel_name, "best_epoch": best_epoch,
           "epochs": args.epochs, "params": n_params, "test_song_acc": round(test_acc, 4),
           "test_slice_acc": round(test_seg_acc, 4), "test_macro_f1": round(macro_f1, 4)}
with open(f"reports/logs/crnn_{tag}_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)

print(f"\n=== FINAL CRNN [{args.variant}] | Test Song-Level Acc: {test_acc:.3f} ({test_acc:.1%}) | "
      f"Test Slice Acc: {test_seg_acc:.3f} | Macro F1: {macro_f1:.3f} | best epoch {best_epoch} (picked on {sel_name}) ===")
print(classification_report(test_labels, test_preds, target_names=genres))
print(f"Saved -> reports/logs/crnn_{tag}_history.csv, reports/logs/crnn_{tag}_summary.json")

if args.variant == "full":
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_preds, display_labels=genres, xticks_rotation=45, cmap="viridis", ax=ax
    )
    plt.title(f"GTZAN Genre Classification: Attention-CRNN (Acc: {test_acc:.1%}, Macro F1: {macro_f1:.2f})")
    plt.tight_layout()
    plt.savefig("reports/confusion_CRNN.png", dpi=150)
    plt.close()
    print("Saved -> reports/confusion_CRNN.png")
