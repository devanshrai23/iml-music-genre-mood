"""Neural network definitions.

Attribute names are identical to src/train_cnn.py and src/train_crnn.py so the
saved state_dicts load without any key remapping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioAttentionCNN(nn.Module):
    """2D CNN + temporal attention over the full 30 s spectrogram (src/train_cnn.py)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 40)),
        )
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        feat = self.features(x).squeeze(2)            # (B, 256, 40)
        feat_t = feat.permute(0, 2, 1)                # (B, 40, 256)
        attn_weights = torch.softmax(self.attention(feat_t), dim=1)
        context = torch.sum(feat_t * attn_weights, dim=1)
        return self.classifier(context), attn_weights.squeeze(-1)


class AttentionCRNN(nn.Module):
    """2D CNN + BiGRU + temporal attention on 3 s slices (src/train_crnn.py)."""

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
        x = x.squeeze(2).permute(0, 2, 1)             # (B, 16, 256)
        gru_out, _ = self.gru(x)
        u = torch.tanh(self.attn_W(gru_out))
        alpha = F.softmax(self.attn_v(u).squeeze(-1), dim=-1)
        context = torch.sum(gru_out * alpha.unsqueeze(-1), dim=1)
        return self.classifier(context), alpha


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
