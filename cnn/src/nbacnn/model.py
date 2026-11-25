import torch, torch.nn as nn, torch.nn.functional as F

class PlayerEncoder(nn.Module):
    def __init__(self, in_ch=23, emb=32, dropout=0.2):
        super().__init__()
        self.career = nn.Sequential(                            # 1D: slides along age
            nn.Conv1d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
        self.shots = nn.Sequential(                             # 2D: slides across the court
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.combine = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, emb))
        self.head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, emb))   # training only

    def forward(self, grid, mask, chart):
        c = self.career(grid)                                   # (B, 64, 26)
        m = mask.unsqueeze(1)                                   # (B, 1, 26)
        c = (c * m).sum(-1) / m.sum(-1).clamp(min=1)            # mean over played ages only
        h = self.combine(torch.cat([c, self.shots(chart)], 1))  # (B, 32)  <- the player summary
        z = F.normalize(self.head(h), dim=1)                    # (B, 32)  <- for the loss only
        return F.normalize(h, dim=1), z