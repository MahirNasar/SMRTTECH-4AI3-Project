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
# file: tests/test_model.py
import torch
from nbacnn.model import PlayerEncoder

def fake(B=4):
    mask = torch.zeros(B, 26); mask[:, 3:10] = 1
    grid = torch.randn(B, 23, 26) * mask[:, None, :]
    return grid, mask, torch.randn(B, 2, 24, 25)

def test_output_shapes_and_unit_length():
    h, z = PlayerEncoder()(*fake())
    assert h.shape == z.shape == (4, 32)
    assert torch.allclose(h.norm(dim=1), torch.ones(4)) and torch.allclose(z.norm(dim=1), torch.ones(4))

def test_parameter_count_is_small():
    n = sum(p.numel() for p in PlayerEncoder().parameters())
    assert 15_000 < n < 25_000                     # ~18,600 including the projection head

def test_unplayed_ages_do_not_matter():
    """Changing values at ages the player didn't play (outside the conv's reach) must not change h."""
    model = PlayerEncoder().eval()
    grid, mask, chart = fake(2)
    h1, _ = model(grid, mask, chart)
    grid2 = grid.clone(); grid2[:, :, 20:] = 99.0   # ages 38-43: far from the last played age
    h2, _ = model(grid2, mask, chart)
    assert torch.allclose(h1, h2, atol=1e-5)

def test_gradients_reach_both_branches():
    model = PlayerEncoder()
    h, z = model(*fake())
    z.sum().backward()
    assert model.career[0].weight.grad.abs().sum() > 0 and model.shots[0].weight.grad.abs().sum() > 0