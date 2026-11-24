# file: src/nbacnn/fake.py
"""Fake players with the real tensor shapes, so the engine can be built before real data/views exist.
Each fake player has a hidden 'profile'; every season is profile + noise, so two halves of the same
player really are alike and a working training loop should learn to match them."""
import torch
from .data import PlayerTensors

def make_fake_split(n_players=256, seed=0, noise=1.0):
    g = torch.Generator().manual_seed(seed)
    profile = torch.randn(n_players, 15, 1, generator=g)                     # what makes a player "him"
    seasons = profile + noise * torch.randn(n_players, 15, 26, generator=g)  # each season = profile + noise
    start = torch.randint(0, 10, (n_players, 1), generator=g)
    length = torch.randint(2, 15, (n_players, 1), generator=g)
    ages = torch.arange(26)[None, :]
    mask = ((ages >= start) & (ages < start + length)).float()               # contiguous careers
    grid = torch.zeros(n_players, 23, 26)
    grid[:, :15] = seasons * mask[:, None, :]                                # 15 stat channels carry the profile
    pos = torch.randint(0, 5, (n_players,), generator=g)                     # weight (channel 15) left at 0
    grid[torch.arange(n_players), 16 + pos] = 1.0
    grid[:, 16:21] *= mask[:, None, :]                                       # one-hot position per season
    grid[:, 21] = 0.6 * mask                                                  # mp_scaled
    grid[:, 22] = mask                                                        # played_mask
    chart = torch.randn(n_players, 2, 24, 25, generator=g)
    return PlayerTensors([f"fake{i:04d}" for i in range(n_players)], grid, mask, chart)