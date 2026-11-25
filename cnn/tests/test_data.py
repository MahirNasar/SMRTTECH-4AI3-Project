import pytest, torch
from nbacnn.data import load_split, MASK_CH

DATA = "../data/processed"

@pytest.fixture(scope="module")
def train():
    return load_split("train", DATA)

def test_shapes(train):
    P = len(train)
    assert P == 844
    assert train.grid.shape == (P, 23, 26) and train.mask.shape == (P, 26)
    assert train.chart.shape == (P, 2, 24, 25)
    assert train.grid.dtype == torch.float32 and train.chart.dtype == torch.float32   # MPS needs float32

def test_mask_channel_matches_mask(train):
    assert torch.equal(train.grid[:, MASK_CH, :], train.mask)

def test_every_player_has_two_seasons(train):
    assert (train.mask.sum(1) >= 2).all()

def test_unplayed_ages_are_zero(train):
    stats = train.grid[:, :21, :]                                  # z-scores + position
    assert (stats * (1 - train.mask[:, None, :])).abs().max() == 0

def test_shot_chart_standardized(train):
    m = train.chart.mean(dim=(0, 2, 3)); s = train.chart.std(dim=(0, 2, 3))
    assert m.abs().max() < 1e-3 and (s - 1).abs().max() < 1e-2

def test_splits_disjoint():
    ids = [set(load_split(s, DATA).ids) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(i) for i in ids) == 1207

def test_subset_keeps_rows_aligned(train):
    sub = train.subset(torch.tensor([5, 2]))
    assert sub.ids == [train.ids[5], train.ids[2]] and torch.equal(sub.grid[0], train.grid[5])