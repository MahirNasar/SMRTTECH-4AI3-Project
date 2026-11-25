import torch
from nbacnn.data import load_split, WEIGHT_CH, POS_CH, MASK_CH
from nbacnn.views import make_views, full_view, season_split

DATA = "../data/processed"
CFG = {"split_mode": "odd_even", "season_drop": 0.2, "weight_noise": 0.5, "weight_drop": 0.5,
       "pos_drop": 0.5, "chart_mode": "drop_one", "chart_drop": 0.5, "chart_noise": 0.5}
batch = load_split("val", DATA)

def test_halves_are_disjoint_and_cover_career():
    a, b = season_split(batch.mask, "odd_even", 0.0, flip=False)
    assert (a * b).sum() == 0                          # no season in both views
    assert torch.equal(a + b, batch.mask)              # together they are the whole career
    assert (a.sum(1) >= 1).all() and (b.sum(1) >= 1).all()

def test_season_dropout_never_empties_a_view():
    for seed in range(5):
        a, b = season_split(batch.mask, "random", 0.9, gen=torch.Generator().manual_seed(seed))
        assert (a.sum(1) >= 1).all() and (b.sum(1) >= 1).all()

def test_hidden_seasons_are_zero_in_every_channel():
    va, vb = make_views(batch, CFG)
    for v in (va, vb):
        hidden = 1 - v.mask
        assert (v.grid * hidden[:, None, :]).abs().max() == 0
        assert torch.equal(v.grid[:, MASK_CH, :], v.mask)

def test_validation_view_b_is_shortcut_free():
    va, vb = make_views(batch, CFG, train=False)
    assert vb.grid[:, WEIGHT_CH].abs().max() == 0 and vb.grid[:, POS_CH].abs().max() == 0
    assert vb.chart.abs().max() == 0
    va2, vb2 = make_views(batch, CFG, train=False)     # validation views are deterministic
    assert torch.equal(va.grid, va2.grid) and torch.equal(vb.grid, vb2.grid)

def test_full_view_is_the_whole_qualified_career():
    v = full_view(batch)
    assert torch.equal(v.mask, batch.mask) and torch.equal(v.chart, batch.chart)
    # seasons under 500 minutes (mask 0) are dropped here too, so embedding sees what training saw
    assert torch.equal(v.grid, batch.grid * batch.mask[:, None, :])
def test_unknown_chart_mode_fails_loudly():
    import pytest
    with pytest.raises(ValueError):
        make_views(batch, {**CFG, "chart_mode": False})   # what an unquoted YAML `off` used to become

def test_chart_off_hides_chart_in_training_and_embedding():
    cfg = {**CFG, "chart_mode": "off"}
    va, vb = make_views(batch, cfg)
    assert va.chart.abs().max() == 0 and vb.chart.abs().max() == 0
    assert full_view(batch, cfg).chart.abs().max() == 0
