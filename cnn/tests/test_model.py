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