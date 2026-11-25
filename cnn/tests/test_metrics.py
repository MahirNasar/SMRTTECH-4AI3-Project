import torch, torch.nn.functional as F
from nbacnn.fake import make_fake_split
from nbacnn.metrics import two_halves

class Const(torch.nn.Module):
    """Returns the same embedding for every input (a collapsed model), or NaN."""
    def __init__(self, value): super().__init__(); self.value = value
    def forward(self, grid, mask, chart):
        z = torch.full((grid.shape[0], 32), self.value)
        return F.normalize(z, dim=1) if self.value == self.value else z, z

def test_collapsed_or_nan_model_scores_zero_not_one():
    data = make_fake_split(50)
    for model in (Const(1.0), Const(float("nan"))):
        m = two_halves(model, data, {})
        assert m["top1"] == 0.0 and m["top5"] == 0.0     # random would be 0.02 / 0.1

class Oracle(torch.nn.Module):
    """Embeds each player by his index, hidden in stat channel 0 (which both validation halves keep)."""
    def forward(self, grid, mask, chart):
        h = F.one_hot(grid[:, 0].max(1).values.round().long(), 64).float()
        return h, h

def test_perfect_model_scores_one():
    data = make_fake_split(50)
    data.grid[:, 0] = torch.arange(50).float()[:, None] * data.mask
    m = two_halves(Oracle(), data, {})
    assert m["top1"] == 1.0 and m["top5"] == 1.0
