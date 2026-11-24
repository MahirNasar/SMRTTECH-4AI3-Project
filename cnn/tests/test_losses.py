# file: tests/test_losses.py
import math, torch, torch.nn.functional as F
from nbacnn.losses import nt_xent, chance_loss

def unit(x): return F.normalize(x, dim=1)

def test_perfect_match_is_near_zero():
    z = unit(torch.randn(64, 32))
    assert nt_xent(z, z) < 0.1                          # each view's partner is identical

def test_uniform_scores_give_chance_loss():
    z = unit(torch.ones(64, 32))                         # everything identical -> no information
    assert abs(nt_xent(z, z) - chance_loss(64)) < 1e-4

def test_symmetric():
    a, b = unit(torch.randn(16, 32)), unit(torch.randn(16, 32))
    assert torch.isclose(nt_xent(a, b), nt_xent(b, a), atol=1e-5)

def test_gradient_flows():
    a = unit(torch.randn(8, 32)).requires_grad_()
    nt_xent(a, unit(torch.randn(8, 32))).backward()
    assert a.grad is not None and a.grad.abs().sum() > 0