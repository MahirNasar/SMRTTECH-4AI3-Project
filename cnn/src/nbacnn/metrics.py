# file: src/nbacnn/metrics.py
import torch
from .views import make_views
from .losses import nt_xent, chance_loss

@torch.no_grad()
def two_halves(model, data, cfg, tau=0.1):
    """Embed each player's odd and even halves (shortcut-free). Does half A find half B?"""
    was_training = model.training
    model.eval()
    va, vb = make_views(data, cfg, train=False)
    ha, za = model(*va)
    hb, zb = model(*vb)
    sim = torch.nan_to_num(ha @ hb.T, nan=float("-inf"))    # (P, P) cosine similarity; NaN = worst
    rank = (sim >= sim.diag()[:, None]).sum(1) - 1          # players that beat OR tie the true partner
    # Ties count against the model: a collapsed (all-identical) or NaN model must score ~0, not 1.0.
    loss = nt_xent(za, zb, tau).item()
    model.train(was_training)
    return {"top1": (rank == 0).float().mean().item(),
            "top5": (rank < 5).float().mean().item(),
            "loss_norm": loss / chance_loss(len(data))}