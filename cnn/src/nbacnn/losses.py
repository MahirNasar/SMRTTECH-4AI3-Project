# file: src/nbacnn/losses.py
import math, torch, torch.nn.functional as F

def nt_xent(z_a, z_b, tau=0.1):
    """SimCLR / InfoNCE loss. z_a[i] and z_b[i] are two views of player i, L2-normalized, shape (n, d)."""
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)                               # (2n, d)
    sim = (z @ z.T) / tau                                          # cosine similarity / temperature
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)                         # a view can't pick itself
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)                            # "which row is my partner?"

def chance_loss(n):
    """Loss when every candidate gets equal score: ln(2n - 1). Used to normalize."""
    return math.log(2 * n - 1)