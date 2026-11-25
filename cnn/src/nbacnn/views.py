"""Turn a batch of players into two partial views (a positive pair) for contrastive training."""
from collections import namedtuple
import torch
from .data import WEIGHT_CH, POS_CH, MASK_CH

View = namedtuple("View", "grid mask chart")   # (B,23,26), (B,26), (B,2,24,25)
CHART_MODES = ("drop_one", "same", "off")

def _rand(shape, gen, device):
    return torch.rand(shape, generator=gen, device=device) if gen is not None else torch.rand(shape, device=device)

def _randn(shape, gen, device):
    return torch.randn(shape, generator=gen, device=device) if gen is not None else torch.randn(shape, device=device)

def season_split(mask, mode="odd_even", drop_p=0.0, gen=None, flip=True):
    """mask (B,26) of played seasons -> (mask_a, mask_b) with no season in both."""
    order = torch.cumsum(mask, dim=1) * mask                 # 1,2,3,... on played ages, 0 elsewhere
    odd = mask * (order % 2 == 1)
    even = mask * ((order % 2 == 0) & (order > 0))
    if mode == "odd_even":
        a, b = odd, even
    elif mode == "random":
        coin = _rand(mask.shape, gen, mask.device) < 0.5
        a, b = mask * coin, mask * ~coin
        empty = ((a.sum(1) == 0) | (b.sum(1) == 0))[:, None]  # a side got nothing: use odd/even
        a, b = torch.where(empty, odd, a), torch.where(empty, even, b)
    else:
        raise ValueError(f"unknown split_mode {mode}")
    if flip:                                                  # which half is "A" shouldn't matter
        f = (_rand((mask.shape[0], 1), gen, mask.device) < 0.5)
        a, b = torch.where(f, b, a), torch.where(f, a, b)
    if drop_p > 0:
        a, b = _season_dropout(a, drop_p, gen), _season_dropout(b, drop_p, gen)
    return a.float(), b.float()

def _season_dropout(m, p, gen):
    keep = m * (_rand(m.shape, gen, m.device) >= p)
    return torch.where(keep.sum(1, keepdim=True) == 0, m, keep)   # never leave a view empty

def apply_view(grid, chart, view_mask, cfg, gen=None, train=True):
    """Hide seasons outside the view, then apply the shortcut controls (training only)."""
    g = grid * view_mask[:, None, :]                          # zero every channel at hidden ages
    g[:, MASK_CH, :] = view_mask                              # played flag = seasons in this view
    c = chart
    mode = cfg.get("chart_mode", "drop_one")
    if mode not in CHART_MODES:                               # e.g. False from an unquoted YAML `off`
        raise ValueError(f"unknown chart_mode {mode!r}; expected one of {CHART_MODES}")
    if mode == "off":                                         # ablation: no shot chart at all
        c = torch.zeros_like(chart)
    if train:
        B, dev = grid.shape[0], grid.device
        if cfg.get("weight_noise", 0) > 0:                    # weight barely changes within a player
            g[:, WEIGHT_CH, :] += _randn((B, 1), gen, dev) * cfg["weight_noise"] * view_mask
        if cfg.get("weight_drop", 0) > 0:                     # ...so also hide it in some views
            hide = (_rand((B, 1), gen, dev) < cfg["weight_drop"]).float()
            g[:, WEIGHT_CH, :] = g[:, WEIGHT_CH, :] * (1 - hide)
        if cfg.get("pos_drop", 0) > 0:                        # position barely changes either
            hide = (_rand((B, 1, 1), gen, dev) < cfg["pos_drop"]).float()
            g[:, POS_CH, :] = g[:, POS_CH, :] * (1 - hide)
        if mode == "drop_one":                                # career chart is identical in both views
            hide = (_rand((B, 1, 1, 1), gen, dev) < cfg.get("chart_drop", 0.5)).float()
            c = c * (1 - hide)
        if mode != "off" and cfg.get("chart_noise", 0) > 0:
            c = c + _randn(c.shape, gen, dev) * cfg["chart_noise"]
    return g, c

def _view(batch, m, cfg, gen, train):
    g, c = apply_view(batch.grid, batch.chart, m, cfg, gen, train)
    return View(g, m, c)

def make_views(batch, cfg, gen=None, train=True):
    """batch: PlayerTensors -> (View A, View B).
    train=True : random augmented halves (for the loss).
    train=False: fixed, shortcut-free halves for validation. A = odd seasons with everything;
                 B = even seasons with weight, position and shot chart hidden, so B can only be
                 matched to A through the season-by-season stats."""
    if train:
        ma, mb = season_split(batch.mask, cfg.get("split_mode", "odd_even"), cfg.get("season_drop", 0.0), gen)
        return _view(batch, ma, cfg, gen, True), _view(batch, mb, cfg, gen, True)
    ma, mb = season_split(batch.mask, "odd_even", 0.0, gen, flip=False)
    a = _view(batch, ma, cfg, None, False)
    gb, cb = apply_view(batch.grid, batch.chart, mb, {"chart_mode": "off"}, train=False)
    gb[:, WEIGHT_CH, :] = 0
    gb[:, POS_CH, :] = 0
    return a, View(gb, mb, cb)

def full_view(batch, cfg=None):
    """Whole career in one view, no augmentation (the guide's original embedding input).
    Note: odd/even training never shows two adjacent seasons, but full careers mostly have them,
    so this is out of distribution for an odd_even-trained model. embed.py uses career_halves."""
    return _view(batch, batch.mask, cfg or {}, None, train=False)

def career_halves(batch, cfg=None):
    """The whole career as its two fixed halves (odd / even qualified seasons), each with full
    information (weight, position, shot chart) and no augmentation. Each half looks like the inputs
    the model trained on, and together they cover every season. Used for embedding."""
    ma, mb = season_split(batch.mask, "odd_even", 0.0, flip=False)
    return _view(batch, ma, cfg or {}, None, train=False), _view(batch, mb, cfg or {}, None, train=False)