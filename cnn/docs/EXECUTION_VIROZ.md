# Execution Guide: Viroz

**Your role:** data, model and evaluation. You turn the cleaned CSVs into tensors, design the two "views" the model learns from (including the shortcut controls), build the encoder, then turn trained models into player comparisons and score them.
**Your files:** `data.py`, `views.py`, `model.py`, `embed.py`, `evaluate.py`, `diagnostics.py`, their tests and scripts, and the `model:` / `views:` blocks of the config.
**Read first:** [`../ROADMAP.md`](../ROADMAP.md), especially the interface contract (§4), the data issues (§2) and the prototype results (§9).

How to use this guide:
- Each task (`VI-1` … `VI-7`) has **Goal → Steps → Code → Done when → 💾 Commit points → 🧠 Theory → 🔍 Research keywords**.
- 💾 marks a good moment to `git add`, `git commit` and `git push`. Commit at each one, even if the task isn't finished.
- All code in this guide (except the VI-7 stretch sketch) was run on the real dataset in a prototype, and the tests pass. Expected numbers are given so you know what "working" looks like.
- Paths are relative to the `cnn/` folder. Mahir's guide (M0) covers the environment; follow it on your laptop (any machine is fine for your tasks).

---

## Task overview

| Task | Roadmap milestone | What you build | Week |
|---|---|---|---|
| M0 | M0 | Clone, environment, kickoff call | 1 |
| VI-1 | M1 | CSVs → tensors (`load_split`) | 1 |
| VI-2 | M2 | Two views per player + shortcut controls | 2 |
| VI-3 | M3 | Encoder: two-branch CNN + projection head | 3 |
| Integration | M4 | First real run, jointly with Mahir | 3 |
| VI-4 | M5 | Embeddings + top-5 neighbours for every player | 4 |
| VI-5 | M7 | BPM hidden-clue evaluator + baselines | 4 |
| VI-6 | M7 | Diagnostics: probes, UMAP, neighbour tables | 5 |
| VI-7 | stretch | Per-season shot charts | 4–5, if time |
| Test evaluation | M9 | Jointly with Mahir | 5 |

---

## M0: Setup

1. Clone the repo and follow Mahir's guide, section M0 steps 2–3 (`conda env create -f environment.yml`, `pip install -e .`).
2. **Kickoff call:** go through the interface contract (ROADMAP §4) together and commit it. You own `model:` and `views:` in `configs/default.yaml`.
3. On day 1, push a **first draft** of `model.py` (VI-3's code works as-is) so Mahir's engine has something to import while you build VI-1 and VI-2 properly.

💾 `model: add first draft of PlayerEncoder for integration`

---

## VI-1: Data → tensors

### Goal
`load_split("train")` returns every player as tensors in the exact shapes the contract promises, ready for the GPU.

### What the CSVs look like, and what the model needs

| File | Rows | Becomes |
|---|---|---|
| `cnn_stats_{split}.csv` | one row per player per age (26 per player) | `grid` (P, 23, 26) and `mask` (P, 26) |
| `cnn_shots_{split}.csv` | one row per player per court cell (600 per player) | `chart` (P, 2, 24, 25) |
| `cnn_shots_train_scaler.csv` | mean/std of the two shot channels on **train** | standardizes all three splits |

The 23 channels in order: 16 z-scored stats (`pts_per_100_poss_z` … `weight_z`), 5 position one-hots, `mp_scaled`, `played_mask`.

### Code

```python
# file: src/nbacnn/data.py
"""Load the cleaned CSVs into tensors. One row per player, channels-first."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, torch

STAT_COLS = ["pts_per_100_poss_z", "usg_percent_z", "ts_percent_z", "x3p_ar_z", "f_tr_z",
             "x2p_pct_adj_z", "ft_pct_adj_z", "x3p_pct_adj_z", "ast_percent_z", "tov_percent_z",
             "orb_percent_z", "drb_percent_z", "stl_percent_z", "blk_percent_z",
             "pf_per_100_poss_z", "weight_z"]
POS_COLS = ["pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"]
CHANNELS = STAT_COLS + POS_COLS + ["mp_scaled", "played_mask"]   # 23
N_AGES, N_ROWS, N_COLS = 26, 24, 25
WEIGHT_CH = CHANNELS.index("weight_z")
POS_CH = [CHANNELS.index(c) for c in POS_COLS]
MASK_CH = CHANNELS.index("played_mask")

@dataclass
class PlayerTensors:
    ids: list            # player_id for each row of every tensor
    grid: torch.Tensor   # (P, 23, 26)    stats x ages
    mask: torch.Tensor   # (P, 26)        1 = qualified season at that age
    chart: torch.Tensor  # (P, 2, 24, 25) standardized freq_rel, eff_rel

    def __len__(self):
        return len(self.ids)

    def to(self, device):
        return PlayerTensors(self.ids, self.grid.to(device), self.mask.to(device), self.chart.to(device))

    def subset(self, idx):
        return PlayerTensors([self.ids[i] for i in idx.tolist()], self.grid[idx], self.mask[idx], self.chart[idx])

def load_split(split: str, data_dir="data/processed") -> PlayerTensors:
    d = Path(data_dir) / "cnn"
    stats = pd.read_csv(d / f"cnn_stats_{split}.csv").sort_values(["player_id", "age"])
    shots = pd.read_csv(d / f"cnn_shots_{split}.csv").sort_values(["player_id", "row", "col"])
    ids = stats.player_id.unique().tolist()
    assert ids == shots.player_id.unique().tolist(), "stats and shots players differ"
    P = len(ids)
    grid = stats[CHANNELS].to_numpy(np.float32).reshape(P, N_AGES, len(CHANNELS)).transpose(0, 2, 1)
    mask = stats["played_mask"].to_numpy(np.float32).reshape(P, N_AGES)
    scaler = pd.read_csv(d / "cnn_shots_train_scaler.csv", index_col=0)   # fitted on TRAIN only
    for c in ["freq_rel", "eff_rel"]:
        shots[c] = (shots[c] - scaler.loc[c, "mean"]) / scaler.loc[c, "std"]
    chart = shots[["freq_rel", "eff_rel"]].to_numpy(np.float32).reshape(P, N_ROWS, N_COLS, 2).transpose(0, 3, 1, 2)
    return PlayerTensors(ids, torch.from_numpy(grid.copy()), torch.from_numpy(mask.copy()),
                         torch.from_numpy(chart.copy()))
```

```python
# file: tests/test_data.py
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
```

**Sanity notebook (`notebooks/01_data_check.ipynb`):** load the val split, find Paul Pierce (`piercpa01`, he's in val), and plot his grid with `plt.imshow(grid[i], aspect="auto")` and his two chart channels. You should see qualified seasons from age 21 and a mostly blank grid outside his career. This is your "eyes on the data" check, and a good figure for the report.

### Done when
`pytest tests/test_data.py` passes (7 tests) and the Pierce plot looks right.

### 💾 Commit points
1. `data: add PlayerTensors and channel constants`
2. `data: load stats grid and mask`
3. `data: load and standardize shot charts with train scaler`
4. `tests: add data shape and integrity tests`
5. `notebooks: add data check with Pierce example` (clear outputs) → PR, merge, tag `m1-foundations` with Mahir's MA-2.

### 🧠 Theory
- **Tensors and shapes.** A tensor is an n-dimensional array. PyTorch convolutions expect **channels first**: `Conv1d` takes (batch, channels, length), `Conv2d` takes (batch, channels, height, width). The CSV gives (player, age, stat), so `transpose(0, 2, 1)` swaps the last two axes to (player, stat, age).
- **`reshape` only works if the rows are sorted.** Sorting by `player_id, age` guarantees each block of 26 rows is one player in age order. If the sort were wrong, reshape would silently mix players. That's why the test checks `subset` alignment and the mask channel.
- **Standardization with the *train* scaler.** Neural networks train best when inputs are roughly mean 0, std 1. The raw shot values are tiny (std ≈ 0.005), so without scaling, the shot branch would barely affect the output. The mean and std come from **train only**, then get applied to val and test too. Computing them on all the data would leak information about validation and test players into training.
- **float32:** the M2's GPU doesn't support float64, and float32 halves the memory anyway.
- **Load once, keep in memory.** The whole dataset is about 20 MB as tensors, so there's no need for PyTorch's `DataLoader` and worker processes. Mahir moves everything to the GPU once, which is the fastest option on the M2.

### 🔍 Research keywords
`PyTorch tensor shapes NCL NCHW` · `numpy reshape transpose` · `feature standardization data leakage` · `PyTorch Dataset vs in-memory tensors`

---

## VI-2: Views and shortcut controls

### Goal
Given a batch of players, return **two partial views of each player** that the model must match, designed so the only way to match them is to learn real basketball traits.

### Why this is the most important task in the project
SimCLR's main finding was that **the choice of views decides what the model learns**. Our prototype confirmed it (ROADMAP §9): with no shortcut controls, the shortcut-free validation score was **0.16**; with them it was **0.71**.

| Shortcut found in the data | Evidence | Control |
|---|---|---|
| Weight barely changes within a player (spread 0.05 vs 1.0 across players) | weight alone matched 52% of validation players | add noise (`weight_noise`) and hide it in some views (`weight_drop`) |
| Position is usually the same every season | five positions alone narrow 181 players to about 36 | hide position in some views (`pos_drop`) |
| The shot chart is career-level, so identical in both views | identical charts → score fell to 0.41 | hide the chart in one view at random (`chart_drop`) + noise (`chart_noise`) |

### How the views are built
1. **Split the seasons.** Number each player's qualified seasons 1, 2, 3… View A gets the odd ones, view B the even ones (randomly swapped, so "A" carries no meaning). `split_mode: random` instead assigns each season by coin flip.
2. **Season dropout.** Each season in a view is hidden with probability `season_drop` (0.2), but a view is never left empty. This creates more variety, so the model can't memorize exact halves.
3. **Hide everything outside the view.** All 23 channels are zeroed at hidden ages, and the `played_mask` channel is set to the view's mask.
4. **Shortcut controls** (training only): weight noise/drop, position drop, chart drop + noise.
5. **Validation views are fixed and shortcut-free** (`train=False`): A = odd seasons with everything; B = even seasons with weight, position and chart hidden. B can only be matched through the season-by-season stats, so the validation score can't be inflated by a shortcut. They're also deterministic, so scores are comparable from epoch to epoch.

Seasons under 500 minutes (`played_mask` = 0) are hidden in every view, including the full career used for embedding (`full_view`), so training and embedding always see the same kind of input.

### Code

```python
# file: src/nbacnn/views.py
"""Turn a batch of players into two partial views (a positive pair) for contrastive training."""
from collections import namedtuple
import torch
from .data import WEIGHT_CH, POS_CH, MASK_CH

View = namedtuple("View", "grid mask chart")   # (B,23,26), (B,26), (B,2,24,25)

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
    """Whole career, no augmentation: what we embed after training."""
    return _view(batch, batch.mask, cfg or {}, None, train=False)
```

```python
# file: tests/test_views.py
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
```

### Done when
`pytest tests/test_views.py` passes (5 tests).

### 💾 Commit points
1. `views: add odd/even and random season split with season dropout`
2. `views: zero hidden seasons and set view mask`
3. `views: add weight, position and shot-chart shortcut controls`
4. `views: add shortcut-free validation views and full_view`
5. `tests: add view tests` → PR, merge, tag `m2-engine` with Mahir's MA-3.

### 🧠 Theory
- **Positive pairs and data augmentation.** In contrastive learning, the two views of the same item are a *positive pair*; all other items in the batch are *negatives*. Augmentation (here: season splitting, dropout, noise) defines what the model is told to ignore. Whatever differs between two views of the same player, the model learns to treat as irrelevant; whatever stays the same, it learns to rely on.
- **Shortcut learning.** If an easy signal is identical in both views, the network will use it, because that's the quickest way to reduce the loss. SimCLR's colour histograms are the image version; our weight and career shot chart are the basketball version. The fix is the same: make the easy signal unreliable so the model has to learn the harder one.
- **Why the controls are random (and only sometimes on).** Hiding weight in *every* view would throw away a legitimate trait. Hiding it in *some* views means the model can use weight when it's there, but can't depend on it.
- **Masking.** A mask is a 0/1 tensor that says which entries are real. Multiplying by it zeroes the rest; the model then uses the mask again (in pooling) to ignore those zeros.
- **Vectorized code.** Everything above works on the whole batch at once with tensor operations (`cumsum`, `where`, broadcasting), with no Python loop over players. That's what makes it fast on a GPU.

### 🔍 Research keywords
`data augmentation contrastive learning` · `positive and negative pairs` · `shortcut learning Geirhos` · `SimCLR augmentation composition` · `tensor broadcasting PyTorch` · `masking variable-length sequences`

---

## VI-3: The encoder

### Goal
A small two-branch CNN that turns (career grid, mask, shot chart) into a 32-number summary `h`, plus a projection head `z` for training.

### Code

```python
# file: src/nbacnn/model.py
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
```

```python
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
```

The model has **18,608 weights** (16,496 without the projection head). Write the shape after every layer in a comment, as above; it's the fastest way to catch mistakes.

### Done when
`pytest tests/test_model.py` passes (4 tests).

### 💾 Commit points
1. `model: add career (1D) and shot (2D) branches`
2. `model: add masked mean pooling over played ages`
3. `model: add combine layer and projection head`
4. `tests: add model shape, masking and gradient tests` → PR, merge, tag `m3-model` with Mahir's MA-4.

### 🧠 Theory
- **Conv1d along age.** Each of the 32 first-layer filters covers 23 channels × 3 ages (69 weights + bias) and slides along the 26 ages, detecting short-term patterns like "three-point rate rising". The second layer combines those into 5-season patterns. `padding=1` keeps the length at 26.
- **Conv2d across the court.** 3×3 filters find local shooting patterns. `MaxPool2d(2)` halves the court size, so the second layer sees larger regions; `AdaptiveAvgPool2d(1)` averages over the whole court, giving one number per pattern.
- **Masked mean pooling.** A plain average over all 26 ages would count retired or pre-debut years (zeros) and penalize short careers. Multiplying by the mask and dividing by the number of played ages averages only real seasons. It also makes the summary independent of *when* a pattern happened, which is what lets the CNN match two players who peaked at different ages. The `test_unplayed_ages_do_not_matter` test checks this property.
- **ReLU** sets negative values to zero: a filter only "speaks" when its pattern is present.
- **Dropout** randomly zeroes 20% of the combined features during training, so the model can't depend on any single one. It's switched off in `model.eval()`.
- **Projection head and `h` vs `z`.** The loss uses `z`; comparisons use `h`. SimCLR showed the layer *before* the head keeps more useful information, because the head absorbs whatever the matching task wants discarded.
- **L2 normalization** scales each vector to length 1, so dot products become cosine similarities and only direction matters.
- **Receptive field.** How much of the input one output can "see". Two 3-wide layers see 5 ages; adding layers or using `dilation` would let it see longer career arcs.

### 🔍 Research keywords
`Conv1d time series PyTorch` · `Conv2d pooling receptive field` · `masked mean pooling` · `projection head SimCLR` · `L2 normalization embeddings` · `dropout regularization`

---

## Integration (M4, joint session with Mahir)

Mahir runs the training (his guide, "Integration"). Your part:
1. Make sure your three modules are merged and `pytest` passes on both machines.
2. During the overfit-one-batch check and the first real run, watch the **alarms**. If the GAP alarm fires on the *default* config, look at your views first.
3. Expected from the prototype: validation top-1 ≈ 0.62–0.72, converging around epochs 90–120.

---

## VI-4: Embeddings and top-5 neighbours

### Goal
For a trained checkpoint, write one 32-number summary for each of the 1,207 players and each player's 5 nearest neighbours, in a format the KNN team can match.

### Code

```python
# file: src/nbacnn/embed.py
"""Turn every player into a summary h, then find each player's 5 nearest neighbours."""
import pandas as pd, torch
from .views import full_view

@torch.no_grad()
def embed_all(model, splits):
    """splits: {"train": PlayerTensors, "val": ..., "test": ...} -> DataFrame, one row per player."""
    model.eval()
    frames = []
    for name, data in splits.items():
        h, _ = model(*full_view(data))                     # h, not z: the projection head is discarded
        df = pd.DataFrame(h.cpu().numpy(), columns=[f"e{i}" for i in range(h.shape[1])])
        df.insert(0, "split", name)
        df.insert(0, "player_id", data.ids)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def top_k(emb, query_ids, k=5):
    """k nearest players by cosine similarity, searching ALL players except the query himself."""
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    E = torch.nn.functional.normalize(torch.tensor(emb[cols].to_numpy(), dtype=torch.float32), dim=1)
    pos = {pid: i for i, pid in enumerate(emb.player_id)}
    q = torch.tensor([pos[p] for p in query_ids])
    sim = E[q] @ E.T                                         # (queries, all players)
    sim[torch.arange(len(q)), q] = -2                        # never return the player himself
    s, j = sim.topk(k, dim=1)
    rows = [(qid, r + 1, emb.player_id.iloc[j[a, r].item()], round(s[a, r].item(), 4))
            for a, qid in enumerate(query_ids) for r in range(k)]
    return pd.DataFrame(rows, columns=["query_id", "rank", "neighbour_id", "similarity"])
```

```python
# file: scripts/export_embeddings.py
"""Embed all 1,207 players with a trained model and write embeddings + top-5 neighbours.
   python scripts/export_embeddings.py --checkpoint results/final/best_seed0.pt --tag seed0"""
import argparse
from pathlib import Path
import torch
from nbacnn.config import load_config
from nbacnn.data import load_split
from nbacnn.model import PlayerEncoder
from nbacnn.embed import embed_all, top_k

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--tag", default="seed0")
p.add_argument("--out_dir", default="results/final")
a = p.parse_args()
cfg = load_config(a.config)
model = PlayerEncoder(**cfg["model"])
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
emb = embed_all(model, {s: load_split(s, cfg["data_dir"]) for s in ("train", "val", "test")})
nb = top_k(emb, emb.player_id.tolist(), k=5)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
emb.to_csv(out / f"embeddings_{a.tag}.csv", index=False)
nb.to_csv(out / f"neighbours_{a.tag}.csv", index=False)
print(f"{len(emb)} players embedded -> {out}")
```

### Agree the neighbours format with the KNN team now
Both teams write `neighbours.csv` with exactly these columns: **`query_id, rank, neighbour_id, similarity`**, one row per (player, rank), 5 rows per player, all 1,207 players, using `player_id` from `player_splits.csv`, never returning the player himself, and searching the **whole** pool (all splits). Then VI-5's evaluator scores both with the same code, which is what makes the comparison fair.

### Eye test (prototype, validation players)
Paul Pierce → Paul George, Jayson Tatum, Antoine Walker, Tracy McGrady, Kobe Bryant. Steve Nash → Mo Williams, Deron Williams, José Calderón, J.J. Barea, Jarrett Jack. If your output looks nothing like this kind of list, check the views and the checkpoint before anything else.

### Done when
`results/final/embeddings_<tag>.csv` (1,207 × 34) and `neighbours_<tag>.csv` (6,035 rows) exist for the integration checkpoint; the KNN team has confirmed the format.

### 💾 Commit points
1. `embed: add embed_all and top_k cosine retrieval`
2. `scripts: add embedding export`
3. `docs: record shared neighbours.csv format with KNN team` → tag `m5-retrieval` with Mahir's MA-5.

### 🧠 Theory
- **Why embed with `full_view`:** training saw partial careers, but a comparison should use everything we know about a player.
- **Cosine similarity** measures direction, not length, and because `h` is unit-length it's just a dot product. One matrix multiplication (`E[q] @ E.T`) compares every query with all 1,207 players at once.
- **`topk`** returns the k largest values and their positions without sorting the whole row.
- **Search pool = all players.** A test player's neighbours can come from train. That isn't leakage: no BPM was used anywhere, and the CNN's weights only learned from train players' stats.

### 🔍 Research keywords
`cosine similarity retrieval` · `nearest neighbour search embeddings` · `torch.topk` · `embedding export inference`

---

## VI-5: BPM hidden-clue evaluator

### Goal
One scoring function for **any** set of neighbours (CNN or KNN): for each query player, how far are his 5 matches' BPM values from his own, at the same ages? Lower is better. Compare against random and same-position baselines, with confidence intervals.

### Rules (from the project decisions)
- `hidden_clue_bpm.csv` is read **only** in `evaluate.py`. No training or embedding code may touch it.
- BPM is **never** used to choose settings. On validation you may only run the sanity checks below (random and oracle embeddings), never to compare CNN configurations.
- The test split is scored **once**, together, with `--final` (roadmap M9). The script refuses otherwise.

### The score
For query player *q* and neighbour *n*: take every age where both had a qualified season, average |BPM_q − BPM_n| over those ages. Average over his 5 neighbours (skipping any with no shared ages) to get *q*'s score. The model's score is the mean over all query players, with a bootstrap 95% confidence interval. **Coverage** = the fraction of neighbour pairs with at least one shared age; report it, because a model whose matches rarely overlap in age is being scored on fewer pairs.

### Code

```python
# file: src/nbacnn/evaluate.py
"""BPM hidden-clue test: do a player's 5 matches have similar BPM at the same ages?"""
from pathlib import Path
import numpy as np, pandas as pd
from .data import POS_COLS

def bpm_table(data_dir="data/processed"):
    """{player_id: {age: bpm}} for qualified seasons. The ONLY place hidden_clue_bpm.csv is read."""
    clue = pd.read_csv(Path(data_dir) / "hidden_clue_bpm.csv")
    return {pid: dict(zip(g.age, g.bpm)) for pid, g in clue.groupby("player_id")}

def primary_position(data_dir="data/processed"):
    """Each player's main position: minutes-weighted over his qualified seasons."""
    frames = [pd.read_csv(Path(data_dir) / "cnn" / f"cnn_stats_{s}.csv") for s in ("train", "val", "test")]
    st = pd.concat(frames)
    w = st.mp_scaled * st.played_mask
    pos = st[POS_COLS].mul(w, axis=0).groupby(st.player_id).sum()
    return pos.idxmax(axis=1).str.replace("pos_", "").to_dict()

def pair_gap(bpm, a, b):
    """Mean |BPM_a - BPM_b| over ages both players had a qualified season; None if no shared ages."""
    shared = bpm[a].keys() & bpm[b].keys()
    return float(np.mean([abs(bpm[a][x] - bpm[b][x]) for x in shared])) if shared else None

def score_neighbours(nb, bpm):
    """nb: DataFrame with query_id, neighbour_id. Returns one row per query: score (lower = better), coverage."""
    nb = nb.assign(gap=[pair_gap(bpm, q, n) for q, n in zip(nb.query_id, nb.neighbour_id)])
    g = nb.groupby("query_id").gap
    return pd.DataFrame({"score": g.mean(), "coverage": g.apply(lambda x: x.notna().mean())})

def random_neighbours(query_ids, pool_ids, k=5, seed=0, groups=None):
    """Baseline: k random players (optionally from the same group, e.g. position), never the query."""
    rng = np.random.default_rng(seed); rows = []
    pool = np.array(pool_ids)
    for q in query_ids:
        cand = pool[(pool != q) & (np.array([groups[p] == groups[q] for p in pool]) if groups else (pool != q))]
        rows += [(q, r + 1, n) for r, n in enumerate(rng.choice(cand, size=k, replace=False))]
    return pd.DataFrame(rows, columns=["query_id", "rank", "neighbour_id"])

def baseline_scores(query_ids, pool_ids, bpm, groups=None, draws=20, k=5):
    """Average the random baseline over several draws so it isn't one lucky/unlucky sample."""
    runs = [score_neighbours(random_neighbours(query_ids, pool_ids, k, seed=d, groups=groups), bpm).score
            for d in range(draws)]
    return pd.concat(runs, axis=1).mean(axis=1)

def bootstrap_ci(x, n=2000, seed=0):
    """Mean and 95% confidence interval by resampling queries."""
    rng = np.random.default_rng(seed); x = np.asarray(pd.Series(x).dropna())
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def compare(model_scores, baseline, n=2000, seed=0):
    """Paired comparison on the same queries: mean(model - baseline) with 95% CI. Negative = model better."""
    d = (model_scores - baseline).dropna()
    return bootstrap_ci(d, n, seed)
```

```python
# file: scripts/evaluate.py
"""Score a neighbours file on the BPM hidden-clue test.
   python scripts/evaluate.py --neighbours results/final/neighbours_seed0.csv --split val
   Test split only with --final, once, together (roadmap M9)."""
import argparse, json, sys
import pandas as pd
from nbacnn.evaluate import bpm_table, primary_position, score_neighbours, baseline_scores, bootstrap_ci, compare

p = argparse.ArgumentParser()
p.add_argument("--neighbours", required=True, help="CSV with query_id, rank, neighbour_id (CNN or KNN)")
p.add_argument("--split", choices=["val", "test"], default="val")
p.add_argument("--data_dir", default="../data/processed")
p.add_argument("--final", action="store_true", help="required for the one-time test evaluation")
p.add_argument("--out", default=None)
a = p.parse_args()
if a.split == "test" and not a.final:
    sys.exit("Test evaluation runs once, together, with --final (roadmap M9).")

splits = pd.read_csv(f"{a.data_dir}/player_splits.csv")
queries = splits.loc[splits.split == a.split, "player_id"].tolist()
pool = splits.player_id.tolist()
nb = pd.read_csv(a.neighbours)
nb = nb[nb.query_id.isin(queries)]
assert set(nb.query_id) == set(queries), "neighbours file is missing some query players"
assert not (nb.query_id == nb.neighbour_id).any(), "a player was returned as his own neighbour"

bpm, pos = bpm_table(a.data_dir), primary_position(a.data_dir)
model = score_neighbours(nb, bpm)
rand = baseline_scores(queries, pool, bpm)
same_pos = baseline_scores(queries, pool, bpm, groups=pos)
res = {
    "split": a.split, "n_queries": len(queries), "coverage": float(model.coverage.mean()),
    "model": bootstrap_ci(model.score), "random": bootstrap_ci(rand), "same_position": bootstrap_ci(same_pos),
    "model_minus_random": compare(model.score, rand), "model_minus_same_position": compare(model.score, same_pos),
}
print(json.dumps(res, indent=2))
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
```

### Sanity checks (validation only, required before M9)
1. **Random embeddings ≈ random baseline.** Replace the embedding columns with random numbers, run `top_k`, and score. The difference from the random baseline must have a CI that includes 0. Prototype: +0.13, CI (−0.004, 0.26). ✓
2. **An "oracle" must win.** Build fake embeddings from each player's BPM-by-age curve itself; its score must be clearly lower than the random baseline. Prototype: 2.20 vs ≈ 2.85. ✓
3. **The test guard works:** `--split test` without `--final` exits with an error. ✓

Write these as a small notebook (`notebooks/04_embeddings_eval.ipynb`) so they're repeatable, and save the outputs to `results/eval_sanity/`.

### At M9 (joint, once)
```bash
for s in 0 1 2 3 4; do
  python scripts/evaluate.py --neighbours results/final/neighbours_seed$s.csv --split test --final --out results/final/test_cnn_seed$s.json
done
python scripts/evaluate.py --neighbours ../knn/results/neighbours.csv --split test --final --out results/final/test_knn.json
```
Report the CNN as mean ± std over the 5 seeds, next to KNN and both baselines. For CNN vs KNN on the same players, `compare(cnn_scores, knn_scores)` gives the paired difference with a CI.

### Done when
The three sanity checks pass and are saved; the KNN team has run the evaluator on one of their validation neighbour files without errors (format check only).

### 💾 Commit points
1. `evaluate: add BPM pair gap and per-query scoring`
2. `evaluate: add random and same-position baselines`
3. `evaluate: add bootstrap CI and paired comparison`
4. `scripts: add evaluate CLI with test-split guard`
5. `results: add evaluator sanity checks` → tag `m7-eval` with VI-6.

### 🧠 Theory
- **Held-out evaluation ("hidden clue").** The models never see BPM, so if their matches turn out to have similar BPM, the similarity captured something real. Known limitation: BPM is computed from box-score stats, so it isn't fully independent of the inputs; the baselines and the KNN comparison put the numbers in context.
- **Baselines.** "Random" answers "is this better than nothing?"; "same position" answers "is this better than the most obvious shortcut a fan would use?". A model that only beats random isn't saying much.
- **Bootstrap confidence intervals.** Resample the query players with replacement thousands of times and recompute the mean each time; the middle 95% of those means is the CI. It shows how much the score would wobble with a different set of players.
- **Paired comparison.** CNN and KNN are scored on the *same* players, so compare the per-player difference rather than two separate means. This removes the variation between players and makes real differences easier to detect.
- **Why test is used once.** Every time you look at a test score and then change something, the test set becomes part of tuning, and the final number becomes optimistic.

### 🔍 Research keywords
`held-out evaluation representation learning` · `bootstrap confidence interval` · `paired bootstrap test` · `baseline comparison machine learning` · `test set reuse overfitting`

---

## VI-6: Diagnostics

### Goal
Show what the summaries learned (and didn't), with figures for the report.

### Code

```python
# file: src/nbacnn/diagnostics.py
"""What did the summaries learn? UMAP map, linear probes, and neighbour tables."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from .data import STAT_COLS

def player_traits(data_dir="data/processed"):
    """Career-average traits per player (minutes-weighted over qualified seasons) + primary position + first age."""
    st = pd.concat([pd.read_csv(Path(data_dir) / "cnn" / f"cnn_stats_{s}.csv") for s in ("train", "val", "test")])
    st = st[st.played_mask == 1]
    w = st.mp_scaled
    avg = st[STAT_COLS].mul(w, axis=0).groupby(st.player_id).sum().div(w.groupby(st.player_id).sum(), axis=0)
    pos = st[["pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"]].mul(w, axis=0).groupby(st.player_id).sum()
    avg["position"] = pos.idxmax(axis=1).str.replace("pos_", "")
    avg["first_age"] = st.groupby("player_id").age.min()
    return avg

def probe(emb, traits, target):
    """Fit a simple linear model on TRAIN embeddings, score on VAL. High score = h contains that information."""
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    df = emb.set_index("player_id").join(traits[[target]])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if not pd.api.types.is_numeric_dtype(df[target]):       # categorical -> accuracy
        m = LogisticRegression(max_iter=2000).fit(tr[cols], tr[target])
        chance = va[target].value_counts(normalize=True).max()
        return {"target": target, "metric": "accuracy", "score": accuracy_score(va[target], m.predict(va[cols])),
                "chance": chance}
    m = Ridge(alpha=1.0).fit(tr[cols], tr[target])          # numeric -> R^2 (0 = no information, 1 = perfect)
    return {"target": target, "metric": "R2", "score": r2_score(va[target], m.predict(va[cols])), "chance": 0.0}

def umap_map(emb, traits, color_by, out_path, seed=0):
    import umap, matplotlib.pyplot as plt
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(emb[cols].to_numpy())
    c = emb.player_id.map(traits[color_by])
    fig, ax = plt.subplots(figsize=(7, 6))
    if not pd.api.types.is_numeric_dtype(c):
        for lab in sorted(c.dropna().unique()):
            k = (c == lab).to_numpy(); ax.scatter(xy[k, 0], xy[k, 1], s=8, label=lab)
        ax.legend(markerscale=2)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=8, cmap="viridis"); fig.colorbar(sc, label=color_by)
    ax.set(title=f"Player summaries (UMAP), coloured by {color_by}", xticks=[], yticks=[])
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return xy
```

### What to produce (`results/diagnostics/`)
1. **Probe table** for `weight_z, x3p_ar_z, blk_percent_z, ast_percent_z, position, first_age`. Prototype (default config): 3PA rate R² 0.89, assist % 0.87, block % 0.85, weight 0.60, position accuracy 0.66 (chance 0.24), first age 0.16.
   - Style traits should be high: the summary knows how a player plays.
   - Weight in the middle is healthy (size relates to role). **If weight R² is near 1.0 and the style traits are low, that's the shortcut.**
   - Run the probes for the `career_only` and `no_controls` checkpoints from Mahir's MA-5 too, and compare.
2. **UMAP maps** coloured by position, by `x3p_ar_z` and by `first_age`. Prototype: point guards cluster at one end, centres at the other, wings in between. If the map separates mainly by first age or era, the season z-scoring isn't doing its job.
3. **Neighbour tables** for ~12 familiar players across positions and eras (Pierce, Nash, Curry, Ben Wallace, Dirk, LeBron, Chris Paul, Dwight Howard, Kawhi, Jokić…). Check which split each player is in first; use validation or train players for discussion during development.
4. **Seed stability:** for the 5 final seeds, how many of each player's top-5 are shared between seeds? Low overlap means the comparisons depend on luck.

### Done when
The three figures, the probe table (default vs. ablations) and the neighbour tables are in `results/diagnostics/` with a short README of what they show.

### 💾 Commit points
1. `diagnostics: add trait table and linear probes`
2. `diagnostics: add UMAP map`
3. `results: add probe table and UMAP figures`
4. `results: add neighbour tables and seed stability` → tag `m7-eval`.

### 🧠 Theory
- **Linear probe.** Fit the simplest possible model (linear) from the 32 numbers to a known trait. If even a linear model can recover the trait on players it wasn't fitted on, the information is clearly in the summary. Probes are fitted on train and scored on validation so they measure information, not memorization.
- **R²** = fraction of the trait's variation explained; 0 is no better than guessing the average, 1 is perfect.
- **UMAP** squashes 32 dimensions into 2 while trying to keep neighbours as neighbours. Good for spotting structure (clusters by position), but distances between far-apart clusters in a UMAP plot aren't meaningful, so don't read too much into the overall layout.

### 🔍 Research keywords
`linear probe representation learning` · `UMAP vs t-SNE` · `embedding visualization` · `representation stability across seeds`

---

## VI-7 (stretch): Per-season shot charts

### Why
The career shot chart is the same in both views, which is why it has to be hidden in one of them. With a chart per season, each view could get a chart built from *its own* seasons, so the chart becomes useful evidence instead of a shortcut.

### What's needed
From the data pipeline (README section 7.4), the per-player-per-season binned shots (`shots_player_season_grid.csv`: player, season, row, col, attempts, makes) and the league grid. Convert season → age for each player (the stats file has both).

### Design (sketch; not yet run, unlike the rest of this guide)
1. Store four per-age tensors per player, shape (P, 26, 24, 25): attempts, makes, expected makes (attempts × league FG% for that season and cell), and league share × that season's attempts.
2. For a view mask m (P, 26), sum each tensor over the view's ages: `A = (att * m[:, :, None, None]).sum(1)` and likewise for the others.
3. `freq_rel = A / A.sum((1, 2)) − league_share_w / A.sum((1, 2))` and `eff_rel = (made − expected) / (A + 10)`, then standardize with the train scaler.
4. Add `season_shots` to `PlayerTensors` and build `chart` inside `make_views` from it. `full_view` with the whole career should reproduce `cnn_shots_*.csv` (within rounding), which is a built-in test.
5. Set `chart_mode: per_season` and rerun MA-5's ablation to measure the gain.

Memory: 844 players × 26 × 600 × 4 × 4 bytes ≈ 210 MB for train, fine on the M2.

💾 `views: add per-season shot charts` · `results: ablation with per-season charts`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `reshape` error in `load_split` | a player doesn't have exactly 26 ages or 600 cells | check the CSV; `groupby("player_id").size().unique()` |
| Model output identical for every player | inputs all zero (wrong channel order or mask) | print `grid[i]` for one player; check `CHANNELS` order |
| `test_unplayed_ages_do_not_matter` fails | pooling not masked, or mask is float64/wrong shape | `mask.unsqueeze(1)` must be (B, 1, 26) |
| GAP alarm on the default config | a shortcut control isn't firing | print what fraction of views have weight/position/chart hidden in a batch |
| Very high validation top-1 (≥ 0.90) | something leaks into validation view B | check `make_views(train=False)` zeroes weight, position and chart |
| Evaluator: "missing some query players" | neighbours file built from a subset | export neighbours for all 1,207 players |
| Probe: `could not convert string to float` | older code checking `dtype == object` | use `pd.api.types.is_numeric_dtype` (pandas 3 uses a string dtype) |
