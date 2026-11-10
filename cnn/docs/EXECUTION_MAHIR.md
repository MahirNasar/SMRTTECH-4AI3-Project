# Execution Guide: Mahir

**Your role:** training engine and experiments. You own the M2 machine, so every real training run happens on your computer.
**Your files:** `device.py`, `losses.py`, `metrics.py`, `convergence.py`, `fake.py`, `config.py`, `train.py`, `live_plot.py`, `study.py`, the `scripts/` that run them, and the `train:` / `convergence:` blocks of the config.
**Read first:** [`../ROADMAP.md`](../ROADMAP.md), especially the interface contract (§4) and the prototype results (§9).

How to use this guide:
- Each task (`MA-1` … `MA-7`) has **Goal → Steps → Code → Done when → 💾 Commit points → 🧠 Theory → 🔍 Research keywords**.
- 💾 marks a good moment to `git add`, `git commit` and `git push`. Commit at each one, even if the task isn't finished.
- All code in this guide was run on the real dataset in a prototype before writing it down. If something doesn't work for you, it's most likely an environment difference; check [Troubleshooting](#troubleshooting).
- Paths are relative to the `cnn/` folder unless they start with the repo root.

---

## Task overview

| Task | Roadmap milestone | What you build | Week |
|---|---|---|---|
| M0 | M0 | Repo scaffold, M2 environment | 1 |
| MA-1 | M1 | Device selection + CPU vs MPS benchmark | 1 |
| MA-2 | M1 | NT-Xent contrastive loss + tests | 1 |
| MA-3 | M2 | Training engine, tested on fake data | 2 |
| MA-4 | M3 | Live training dashboard + convergence target | 3 |
| Integration | M4 | First real run, jointly with Viroz | 3 |
| MA-5 | M5 | Shortcut ablation runs | 4 |
| MA-6 | M6 | Parametric study: learning rate × epochs × seeds | 4 |
| MA-7 | M8 | Final 5-seed training + handoff | 5 |
| Test evaluation | M9 | Jointly with Viroz | 5 |

---

## M0: Repo scaffold and M2 setup

### Goal
A repo both of you can clone and run, and a Python environment on the M2 that uses Apple's GPU.

### Steps

**1. Set up the M2 for training.**

| Setting | Why |
|---|---|
| **Use native arm64 Python, not Rosetta.** Install [Miniforge](https://github.com/conda-forge/miniforge) (Apple Silicon build). | An x86 Python running under Rosetta can't use the GPU (MPS) and is much slower. |
| Check: `python -c "import platform; print(platform.machine())"` must print `arm64`. | If it prints `x86_64`, your terminal or Python is running under Rosetta. |
| macOS 14 or newer. | MPS support in PyTorch keeps improving with newer macOS. |
| Plug in the charger; turn **Low Power Mode off** (System Settings → Battery). | On battery, macOS throttles the CPU and GPU. |
| Run long jobs with `caffeinate -dims python ...`. | Stops the Mac sleeping in the middle of the parametric study. |
| Close heavy apps (browser tabs, Docker, video calls). | The M2's CPU and GPU share one pool of memory (unified memory). |
| Watch **Activity Monitor → Window → GPU History** during training. | Confirms the GPU is actually busy when you pick MPS. |

**2. Create the environment.** Save as `cnn/environment.yml`:

```yaml
name: nbacnn
channels: [conda-forge]
dependencies:
  - python=3.11
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - pyyaml
  - pytest
  - jupyterlab
  - ipympl            # interactive plots in Jupyter
  - pip
  - pip:
      - torch         # the standard macOS arm64 wheel includes MPS support
      - tensorboard
      - umap-learn    # for Viroz's diagnostics
```

```bash
cd cnn
conda env create -f environment.yml
conda activate nbacnn
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# expect: 2.x.x True
```

**3. Make the package importable.** Save as `cnn/pyproject.toml`:

```toml
[project]
name = "nbacnn"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Then run `pip install -e .` once. The `-e` (editable) means your code edits take effect without reinstalling.

**4. Scaffold the folders** from ROADMAP §10, add the `.gitignore` lines, and unzip `nba_model_data.zip` into `data/processed/` at the repo root. Add an empty `src/nbacnn/__init__.py`.

**5. Kickoff call with Viroz (30 min).** Go through the interface contract (ROADMAP §4) line by line and commit it. Also commit `configs/default.yaml` (below). Your engine imports `PlayerTensors`, `make_views` and `PlayerEncoder`; until Viroz's versions land, use the fake data in MA-3 and Viroz's first draft of the model.

```yaml
# file: configs/default.yaml
seed: 0
device: auto            # auto | mps | cpu
data_dir: ../data/processed
out_dir: outputs/runs
run_name: baseline
tensorboard: true
early_stop: true

model:                  # Viroz
  in_ch: 23
  emb: 32
  dropout: 0.2

views:                  # Viroz
  split_mode: odd_even  # odd_even | random
  season_drop: 0.2      # chance of hiding each season in a view (training only)
  weight_noise: 0.5     # std of noise added to weight_z per view
  weight_drop: 0.5      # chance of hiding weight in a view
  pos_drop: 0.5         # chance of hiding position channels in a view
  chart_mode: drop_one  # drop_one | same | off
  chart_drop: 0.5       # chance of hiding the shot chart in a view
  chart_noise: 0.5      # std of noise added to the standardized shot chart

train:                  # Mahir
  lr: 1.0e-3
  weight_decay: 1.0e-4
  batch_size: 128
  epochs: 200
  warmup_epochs: 5
  tau: 0.1

convergence:            # Mahir
  patience: 20
  min_delta: 0.005
  target_top1: 0.65
  too_good_top1: 0.90
  gap_alarm: 0.30
```

`data_dir` is relative to `cnn/`, where you run the scripts. In YAML, write small numbers with a decimal point (`1.0e-3`, not `1e-3`), or YAML reads them as text.

### Done when
- `python -c "import nbacnn; import torch; print(torch.backends.mps.is_available())"` prints `True` on the M2.
- Viroz can clone, create the environment and run `pytest` (zero tests is fine at this stage).

### 💾 Commit points
1. `chore: add cnn folder structure and gitignore`
2. `chore: add conda environment and pyproject`
3. `data: add processed dataset (nba_model_data.zip)`
4. `docs: add roadmap and execution guides`
5. `config: add shared default.yaml` → open PR, Viroz approves, merge, tag `m0-setup`.

### 🧠 Theory: why Apple silicon is different
- **Unified memory.** On a normal PC, the GPU has its own memory and data must be copied across a slow connection. On the M2, the CPU and GPU share one pool of memory, so copies are cheaper, but they still compete for the same memory and bandwidth.
- **MPS (Metal Performance Shaders)** is Apple's GPU backend. PyTorch calls it the `"mps"` device. Moving data there is `tensor.to("mps")`.
- **float32 only.** MPS doesn't support float64. Pandas produces float64 by default, which is why `load_split` converts everything to `float32`.
- **Launch overhead.** Each GPU operation has a small fixed start-up cost. Our model is tiny (~18,600 weights) and batches are small, so the GPU may spend more time starting work than doing it. That's why MA-1 benchmarks CPU vs MPS instead of assuming the GPU wins.

### 🔍 Research keywords
`PyTorch MPS backend` · `Apple unified memory architecture` · `conda-forge miniforge arm64` · `pip install -e editable install`

---

## MA-1: Device selection and benchmark

### Goal
One function that picks the device, and a measured answer to "is MPS or CPU faster for *our* model?"

### Code

```python
# file: src/nbacnn/device.py
import random
import numpy as np, torch

def pick_device(pref="auto"):
    """'auto' -> mps if available, else cpu. 'cpu' / 'mps' force a choice."""
    if pref == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "mps":
        raise RuntimeError("MPS requested but not available (check arm64 Python + PyTorch build)")
    return torch.device("cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # also seeds MPS

def sync(device):
    """Wait for queued GPU work to finish (needed for honest timing on MPS)."""
    if device.type == "mps":
        torch.mps.synchronize()
```

```python
# file: scripts/benchmark_device.py
"""Time one training step on CPU and on MPS to decide which device to use."""
import time, torch
from nbacnn.fake import make_fake_split
from nbacnn.model import PlayerEncoder
from nbacnn.views import make_views
from nbacnn.losses import nt_xent
from nbacnn.device import sync

def bench(device, batch=128, steps=50):
    data = make_fake_split(batch).to(device)
    model = PlayerEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = {"chart_mode": "drop_one"}
    for i in range(steps + 10):                     # first 10 steps are warm-up (not timed)
        if i == 10:
            sync(device); t0 = time.perf_counter()
        a, b = make_views(data, cfg)
        loss = nt_xent(model(*a)[1], model(*b)[1])
        opt.zero_grad(); loss.backward(); opt.step()
    sync(device)
    return (time.perf_counter() - t0) / steps * 1000

devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
for batch in (64, 128, 256):
    print(f"batch {batch:3d}: " + "  ".join(f"{d}={bench(torch.device(d), batch):6.1f} ms/step" for d in devices))
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
```

This needs `fake.py` (MA-3), `model.py` and `views.py`. Run it once those exist, then **set `device:` in `default.yaml` to whichever is faster at batch 128.** For reference, a 2-thread cloud CPU took about 39 ms per step at batch 128; the M2's CPU should be faster.

### Done when
The benchmark output is pasted into `results/benchmark.txt` and `device:` is set from it.

### 💾 Commit points
1. `device: add pick_device, seeding and sync`
2. `scripts: add CPU vs MPS benchmark`
3. `results: record M2 benchmark` (after you run it)

### 🧠 Theory
- **Why `sync` before timing:** GPU work is *asynchronous*. Python queues the operation and moves on immediately, so without `torch.mps.synchronize()` you'd time the queuing, not the work.
- **Warm-up steps:** the first few steps include one-off costs (compiling GPU kernels, allocating memory), so they're excluded from the timing.
- **Seeds:** `torch.manual_seed` makes random numbers repeatable. On MPS, some operations still aren't perfectly deterministic, which is one more reason the study uses 3 seeds and reports a spread.

### 🔍 Research keywords
`asynchronous GPU execution` · `torch.mps.synchronize` · `benchmarking warmup` · `PyTorch reproducibility`

---

## MA-2: Contrastive loss (NT-Xent)

### Goal
The loss function from SimCLR, with tests that prove it behaves correctly.

### Code

```python
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
```

```python
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
```

### Done when
`pytest tests/test_losses.py` passes.

### 💾 Commit points
1. `losses: add NT-Xent and chance_loss`
2. `tests: add loss sanity tests` → PR, merge, tag `m1-foundations` once Viroz's VI-1 is also merged.

### 🧠 Theory: what the loss computes
For a batch of N players there are 2N views. Stack them into one matrix `z` (2N × 32) and compute every pairwise cosine similarity at once with `z @ z.T`. Because each row is unit length, the dot product *is* the cosine similarity.

For row *i*, the "classes" are the other 2N − 1 views, and the correct class is its partner. `F.cross_entropy` turns each row into probabilities with **softmax** and takes −log(probability of the correct partner). That's exactly SimCLR's equation 1.

- **Masking the diagonal:** a view is always identical to itself (similarity 1). Without the mask it would always "find itself" and the task would be trivial.
- **Temperature τ:** dividing by 0.1 multiplies differences by 10 before softmax. Small τ makes the model focus on the hardest wrong answers; too small and training becomes unstable. SimCLR found 0.1 works well.
- **Normalized loss:** if the model gave every candidate equal probability, the loss would be ln(2N − 1). Dividing by that gives a number where 1.0 = random guessing and 0 = perfect, so batches of different sizes (128 for training, 181 for validation) can be compared. An untrained network can start slightly *above* 1.0, because random similarities divided by τ aren't uniform.
- **Why `cross_entropy` rather than writing softmax yourself:** it uses the log-sum-exp trick internally, which avoids overflow when values like 10/τ are exponentiated.

### 🔍 Research keywords
`InfoNCE loss` · `NT-Xent SimCLR` · `softmax cross entropy` · `log-sum-exp trick` · `temperature contrastive learning` · `hard negatives`

---

## MA-3: Training engine on fake data

### Goal
A complete training loop: batches, views, loss, optimizer, learning-rate schedule, validation, logging and checkpoints. Build it on **fake data** so you don't wait for Viroz.

### Steps
1. Write `fake.py`: fake players with the real tensor shapes.
2. Write `metrics.py`: the two-halves accuracy used for validation.
3. Write `config.py`: load a YAML config with command-line overrides.
4. Write `train.py` and `scripts/train.py`.
5. Write a smoke test: the loss must fall on fake data.

### Code

```python
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
```

```python
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
    sim = ha @ hb.T                                         # (P, P) cosine similarity of summaries
    rank = (sim > sim.diag()[:, None]).sum(1)               # how many players beat the true partner
    loss = nt_xent(za, zb, tau).item()
    model.train(was_training)
    return {"top1": (rank == 0).float().mean().item(),
            "top5": (rank < 5).float().mean().item(),
            "loss_norm": loss / chance_loss(len(data))}
```

```python
# file: src/nbacnn/config.py
import yaml

def load_config(path, overrides=()):
    """Load a YAML config and apply command-line overrides like 'train.lr=3e-3' or 'seed=1'."""
    cfg = yaml.safe_load(open(path))
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        v = yaml.safe_load(value)                 # "true" -> True, "200" -> 200
        if isinstance(v, str):
            try:
                v = float(v)                      # YAML reads "3e-3" as text; make it a number
            except ValueError:
                pass
        node[leaf] = v
    return cfg
```

```python
# file: src/nbacnn/train.py
"""Contrastive training loop with validation, checkpoints, logging and convergence tracking."""
import csv, json, math, time
from pathlib import Path
import torch, yaml

from .data import load_split
from .views import make_views
from .model import PlayerEncoder
from .losses import nt_xent, chance_loss
from .metrics import two_halves
from .device import pick_device, seed_everything, sync
from .convergence import ConvergenceTracker

def warmup_cosine(opt, warmup_steps, total_steps):
    """LR rises linearly for warmup_steps, then follows a cosine curve down to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)

def train(cfg, data=None, live=None):
    """cfg: dict from configs/*.yaml. data: optional (train, val) PlayerTensors, e.g. fake data in tests."""
    t_cfg, conv_cfg = cfg["train"], cfg["convergence"]
    device = pick_device(cfg.get("device", "auto"))
    seed_everything(cfg["seed"])
    tr, va = data if data is not None else (load_split("train", cfg["data_dir"]), load_split("val", cfg["data_dir"]))
    tr, va = tr.to(device), va.to(device)            # the whole dataset fits in memory: move it once
    g = torch.Generator().manual_seed(0)              # fixed train subset the same size as val,
    tr_eval = tr.subset(torch.randperm(len(tr), generator=g)[:len(va)].to(device))   # so top-1 is comparable

    model = PlayerEncoder(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(tr) / t_cfg["batch_size"])
    sched = warmup_cosine(opt, t_cfg["warmup_epochs"] * steps_per_epoch, t_cfg["epochs"] * steps_per_epoch)
    tracker = ConvergenceTracker(**conv_cfg)

    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    writer = None
    if cfg.get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    log = open(run_dir / "metrics.csv", "w", newline="")
    fields = ["epoch", "lr", "train_loss_norm", "val_loss_norm", "val_top1", "val_top5", "train_top1", "seconds"]
    out = csv.DictWriter(log, fieldnames=fields); out.writeheader()

    t0 = time.time()
    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(tr)).to(device)     # shuffle on CPU, then move (works on every device)
        total, seen = 0.0, 0
        for i in range(0, len(tr), t_cfg["batch_size"]):
            idx = perm[i:i + t_cfg["batch_size"]]
            if len(idx) < 8:                         # a tiny last batch gives a noisy loss: skip it
                continue
            view_a, view_b = make_views(tr.subset(idx), cfg["views"])
            _, z_a = model(*view_a)
            _, z_b = model(*view_b)
            loss = nt_xent(z_a, z_b, t_cfg["tau"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() / chance_loss(len(idx)) * len(idx)
            seen += len(idx)

        val = two_halves(model, va, cfg["views"], t_cfg["tau"])
        trm = two_halves(model, tr_eval, cfg["views"], t_cfg["tau"])
        sync(device)
        row = dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss_norm=total / seen,
                   val_loss_norm=val["loss_norm"], val_top1=val["top1"], val_top5=val["top5"],
                   train_top1=trm["top1"], seconds=time.time() - t0)
        out.writerow(row); log.flush()
        if tracker.update(epoch, row["train_loss_norm"], val["loss_norm"], val["top1"]):
            torch.save(model.state_dict(), run_dir / "best.pt")
        if writer:
            writer.add_scalars("loss_norm", {"train": row["train_loss_norm"], "val": row["val_loss_norm"]}, epoch)
            writer.add_scalars("top1", {"val": val["top1"], "train": trm["top1"], "target": conv_cfg["target_top1"]}, epoch)
            writer.add_scalar("lr", row["lr"], epoch)
        if live:
            live.update(row, tracker.converged_epoch, tracker.target_epoch)
        if tracker.converged and cfg.get("early_stop", True):
            print(f"converged at epoch {epoch} (best val loss at epoch {tracker.best_epoch})")
            break

    if live:
        live.update(None, tracker.converged_epoch, tracker.target_epoch, force=True)   # final redraw
    torch.save(model.state_dict(), run_dir / "last.pt")
    summary = dict(best_epoch=tracker.best_epoch, best_val_loss_norm=tracker.best,
                   converged_epoch=tracker.converged_epoch, target_epoch=tracker.target_epoch,
                   alarms=tracker.alarms, final=row, device=str(device))
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)
    log.close()
    if writer:
        writer.close()
    return run_dir
```

`train.py` imports `ConvergenceTracker`, which you write in MA-4. For MA-3, create `convergence.py` now with the MA-4 code below; you'll understand it fully in MA-4.

```python
# file: scripts/train.py
"""Train one model.  python scripts/train.py --config configs/default.yaml --live window --set train.lr=3e-3"""
import argparse
from nbacnn.config import load_config
from nbacnn.train import train

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--live", choices=["window", "file", "none"], default="window")
p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=3e-3 seed=1 run_name=lr3e-3")
args = p.parse_args()
cfg = load_config(args.config, args.set)

live = None
if args.live != "none":
    if args.live == "file":
        import matplotlib; matplotlib.use("Agg")          # no window: just save a PNG
    from nbacnn.live_plot import LivePlot
    c = cfg["convergence"]
    live = LivePlot(c["target_top1"], c["too_good_top1"], mode=args.live,
                    save_path=f"{cfg['out_dir']}/{cfg['run_name']}_live.png")
run_dir = train(cfg, live=live)
print("saved to", run_dir)
if live and args.live == "window":
    input("Training finished. Press Enter to close the plot...")
```

```python
# file: tests/test_train_smoke.py
import pandas as pd
from nbacnn.config import load_config
from nbacnn.fake import make_fake_split
from nbacnn.train import train

def test_loss_falls_on_fake_data(tmp_path):
    cfg = load_config("configs/default.yaml",
                      [f"out_dir={tmp_path}", "run_name=smoke", "tensorboard=false",
                       "early_stop=false", "train.epochs=20", "views.chart_mode=off", "device=cpu"])
    run = train(cfg, data=(make_fake_split(512, seed=0), make_fake_split(128, seed=1)))
    m = pd.read_csv(run / "metrics.csv")
    assert len(m) == 20
    assert m.train_loss_norm.iloc[-1] < m.train_loss_norm.iloc[0] - 0.15   # it learned something
    assert (run / "best.pt").exists() and (run / "summary.json").exists()
```

On fake data the normalized training loss should fall from about 1.0 to about 0.7 in 20 epochs. Validation top-1 on fake data jumps around and isn't meaningful; only the loss matters for this test.

### Done when
- `pytest tests/test_train_smoke.py` passes on CPU.
- A run folder contains `config.yaml`, `metrics.csv`, `best.pt`, `last.pt`, `summary.json`.

### 💾 Commit points
1. `fake: add fake player generator with real tensor shapes`
2. `metrics: add two-halves accuracy`
3. `config: add YAML loader with command-line overrides`
4. `train: add training loop with AdamW and warmup-cosine schedule`
5. `train: add checkpoints, CSV logging and TensorBoard`
6. `tests: add fake-data smoke test` → PR, merge, tag `m2-engine` with Viroz's VI-2.

### 🧠 Theory: the training loop, line by line
One **epoch** = one pass over all 844 training players, in batches of 128 (7 steps). Each **step**:

| Line | What happens |
|---|---|
| `make_views(...)` | two partial views of each player in the batch |
| `model(*view)` | **forward pass**: compute the embeddings |
| `nt_xent(...)` | how wrong the matching was |
| `opt.zero_grad()` | clear last step's gradients (PyTorch adds to them otherwise) |
| `loss.backward()` | **backpropagation**: compute ∂loss/∂weight for all ~18,600 weights |
| `opt.step()` | **update**: move each weight a small step to reduce the loss |
| `sched.step()` | adjust the learning rate for the next step |

- **AdamW:** an optimizer that adapts the step size for each weight from the history of its gradients. "W" means *decoupled weight decay*: weights are gently pulled toward zero each step, which discourages memorizing.
- **Learning rate schedule (warmup + cosine):** start with a tiny rate for 5 epochs so the random initial weights don't get thrown around, then decay smoothly to zero so the final epochs make fine adjustments. Because the curve stretches to fit `epochs`, a 50-epoch run and a 400-epoch run follow *different* schedules, which is why the study trains each epoch count separately.
- **`model.train()` vs `model.eval()`:** dropout is only active in training mode. Validation must use eval mode, or dropout adds noise to the measurement.
- **`torch.no_grad()`:** skips gradient bookkeeping during validation; faster and uses less memory.
- **Checkpoint on best validation loss:** the final epoch isn't always the best one. `best.pt` keeps the best weights seen.
- **Why validation uses a train subset of 181 players for "train top-1":** top-1 accuracy depends on how many candidates there are. Finding your partner among 844 players is harder than among 181. Using equal-size pools makes train and validation accuracy comparable.

### 🔍 Research keywords
`PyTorch training loop` · `backpropagation intuition` · `AdamW decoupled weight decay` · `learning rate warmup cosine annealing` · `model.eval dropout` · `checkpointing best model`

---

## MA-4: Live dashboard and convergence target

### Goal
Watch training as it happens, with the convergence rule and target drawn on the plot, plus alarms for the known failure modes.

### The convergence target, and why these numbers

| Signal | Rule | Why this value |
|---|---|---|
| **Converged** | validation normalized loss hasn't improved by more than `min_delta` = 0.005 for `patience` = 20 epochs | validation loss is smooth enough that a 0.005 change is real; 20 epochs rides out noise. Prototype converged around epochs 90–120. |
| **Target** | shortcut-free validation top-1 ≥ `target_top1` = 0.65 | prototype default runs reached 0.62–0.71 (3 seeds). The target says "working as well as the prototype or better". Random is 0.006. |
| **Gap alarm** | validation loss − train loss > `gap_alarm` = 0.30 | in the prototype, the two shortcut runs hit this at epoch 6 (gap ≈ 0.35, growing to 1.1); the default runs stayed below 0.05. |
| **Too-good alarm** | validation top-1 ≥ 0.90 | view B hides weight, position and the chart, so ≥ 0.90 suggests something leaked into it. |

Treat the target as a starting point. After the integration run (M4) and the study (MA-6), update `target_top1` to the value the chosen configuration reliably reaches, and note the change in the PR.

### Code

```python
# file: src/nbacnn/convergence.py
import math

class ConvergenceTracker:
    """Decides when validation loss has plateaued, and raises alarms."""
    def __init__(self, patience=20, min_delta=0.005, target_top1=0.65, too_good_top1=0.90, gap_alarm=0.30):
        self.patience, self.min_delta, self.target_top1 = patience, min_delta, target_top1
        self.too_good_top1, self.gap_alarm = too_good_top1, gap_alarm
        self.best, self.best_epoch = math.inf, 0
        self.converged_epoch = self.target_epoch = None
        self.alarms = []

    def update(self, epoch, train_loss_norm, val_loss_norm, val_top1):
        improved = val_loss_norm < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = val_loss_norm, epoch
        if self.target_epoch is None and val_top1 >= self.target_top1:
            self.target_epoch = epoch
        if self.converged_epoch is None and epoch - self.best_epoch >= self.patience:
            self.converged_epoch = epoch
        if val_loss_norm - train_loss_norm > self.gap_alarm:
            self._alarm("GAP", f"val loss {val_loss_norm:.2f} vs train {train_loss_norm:.2f} at epoch {epoch}: "
                               "shortcut or overfitting")
        if val_top1 >= self.too_good_top1:
            self._alarm("TOO GOOD", f"val top-1 {val_top1:.2f} at epoch {epoch}: check that view B hides "
                                    "weight, position and the shot chart")
        return improved

    def _alarm(self, kind, msg):
        if not any(a.startswith(kind) for a in self.alarms):   # report each kind once
            self.alarms.append(f"{kind}: {msg}")
            print(f"  !! {kind}: {msg}")

    @property
    def converged(self):
        return self.converged_epoch is not None
```

```python
# file: src/nbacnn/live_plot.py
"""Live training dashboard: loss and accuracy vs. epoch, with the convergence target."""
import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self, target_top1=0.65, too_good_top1=0.90, mode="window", save_path=None, every=1):
        self.mode, self.save_path, self.every = mode, save_path, every   # mode: window | notebook | file
        self.target_top1, self.too_good_top1 = target_top1, too_good_top1
        self.h = {k: [] for k in ["epoch", "train_loss_norm", "val_loss_norm", "val_top1", "val_top5", "train_top1"]}
        if mode == "window":
            plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    def update(self, row, converged_epoch=None, target_epoch=None, force=False):
        if row is not None:
            for k in self.h:
                self.h[k].append(row[k])
        if not force and self.h["epoch"][-1] % self.every:
            return
        e = self.h["epoch"]
        a = self.ax_loss; a.clear()
        a.plot(e, self.h["train_loss_norm"], label="train")
        a.plot(e, self.h["val_loss_norm"], label="validation")
        a.axhline(1.0, ls=":", c="grey", label="random guessing")
        a.set(title="Normalized contrastive loss", xlabel="epoch", ylabel="loss ÷ ln(2N−1)", ylim=(0, None))
        b = self.ax_acc; b.clear()
        b.plot(e, self.h["val_top1"], label="val top-1")
        b.plot(e, self.h["val_top5"], label="val top-5", alpha=0.6)
        b.plot(e, self.h["train_top1"], label="train top-1", ls="--", alpha=0.6)
        b.axhline(self.target_top1, c="green", ls="--", label=f"target {self.target_top1:.2f}")
        b.axhline(self.too_good_top1, c="red", ls=":", label="too-good alarm (top-1)")
        b.set(title="Two-halves matching accuracy", xlabel="epoch", ylim=(0, 1.02))
        for ax in (a, b):
            if converged_epoch: ax.axvline(converged_epoch, c="purple", ls="-.", label="converged")
            if target_epoch and ax is b: ax.axvline(target_epoch, c="green", alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        self.fig.tight_layout()
        if self.mode == "window":
            self.fig.canvas.draw_idle(); plt.pause(0.001)
        elif self.mode == "notebook":
            from IPython.display import display, clear_output
            clear_output(wait=True); display(self.fig)
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=120)

    def close(self):
        if self.mode == "window":
            plt.ioff()
        plt.close(self.fig)
```

### Three ways to watch training

1. **Pop-up window (default):** `python scripts/train.py --live window`. A matplotlib window redraws every epoch.
2. **Jupyter (`notebooks/02_live_training.ipynb`):**
   ```python
   from nbacnn.config import load_config
   from nbacnn.train import train
   from nbacnn.live_plot import LivePlot
   cfg = load_config("../configs/default.yaml", ["data_dir=../../data/processed", "out_dir=../outputs/runs"])
   live = LivePlot(cfg["convergence"]["target_top1"], cfg["convergence"]["too_good_top1"], mode="notebook")
   run_dir = train(cfg, live=live)
   ```
3. **TensorBoard (best for comparing runs):** with `tensorboard: true` in the config, run `tensorboard --logdir outputs` in a second terminal and open http://localhost:6006. Curves update live and every run appears side by side, which is ideal for the study. The `top1/target` line shows the target.

What a healthy run looks like (prototype, default config): both losses fall together from ~1.0 to ~0.3; validation top-1 rises past 0.65 around epoch 30–35 and levels off near 0.7; train top-1 ends ~0.15 above validation.

### Done when
- The live plot shows both panels with the target line and a converged marker.
- Running with `--set views.chart_mode=same` triggers the GAP alarm within ~10 epochs (it's a deliberate shortcut, so the alarm *should* fire).

### 💾 Commit points
1. `convergence: add plateau detection, target and alarms`
2. `viz: add live training plot (window/notebook/file)`
3. `train: wire live plot and TensorBoard target line`
4. `notebooks: add live training notebook` (clear outputs first) → PR, merge, tag `m3-model` with Viroz's VI-3.

### 🧠 Theory
- **Training vs validation curves:** training loss shows whether the model can fit the data it sees; validation loss shows whether that carries over to players it hasn't seen. Validation flattening out means more training isn't helping. Validation rising while training falls means overfitting or a shortcut.
- **Early stopping with patience:** stopping the moment validation doesn't improve would react to noise. Patience waits a set number of epochs before deciding the improvement has really stopped. `min_delta` ignores improvements too small to matter.
- **Why the target is on top-1 accuracy, not loss:** loss depends on τ and batch size, while "fraction of players matched to their other half" is easy to interpret and to explain.
- **What the alarms catch:** a shortcut makes the *training* task easy (training loss drops fast) while the shortcut-free validation task stays hard (validation loss stalls or rises). That growing gap is the signature.

### 🔍 Research keywords
`learning curves interpretation` · `early stopping patience` · `TensorBoard PyTorch SummaryWriter` · `matplotlib interactive mode plt.ion` · `generalization gap`

---

## Integration (M4, joint session with Viroz)

### Steps
1. Merge Viroz's `data.py`, `views.py`, `model.py` (VI-1 to VI-3).
2. Run all tests: `pytest`.
3. **Overfit one batch** (a classic check that the whole pipeline can learn):
   ```bash
   python scripts/train.py --live window --set run_name=overfit_check train.epochs=300 \
       train.batch_size=32 views.season_drop=0 views.weight_noise=0 views.weight_drop=0 \
       views.pos_drop=0 views.chart_mode=same views.chart_noise=0 early_stop=false
   ```
   With all controls off, training loss should fall close to 0. If it doesn't, something in the pipeline is broken (data, views or gradients). The GAP alarm is expected here.
4. **First real run** with the default config: `python scripts/train.py --live window --set run_name=baseline`. Compare with the prototype: validation top-1 should reach roughly 0.62–0.72 and converge around epochs 90–120.
5. Copy `outputs/runs/baseline/{metrics.csv,summary.json,config.yaml}` and the live plot PNG into `results/first_run/`.

### 💾 Commit points
`results: add first real training run` → tag `m4-first-run`.

---

## MA-5: Shortcut ablation runs

### Goal
Show, with evidence, that the shortcut controls matter and what each input contributes. This reproduces the prototype table (ROADMAP §9) on the M2 with 3 seeds, which becomes a results table for the report.

### Runs (3 seeds each, default 200 epochs with early stopping)

| Name | Overrides |
|---|---|
| `default` | none |
| `career_only` | `views.chart_mode=off` |
| `identical_chart` | `views.chart_mode=same views.chart_noise=0` |
| `no_controls` | `views.season_drop=0 views.weight_noise=0 views.weight_drop=0 views.pos_drop=0 views.chart_mode=same views.chart_noise=0` |

```bash
for s in 0 1 2; do
  caffeinate -i python scripts/train.py --live none --set seed=$s run_name=abl_default_s$s
  caffeinate -i python scripts/train.py --live none --set seed=$s run_name=abl_career_only_s$s views.chart_mode=off
  # ...same pattern for the other two
done
```

Collect `final.val_top1` and `alarms` from each `summary.json` into `results/ablations/ablations.csv` (mean ± std per name). Viroz uses the `career_only` and `default` checkpoints for the probe test in VI-6.

### 💾 Commit points
1. `configs: add ablation overrides`
2. `results: add ablation table (3 seeds)`

### 🧠 Theory
An **ablation** removes one part of a system and measures the change. It's how you show which parts matter, rather than claiming it. The `no_controls` run is a *negative control*: it should fail, and failing in the expected way is evidence the validation metric works.

### 🔍 Research keywords
`ablation study machine learning` · `shortcut learning Geirhos` · `negative control experiment`

---

## MA-6: Parametric study (learning rate × epochs)

### Goal
Measure how learning rate and training length affect the result, and pick the best combination **using validation only**.

### Design

| Factor | Values | Why |
|---|---|---|
| Learning rate | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | evenly spaced on a log scale; the useful range for AdamW on small models usually lies inside it |
| Epochs | 50, 100, 200, 400 | doubles each step; covers "too short" to "probably too long" |
| Seeds | 0, 1, 2 | the prototype showed ~0.1 difference between seeds, so one seed per cell can mislead |

5 × 4 × 3 = **60 runs**, early stopping **off** (so "epochs" really means the run length). About 11,250 epochs in total: roughly an hour on the M2 (the prototype did ~0.3 s per epoch on a plain CPU).

### Code

```python
# file: src/nbacnn/study.py
"""Parametric study: learning rate x epochs x seeds."""
import copy, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd
from .train import train

def run_study(base_cfg, lrs, epochs_list, seeds, out_dir="outputs/study"):
    rows = []
    for lr, epochs, seed in itertools.product(lrs, epochs_list, seeds):
        cfg = copy.deepcopy(base_cfg)
        cfg["train"].update(lr=lr, epochs=epochs)
        cfg.update(seed=seed, out_dir=out_dir, run_name=f"lr{lr:g}_ep{epochs}_s{seed}", early_stop=False)
        run_dir = Path(out_dir) / cfg["run_name"]
        if not (run_dir / "summary.json").exists():          # resume: skip runs that already finished
            t = time.time(); train(cfg); print(f"{cfg['run_name']}: {time.time() - t:.0f}s")
        s = json.load(open(run_dir / "summary.json"))
        m = pd.read_csv(run_dir / "metrics.csv")
        rows.append(dict(lr=lr, epochs=epochs, seed=seed,
                         final_val_top1=m.val_top1.iloc[-1], best_val_top1=m.val_top1.max(),
                         final_val_loss_norm=m.val_loss_norm.iloc[-1], final_train_top1=m.train_top1.iloc[-1],
                         converged_epoch=s["converged_epoch"], target_epoch=s["target_epoch"],
                         seconds=m.seconds.iloc[-1], alarms="; ".join(s["alarms"])))
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "study_runs.csv", index=False)
    summary = (df.groupby(["lr", "epochs"])
                 .agg(val_top1_mean=("final_val_top1", "mean"), val_top1_std=("final_val_top1", "std"),
                      train_top1_mean=("final_train_top1", "mean"),
                      conv_epoch=("converged_epoch", "median"), seconds=("seconds", "mean"))
                 .reset_index())
    summary["gap"] = summary.train_top1_mean - summary.val_top1_mean     # big gap = overfitting
    summary.to_csv(Path(out_dir) / "study_summary.csv", index=False)
    return df, summary

def plot_study(summary, runs_dir, out_dir):
    """Heatmap of mean val top-1 (lr x epochs) + validation curves for every learning rate."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    piv = summary.pivot(index="lr", columns="epochs", values="val_top1_mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="k" if v > np.nanmean(piv.values) else "w")
    ax.set(xlabel="epochs", ylabel="learning rate", title="Validation top-1 (mean over seeds)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_dir / "heatmap_val_top1.png", dpi=150); plt.close(fig)

    longest = summary.epochs.max()
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr in sorted(summary.lr.unique()):
        curves = [pd.read_csv(p) for p in Path(runs_dir).glob(f"lr{lr:g}_ep{longest}_s*/metrics.csv")]
        m = pd.concat(curves).groupby("epoch").val_loss_norm.mean()
        ax.plot(m.index, m.values, label=f"lr {lr:g}")
    ax.set(xlabel="epoch", ylabel="val loss (normalized)", title=f"Validation loss, {longest}-epoch runs"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "curves_by_lr.png", dpi=150); plt.close(fig)
```

```yaml
# file: configs/study_lr_epochs.yaml
base_config: configs/default.yaml
out_dir: outputs/study
figures_dir: results/study
tensorboard: true
lrs: [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]
epochs: [50, 100, 200, 400]
seeds: [0, 1, 2]
```

```python
# file: scripts/run_study.py
"""Learning rate x epochs x seeds study.  python scripts/run_study.py --study configs/study_lr_epochs.yaml"""
import argparse, yaml
import matplotlib; matplotlib.use("Agg")
from nbacnn.config import load_config
from nbacnn.study import run_study, plot_study

p = argparse.ArgumentParser()
p.add_argument("--study", default="configs/study_lr_epochs.yaml")
p.add_argument("--set", nargs="*", default=[], help="overrides for the base config")
args = p.parse_args()
s = yaml.safe_load(open(args.study))
base = load_config(s["base_config"], args.set)
base["tensorboard"] = s.get("tensorboard", True)
runs, summary = run_study(base, s["lrs"], s["epochs"], s["seeds"], out_dir=s["out_dir"])
plot_study(summary, s["out_dir"], s["figures_dir"])
print(summary.sort_values("val_top1_mean", ascending=False).round(3).to_string(index=False))
```

Run it with `caffeinate -dims python scripts/run_study.py`. If it's interrupted, run the same command again: finished runs are skipped. Watch everything live with `tensorboard --logdir outputs/study`.

**Before the full study,** do a 2-minute dry run with a tiny grid (copy the YAML, set `lrs: [1.0e-3, 3.0e-3]`, `epochs: [10, 20]`, `seeds: [0]`) to check the whole pipeline end to end.

### How to read the results and choose
1. **Heatmap:** the best cell is the highest mean validation top-1. Look for a *region* of good cells, not one lucky cell.
2. **Overlapping cells:** if two cells differ by less than about one standard deviation (`val_top1_std`), treat them as tied and pick the cheaper one (fewer epochs).
3. **`curves_by_lr.png`:** too-low learning rates are still falling at the end (under-trained); too-high ones are jagged or rise again (unstable). The best rate falls smoothly and flattens.
4. **`gap` column:** a large train−validation gap at 400 epochs means extra epochs only memorize.
5. **Alarms column:** a cell with alarms is disqualified, whatever its score.
6. Write the choice and the reasoning into `results/study/README.md`, and update `configs/default.yaml` (`train.lr`, `train.epochs`, and `convergence.target_top1` if appropriate). **BPM is not used in any of this.**

### Done when
`results/study/` contains `study_summary.csv`, both figures and the README with the chosen configuration.

### 💾 Commit points
1. `study: add learning rate x epochs runner with resume`
2. `study: add heatmap and curve plots`
3. `configs: add study grid`
4. `results: add parametric study results and choice` → tag `m6-study`.

### 🧠 Theory
- **Learning rate** is the size of each weight update. Too small: slow progress, under-trained at the end. Too large: steps overshoot, and loss bounces or blows up. It usually matters more than any other setting, which is why it's searched on a log scale (the difference between 0.0001 and 0.001 matters as much as between 0.001 and 0.01).
- **Epochs** set how long training lasts. Too few under-fits; too many can overfit, especially with only 844 players. With a cosine schedule, run length also changes *how fast* the learning rate decays, so learning rate and epochs interact. That's why it's a grid, not two separate sweeps.
- **Grid search vs. alternatives:** a grid is simple and easy to visualize as a heatmap. Random search and Bayesian optimization are more efficient for many settings but harder to present.
- **Selecting on validation, not test:** if you picked settings by test score, the test score would be optimistic, because you'd have tuned to it. Validation exists so the test stays untouched until M9.
- **Seeds and variance:** different seeds change the initial weights and batch order. Reporting mean ± std across seeds tells you whether a difference between settings is bigger than the randomness.

### 🔍 Research keywords
`hyperparameter grid search` · `learning rate range test` · `learning rate and batch size interaction` · `random seed variance deep learning` · `model selection validation set`

---

## MA-7: Final training and handoff

### Steps
1. Set the chosen `lr` and `epochs` in `configs/default.yaml`.
2. Train 5 seeds: `for s in 0 1 2 3 4; do caffeinate -i python scripts/train.py --live none --set seed=$s run_name=final_s$s; done`
3. Copy each `best.pt` to `results/final/best_seed{s}.pt` (about 80 KB each) and record validation top-1 mean ± std in `results/final/training_summary.csv`.
4. Hand off to Viroz: VI-4 embeds every player with each seed's model.

### 💾 Commit points
`results: add final 5-seed checkpoints and training summary` → tag `m8-final`.

---

## M9: Test evaluation (joint)

Sit down together and run Viroz's evaluation on the test split **once**, with the `--final` flag. Commit the outputs as they come out, and don't change anything afterwards to improve the test number; that would make it a validation number. Tag `v1.0`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.backends.mps.is_available()` is `False` | x86 Python under Rosetta, or an old macOS | reinstall with arm64 Miniforge; check `platform.machine()` |
| `TypeError: Cannot convert a MPS Tensor to float64` | a float64 tensor (often from pandas/NumPy) | `.float()` or `np.float32` when creating tensors |
| `NotImplementedError: ... not currently implemented for the MPS device` | an operation MPS doesn't support yet | `export PYTORCH_ENABLE_MPS_FALLBACK=1` (runs that one op on CPU), or use `device: cpu` |
| MPS slower than CPU | launch overhead with a tiny model (see MA-1) | use CPU; try `torch.set_num_threads(8)` |
| Loss is `nan` | learning rate too high | lower `train.lr`; check for `nan` in the inputs |
| Loss stuck at ~1.0 | not learning: views broken, or learning rate far too low | overfit-one-batch test (Integration step 3) |
| GAP alarm in a default run | shortcut or overfitting | check Viroz's view controls; try more dropout or fewer epochs |
| Mac slows during the study | thermal throttling | keep it plugged in, on a hard surface; close other apps |
| `train.lr=3e-3` override ignored | YAML reads `3e-3` as text | `load_config` converts it; in YAML files write `3.0e-3` |
