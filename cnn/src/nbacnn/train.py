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