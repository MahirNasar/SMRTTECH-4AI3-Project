"""Experiment runners: shortcut ablations (MA-5) and the learning rate x epochs x seeds study (MA-6).
Both resume: a run whose folder already has summary.json is not trained again."""
import copy, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd
from .config import apply_overrides
from .train import train

def _run(cfg, data=None):
    """Train cfg unless it already finished; return its run folder."""
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    if not (run_dir / "summary.json").exists():
        t = time.time()
        train(cfg, data=data)
        print(f"{cfg['run_name']}: {time.time() - t:.0f}s", flush=True)
    return run_dir

def run_record(run_dir):
    """One row of results for a finished run."""
    s = json.load(open(Path(run_dir) / "summary.json"))
    m = pd.read_csv(Path(run_dir) / "metrics.csv")
    best = m[m.epoch == s["best_epoch"]].iloc[0]                 # the epoch best.pt was saved at
    return dict(final_val_top1=m.val_top1.iloc[-1], final_val_top5=m.val_top5.iloc[-1],
                best_ckpt_val_top1=best.val_top1, peak_val_top1=m.val_top1.max(),
                final_val_loss_norm=m.val_loss_norm.iloc[-1], final_train_top1=m.train_top1.iloc[-1],
                max_gap=(m.val_loss_norm - m.train_loss_norm).max(), epochs_run=len(m),
                best_epoch=s["best_epoch"], converged_epoch=s["converged_epoch"], target_epoch=s["target_epoch"],
                seconds=m.seconds.iloc[-1], n_alarms=len(s["alarms"]), alarms="; ".join(s["alarms"]))

def _summarize(df, keys):
    out = (df.groupby(keys, sort=False)
             .agg(n=("seed", "size"),
                  val_top1_mean=("final_val_top1", "mean"), val_top1_std=("final_val_top1", "std"),
                  best_ckpt_val_top1_mean=("best_ckpt_val_top1", "mean"),
                  val_top5_mean=("final_val_top5", "mean"), train_top1_mean=("final_train_top1", "mean"),
                  max_gap_mean=("max_gap", "mean"), conv_epoch=("converged_epoch", "median"),
                  target_epoch=("target_epoch", "median"), runs_with_alarms=("n_alarms", lambda x: int((x > 0).sum())),
                  seconds=("seconds", "mean"))
             .reset_index())
    out["gap"] = out.train_top1_mean - out.val_top1_mean            # big gap = overfitting
    return out

def run_ablations(base_cfg, runs, seeds, out_dir="outputs/ablations", data=None):
    """runs: {name: [override, ...]}. Trains every (name, seed) with the base config's early stopping."""
    rows = []
    for name, overrides in runs.items():
        for seed in seeds:
            cfg = apply_overrides(copy.deepcopy(base_cfg), overrides or [])
            cfg.update(seed=seed, out_dir=str(out_dir), run_name=f"{name}_s{seed}")
            rows.append(dict(name=name, seed=seed, overrides=" ".join(overrides or []), **run_record(_run(cfg, data))))
    df = pd.DataFrame(rows)
    return df, _summarize(df, ["name"])

def run_study(base_cfg, lrs, epochs_list, seeds, out_dir="outputs/study", data=None):
    """Grid of learning rate x epochs x seeds, early stopping off (so 'epochs' is the real run length)."""
    rows = []
    for lr, epochs, seed in itertools.product(lrs, epochs_list, seeds):
        cfg = copy.deepcopy(base_cfg)
        cfg["train"].update(lr=lr, epochs=epochs)
        cfg.update(seed=seed, out_dir=str(out_dir), run_name=f"lr{lr:g}_ep{epochs}_s{seed}", early_stop=False)
        rows.append(dict(lr=lr, epochs=epochs, seed=seed, **run_record(_run(cfg, data))))
    df = pd.DataFrame(rows)
    return df, _summarize(df, ["lr", "epochs"])

def plot_study(summary, runs_dir, out_dir):
    """Heatmap of mean val top-1 (lr x epochs, with std) + validation curves for every learning rate."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    piv = summary.pivot(index="lr", columns="epochs", values="val_top1_mean")
    sd = summary.pivot(index="lr", columns="epochs", values="val_top1_std")
    alarms = summary.pivot(index="lr", columns="epochs", values="runs_with_alarms")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            label = f"{v:.2f}\n±{sd.values[i, j]:.2f}" + ("\n(alarm)" if alarms.values[i, j] else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="k" if v > np.nanmean(piv.values) else "w")
    ax.set(xlabel="epochs", ylabel="learning rate", title="Validation top-1 (mean ± std over seeds)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_dir / "heatmap_val_top1.png", dpi=150); plt.close(fig)

    longest = summary.epochs.max()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    for lr in sorted(summary.lr.unique()):
        curves = [pd.read_csv(p) for p in Path(runs_dir).glob(f"lr{lr:g}_ep{longest}_s*/metrics.csv")]
        if not curves:
            continue
        m = pd.concat(curves).groupby("epoch")[["val_loss_norm", "val_top1"]].mean()
        a.plot(m.index, m.val_loss_norm, label=f"lr {lr:g}")
        b.plot(m.index, m.val_top1, label=f"lr {lr:g}")
    a.set(xlabel="epoch", ylabel="val loss (normalized)", title=f"Validation loss, {longest}-epoch runs (mean of seeds)")
    b.set(xlabel="epoch", ylabel="val top-1", title=f"Validation top-1, {longest}-epoch runs (mean of seeds)", ylim=(0, 1))
    a.legend(); b.legend()
    fig.tight_layout(); fig.savefig(out_dir / "curves_by_lr.png", dpi=150); plt.close(fig)
