"""Learning rate x epochs x seeds study (MA-6).
   caffeinate -dims python scripts/run_study.py --study configs/study_lr_epochs.yaml
   Resumable: run the same command again after an interruption and finished runs are skipped."""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
from nbacnn.config import load_config
from nbacnn.data import load_split
from nbacnn.study import run_study, plot_study

p = argparse.ArgumentParser()
p.add_argument("--study", default="configs/study_lr_epochs.yaml")
p.add_argument("--set", nargs="*", default=[], help="overrides for the base config")
args = p.parse_args()
s = load_config(args.study)
base = load_config(s["base_config"], args.set)
base["tensorboard"] = s.get("tensorboard", True)
data = (load_split("train", base["data_dir"]), load_split("val", base["data_dir"]))   # load the CSVs once
runs, summary = run_study(base, s["lrs"], s["epochs"], s["seeds"], out_dir=s["out_dir"], data=data)
fig_dir = Path(s["figures_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
for d in (Path(s["out_dir"]), fig_dir):
    runs.to_csv(d / "study_runs.csv", index=False)
    summary.to_csv(d / "study_summary.csv", index=False)
plot_study(summary, s["out_dir"], fig_dir)
cols = ["lr", "epochs", "val_top1_mean", "val_top1_std", "best_ckpt_val_top1_mean", "gap", "max_gap_mean",
        "conv_epoch", "runs_with_alarms", "seconds"]
print(summary.sort_values("val_top1_mean", ascending=False)[cols].round(3).to_string(index=False))
