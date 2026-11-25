"""Shortcut ablations, 3 seeds each (MA-5).
   caffeinate -dims python scripts/run_ablations.py --ablations configs/ablations.yaml
   Resumable: finished runs are skipped."""
import argparse
from pathlib import Path
from nbacnn.config import load_config
from nbacnn.data import load_split
from nbacnn.study import run_ablations

p = argparse.ArgumentParser()
p.add_argument("--ablations", default="configs/ablations.yaml")
p.add_argument("--set", nargs="*", default=[], help="overrides for the base config")
args = p.parse_args()
s = load_config(args.ablations)
base = load_config(s["base_config"], args.set)
base["tensorboard"] = s.get("tensorboard", False)
data = (load_split("train", base["data_dir"]), load_split("val", base["data_dir"]))   # load the CSVs once
runs, summary = run_ablations(base, s["runs"], s["seeds"], out_dir=s["out_dir"], data=data)
out = Path(s["results_dir"]); out.mkdir(parents=True, exist_ok=True)
runs.to_csv(out / "ablations_runs.csv", index=False)
summary.to_csv(out / "ablations.csv", index=False)
cols = ["name", "n", "val_top1_mean", "val_top1_std", "best_ckpt_val_top1_mean", "max_gap_mean",
        "conv_epoch", "runs_with_alarms"]
print(summary[cols].round(3).to_string(index=False))
