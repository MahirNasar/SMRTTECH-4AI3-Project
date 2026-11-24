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