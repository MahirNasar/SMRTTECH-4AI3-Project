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