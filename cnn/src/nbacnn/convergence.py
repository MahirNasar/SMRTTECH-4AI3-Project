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