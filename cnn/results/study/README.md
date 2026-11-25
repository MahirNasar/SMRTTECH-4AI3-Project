# M6: parametric study, learning rate × epochs × seeds (MA-6)

`caffeinate -dims python scripts/run_study.py --study configs/study_lr_epochs.yaml`. Date 2026-11-25, Apple M2, MPS.
5 learning rates × 4 run lengths × 3 seeds = **60 runs**, early stopping **off** (so "epochs" is the real run length), everything else as in `configs/default.yaml` at the time (batch 128, warmup 5 epochs then cosine decay to 0, τ 0.1, all shortcut controls on). 11,250 epochs in 37 min of training.
Score: **shortcut-free validation two-halves top-1 at the last epoch** (181 players; random ≈ 0.006). **BPM was not used.**

## Results

Mean ± std over seeds 0/1/2 (`study_summary.csv`; every run in `study_runs.csv`; `heatmap_val_top1.png`, `curves_by_lr.png`):

| lr \ epochs | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| 1e-4 | 0.341 ± 0.038 | 0.477 ± 0.039 | 0.610 ± 0.043 | 0.669 ± 0.006 |
| 3e-4 | 0.549 ± 0.056 | 0.652 ± 0.042 | 0.681 ± 0.025 | **0.722 ± 0.025** |
| 1e-3 | 0.648 ± 0.039 | 0.685 ± 0.029 | 0.687 ± 0.031 | 0.691 ± 0.010 |
| 3e-3 | 0.626 ± 0.008 | 0.665 ± 0.003 | 0.683 ± 0.026 | 0.674 ± 0.006 |
| 1e-2 | 0.591 ± 0.035 | 0.606 ± 0.046 | 0.615 ± 0.031 | 0.622 ± 0.030 |

No run raised an alarm (GAP or too-good), and no run produced NaN.

## Reading it (the guide's six checks)

1. **Heatmap.** The best cell is **lr 3e-4, 400 epochs: 0.722 ± 0.025**. Around it is a broad plateau of good cells at lr 1e-3 (0.685–0.691 for 100–400 epochs) and lr 3e-3 / 3e-4 at 200 epochs (0.681–0.683).
2. **Is it a real win or a tie?** It beats every other cell by more than one standard deviation, and it does so **in every seed**. Seeds share the same initial weights across cells, so this is a paired comparison:

   | 3e-4 × 400 minus | seed 0 | seed 1 | seed 2 | mean |
   |---|---|---|---|---|
   | 1e-3 × 400 | +0.039 | +0.044 | +0.011 | +0.031 |
   | 1e-3 × 200 | +0.011 | +0.050 | +0.044 | +0.035 |
   | 1e-3 × 100 | +0.006 | +0.077 | +0.028 | +0.037 |
   | 3e-3 × 200 | +0.011 | +0.077 | +0.028 | +0.039 |

   With only 3 seeds this is modest evidence; M8's 5 seeds will confirm or revise it.
3. **Curves (`curves_by_lr.png`).** 1e-4 is still improving at epoch 400 (under-trained). 1e-2 levels off lowest (0.62), with the noisiest curve. 1e-3 and 3e-3 reach their top-1 by about epoch 50–100 and then stay flat or drift down slightly. 3e-4 rises more slowly, passes 1e-3 around epoch 200 and is flat from about epoch 250, so it is not still climbing at the end of the grid.
   Note that lr 1e-3 reaches a *lower* validation loss (≈ 0.265) than 3e-4 (≈ 0.29) but a lower top-1. The loss and the matching accuracy disagree slightly; selection follows top-1, as planned.
4. **Overfitting gap.** Train top-1 − val top-1 is 0.16 for the chosen cell, against 0.27 for 1e-3 × 400 and 0.34 for 1e-2. Its loss gap peaks at 0.041, far below the 0.30 alarm.
5. **Alarms.** None in any cell, so no cell is disqualified.
6. **Cost.** 80 s per run against 20 s for 1e-3 × 100. That is negligible here (5 final seeds = 7 min), so it doesn't outweigh a consistent 0.03–0.04 gain.

## Choice

**lr = 3e-4, 400 epochs, early stopping off**, now set in `configs/default.yaml` (M4 baseline: lr 1e-3, 200 epochs, early stopping on).

Consequences for M8 (final 5-seed training):
- **Early stopping must stay off.** With this learning rate, the plateau rule (no val-loss gain > 0.005 for 20 epochs) fires at epochs 129–182, long before the cosine schedule ends. Stopping there loses up to 0.03 top-1 (seed 2: 0.641 at the rule vs 0.696 at epoch 400).
- **Ship `last.pt`, not `best.pt`.** The study selected on last-epoch top-1, so `last.pt` is the checkpoint whose quality it measured (0.724 / 0.746 / 0.696 here). The lowest-val-loss epoch gives about the same top-1 (0.729 / 0.746 / 0.674) but is a different selection rule. This replaces "copy each best.pt" in MA-7.
- **`convergence.target_top1` stays 0.65.** The chosen config clears it in every seed (by epoch 56–62). 0.70 is not reliable yet, because seed 2 ends at 0.696; revisit after M8.

## Caveat
The best cell sits at the edge of the grid (the longest runs). The flat top-1 curve from about epoch 250 suggests longer runs would add little, so the grid was not extended. Extending it to look for a slightly better cell would also start to tune to the validation set.
