# M5: shortcut ablations (MA-5)

`python scripts/run_ablations.py` (settings in `configs/ablations.yaml`). Date 2026-11-25, Apple M2, MPS.
Each ablation is the M4 baseline config (lr 1e-3, up to 200 epochs, early stopping on) with one change, trained with seeds 0, 1 and 2. The score is **shortcut-free validation two-halves top-1** at the last epoch (181 players; random ≈ 0.006). BPM was not used.

| Run | What changes | Val top-1 (mean ± std) | Per seed | GAP alarm | Prototype (ROADMAP §9) |
|---|---|---|---|---|---|
| `default` | nothing | **0.696 ± 0.025** | 0.724 / 0.691 / 0.674 | 0 / 3 | 0.71 / 0.70 / 0.62 |
| `career_only` | `views.chart_mode=off` (no shot chart anywhere) | **0.698 ± 0.021** | 0.707 / 0.713 / 0.674 | 0 / 3 | 0.66 (1 seed) |
| `identical_chart` | `chart_mode=same`, `chart_noise=0` | **0.414 ± 0.020** | 0.420 / 0.431 / 0.392 | 3 / 3 (epochs 5–6) | 0.41 |
| `no_controls` | all shortcut controls off, identical chart | **0.193 ± 0.072** | 0.193 / 0.265 / 0.122 | 3 / 3 (epoch 5) | 0.16 |

Full per-run numbers: `ablations_runs.csv`; summary: `ablations.csv`.

## What it shows

1. **The shortcut controls are essential.** With no controls, the model learns shortcuts (weight, position, identical chart) and validation top-1 drops from 0.70 to 0.19. The GAP alarm fires by epoch 5–6 in every shortcut run and never in the default runs (largest gap 0.047). This matches the prototype.
2. **The career-level shot chart is a shortcut when it is identical in both views.** Showing the same chart to both halves costs about 0.28 top-1 (0.70 → 0.41), so it has to be hidden in one view (`chart_mode: drop_one`).
3. **The shot chart adds no measurable two-halves accuracy.** `career_only` (0.698) and `default` (0.696) are tied; the difference is far smaller than the seed spread. The prototype's single-seed gap (0.71 vs 0.66) was seed noise.

Caveat on point 3: the validation metric is built to be shortcut-free, so validation half B **never** contains the shot chart. The chart can only help through half A, so this metric cannot measure everything the chart contributes to the final comparisons (which embed both halves with the chart). The VI-6 linear probes (default vs career_only checkpoints) and the M9 BPM evaluation are where the chart's value shows up, or doesn't.

## Checkpoints for VI-6
Kept locally (not committed): `outputs/ablations/{default,career_only,no_controls}_s{0,1,2}/best.pt`, each with its `config.yaml`. `scripts/export_embeddings.py` reads that `config.yaml`, so `career_only` models are embedded without a shot chart, as trained.
