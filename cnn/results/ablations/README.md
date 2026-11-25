# M5: shortcut ablations (MA-5)

`python scripts/run_ablations.py` (settings in `configs/ablations.yaml`). Date 2026-11-25, Apple M2, MPS.
Each ablation is the **M6-chosen config** (lr 3e-4, 400 epochs, early stopping off, every run trained to the end of its cosine schedule) with one change, trained with seeds 0, 1 and 2. The score is **shortcut-free validation two-halves top-1** at the last epoch (181 players; random ≈ 0.006). BPM was not used.

| Run | What changes | Val top-1 (mean ± std) | Per seed | Train top-1 | GAP alarm | Prototype (ROADMAP §9) |
|---|---|---|---|---|---|---|
| `default` | nothing | **0.722 ± 0.025** | 0.724 / 0.746 / 0.696 | 0.89 | 0 / 3 | 0.71 / 0.70 / 0.62 |
| `career_only` | `views.chart_mode=off` (no shot chart anywhere) | **0.713 ± 0.043** | 0.735 / 0.740 / 0.663 | 0.88 | 0 / 3 | 0.66 (1 seed) |
| `identical_chart` | `chart_mode=same`, `chart_noise=0` | **0.435 ± 0.070** | 0.354 / 0.470 / 0.481 | 0.59 | 3 / 3 (epochs 8–10) | 0.41 |
| `no_controls` | all shortcut controls off, identical chart | **0.175 ± 0.018** | 0.188 / 0.155 / 0.182 | 0.26 | 3 / 3 (epochs 7–10) | 0.16 |

Per run: `ablations_runs.csv`; summary: `ablations.csv`. Every final score equals the run's mean over its last 50 epochs to within 0.005, so these are settled values, not snapshots of a noisy curve. The `default` row reproduces the M6 study's chosen cell exactly, seed by seed, because training on MPS is deterministic.

## What it shows

1. **The shortcut controls are essential.** Without them, the model learns shortcuts and validation top-1 drops from 0.72 to **0.18** (−0.55). The GAP alarm fires by epoch 7–10 in every shortcut run and never in the default or chart-off runs (largest gap 0.06).
2. **The identical career shot chart alone is a strong shortcut.** It costs **0.29** top-1 (0.72 → 0.44), so the chart must be hidden in one view (`chart_mode: drop_one`).
3. **Shortcut runs get worse with more training.** `no_controls` peaks at 0.27–0.37 early on, then falls to 0.16–0.19 as it learns the shortcut. This is the signature the GAP alarm is designed to catch.
4. **The shot chart's contribution can't be measured with this metric.** `default` and `career_only` are tied (default − career_only per seed: −0.011, +0.006, +0.033; mean +0.009, well inside the seed spread). Two direct checks on the three `default` models (`last.pt`) explain why:
   - Blanking the chart at inference changes two-halves top-1 by +0.000 / +0.005 / −0.022, so it has no consistent effect. The metric is shortcut-free by design: validation half B never contains the chart, so it can only credit the chart through half A.
   - The model does use the chart, lightly. Blanking it moves each summary only slightly (cosine 0.99 to the original), but **changes about 1.2 of each player's 5 neighbours** (3.75–3.88 of 5 kept, train+val pool).

   So the chart changes roughly one comparison in five, and whether those changes are better is a question for the VI-6 probes (default vs `career_only`) and the M9 BPM test, not for two-halves accuracy.

## Earlier early-stopped table (superseded)
The first version (`early_stopped/`) used the M4 baseline config (lr 1e-3, up to 200 epochs, early stopping on): default 0.696 ± 0.025, career_only 0.698 ± 0.021, identical_chart 0.414 ± 0.020, no_controls 0.193 ± 0.072. The conclusions were the same, but the runs weren't comparable. Early stopping cut every shortcut run at epoch 26–36 while the learning rate was still at 94–97% of its peak, against 88–130 epochs for the default runs. Those shortcut scores were snapshots of a fluctuating curve (e.g. `identical_chart` seed 1: 0.431 at the stop vs 0.486 over its last 10 epochs). The table above replaces it.

## Checkpoints for VI-6 (kept locally, not committed)
`outputs/ablations_final/{default,career_only,identical_chart,no_controls}_s{0,1,2}/`, each with `config.yaml`.
- **Use `last.pt`.** It is the fully trained model the table scores. For the shortcut runs, `best.pt` is from the validation-loss minimum at epochs 1–49, before or while the shortcut was being learned (val top-1 0.08–0.55), so it is not a "shortcut model".
- `scripts/export_embeddings.py` reads each run's `config.yaml`, so `career_only` models are embedded without a shot chart, as trained.
