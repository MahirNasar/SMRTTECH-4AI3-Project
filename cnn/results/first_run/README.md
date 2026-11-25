# M4: first real training run

Date: 2026-11-25 · Apple M2, MPS (`results/benchmark.txt`) · torch 2.14.1 · commit history: see `git log` at tag `m4-first-run`.
All numbers are on the **shortcut-free validation halves** (181 players; random top-1 ≈ 0.006). BPM was not used.

## 1. Overfit check (`overfit_check/`)

Does the whole pipeline (data → views → model → loss → optimizer) learn when nothing stops it?

```bash
python scripts/train.py --live file --set run_name=overfit_check train.epochs=300 \
    train.batch_size=32 views.season_drop=0 views.weight_noise=0 views.weight_drop=0 \
    views.pos_drop=0 views.chart_mode=same views.chart_noise=0 early_stop=false
```

| Epoch | Train loss (norm.) | Val loss (norm.) | Val top-1 | Train top-1 |
|---:|---:|---:|---:|---:|
| 1 | 0.985 | 1.002 | 0.10 | 0.08 |
| 10 | 0.169 | 1.095 | 0.24 | 0.26 |
| 100 | 0.027 | 1.446 | 0.11 | 0.21 |
| 300 | **0.017** | 1.451 | 0.12 | 0.22 |

**Pass.** Training loss falls to 0.017 (1.0 = random guessing), so gradients reach the model and it can fit the training task. The GAP alarm fires at epoch 3, as expected: with every shortcut control off, the model matches halves by weight, position and the identical shot chart, which does not transfer to the shortcut-free validation halves (val top-1 stalls at ~0.12; prototype "no controls" run: 0.16). Wall time 2 min 11 s.

## 2. Baseline run (`baseline/`)

```bash
python scripts/train.py --live file --set run_name=baseline      # configs/default.yaml, seed 0
```

| Epoch | LR | Train loss | Val loss | Gap | Val top-1 | Val top-5 | Train top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2e-4 | 0.994 | 1.002 | 0.01 | 0.07 | 0.25 | 0.08 |
| 10 | 1e-3 | 0.601 | 0.550 | −0.05 | 0.40 | 0.77 | 0.45 |
| 20 | 1e-3 | 0.454 | 0.419 | −0.03 | 0.63 | 0.86 | 0.59 |
| 35 | 9e-4 | 0.380 | 0.349 | −0.03 | 0.67 | 0.90 | 0.72 |
| 50 | 9e-4 | 0.319 | 0.336 | 0.02 | 0.69 | 0.89 | 0.80 |
| 100 | 5e-4 | 0.264 | 0.294 | 0.03 | 0.72 | 0.91 | 0.89 |
| 110 (best) | 4e-4 | | 0.283 | | 0.71 | | |
| 130 (stop) | 3e-4 | 0.246 | 0.285 | 0.04 | **0.72** | 0.91 | 0.92 |

Compared with the prototype (ROADMAP §9):

| | Prototype (CPU, 3 seeds) | This run (MPS, seed 0) |
|---|---|---|
| Final val top-1 | 0.71 / 0.70 / 0.62 | **0.72** (best.pt at epoch 110: 0.71; peak 0.73 at epoch 54) |
| Target 0.65 first reached | epoch 30–35 | **epoch 31** |
| Converged (no val-loss gain > 0.005 for 20 epochs) | epoch 90–120 | best epoch 110, rule met at **epoch 130** |
| Alarms | none | **none** (max gap 0.04) |
| Train top-1 − val top-1 at end | ~0.15 | 0.19 |
| Time | 30–50 s | 45 s (incl. data loading) |

**Pass.** The pipeline reproduces the prototype on the M2. `best.pt` (lowest validation loss, epoch 110) is the checkpoint to use downstream.

### Note on the dashboard
Validation **top-5** runs along the red 0.90 line in `baseline/live.png`. That line is the too-good alarm for **top-1** only; top-5 ≈ 0.9 is normal. The alarm did not fire.

### Target recalibration
`convergence.target_top1` stays at **0.65** for now. One seed at 0.72 is within the ~0.1 seed spread, so it isn't enough evidence to raise the target. Revisit after the 3-seed default runs (MA-5) and the parametric study (MA-6).

## 3. End-to-end check: embeddings and eye test

`scripts/export_embeddings.py --checkpoint outputs/runs/baseline/best.pt` produced 1,207 × 34 embeddings and 6,035 neighbour rows (5 per player, no self-matches, searching all splits).

### Embedding method: average of the two halves (decision, 2026-11-25)
The guide embedded each player's whole career in one view (`full_view`). A pre-integration audit found that this is out of distribution for an odd/even-trained model: training and validation views **never** contain two played seasons at adjacent ages (0%), but whole careers mostly do (79% of validation seasons). On this checkpoint, on validation:

| Check | Value |
|---|---|
| odd half A → even half B (the validation metric) | top-1 0.713 |
| whole career → B (contains all of B's seasons) | top-1 **0.657**, worse than half A alone |
| cosine(whole career, own halves) vs cosine(half, half) | 0.948 / 0.943 vs 0.962 |

So players are now embedded as **h = normalize(h(odd seasons) + h(even seasons))**, each half with full information (weight, position, shot chart) and no augmentation (`embed.method: halves` in `configs/default.yaml`; `src/nbacnn/embed.py`). Each half looks like the training inputs, and together they use every season. The guide's method remains available as `--method full` for comparison; the two methods share only 2.1 of each player's 5 neighbours on this checkpoint. The choice was made on this distribution argument, **not** on BPM.

### Eye test (halves method, train/validation players only; test players are not inspected before M9)

| Player (split) | Top-5 |
|---|---|
| Paul Pierce (val) | Paul George, Jayson Tatum, Kobe Bryant, Kevin Durant, Tracy McGrady |
| Steve Nash (train) | Mo Williams, José Calderón, Greivis Vásquez, Deron Williams, Jameer Nelson |
| Stephen Curry (train) | Kyrie Irving, Damian Lillard, Ray Allen, Chauncey Billups, D'Angelo Russell |
| Ben Wallace (train) | Nic Claxton, Kwame Brown, Willie Cauley-Stein, Precious Achiuwa, Lou Amundson |
| Dirk Nowitzki (train) | Keith Van Horn, Kevin Durant, Michael Porter Jr., Charlie Villanueva, Carmelo Anthony |
| LeBron James (train) | Luka Dončić, Russell Westbrook, Dwyane Wade, Tracy McGrady, Dejounte Murray |
| Dwight Howard (train) | Rudy Gobert, DeAndre Jordan, Tyson Chandler, Evan Mobley, Anthony Randolph |
| Kawhi Leonard (train) | Tracy McGrady, Rudy Gay, Anthony Edwards, Kobe Bryant, Gordon Hayward |
| Nikola Jokić (val) | Luka Dončić, Domantas Sabonis, Karl-Anthony Towns, DeMarcus Cousins, LeBron James |

These overlap with the prototype's lists (Pierce → George, Tatum, McGrady, Kobe; Nash → Mo Williams, Deron Williams, Calderón; Curry → Irving, Lillard; Wallace → Cauley-Stein, Claxton). This is a sanity check, not an evaluation; the embeddings files themselves are produced at M5 (VI-4).
