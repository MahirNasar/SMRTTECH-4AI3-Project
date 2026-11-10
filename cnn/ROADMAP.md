# CNN Development Roadmap

**Project:** NBA player comparison: does a contrastive CNN find more meaningful comparisons than KNN?
**Team:** Mahir (training engine and experiments; owns the M2 training machine) · Viroz (data, model and evaluation)
**Scope:** the CNN only. The KNN baseline is built separately; the two meet at the shared evaluation in Milestone 9.

The detailed walkthroughs are in:
- [`docs/EXECUTION_MAHIR.md`](docs/EXECUTION_MAHIR.md)
- [`docs/EXECUTION_VIROZ.md`](docs/EXECUTION_VIROZ.md)

---

## 1. What "done" means

By the end of this roadmap we have:

1. A trained encoder that turns every eligible player (1,207) into a 32-number summary `h`.
2. A top-5 comparison list for every player, using cosine similarity between summaries.
3. A **live training dashboard** showing train/validation loss against a **convergence target**.
4. A **parametric study** of learning rate × epochs, with a chosen configuration and the evidence for it.
5. A **test-set score** on the BPM hidden-clue test, in the same format as the KNN team's, with random and same-position baselines and confidence intervals.
6. Diagnostics showing what the model learned and that it didn't take a shortcut.

---

## 2. Starting point: the cleaned dataset

`nba_model_data.zip` → `data/processed/` in the repo.

| File | Shape | Contents |
|---|---|---|
| `player_splits.csv` | 1,207 rows | `player_id`, `split` (train 844 / val 181 / test 182) |
| `cnn/cnn_stats_{split}.csv` | 26 rows per player | ages 18–43 × 23 channels (16 z-scored stats, 5 position one-hots, `mp_scaled`, `played_mask`) |
| `cnn/cnn_shots_{split}.csv` | 600 rows per player | 24 × 25 court cells × 2 channels (`freq_rel`, `eff_rel`), **career-level** |
| `cnn/cnn_shots_train_scaler.csv` | 2 rows | train mean/std for the two shot channels (the shot CSVs are **not yet standardized**) |
| `hidden_clue_bpm.csv` | 8,243 rows | BPM per qualified season. **Evaluation only; never read by training code** |
| `knn/*` | | KNN team's inputs (not used by the CNN) |

Checks already run on this data: no missing values; every player has exactly 26 age rows and 600 shot cells; the stats and shot files contain the same players in every split; every player has 2–23 qualified seasons (median 6).

### Three issues found in the data that shape the plan

1. **Weight is a shortcut.** Within a player, `weight_z` barely changes (average spread 0.05) while across players it spreads by about 1.0. In the prototype, **weight alone matched 52% of validation players** to their other half. → Viroz adds weight noise and weight/position dropout to the views (VI-2).
2. **The shot chart is career-level, so it's identical in both views.** If both halves carry the same chart, the model matches them by chart and learns little else (prototype: the shortcut-free score fell from 0.71 to 0.41). → Hide the chart in one of the two views at random and add noise. Proper fix: per-season shot grids so each half gets its own chart (stretch task VI-7, needs the pipeline's per-season shot table).
3. **Shot values are tiny and unscaled** (`freq_rel` std ≈ 0.005). → Standardize with the train scaler when loading (VI-1).

Because of 1 and 2, **validation uses shortcut-free halves**: half A is the odd seasons with everything; half B is the even seasons with weight, position and the shot chart hidden. B can only be matched to A through the season-by-season stats, so the validation score can't be inflated by a shortcut.

The prototype results that back these up are in [§9](#9-prototype-results-what-we-already-know).

---

## 3. The model in one picture

```
            ┌──────────── view A (odd seasons) ────────────┐
player  ──► │  career grid 23×26  ─► Conv1d ─► Conv1d ─► masked pool ─► 64 ┐
            │  shot chart 2×24×25 ─► Conv2d ─► pool ─► Conv2d ─► pool ─► 32 ┴► Linear ─► h (32) ─► head ─► z
            └──────────────────────────────────────────────┘
            same for view B (even seasons) ─────────────────────────────────────────────────────────────► z'
                                              NT-Xent loss: z and z' should match, other players shouldn't
After training: full career ─► h  ─►  cosine similarity  ─►  5 nearest players  ─►  BPM hidden-clue test
```

18,608 trainable weights (16,496 without the projection head, which is used only for training).

---

## 4. Who does what

| | **Mahir: training engine and experiments** | **Viroz: data, model and evaluation** |
|---|---|---|
| Owns | `device.py`, `losses.py`, `metrics.py`, `convergence.py`, `train.py`, `live_plot.py`, `study.py`, configs | `data.py`, `views.py`, `model.py`, `embed.py`, `evaluate.py`, `diagnostics.py` |
| Builds against | fake tensors with the agreed shapes, until Viroz's pieces land | the real data; no training loop needed until integration |
| Machine | M2 (all real training runs) | any laptop (data, model shapes, evaluation are light) |

The split works because of the **interface contract** below. As long as both sides keep to it, neither waits on the other until integration.

### Interface contract (agree on this on day 1; change it only with a PR both approve)

```python
# --- Viroz provides ------------------------------------------------------------
# nbacnn/data.py
@dataclass
class PlayerTensors:
    ids:   list[str]          # player_id for each row
    grid:  Tensor             # (P, 23, 26) float32: channels = stats, length = ages 18–43
    mask:  Tensor             # (P, 26)     float32: 1 = qualified season at that age
    chart: Tensor             # (P, 2, 24, 25) float32: standardized freq_rel, eff_rel
    def to(self, device) -> "PlayerTensors": ...
    def subset(self, idx: Tensor) -> "PlayerTensors": ...
def load_split(split: str, data_dir: str = "data/processed") -> PlayerTensors: ...

# nbacnn/views.py
View = namedtuple("View", "grid mask chart")      # grid (B,23,26), mask (B,26), chart (B,2,24,25)
def make_views(batch: PlayerTensors, cfg: dict, gen: torch.Generator | None = None,
               train: bool = True) -> tuple[View, View]: ...
    # train=True : random augmented halves + shortcut controls (for the loss)
    # train=False: fixed shortcut-free halves (for validation metrics)
def full_view(batch: PlayerTensors) -> View: ...  # whole career, no augmentation (for embedding)

# nbacnn/model.py
class PlayerEncoder(nn.Module):                  # forward(grid, mask, chart) -> (h, z)
    ...                                          # h, z: (B, 32), both L2-normalized

# --- Mahir provides ------------------------------------------------------------
# nbacnn/losses.py
def nt_xent(z_a: Tensor, z_b: Tensor, tau: float) -> Tensor: ...   # scalar
def chance_loss(n: int) -> float: ...                                # ln(2n - 1)

# nbacnn/metrics.py
def two_halves(model, data: PlayerTensors, cfg: dict) -> dict: ...   # {"top1", "top5", "loss_norm"}

# nbacnn/train.py
def train(cfg: dict) -> Path: ...    # writes outputs/runs/<name>/{config.yaml, metrics.csv, best.pt, last.pt}
```

The config file `configs/default.yaml` is shared too. Viroz owns the `model:` and `views:` blocks; Mahir owns `train:` and `convergence:`.

---

## 5. Milestones

Tags mark the end of each milestone on `main` (`git tag m3-model && git push --tags`).

Personal task IDs (`MA-` for Mahir, `VI-` for Viroz) match the sections of each execution guide.

| # | Milestone | Tasks | Depends on | Done when | Tag |
|---|---|---|---|---|---|
| **M0** | Repo, folder structure, environments, data committed | both (Mahir scaffolds) | n/a | both machines run `pytest` and `python -c "import nbacnn"`; Mahir's prints `mps` | `m0-setup` |
| **M1** | Foundations | VI-1 data → tensors · MA-1 device + benchmark · MA-2 loss | M0 | `load_split` passes shape tests; `nt_xent` passes sanity tests; CPU vs MPS benchmark recorded | `m1-foundations` |
| **M2** | Views and engine | VI-2 views + shortcut controls · MA-3 training engine on fake data | M1 | view tests pass; fake-data training runs end to end and writes checkpoints | `m2-engine` |
| **M3** | Model and monitoring | VI-3 encoder · MA-4 live visualizer + convergence target | M2 | model shape and masking tests pass; live plot updates each epoch with the target line | `m3-model` |
| **M4** | **Integration**: first real training run | **joint session** | M3 | overfit-one-batch test passes; a baseline run reaches the convergence rule; run summary committed to `results/` | `m4-first-run` |
| **M5** | Retrieval and ablations | VI-4 embeddings + top-5 · MA-5 shortcut ablation runs | M4 | `embeddings.csv` + `neighbours.csv` for all 1,207 players; ablation table filled | `m5-retrieval` |
| **M6** | Parametric study: learning rate × epochs × seeds | MA-6 | M4 | 60 runs logged; heatmap and curve overlays; best config chosen **by validation only** | `m6-study` |
| **M7** | Evaluation and diagnostics | VI-5 BPM evaluator + baselines · VI-6 diagnostics | M5 | evaluator passes its sanity checks; UMAP, probes and neighbour tables produced | `m7-eval` |
| **M8** | Final training: best config × 5 seeds | MA-7 | M6, M7 | 5 checkpoints + embeddings; seed spread reported | `m8-final` |
| **M9** | **Test evaluation, run once**, compared with KNN | **joint session** | M8, KNN team | `results/final/` has scores, CIs, figures; joint check with the KNN team | `v1.0` |

**VI-7 (stretch):** per-season shot charts, so each view gets its own chart. Do it during M5–M7 if the pipeline's per-season shot table is available. Otherwise keep the "hide in one view" control and say so in the report.

---

## 6. Timeline

Suggested dates for the current week (starting Mon, Nov 9) onward. Shift them to fit course deadlines; keep the order.

| Week | Dates | Mahir | Viroz | Joint |
|---|---|---|---|---|
| 1 | Nov 9 – 15 | M0 scaffold; MA-1 environment + benchmark; MA-2 loss | VI-1 `load_split` + tests, Pierce sanity plot | M0 call: agree on interface contract and config |
| 2 | Nov 16 – 22 | MA-3 training engine on fake data | VI-2 views + shortcut controls | mid-week check: shapes still match |
| 3 | Nov 23 – 29 | MA-4 live plot + convergence target | VI-3 encoder | **M4 integration session** (pair-program, ~2 h) |
| 4 | Nov 30 – Dec 6 | MA-5 ablations, MA-6 parametric study | VI-4 embeddings + retrieval, VI-5 evaluator | review study results together |
| 5 | Dec 7 – 13 | MA-7 final 5-seed training | VI-6 diagnostics, VI-7 if feasible | **M9 test evaluation**, figures, write-up |

---

## 7. Training targets (summary)

All loss values are reported as **normalized loss = loss ÷ ln(2N − 1)**, where N is the number of players in the batch. 1.0 means random guessing and 0 means perfect matching, so runs with different batch sizes stay comparable.

| Signal | Definition | Value |
|---|---|---|
| **Converged** | validation normalized loss hasn't improved by more than 0.005 for 20 epochs | stop (or record the epoch, in the study); prototype converged around epoch 90–120 |
| **Performance target** | shortcut-free validation two-halves top-1 accuracy (181 players; random = 0.006) | **≥ 0.65**, stretch 0.75. Prototype reached 0.62–0.71 over 3 seeds; recalibrate at M4 |
| **Gap alarm** (shortcut or overfitting) | validation normalized loss − train normalized loss > 0.30 | fired at epoch 6 in both shortcut prototypes; never in the default runs |
| **Too-good alarm** | validation top-1 ≥ 0.90 | shortcut-free halves shouldn't get this high; check that view B really hides weight, position and the chart |

Details and reasoning: `docs/EXECUTION_MAHIR.md`, task MA-4.

## 8. Parametric study (summary)

- **Learning rate:** 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 (evenly spaced on a log scale).
- **Epochs:** 50, 100, 200, 400 (each trained separately, because the cosine schedule depends on the total).
- **Seeds:** 3 per cell → 5 × 4 × 3 = **60 runs**. The prototype took about 0.3 s per epoch on a plain CPU, so the whole study (11,250 epochs) is roughly an hour on the M2.
- **Selection:** by validation two-halves top-1 (mean over seeds). **BPM is never used to choose settings.**

Details: `docs/EXECUTION_MAHIR.md`, task MA-6.

---

## 9. Prototype results: what we already know

A CPU prototype using the code in the execution guides was run on this data (lr 1e-3, batch 128, 100–200 epochs, about 30–50 s per run). Score = shortcut-free validation top-1 (181 players; random ≈ 0.006).

| Run | Validation top-1 | What it shows |
|---|---|---|
| **Default config** (all shortcut controls), seeds 0 / 1 / 2 | **0.71 / 0.70 / 0.62** | the method works; seeds vary by ~0.1, so always run several |
| Career grid only (no shot chart) | 0.66 | the shot chart adds a little on top of the stats |
| Identical shot chart in both views | 0.41 (gap alarm at epoch 6) | the chart shortcut costs a lot |
| No shortcut controls at all | 0.16 (gap alarm at epoch 6) | the model learns shortcuts instead of basketball |
| Weight channel only, visible in both halves | 0.52 | weight alone identifies half the players |

Eye test (validation players, default run; not evaluated, just a sanity check):
- **Paul Pierce** → Paul George, Jayson Tatum, Antoine Walker, Tracy McGrady, Kobe Bryant
- **Steve Nash** → Mo Williams, Deron Williams, José Calderón, J.J. Barea, Jarrett Jack
- **Stephen Curry** → Desmond Bane, Khris Middleton, Damian Lillard, Kyrie Irving, Mo Williams
- **Ben Wallace** → Willie Cauley-Stein, Khem Birch, Nic Claxton, Moussa Diabaté, Cody Zeller

What this means for the plan:
- The shortcut controls aren't optional. They're the difference between 0.16 and 0.71.
- The target of 0.65 is reachable but not guaranteed; the parametric study should push it up.
- Training is fast, so the bottleneck is our time, not compute. Seeds and ablations are cheap.
- The BPM evaluator was checked too: random embeddings score the same as the random baseline, and an "oracle" built from BPM itself scores much better, so the evaluator measures what it should.

---

## 10. Repository folder structure

The repo currently holds the front end in `project/`. Leave it where it is (or rename it to `frontend/` in a separate PR) and add:

```
SMRTTECH-4AI3-Project/
├── README.md                     # project overview + links to the roadmaps
├── .gitignore                    # see below
├── project/                      # existing front end (unchanged)
├── data/
│   ├── README.md                 # where the data came from, file list, checksums
│   └── processed/                # contents of nba_model_data.zip (committed, ~38 MB)
│       ├── player_splits.csv
│       ├── hidden_clue_bpm.csv
│       ├── cnn/                  # cnn_stats_*, cnn_shots_*, cnn_shots_train_scaler.csv
│       └── knn/                  # KNN team inputs
├── knn/                          # KNN team's code (owned separately)
└── cnn/
    ├── ROADMAP.md                # this file
    ├── README.md                 # how to install and run
    ├── docs/
    │   ├── EXECUTION_MAHIR.md
    │   └── EXECUTION_VIROZ.md
    ├── environment.yml           # conda env (arm64), pinned versions
    ├── pyproject.toml            # makes `nbacnn` importable: pip install -e .
    ├── configs/
    │   ├── default.yaml          # shared: model, views, train, convergence
    │   ├── study_lr_epochs.yaml  # parametric study grid
    │   └── ablations/            # career_only.yaml, no_shortcut_controls.yaml, ...
    ├── src/nbacnn/
    │   ├── __init__.py
    │   ├── data.py               # V  load_split, PlayerTensors
    │   ├── views.py              # V  make_views, full_view, shortcut controls
    │   ├── model.py              # V  PlayerEncoder
    │   ├── config.py             # M  YAML loader with command-line overrides
    │   ├── fake.py               # M  fake players with real shapes (engine tests)
    │   ├── device.py             # M  pick mps/cpu, seeding
    │   ├── losses.py             # M  nt_xent, chance_loss
    │   ├── metrics.py            # M  two_halves accuracy, normalized loss
    │   ├── convergence.py        # M  plateau detection, target, alarms
    │   ├── train.py              # M  training loop, checkpoints, logging
    │   ├── live_plot.py          # M  live loss dashboard
    │   ├── study.py              # M  parametric study runner + summary
    │   ├── embed.py              # V  embeddings + top-5 neighbours
    │   ├── evaluate.py           # V  BPM hidden-clue score, baselines, bootstrap CI
    │   └── diagnostics.py        # V  UMAP, probes, neighbour tables
    ├── scripts/                  # thin command-line entry points
    │   ├── benchmark_device.py   # M
    │   ├── train.py              # M  python scripts/train.py --config configs/default.yaml
    │   ├── run_study.py          # M
    │   ├── export_embeddings.py  # V
    │   └── evaluate.py           # V
    ├── tests/
    │   ├── test_data.py          # V
    │   ├── test_views.py         # V
    │   ├── test_model.py         # V
    │   ├── test_losses.py        # M
    │   └── test_train_smoke.py   # M  2-epoch run on fake data
    ├── notebooks/                # exploration only; real logic lives in src/
    │   ├── 01_data_check.ipynb       # V
    │   ├── 02_live_training.ipynb    # M
    │   ├── 03_study_analysis.ipynb   # M
    │   └── 04_embeddings_eval.ipynb  # V
    ├── outputs/                  # NOT committed: TensorBoard runs, checkpoints, scratch
    └── results/                  # committed: small final CSVs + figures for the report
        ├── benchmark.txt         # CPU vs MPS timing on the M2
        ├── first_run/            # integration run (M4)
        ├── study/                # study_summary.csv, heatmaps, curve overlays, README with the choice
        ├── ablations/
        ├── eval_sanity/          # evaluator sanity checks (random / oracle)
        ├── diagnostics/          # probes, UMAP maps, neighbour tables
        └── final/                # best_seed*.pt, embeddings_*.csv, neighbours_*.csv, test scores
```

### `.gitignore` additions

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
*.egg-info/
# macOS
.DS_Store
# training outputs (large / machine-specific)
cnn/outputs/
*.pt
!cnn/results/final/*.pt
```

Commit only the final checkpoint (`cnn/results/final/best_seed*.pt`, ~80 KB each). Keep every other checkpoint and TensorBoard log local.

---

## 11. Git workflow

- **`main` always works.** No direct pushes once M0 is merged; everything goes through a pull request the other person reviews (even a quick skim).
- **Branch per milestone:** `mahir/m1-device-loss`, `viroz/v2-views`, `both/m4-integration`.
- **Commit small and often**, at the points each execution guide marks with 💾. One idea per commit.
- **Commit messages:** `area: what changed`, e.g. `views: add odd/even season split`, `train: log normalized loss to TensorBoard`.
- **Push at least once a day** while working, even on unfinished branches, so nothing lives only on one laptop.
- **Tag** the merge commit at the end of each milestone (table in §5).
- **Never commit:** `cnn/outputs/`, checkpoints other than the final ones, anything with credentials, or notebook outputs over ~1 MB (clear outputs before committing).
- **Config changes:** if a change touches the interface contract or the other person's config block, mention them in the PR.

---

## 12. Risks and what we do about them

| Risk | Sign | Response |
|---|---|---|
| Shortcut learning (weight, position, identical shot chart) | gap alarm: train loss keeps falling while validation loss rises; shortcut-free validation top-1 stalls low | shortcut controls in views (VI-2); shortcut-free validation; ablation runs (MA-5) |
| Overfitting (844 training players) | gap alarm; train top-1 far above validation top-1 | dropout, weight decay, season dropout, early stopping, small model |
| MPS quirks (float64, missing ops, slower than CPU for tiny models) | errors mentioning MPS; benchmark shows CPU faster | float32 everywhere; `PYTORCH_ENABLE_MPS_FALLBACK=1`; use whichever device the benchmark favours |
| Seed-to-seed noise hides real differences | the spread across seeds is as big as the difference between settings | 3 seeds in the study, 5 in the final; report mean ± std |
| Test-set leakage | anyone looks at test BPM before M9 | the evaluator refuses `split="test"` unless `--final` is passed; M9 run once, jointly |
| CNN doesn't beat KNN | M9 result | still a valid finding; the diagnostics explain why |
