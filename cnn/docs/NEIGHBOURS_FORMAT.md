# Shared `neighbours.csv` format (CNN and KNN)

Both models hand the BPM hidden-clue evaluator (VI-5, `scripts/evaluate.py`) a file in exactly this format, so the same code scores both and the comparison is fair.

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | `player_id` from `data/processed/player_splits.csv` |
| `rank` | int 1–5 | 1 = most similar |
| `neighbour_id` | string | `player_id` of the match |
| `similarity` | float | cosine similarity (higher = closer); informational, not used for scoring |

Rules:
- **All 1,207 players** appear as `query_id`, with **exactly 5 rows each** (6,035 rows in total).
- The search pool is **all players in all splits**: a test player's neighbours may come from train. This is not leakage, because no model sees BPM.
- A player is **never his own neighbour**.
- Rows for one query are ordered by rank, so similarity does not increase from rank 1 to 5.

CNN files: `scripts/export_embeddings.py` writes `neighbours_<tag>.csv` (and `embeddings_<tag>.csv`, 1,207 × 34: `player_id, split, e0 … e31`). The M4 checkpoint's files are in `results/first_run/`; the final ones go in `results/final/`.

KNN: `knn/nba_knn_baseline.py` builds a DataFrame with these columns (`build_neighbours`, cosine on 32 z-scored features, all 1,207 players pooled) but **does not save it yet**, and it is a Colab export that does not run as a plain script. Before M7's format check and M9, the KNN team needs to provide `knn/results/neighbours.csv`, for example by adding `neighbours.to_csv("results/neighbours.csv", index=False)` after it is built.
