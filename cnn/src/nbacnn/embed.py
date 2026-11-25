"""Turn every player into a summary h, then find each player's 5 nearest neighbours."""
import pandas as pd, torch
import torch.nn.functional as F
from .views import full_view, career_halves

EMBED_METHODS = ("halves", "full")

@torch.no_grad()
def embed_players(model, data, views_cfg=None, method="halves"):
    """(P, emb) unit-length summaries h for one PlayerTensors.
    method="halves" (default): normalize(h(odd seasons) + h(even seasons)). Each half matches what an
        odd_even-trained model saw in training (no two adjacent seasons), and together they use every season.
    method="full": h(whole career in one view), the guide's original method. Out of distribution for
        odd_even training (79% of seasons have a played neighbour); kept for comparison only.
    h, not z: the projection head is discarded after training."""
    if method == "halves":
        odd, even = career_halves(data, views_cfg)
        return F.normalize(model(*odd)[0] + model(*even)[0], dim=1)
    if method == "full":
        return model(*full_view(data, views_cfg))[0]
    raise ValueError(f"unknown embed method {method!r}; expected one of {EMBED_METHODS}")

@torch.no_grad()
def embed_all(model, splits, views_cfg=None, method="halves"):
    """splits: {"train": PlayerTensors, "val": ..., "test": ...} -> DataFrame, one row per player.
    views_cfg: the `views:` block the checkpoint was trained with, so e.g. a chart_mode=off model
    is embedded without the shot chart it never learned from."""
    model.eval()
    frames = []
    for name, data in splits.items():
        h = embed_players(model, data, views_cfg, method)
        df = pd.DataFrame(h.cpu().numpy(), columns=[f"e{i}" for i in range(h.shape[1])])
        df.insert(0, "split", name)
        df.insert(0, "player_id", data.ids)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def top_k(emb, query_ids, k=5):
    """k nearest players by cosine similarity, searching ALL players except the query himself."""
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    E = torch.nn.functional.normalize(torch.tensor(emb[cols].to_numpy(), dtype=torch.float32), dim=1)
    pos = {pid: i for i, pid in enumerate(emb.player_id)}
    q = torch.tensor([pos[p] for p in query_ids])
    sim = E[q] @ E.T                                         # (queries, all players)
    sim[torch.arange(len(q)), q] = -2                        # never return the player himself
    s, j = sim.topk(k, dim=1)
    rows = [(qid, r + 1, emb.player_id.iloc[j[a, r].item()], round(s[a, r].item(), 4))
            for a, qid in enumerate(query_ids) for r in range(k)]
    return pd.DataFrame(rows, columns=["query_id", "rank", "neighbour_id", "similarity"])
