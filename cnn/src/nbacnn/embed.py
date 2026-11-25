"""Turn every player into a summary h, then find each player's 5 nearest neighbours."""
import pandas as pd, torch
from .views import full_view

@torch.no_grad()
def embed_all(model, splits):
    """splits: {"train": PlayerTensors, "val": ..., "test": ...} -> DataFrame, one row per player."""
    model.eval()
    frames = []
    for name, data in splits.items():
        h, _ = model(*full_view(data))                     # h, not z: the projection head is discarded
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