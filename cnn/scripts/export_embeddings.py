"""Embed all 1,207 players with a trained model and write embeddings + top-5 neighbours.
   python scripts/export_embeddings.py --checkpoint results/final/best_seed0.pt --tag seed0"""
import argparse
from pathlib import Path
import torch
from nbacnn.config import load_config
from nbacnn.data import load_split
from nbacnn.model import PlayerEncoder
from nbacnn.embed import embed_all, top_k

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--tag", default="seed0")
p.add_argument("--out_dir", default="results/final")
a = p.parse_args()
cfg = load_config(a.config)
model = PlayerEncoder(**cfg["model"])
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
emb = embed_all(model, {s: load_split(s, cfg["data_dir"]) for s in ("train", "val", "test")})
nb = top_k(emb, emb.player_id.tolist(), k=5)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
emb.to_csv(out / f"embeddings_{a.tag}.csv", index=False)
nb.to_csv(out / f"neighbours_{a.tag}.csv", index=False)
print(f"{len(emb)} players embedded -> {out}")