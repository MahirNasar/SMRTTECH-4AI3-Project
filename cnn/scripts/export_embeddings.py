"""Embed all 1,207 players with a trained model and write embeddings + top-5 neighbours.
   python scripts/export_embeddings.py --checkpoint outputs/runs/baseline/best.pt --tag baseline
   By default the config is the run's own config.yaml next to the checkpoint (so model size and
   views settings such as chart_mode match how it was trained); --config overrides that."""
import argparse
from pathlib import Path
import torch
from nbacnn.config import load_config
from nbacnn.data import load_split
from nbacnn.model import PlayerEncoder
from nbacnn.embed import embed_all, top_k

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--config", default=None, help="default: config.yaml next to the checkpoint, else configs/default.yaml")
p.add_argument("--tag", default="seed0")
p.add_argument("--out_dir", default="results/final")
a = p.parse_args()
run_cfg = Path(a.checkpoint).with_name("config.yaml")
cfg_path = a.config or (run_cfg if run_cfg.exists() else "configs/default.yaml")
print(f"config: {cfg_path}")
cfg = load_config(cfg_path)
model = PlayerEncoder(**cfg["model"])
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
emb = embed_all(model, {s: load_split(s, cfg["data_dir"]) for s in ("train", "val", "test")}, cfg["views"])
nb = top_k(emb, emb.player_id.tolist(), k=5)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
emb.to_csv(out / f"embeddings_{a.tag}.csv", index=False)
nb.to_csv(out / f"neighbours_{a.tag}.csv", index=False)
print(f"{len(emb)} players embedded -> {out}")
