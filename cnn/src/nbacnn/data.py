"""Load the cleaned CSVs into tensors. One row per player, channels-first."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, torch

STAT_COLS = ["pts_per_100_poss_z", "usg_percent_z", "ts_percent_z", "x3p_ar_z", "f_tr_z",
             "x2p_pct_adj_z", "ft_pct_adj_z", "x3p_pct_adj_z", "ast_percent_z", "tov_percent_z",
             "orb_percent_z", "drb_percent_z", "stl_percent_z", "blk_percent_z",
             "pf_per_100_poss_z", "weight_z"]
POS_COLS = ["pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"]
CHANNELS = STAT_COLS + POS_COLS + ["mp_scaled", "played_mask"]   # 23
N_AGES, N_ROWS, N_COLS = 26, 24, 25
WEIGHT_CH = CHANNELS.index("weight_z")
POS_CH = [CHANNELS.index(c) for c in POS_COLS]
MASK_CH = CHANNELS.index("played_mask")

@dataclass
class PlayerTensors:
    ids: list            # player_id for each row of every tensor
    grid: torch.Tensor   # (P, 23, 26)    stats x ages
    mask: torch.Tensor   # (P, 26)        1 = qualified season at that age
    chart: torch.Tensor  # (P, 2, 24, 25) standardized freq_rel, eff_rel

    def __len__(self):
        return len(self.ids)

    def to(self, device):
        return PlayerTensors(self.ids, self.grid.to(device), self.mask.to(device), self.chart.to(device))

    def subset(self, idx):
        return PlayerTensors([self.ids[i] for i in idx.tolist()], self.grid[idx], self.mask[idx], self.chart[idx])

def load_split(split: str, data_dir="data/processed") -> PlayerTensors:
    d = Path(data_dir) / "cnn"
    stats = pd.read_csv(d / f"cnn_stats_{split}.csv").sort_values(["player_id", "age"])
    shots = pd.read_csv(d / f"cnn_shots_{split}.csv").sort_values(["player_id", "row", "col"])
    ids = stats.player_id.unique().tolist()
    assert ids == shots.player_id.unique().tolist(), "stats and shots players differ"
    P = len(ids)
    grid = stats[CHANNELS].to_numpy(np.float32).reshape(P, N_AGES, len(CHANNELS)).transpose(0, 2, 1)
    mask = stats["played_mask"].to_numpy(np.float32).reshape(P, N_AGES)
    scaler = pd.read_csv(d / "cnn_shots_train_scaler.csv", index_col=0)   # fitted on TRAIN only
    for c in ["freq_rel", "eff_rel"]:
        shots[c] = (shots[c] - scaler.loc[c, "mean"]) / scaler.loc[c, "std"]
    chart = shots[["freq_rel", "eff_rel"]].to_numpy(np.float32).reshape(P, N_ROWS, N_COLS, 2).transpose(0, 3, 1, 2)
    return PlayerTensors(ids, torch.from_numpy(grid.copy()), torch.from_numpy(mask.copy()),
                         torch.from_numpy(chart.copy()))