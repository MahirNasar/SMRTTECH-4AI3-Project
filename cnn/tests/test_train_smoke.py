# file: tests/test_train_smoke.py
import pandas as pd
from nbacnn.config import load_config
from nbacnn.fake import make_fake_split
from nbacnn.train import train

def test_loss_falls_on_fake_data(tmp_path):
    cfg = load_config("configs/default.yaml",
                      [f"out_dir={tmp_path}", "run_name=smoke", "tensorboard=false",
                       "early_stop=false", "train.epochs=20", "views.chart_mode=off", "device=cpu",
                       "train.lr=1e-3"])          # tests the engine, not the chosen config: fixed lr
    run = train(cfg, data=(make_fake_split(512, seed=0), make_fake_split(128, seed=1)))
    m = pd.read_csv(run / "metrics.csv")
    assert len(m) == 20
    assert m.train_loss_norm.iloc[-1] < m.train_loss_norm.iloc[0] - 0.15   # it learned something
    assert (run / "best.pt").exists() and (run / "summary.json").exists()

    