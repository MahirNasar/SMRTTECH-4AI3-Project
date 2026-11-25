import matplotlib; matplotlib.use("Agg")
import pytest
from nbacnn.config import load_config
from nbacnn.fake import make_fake_split
from nbacnn.study import run_study, run_ablations, plot_study

DATA = (make_fake_split(128, seed=0), make_fake_split(64, seed=1))

def base(tmp_path):
    return load_config("configs/default.yaml", ["device=cpu", "tensorboard=false", f"out_dir={tmp_path}"])

def test_study_grid_resume_and_plots(tmp_path):
    runs, summary = run_study(base(tmp_path), [1e-3, 3e-3], [2, 3], [0, 1], out_dir=tmp_path / "study", data=DATA)
    assert len(runs) == 8 and len(summary) == 4 and (summary.n == 2).all()
    assert set(runs.epochs_run) == {2, 3}                          # early stopping is off in the study
    stamp = (tmp_path / "study" / "lr0.001_ep2_s0" / "metrics.csv").stat().st_mtime
    run_study(base(tmp_path), [1e-3], [2], [0], out_dir=tmp_path / "study", data=DATA)
    assert (tmp_path / "study" / "lr0.001_ep2_s0" / "metrics.csv").stat().st_mtime == stamp   # resumed, not retrained
    plot_study(summary, tmp_path / "study", tmp_path / "figs")
    assert (tmp_path / "figs" / "heatmap_val_top1.png").exists() and (tmp_path / "figs" / "curves_by_lr.png").exists()

def test_ablations_apply_overrides(tmp_path):
    cfg = base(tmp_path); cfg["train"]["epochs"] = 2
    runs, summary = run_ablations(cfg, {"default": [], "career_only": ["views.chart_mode=off"]}, [0],
                                  out_dir=tmp_path / "abl", data=DATA)
    assert list(summary.name) == ["default", "career_only"]
    saved = load_config(tmp_path / "abl" / "career_only_s0" / "config.yaml")
    assert saved["views"]["chart_mode"] == "off" and saved["seed"] == 0

def test_override_typo_is_an_error():
    with pytest.raises(KeyError):
        load_config("configs/default.yaml", ["train.learning_rate=3e-3"])
