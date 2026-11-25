import yaml
from nbacnn.config import load_config, parse_value

def test_override_types_used_by_the_run_commands():
    cfg = load_config("configs/default.yaml",
                      ["views.chart_mode=off", "train.lr=3e-3", "seed=1", "early_stop=false",
                       "train.epochs=300", "run_name=lr3e-3", "views.chart_noise=0"])
    assert cfg["views"]["chart_mode"] == "off"          # YAML 1.1 would make this False
    assert cfg["train"]["lr"] == 0.003 and cfg["seed"] == 1 and cfg["train"]["epochs"] == 300
    assert cfg["early_stop"] is False and cfg["run_name"] == "lr3e-3" and cfg["views"]["chart_noise"] == 0

def test_only_true_false_are_booleans():
    assert parse_value("true") is True and parse_value("False") is False
    for word in ("off", "on", "yes", "no"):
        assert parse_value(word) == word

def test_off_in_a_yaml_file_stays_a_string(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("views:\n  chart_mode: off\n")
    assert load_config(p)["views"]["chart_mode"] == "off"

def test_saved_run_config_round_trips(tmp_path):
    cfg = load_config("configs/default.yaml", ["views.chart_mode=off"])
    p = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(p, "w"))                  # what train() writes into each run folder
    assert load_config(p) == cfg
