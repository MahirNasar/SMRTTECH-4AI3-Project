# file: src/nbacnn/config.py
import yaml

def load_config(path, overrides=()):
    """Load a YAML config and apply command-line overrides like 'train.lr=3e-3' or 'seed=1'."""
    cfg = yaml.safe_load(open(path))
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        v = yaml.safe_load(value)                 # "true" -> True, "200" -> 200
        if isinstance(v, str):
            try:
                v = float(v)                      # YAML reads "3e-3" as text; make it a number
            except ValueError:
                pass
        node[leaf] = v
    return cfg