# file: src/nbacnn/config.py
import re
import yaml

class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false are booleans.
    Plain YAML 1.1 also turns on/off/yes/no into booleans, so `chart_mode: off` (or the override
    `views.chart_mode=off`) would silently become False instead of the string "off"."""

_Loader.yaml_implicit_resolvers = {
    ch: [r for r in resolvers if r[0] != "tag:yaml.org,2002:bool"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool",
                              re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))

def parse_value(text):
    """Parse one override value: '200' -> 200, 'false' -> False, '3e-3' -> 0.003, 'off' -> 'off'."""
    v = yaml.load(text, Loader=_Loader)
    if isinstance(v, str):
        try:
            v = float(v)                          # YAML reads "3e-3" as text; make it a number
        except ValueError:
            pass
    return v

def apply_overrides(cfg, overrides=()):
    """Apply overrides like 'train.lr=3e-3' or 'seed=1' to cfg in place, and return it."""
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:                      # a typo would otherwise silently add a dead key
            raise KeyError(f"override {item!r}: no key {leaf!r} under {'.'.join(parents) or 'top level'}")
        node[leaf] = parse_value(value)
    return cfg

def load_config(path, overrides=()):
    """Load a YAML config and apply command-line overrides like 'train.lr=3e-3' or 'seed=1'."""
    with open(path) as f:
        cfg = yaml.load(f, Loader=_Loader)
    return apply_overrides(cfg, overrides)
