from pathlib import Path
from types import SimpleNamespace

import yaml


def _flatten_dict(d):
    flat = {}
    for _, value in d.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            raise ValueError("Top-level config entries must be dictionaries in this minimal setup.")
    return flat


def load_flat_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    flat = _flatten_dict(raw)
    return SimpleNamespace(**flat)
