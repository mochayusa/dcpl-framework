from __future__ import annotations
from pathlib import Path
import yaml
import json


def load_config(path: str | Path) -> dict:
    """
    Load YAML or JSON config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    raise ValueError("Config must be .yaml, .yml, or .json")
