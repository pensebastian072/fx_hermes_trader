"""YAML config loading for configs/active."""

from pathlib import Path

import yaml

from app.paths import CONFIG_DIR


def load_config(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
