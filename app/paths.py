"""Filesystem locations. Everything log-related is derived from the data dir,
which can be overridden with FXHT_DATA_DIR (used by tests)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "active"


def data_dir() -> Path:
    return Path(os.environ.get("FXHT_DATA_DIR", str(PROJECT_ROOT / "data")))


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
