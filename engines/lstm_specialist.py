"""LSTM trend specialist (PROJECT_PLAN.md 29.2 / 30 follow-up).

Mirrors engines/ml_specialist.MLSpecialist: a model trained offline by
backtests/train_lstm.py, loaded read-only from data/artifacts/models/, with
the same edge gate (a model that doesn't beat the majority-class baseline
out-of-sample is never used for scoring). torch is an optional dependency -
if it isn't installed, or no usable model is on disk, .available is False and
load_ensemble() falls back to trend_rf or the rule-based TrendSpecialist.
"""

import json

import numpy as np

from app.paths import data_dir
from engines.ml_specialist import FEATURES_VERSION, N_FEATURES, build_feature_matrix

TREND_LSTM_MODEL = "trend_lstm.pt"

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch isn't installed
    TORCH_AVAILABLE = False


def models_dir():
    return data_dir() / "artifacts" / "models"


if TORCH_AVAILABLE:

    class LSTMTrendNet(nn.Module):
        """Single-layer LSTM over a window of feature rows -> P(up) logit."""

        def __init__(self, input_size: int = N_FEATURES, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)  # logits


class LSTMSpecialist:
    """Score = (2 * P(up) - 1) * 100, clipped like every other specialist."""

    name = "trend_lstm"

    def __init__(self, model_file: str = TREND_LSTM_MODEL):
        self.model = None
        self.meta: dict | None = None
        if not TORCH_AVAILABLE:
            return
        path = models_dir() / model_file
        meta_path = path.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("features_version") != FEATURES_VERSION:
                return
            if float(meta.get("oos_accuracy", 0.0)) <= float(meta.get("baseline_accuracy", 0.5)):
                return
            net = LSTMTrendNet(
                input_size=meta["input_size"],
                hidden_size=meta["hidden_size"],
                num_layers=meta["num_layers"],
            )
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.eval()
            self.model = net
            self.meta = meta
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            self.model = None
            self.meta = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, closes: np.ndarray, market_returns: np.ndarray | None = None) -> float:
        if self.model is None:
            return 0.0
        seq_len = self.meta["seq_len"]
        fm = build_feature_matrix(closes, market_returns)
        if len(fm) < seq_len:
            return 0.0
        seq = fm[-seq_len:]
        mean = np.array(self.meta["feature_mean"])
        std = np.array(self.meta["feature_std"])
        seq = (seq - mean) / std
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_up = torch.sigmoid(self.model(x)).item()
        return float(np.clip((2.0 * p_up - 1.0) * 100.0, -100.0, 100.0))
