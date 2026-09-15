"""Trained ML specialists for the regime-gated ensemble (PROJECT_PLAN.md 29.2).

A specialist is any object with .name and .score(closes) -> [-100, 100].
MLSpecialist wraps a sklearn classifier trained offline by
backtests/train_specialists.py and loaded read-only from
data/artifacts/models/. If the model file is missing or unreadable the
factory falls back to the deterministic rule-based specialists, so the
system never depends on an untrained model.

The CRISIS slot is never ML: capital preservation stays deterministic.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from app.paths import data_dir
from engines.ensemble import (
    CapitalPreservationSpecialist,
    MeanReversionSpecialist,
    RegimeGatedEnsemble,
    TrendSpecialist,
)
from engines.features import bollinger_percent_b, ema, macd_histogram, rolling_correlation, rsi, zscore
from engines.regime_detection.hmm_regime import CRISIS, RANGE, TREND

# Bump when build_feature_row changes; stored in model metadata and checked
# at load so a stale model is never fed features it wasn't trained on.
# v2: added RSI, MACD histogram, Bollinger %B, EMA100/200 cross, 63-bar
# momentum, short/long vol ratio, and a cross-pair correlation slot.
FEATURES_VERSION = 2
# 70 bars covers the slowest single-series indicator (MACD slow+signal=35)
# plus the 63-bar momentum lookback with margin.
MIN_BARS = 70
# Features always see at most this many bars so training and live inference
# compute identical values regardless of how much history the caller holds.
FEATURE_WINDOW = 200
# Rolling window for the cross-pair correlation feature.
CORR_WINDOW = 30
# Length of the feature vector returned by build_feature_row.
N_FEATURES = 13

TREND_MODEL = "trend_rf.joblib"
RANGE_MODEL = "range_svm.joblib"


def models_dir() -> Path:
    return data_dir() / "artifacts" / "models"


def build_feature_row(
    closes: np.ndarray, market_returns: np.ndarray | None = None
) -> np.ndarray | None:
    """Features describing the latest bar; None if history is too short.

    `market_returns`, if given, must be a per-bar log-return series aligned
    1:1 with `closes` (same length, market_returns[0] unused) - e.g. the mean
    return of other pairs on the same dates. Used for a cross-pair
    correlation feature; omitted (0.0) when not available.
    """
    closes = np.asarray(closes, dtype=float)[-FEATURE_WINDOW:]
    if len(closes) < MIN_BARS:
        return None
    returns = np.diff(np.log(closes))
    fast = ema(closes.tolist(), 21)
    slow = ema(closes.tolist(), 55)
    long_fast = ema(closes.tolist(), 100)
    long_slow = ema(closes.tolist(), 200)

    short_vol = returns[-21:].std()
    long_vol = returns[-100:].std() if len(returns) >= 100 else returns.std()
    vol_ratio = (short_vol / long_vol - 1.0) if long_vol > 0 else 0.0

    corr = 0.0
    if market_returns is not None:
        mret = np.asarray(market_returns, dtype=float)[-FEATURE_WINDOW:]
        if len(mret) == len(closes):
            mret_aligned = mret[1:]  # drop bar 0, matching `returns`
            corr = rolling_correlation(
                returns[-CORR_WINDOW:].tolist(), mret_aligned[-CORR_WINDOW:].tolist()
            )

    return np.array(
        [
            returns[-1],
            returns[-5:].sum(),
            returns[-21:].sum(),
            returns[-20:].std(),
            zscore(closes[-50:].tolist()),
            (fast - slow) / closes[-1],
            rsi(closes[-15:].tolist(), 14) / 50.0 - 1.0,
            macd_histogram(closes.tolist()) / closes[-1],
            bollinger_percent_b(closes[-20:].tolist(), 20) - 0.5,
            (long_fast - long_slow) / closes[-1],
            returns[-63:].sum(),
            vol_ratio,
            corr,
        ]
    )


def build_dataset(
    closes: np.ndarray, market_returns: np.ndarray | None = None, horizon: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) over a whole series: y = 1 if the forward `horizon`-bar return
    is positive. Each row only uses closes[: t + 1] - no lookahead.

    `market_returns`, if given, must be the same length as `closes` (see
    build_feature_row).
    """
    closes = np.asarray(closes, dtype=float)
    mret = np.asarray(market_returns, dtype=float) if market_returns is not None else None
    if mret is not None and len(mret) != len(closes):
        raise ValueError("market_returns must be the same length as closes")
    rows, labels = [], []
    for t in range(MIN_BARS, len(closes) - horizon):
        start = max(0, t + 1 - FEATURE_WINDOW)
        row = build_feature_row(
            closes[start : t + 1], mret[start : t + 1] if mret is not None else None
        )
        if row is None:
            continue
        rows.append(row)
        labels.append(1 if closes[t + horizon] > closes[t] else 0)
    return np.array(rows), np.array(labels)


def build_feature_matrix(
    closes: np.ndarray, market_returns: np.ndarray | None = None
) -> np.ndarray:
    """Feature row for every bar from MIN_BARS-1 onward (no lookahead).

    Row i corresponds to closes[: MIN_BARS + i]. Used to build sliding
    sequences for the LSTM specialist (engines/lstm_specialist.py).
    """
    closes = np.asarray(closes, dtype=float)
    mret = np.asarray(market_returns, dtype=float) if market_returns is not None else None
    if mret is not None and len(mret) != len(closes):
        raise ValueError("market_returns must be the same length as closes")
    rows = []
    for t in range(MIN_BARS - 1, len(closes)):
        start = max(0, t + 1 - FEATURE_WINDOW)
        row = build_feature_row(
            closes[start : t + 1], mret[start : t + 1] if mret is not None else None
        )
        rows.append(row)  # never None: t + 1 >= MIN_BARS by construction
    return np.array(rows)


def build_lstm_sequences(
    closes: np.ndarray,
    market_returns: np.ndarray | None = None,
    horizon: int = 5,
    seq_len: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) sliding-window sequences for the LSTM specialist.

    X has shape (n, seq_len, n_features); each sequence X[i] is the feature
    matrix for bars [t - seq_len + 1, t], y[i] = 1 if the forward
    `horizon`-bar return from bar t is positive. No lookahead.
    """
    closes = np.asarray(closes, dtype=float)
    fm = build_feature_matrix(closes, market_returns)
    sequences, labels = [], []
    for i in range(seq_len - 1, len(fm)):
        t = MIN_BARS - 1 + i
        if t + horizon >= len(closes):
            break
        sequences.append(fm[i - seq_len + 1 : i + 1])
        labels.append(1 if closes[t + horizon] > closes[t] else 0)
    return np.array(sequences), np.array(labels)


class MLSpecialist:
    """Score = (2 * P(up) - 1) * 100, clipped like every other specialist."""

    def __init__(self, name: str, model_file: str):
        self.name = name
        self.model = None
        path = models_dir() / model_file
        meta_path = path.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("features_version") != FEATURES_VERSION:
                return
            # Edge gate: a model that doesn't beat the majority-class baseline
            # out-of-sample stays on disk but is never used for scoring.
            if float(meta.get("oos_accuracy", 0.0)) <= float(
                meta.get("baseline_accuracy", 0.5)
            ):
                return
            self.model = joblib.load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, closes: np.ndarray, market_returns: np.ndarray | None = None) -> float:
        if self.model is None:
            return 0.0
        row = build_feature_row(closes, market_returns)
        if row is None:
            return 0.0
        p_up = float(self.model.predict_proba(row.reshape(1, -1))[0, 1])
        return float(np.clip((2.0 * p_up - 1.0) * 100.0, -100.0, 100.0))


def load_ensemble() -> RegimeGatedEnsemble:
    """Regime-gated ensemble preferring trained models, rule-based otherwise.

    TREND slot prefers an edge-gated LSTM, then an edge-gated RandomForest,
    then the deterministic rule-based specialist - each tier only used if the
    one before it is unavailable or fails its out-of-sample edge gate.
    """
    from engines.lstm_specialist import LSTMSpecialist  # local: avoid import cycle

    lstm = LSTMSpecialist()
    trend_rf = MLSpecialist("trend_rf", TREND_MODEL)
    rng = MLSpecialist("range_svm", RANGE_MODEL)
    if lstm.available:
        trend = lstm
    elif trend_rf.available:
        trend = trend_rf
    else:
        trend = TrendSpecialist()
    return RegimeGatedEnsemble(
        {
            TREND: trend,
            RANGE: rng if rng.available else MeanReversionSpecialist(),
            CRISIS: CapitalPreservationSpecialist(),
        }
    )
