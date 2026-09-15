"""Parity: gpu_meanrev's vectorized features must reproduce engines/features.py.

engines.features.zscore/bollinger_percent_b/rsi are the tested, no-lookahead
primitives this study is built on. If the vectorized versions silently
diverge from them, every downstream signal is wrong in a way nothing else
would catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from engines import features as scalar
from gpu_meanrev.features import recursive, rolling


def _synthetic_prices(n=500, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.001, n)
    return 1.10 + np.cumsum(steps)


def test_rolling_zscore_matches_scalar():
    prices = _synthetic_prices()
    window = 20
    t = torch.tensor(prices, dtype=torch.float64).unsqueeze(0)  # (1, n)
    vec = rolling.rolling_zscore(t, window).squeeze(0).numpy()

    for i in [window - 1, window + 50, window + 200, len(prices) - 1]:
        expected = scalar.zscore(prices[i - window + 1: i + 1].tolist())
        assert vec[i] == pytest.approx(expected, abs=1e-6), f"mismatch at bar {i}"


def test_rolling_bollinger_matches_scalar():
    prices = _synthetic_prices(seed=11)
    window = 20
    t = torch.tensor(prices, dtype=torch.float64).unsqueeze(0)
    vec = rolling.rolling_bollinger_percent_b(t, window).squeeze(0).numpy()

    for i in [window - 1, window + 50, window + 200, len(prices) - 1]:
        expected = scalar.bollinger_percent_b(prices[i - window + 1: i + 1].tolist(), period=window)
        assert vec[i] == pytest.approx(expected, abs=1e-6), f"mismatch at bar {i}"


def test_rolling_zscore_short_series_is_nan_not_error():
    prices = _synthetic_prices(n=5)
    t = torch.tensor(prices, dtype=torch.float64).unsqueeze(0)
    vec = rolling.rolling_zscore(t, window=20).squeeze(0).numpy()
    assert np.isnan(vec).all()


def test_rsi_matches_scalar_past_warmup():
    prices = _synthetic_prices(n=1000, seed=3)
    period = 14
    s = pd.Series(prices)
    vec = recursive.rsi_series(s, period=period)

    # Wilder-seed (SMA) vs ewm-seed (first value) converge exponentially;
    # by bar 300 the gap is far below float precision that matters here.
    for i in [300, 500, 999]:
        expected = scalar.rsi(prices[: i + 1].tolist(), period=period)
        assert vec.iloc[i] == pytest.approx(expected, abs=1e-3), f"mismatch at bar {i}"


def test_atr_matches_scalar_exactly():
    rng = np.random.default_rng(5)
    n = 200
    closes = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    highs = closes + np.abs(rng.normal(0, 0.0005, n))
    lows = closes - np.abs(rng.normal(0, 0.0005, n))
    period = 14

    vec = recursive.atr_series(pd.Series(highs), pd.Series(lows), pd.Series(closes), period=period)
    for i in [period, period + 50, n - 1]:
        expected = scalar.atr(highs[: i + 1].tolist(), lows[: i + 1].tolist(), closes[: i + 1].tolist(), period=period)
        assert vec.iloc[i] == pytest.approx(expected, abs=1e-9), f"mismatch at bar {i}"
