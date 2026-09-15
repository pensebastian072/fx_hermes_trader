"""Tests for new technical-indicator primitives in engines/features.py."""

import math

import pytest

from engines.features import bollinger_percent_b, macd_histogram, rolling_correlation, rsi


def test_rsi_all_gains_is_100():
    closes = [float(i) for i in range(1, 20)]  # strictly increasing
    assert rsi(closes, period=14) == 100.0


def test_rsi_all_losses_is_0():
    closes = [float(i) for i in range(20, 1, -1)]  # strictly decreasing
    assert rsi(closes, period=14) == 0.0


def test_rsi_flat_series_is_neutral():
    closes = [100.0] * 20
    assert rsi(closes, period=14) == 50.0


def test_rsi_requires_period_plus_one():
    with pytest.raises(ValueError):
        rsi([1.0] * 14, period=14)


def test_macd_histogram_finite():
    closes = [100.0 + 0.1 * i for i in range(60)]
    hist = macd_histogram(closes)
    assert math.isfinite(hist)


def test_macd_histogram_requires_enough_history():
    with pytest.raises(ValueError):
        macd_histogram([1.0] * 30, fast=12, slow=26, signal=9)


def test_bollinger_percent_b_bounds():
    # constant series -> flat bands -> neutral 0.5
    assert bollinger_percent_b([100.0] * 20) == 0.5

    # strictly increasing series -> latest close near/above upper band
    closes = [float(i) for i in range(1, 21)]
    assert bollinger_percent_b(closes) > 0.5


def test_bollinger_percent_b_requires_period():
    with pytest.raises(ValueError):
        bollinger_percent_b([1.0] * 10, period=20)


def test_rolling_correlation_identical_series_is_one():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rolling_correlation(a, a) == pytest.approx(1.0)


def test_rolling_correlation_inverted_series_is_negative_one():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert rolling_correlation(a, b) == pytest.approx(-1.0)


def test_rolling_correlation_window_slices_tail():
    a = [0.0, 0.0, 1.0, 2.0, 3.0]
    b = [9.0, -9.0, 1.0, 2.0, 3.0]
    # full series is dragged down by the first two points, windowed is perfectly correlated
    assert rolling_correlation(a, b, window=3) == pytest.approx(1.0)


def test_rolling_correlation_constant_series_is_zero():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert rolling_correlation(a, b) == 0.0


def test_rolling_correlation_mismatched_lengths():
    with pytest.raises(ValueError):
        rolling_correlation([1.0, 2.0], [1.0, 2.0, 3.0])
