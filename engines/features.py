"""Basic feature functions for later milestones.

Pure functions over numeric sequences - no lookahead. The scoring stub does
not use these yet; they are here so backtests and real engines can build on
tested primitives.
"""

import math
from collections.abc import Sequence


def ema(values: Sequence[float], period: int) -> float:
    if period <= 0 or len(values) == 0:
        raise ValueError("period and values must be positive/non-empty")
    alpha = 2.0 / (period + 1)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> float:
    n = len(closes)
    if not (len(highs) == len(lows) == n) or n < 2:
        raise ValueError("need equal-length OHLC series with at least 2 bars")
    true_ranges = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    window = true_ranges[-period:]
    return sum(window) / len(window)


def zscore(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (values[-1] - mean) / std


def rsi(closes: Sequence[float], period: int = 14) -> float:
    """Wilder's RSI in [0, 100]. Needs at least period+1 closes."""
    n = len(closes)
    if n < period + 1:
        raise ValueError("need at least period+1 closes")
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd_histogram(
    closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> float:
    """MACD line minus its signal-line EMA, in price units."""
    n = len(closes)
    if n < slow + signal:
        raise ValueError("need at least slow+signal closes")
    macd_series = []
    for i in range(slow, n + 1):
        window = closes[:i]
        macd_series.append(ema(window, fast) - ema(window, slow))
    return macd_series[-1] - ema(macd_series, signal)


def bollinger_percent_b(closes: Sequence[float], period: int = 20, num_std: float = 2.0) -> float:
    """Position of the latest close within its Bollinger Bands.

    0 = lower band, 0.5 = middle band (SMA), 1 = upper band. Values outside
    [0, 1] mean price is beyond the bands. Flat windows (std == 0) return 0.5.
    """
    n = len(closes)
    if n < period:
        raise ValueError("need at least period closes")
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((v - mean) ** 2 for v in window) / period
    std = math.sqrt(variance)
    if std == 0:
        return 0.5
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (closes[-1] - lower) / (upper - lower)


def rolling_correlation(a: Sequence[float], b: Sequence[float], window: int | None = None) -> float:
    """Pearson correlation between the last `window` (or all) paired values."""
    n = len(a)
    if n != len(b):
        raise ValueError("series must be the same length")
    if window is not None:
        a = a[-window:]
        b = b[-window:]
    n = len(a)
    if n < 2:
        raise ValueError("need at least 2 values")
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        return 0.0
    return cov / denom


# TODO(Milestone 2+): session high/low tracking (Asia/London/NY), VWAP,
# currency strength baskets.
