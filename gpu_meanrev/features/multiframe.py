"""Higher-timeframe trend context joined onto 1-minute bars, leak-guarded.

The idea (user's, 2026-08-04): "go out to the 15m/30m/1h/4h/daily/weekly and
that's where the trend gets found" -- a 1-minute entry should know what the
larger structure is doing, instead of trading a z-score in a vacuum.

THE LEAK IS THE WHOLE PROBLEM HERE. Resampling 1m -> 4h and joining back is
the single easiest way to hand a backtest tomorrow's information: the naive
`resample().ffill()` lets the 09:15 bar see a 4h bar that does not finish
until 12:00. Two guards, both applied:

  1. `label="right", closed="right"` -- a resampled bar is stamped at the
     time it CLOSES, not the time it opens.
  2. `.shift(1)` -- a 1m bar only ever sees the last FULLY COMPLETED higher
     timeframe bar, never the one currently forming. Strictly conservative:
     at 12:00 exactly, the bar that just closed at 12:00 is still withheld.

Same discipline as alpaca_gpu_lab/src/features/build.py's forward-fill +
join_asof(backward) pattern, and tests/test_gpu_meanrev_multiframe.py pins
it with an explicit "future spike must not be visible" assertion.
"""
from __future__ import annotations

import pandas as pd

# Pandas offset aliases. "h"/"min" (not "H"/"T") -- the capitalized forms are
# deprecated in pandas 2.2+ and warn.
TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def resample_closes(closes: pd.Series, rule: str) -> pd.Series:
    """1m close series -> higher-timeframe close series, stamped at bar close.

    Bars are labelled at their right edge so the timestamp is the moment the
    information becomes known. No shift here -- callers get the shift via
    `align_to_1m`, which is the only function that should ever be used to put
    this data next to 1m bars.
    """
    return closes.resample(rule, label="right", closed="right").last().dropna()


def align_to_1m(htf_series: pd.Series, index_1m: pd.DatetimeIndex) -> pd.Series:
    """Broadcast a higher-timeframe series onto a 1m index, leak-guarded.

    `.shift(1)` first, so each 1m bar sees only the last fully completed
    higher-timeframe bar; then reindex-forward-fill onto the 1m grid.
    """
    completed = htf_series.shift(1)
    return completed.reindex(index_1m, method="ffill")


def trend_direction(closes: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """+1 uptrend / -1 downtrend / 0 undecided, from an EMA cross on that
    timeframe's own bars. Uses `adjust=False` so it is the same recursion as
    engines/features.py::ema (see features/recursive.py for that parity note).
    """
    ef = closes.ewm(span=fast, adjust=False).mean()
    es = closes.ewm(span=slow, adjust=False).mean()
    diff = ef - es
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype("float64")


def build_mtf_context(
    closes_1m: pd.Series,
    timeframes: list[str] | None = None,
    fast: int = 10,
    slow: int = 30,
) -> pd.DataFrame:
    """Per-1m-bar trend direction for each requested higher timeframe.

    Returns a DataFrame indexed like `closes_1m`, one column per timeframe
    (e.g. "trend_4h"), each in {-1, 0, +1}, NaN before that timeframe has
    enough completed history.
    """
    timeframes = timeframes or list(TIMEFRAMES)
    out = {}
    for tf in timeframes:
        rule = TIMEFRAMES[tf]
        htf_closes = resample_closes(closes_1m, rule)
        htf_trend = trend_direction(htf_closes, fast=fast, slow=slow)
        out[f"trend_{tf}"] = align_to_1m(htf_trend, closes_1m.index)
    return pd.DataFrame(out, index=closes_1m.index)


def confluence_score(mtf: pd.DataFrame) -> pd.Series:
    """Net agreement across timeframes, in [-1, +1].

    +1 = every timeframe up, -1 = every timeframe down, ~0 = conflicted.
    NaN timeframes (not enough history yet) are ignored rather than counted
    as disagreement, so early bars are not silently biased toward 0.
    """
    return mtf.mean(axis=1, skipna=True)
