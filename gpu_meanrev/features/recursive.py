"""Recursive indicators (EMA, Wilder RSI, ATR) — CPU via pandas, per pair.

These don't vectorize across time as a single tensor op (true recurrence:
each bar depends on the previous smoothed value), so there's no real GPU win
here — pandas's C-implemented .ewm()/.rolling() handle millions of rows in
well under a second per pair. Compute once per pair on CPU, stack into a
(n_pairs, n_bars) tensor afterward for anything downstream that wants it
alongside the GPU rolling-window features.

Vectorized equivalents of engines/features.py's scalar recursive functions:
  - ema: pandas .ewm(alpha=2/(period+1), adjust=False) IS the same recursion
    (y0=x0, yt=(1-a)*y(t-1)+a*xt) — exact match.
  - rsi: pandas .ewm(alpha=1/period, adjust=False) on gains/losses is the
    standard vectorized form of Wilder's smoothing. It differs from engines.
    features.rsi only in how the first `period` bars are seeded (SMA seed
    there vs. first-value seed here); that difference decays like
    (1-1/period)^n and is machine-epsilon by a few hundred bars — negligible
    on multi-million-row series. Parity test checks agreement well past
    warmup, not at bar 0.
  - atr: engines.features.atr is a SIMPLE rolling mean of true range (not
    Wilder-smoothed) — pandas .rolling(period).mean() on TR is an exact
    match, no approximation involved.
"""
from __future__ import annotations

import pandas as pd
import torch


def ema_series(closes: pd.Series, period: int) -> pd.Series:
    alpha = 2.0 / (period + 1)
    return closes.ewm(alpha=alpha, adjust=False).mean()


def rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 & avg_gain == 0 -> rs is NaN (0/0); engines.features.rsi
    # returns 50.0 for that case, and 100.0 when avg_loss==0 & avg_gain>0.
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def atr_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def stack_pairs(series_by_pair: dict[str, pd.Series], device=None) -> tuple[torch.Tensor, list[str]]:
    """dict[pair] -> aligned (n_pairs, n_bars) tensor over the union index."""
    df = pd.DataFrame(series_by_pair)
    arr = df.to_numpy(dtype="float32")
    t = torch.from_numpy(arr.T).contiguous()
    if device is not None:
        t = t.to(device)
    return t, list(df.columns)
