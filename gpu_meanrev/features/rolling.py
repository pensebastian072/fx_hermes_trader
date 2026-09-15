"""GPU-batched rolling zscore / Bollinger %B across (pairs, time).

engines/features.py's zscore()/bollinger_percent_b() are correct, tested,
single-window scalar functions evaluated at the latest point of a slice --
exactly the math this module vectorizes, just batched across every pair and
every lookback window in the grid at once instead of called per-bar per-pair.
This is the layer where GPU actually earns its keep: the indicators are
cheap, the parameter-sweep breadth is not.

Implementation note: an earlier version used `.unfold()` to materialize
every window explicitly -- O(n_pairs * n_bars * window) memory. At this
study's real scale (28 pairs, ~7.9M bars/pair, windows up to 480) that is
~425GB for a single window, nowhere close to fitting. This version uses
`avg_pool1d` for O(n_bars) memory instead: rolling mean via avg_pool1d(x),
rolling E[x^2] via avg_pool1d(x**2), variance = E[x^2] - E[x]^2. Computed in
float64 specifically because FX prices sit near O(1) while the variance
signal is O(1e-8) (std ~1e-4) -- in float32 that magnitude gap eats the
variance in cancellation error (float32 abs precision near 1.21 is ~1.4e-7,
comparable to the variance itself). float64's abs precision near 1.21 is
~2.2e-16, leaving the O(1e-8) signal fully intact.

Definitions kept IDENTICAL to engines/features.py (parity-tested in
tests/test_gpu_meanrev_features.py):
  - zscore: sample std (ddof=1), (last - mean) / std, 0.0 if std==0.
  - bollinger %B: population std (ddof=0), (last - lower) / (upper - lower),
    0.5 if std==0 (flat window).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rolling_mean_and_var(x64: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x64: (n_pairs, n_bars) float64. Returns (mean, pop_var, last), each
    (n_pairs, n_bars-window+1), causal (index t in the output = window
    ending at bar t+window-1 of the input)."""
    x3 = x64.unsqueeze(1)  # (n_pairs, 1, n_bars) -- avg_pool1d wants (N, C, L)
    mean = F.avg_pool1d(x3, kernel_size=window, stride=1).squeeze(1)
    mean_sq = F.avg_pool1d(x3 * x3, kernel_size=window, stride=1).squeeze(1)
    pop_var = (mean_sq - mean * mean).clamp_min(0.0)  # clamp: fp noise can push this slightly negative at var~0
    last = x64[:, window - 1:]
    return mean, pop_var, last


def rolling_zscore(prices: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling z-score of the latest bar within its own trailing window.

    prices: (n_pairs, n_bars) float tensor. Returns (n_pairs, n_bars) with the
    first `window-1` bars NaN (insufficient history) -- vectorized analogue
    of engines.features.zscore's ValueError-on-too-short-window contract.
    """
    n_pairs, n_bars = prices.shape
    out = torch.full((n_pairs, n_bars), float("nan"), dtype=torch.float64, device=prices.device)
    if n_bars < window:
        return out
    x64 = prices.double()
    mean, pop_var, last = _rolling_mean_and_var(x64, window)
    if window > 1:
        sample_var = pop_var * (window / (window - 1))
    else:
        sample_var = pop_var
    std = sample_var.sqrt()
    z = torch.where(std > 0, (last - mean) / std, torch.zeros_like(mean))
    out[:, window - 1:] = z
    return out


def rolling_bollinger_percent_b(prices: torch.Tensor, window: int, num_std: float = 2.0) -> torch.Tensor:
    """Rolling Bollinger %B, population std (ddof=0), matches engines.features."""
    n_pairs, n_bars = prices.shape
    out = torch.full((n_pairs, n_bars), float("nan"), dtype=torch.float64, device=prices.device)
    if n_bars < window:
        return out
    x64 = prices.double()
    mean, pop_var, last = _rolling_mean_and_var(x64, window)
    std = pop_var.sqrt()
    upper = mean + num_std * std
    lower = mean - num_std * std
    band = upper - lower
    pb = torch.where(std > 0, (last - lower) / band, torch.full_like(mean, 0.5))
    out[:, window - 1:] = pb
    return out
