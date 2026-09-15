"""Historical data loading + synthetic series generation.

CSV format: timestamp,open,high,low,close[,volume]. Parquet supported via
pyarrow. For stress tests and unit tests, synthetic_closes() scripts regime
segments (calm / trend / crisis) with a seeded RNG so results are
reproducible.

Reminder (PROJECT_PLAN.md 29.4): real datasets must reach back beyond the
post-2010 ZIRP era - include the 2008 yield high, curve inversions, and JPY
carry unwinds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]


def load_ohlc(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def synthetic_closes(
    segments: list[tuple[str, int]], start_price: float = 150.0, seed: int = 7
) -> np.ndarray:
    """Build a close series from (regime, n_bars) segments.

    Regimes: "calm" (low vol, no drift), "trend_up"/"trend_down" (drift),
    "crisis" (high vol + sharp adverse drift).
    """
    params = {
        "calm": (0.0, 0.0005),
        "trend_up": (0.0008, 0.0015),
        "trend_down": (-0.0008, 0.0015),
        "crisis": (-0.003, 0.012),
    }
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    for regime, n in segments:
        drift, vol = params[regime]
        returns.extend(rng.normal(drift, vol, n))
    return start_price * np.exp(np.cumsum(returns))


def closes_to_ohlc(closes: np.ndarray, freq: str = "h") -> pd.DataFrame:
    """Wrap a close series into an OHLC frame (open=prev close, hi/lo padded)."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    span = np.abs(closes - opens)
    highs = np.maximum(opens, closes) + 0.25 * span
    lows = np.minimum(opens, closes) - 0.25 * span
    timestamps = pd.date_range("2020-01-01", periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"timestamp": timestamps, "open": opens, "high": highs, "low": lows, "close": closes}
    )
