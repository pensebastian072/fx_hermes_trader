"""Tests for the daily panel's holdout re-assertion and the carry battery's
look-ahead guards.

CLAUDE.md gpu_meanrev rule 6 exists because reimplementing the loader is exactly
how the holdout clip gets duplicated away. Before this file the repo had no test
anywhere asserting that clip — on build_panel or on daily.py — so the guard was
protected only by code review. These tests pin it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_meanrev import config
from gpu_meanrev.data import daily as daily_mod
from gpu_meanrev.data import rates as rates_mod


@pytest.fixture
def synthetic_pair(monkeypatch, tmp_path):
    """One pair spanning the holdout boundary, with a temp cache."""
    idx = pd.date_range("2022-12-28", "2023-01-05", freq="1min", tz="UTC")
    prices = pd.Series(np.linspace(1.0, 1.1, len(idx)), index=idx)

    def fake_load(pair, report=None):
        s = prices.copy()
        s.name = pair
        if report is not None:
            report[pair] = {"rows_in": len(s), "rows_out": len(s)}
        return s

    monkeypatch.setattr(daily_mod, "load_pair_closes", fake_load)
    monkeypatch.setattr(daily_mod, "CACHE", tmp_path / "daily.parquet")
    monkeypatch.setattr(daily_mod, "CACHE_REPORT", tmp_path / "report.json")
    return prices


BOUNDARY = pd.Timestamp(config.HOLDOUT_START, tz="UTC")


def test_fresh_build_clips_holdout(synthetic_pair, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)
    df = daily_mod.load_daily_closes(["EURUSD"])
    assert df.index.max() < BOUNDARY


def test_cache_holds_full_panel_but_read_is_clipped(synthetic_pair, monkeypatch):
    """The cache deliberately stores the UNCLIPPED panel; the clip is applied on
    every read. A cache written while unlocked must not leak to a locked caller."""
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", True)
    unlocked = daily_mod.load_daily_closes(["EURUSD"], unlock_holdout=True, refresh=True)
    assert unlocked.index.max() >= BOUNDARY, "unlocked read should see the holdout"

    on_disk = pd.read_parquet(daily_mod.CACHE)
    assert on_disk.index.max() >= BOUNDARY, "cache should hold the full panel"

    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)
    locked = daily_mod.load_daily_closes(["EURUSD"])
    assert locked.index.max() < BOUNDARY, "locked read leaked the holdout from cache"


def test_unlock_flag_alone_is_not_enough(synthetic_pair, monkeypatch):
    """Double gate: unlock_holdout=True in code with the env var unset stays clipped."""
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)
    df = daily_mod.load_daily_closes(["EURUSD"], unlock_holdout=True, refresh=True)
    assert df.index.max() < BOUNDARY


def test_weekend_bars_dropped(synthetic_pair, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)
    df = daily_mod.load_daily_closes(["EURUSD"], refresh=True)
    assert (df.index.dayofweek < 5).all()


def test_signal_rates_are_publication_lagged(monkeypatch):
    """Row for month m must carry the rate of month m-1, never m."""
    months = pd.date_range("2020-01-01", "2020-06-01", freq="MS")
    frame = pd.DataFrame(
        {c: np.arange(1.0, len(months) + 1) for c in rates_mod.CURRENCIES},
        index=months,
    )
    monkeypatch.setattr(rates_mod, "fetch_rates", lambda refresh=False: frame)

    sig = rates_mod.signal_rates(start="2020-02-01", end="2020-06-01")
    assert sig.loc[pd.Timestamp("2020-02-01"), "USD"] == 1.0  # January's value
    assert sig.loc[pd.Timestamp("2020-03-01"), "USD"] == 2.0


def test_signal_rates_pad_keeps_first_in_window_month_usable(monkeypatch):
    """Clipping at `start` before the shift would leave the first month NaN and
    silently cost the battery a rebalance."""
    months = pd.date_range("2019-01-01", "2020-06-01", freq="MS")
    frame = pd.DataFrame(
        {c: np.arange(1.0, len(months) + 1) for c in rates_mod.CURRENCIES},
        index=months,
    )
    monkeypatch.setattr(rates_mod, "fetch_rates", lambda refresh=False: frame)

    sig = rates_mod.signal_rates(start="2020-02-01", end="2020-06-01")
    assert sig.loc[pd.Timestamp("2020-02-01")].notna().all()


def test_signal_rates_rejects_a_real_gap(monkeypatch):
    months = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    frame = pd.DataFrame(
        {c: np.arange(1.0, len(months) + 1) for c in rates_mod.CURRENCIES},
        index=months,
    )
    frame.loc["2020-05-01":"2020-07-01", "USD"] = np.nan  # 3-month hole
    monkeypatch.setattr(rates_mod, "fetch_rates", lambda refresh=False: frame)

    with pytest.raises(RuntimeError, match="gaps longer than"):
        rates_mod.signal_rates(start="2020-01-01", end="2020-12-01")


def test_redact_strips_the_api_key():
    msg = "400 Client Error for url: https://x?series_id=A&api_key=SECRET123&file_type=json"
    assert "SECRET123" not in rates_mod._redact(msg, "SECRET123")
