"""Tests for D01's selection/scoring separation.

The one property whose silent breakage would be INVISIBLE in the output: if the
selection window leaked into the scored window, every number in the scorecard
would simply look better and nothing would flag it. These tests plant a combo
that is brilliant only during the selection window and terrible afterwards, and
assert the scorecard reports the terrible half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_meanrev.analysis import rule_persistence as rp


def _daily(year: int, n_combos: int = 4) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return pd.DataFrame(
        np.zeros((len(days), n_combos)),
        index=[d.date() for d in days],
        columns=pd.Index(range(n_combos), name="combo_idx"),
    )


def test_across_years_scores_only_the_traded_year():
    """Combo 0 is planted to win 2010 hugely and lose 2011. Selecting on 2010
    must book 2011's LOSS, never 2010's win."""
    y0, y1 = _daily(2010), _daily(2011)
    y0.iloc[:, 0] = 10.0     # combo 0 wins the selection year
    y1.iloc[:, 0] = -5.0     # and loses the traded year
    y1.iloc[:, 1] = 1.0      # combo 1 is the traded year's real winner

    out = rp.across_years({2010: y0, 2011: y1})
    t = out["transitions"][0]
    assert t["picked_combo"] == 0
    assert t["picked"] < 0, "selection-year PnL leaked into the scored year"
    assert t["oracle"] > t["picked"]


def test_burn_in_scores_only_the_post_burn_window():
    y = _daily(2010)
    burn = np.array([d.month <= 3 for d in pd.to_datetime(pd.Series(y.index))])
    y.iloc[burn, 0] = 10.0       # combo 0 wins the burn-in
    y.iloc[~burn, 0] = -5.0      # and loses the traded remainder
    y.iloc[~burn, 1] = 1.0

    out = rp.burn_in({2010: y}, burn_months=3)
    r = out["years"][0]
    assert r["picked_combo"] == 0
    assert r["picked"] < 0, "burn-in PnL leaked into the traded window"
    assert r["burn_days"] + r["trade_days"] == len(y)


def test_burn_in_windows_are_complementary():
    y = _daily(2010)
    out = rp.burn_in({2010: y}, burn_months=6)
    r = out["years"][0]
    assert r["burn_days"] > 0 and r["trade_days"] > 0
    assert r["burn_days"] + r["trade_days"] == len(y)


def test_spearman_matches_a_known_case():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rp._spearman(a, a) == pytest.approx(1.0)
    assert rp._spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(rp._spearman(a, np.ones(5)))  # constant -> nan, like scipy


def test_partial_spearman_removes_the_control():
    """y driven entirely by z: partialling z out must collapse the correlation."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    x = z + 0.01 * rng.normal(size=200)
    y = z + 0.01 * rng.normal(size=200)
    assert rp._spearman(x, y) > 0.9
    assert abs(rp._partial_spearman(x, y, z)) < 0.5


def test_available_years_refuses_the_holdout(monkeypatch, tmp_path):
    from gpu_meanrev import config

    (tmp_path / "year_2022.parquet").touch()
    (tmp_path / "year_2023.parquet").touch()
    monkeypatch.setattr(rp, "CHECKPOINT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)

    with pytest.raises(RuntimeError, match="holdout years"):
        rp.available_years()
