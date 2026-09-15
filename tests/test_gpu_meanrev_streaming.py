"""Coverage for the two paths that only appear at full battery scale.

Both were written to fix a real OOM on the B01 2008-2022 log (107M rows) and
neither was exercised by the existing suite, which runs on pilot-sized data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_meanrev.backtest.bootstrap_ci import confidence


def _write_year(dirpath, year, combo_pnls):
    """combo_pnls: {combo_idx: [pnl, ...]} -> one year_<Y>.parquet."""
    rows = []
    ts0 = pd.Timestamp(f"{year}-03-01", tz="UTC")
    for combo_idx, pnls in combo_pnls.items():
        for i, v in enumerate(pnls):
            rows.append({"ts": ts0 + pd.Timedelta(minutes=i), "pair": "EURUSD",
                         "combo_idx": combo_idx, "pnl": float(v)})
    pd.DataFrame(rows).to_parquet(dirpath / f"year_{year}.parquet")


def _patch_paths(monkeypatch, tmp_path, battery_id):
    from gpu_meanrev import config
    monkeypatch.setattr(config, "DATA_REPORTS", tmp_path)
    d = tmp_path / f"{battery_id}_checkpoint" / "trades"
    d.mkdir(parents=True)
    return d


def test_best_combo_streaming_finds_the_known_winner(monkeypatch, tmp_path):
    """Combo 1 has the best mean/std; streaming must find it across files."""
    from gpu_meanrev.reporting import regime_table

    d = _patch_paths(monkeypatch, tmp_path, "BSTREAM")
    rng = np.random.default_rng(7)
    for year in (2019, 2020):
        _write_year(d, year, {
            0: rng.normal(0.0, 1e-4, 400),        # no edge
            1: rng.normal(9e-5, 1e-4, 400),       # clear edge  <- winner
            2: rng.normal(-5e-5, 1e-4, 400),      # negative
        })
    assert regime_table.best_combo_streaming("BSTREAM") == 1


def test_best_combo_streaming_matches_full_load(monkeypatch, tmp_path):
    """Streaming mean/std must agree with the naive whole-log computation."""
    from gpu_meanrev.reporting import regime_table

    d = _patch_paths(monkeypatch, tmp_path, "BMATCH")
    rng = np.random.default_rng(11)
    for year in (2018, 2019, 2020):
        _write_year(d, year, {i: rng.normal(i * 2e-5, 1e-4, 300) for i in range(4)})

    df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("year_*.parquet"))],
                   ignore_index=True)
    stats = df.groupby("combo_idx")["pnl"].agg(["mean", "std", "count"])
    stats = stats[stats["count"] >= 8]
    naive = int((stats["mean"] / stats["std"]).idxmax())
    assert regime_table.best_combo_streaming("BMATCH") == naive


def test_best_combo_streaming_ignores_block_rows_for_covered_years(monkeypatch, tmp_path):
    """A block file overlapping a year-file's year must not be double-counted.

    The block rows here are hugely positive for combo 0. If they leaked into
    the ranking, combo 0 would win; correct precedence keeps combo 1.
    """
    from gpu_meanrev.reporting import regime_table

    d = _patch_paths(monkeypatch, tmp_path, "BDEDUP")
    rng = np.random.default_rng(3)
    _write_year(d, 2020, {0: rng.normal(0.0, 1e-4, 400),
                          1: rng.normal(9e-5, 1e-4, 400)})
    ts0 = pd.Timestamp("2020-06-01", tz="UTC")
    pd.DataFrame([{"ts": ts0 + pd.Timedelta(minutes=i), "pair": "EURUSD",
                   "combo_idx": 0, "pnl": 5e-3} for i in range(400)]
                 ).to_parquet(d / "block_00000.parquet")

    assert regime_table.best_combo_streaming("BDEDUP") == 1


@pytest.mark.parametrize("n", [500, 40_000])
def test_bootstrap_chunking_is_statistically_equivalent(n):
    """Forces rows_per_chunk < n_boot at n=40k; CI must stay stable."""
    rng = np.random.default_rng(23)
    pnls = rng.normal(3e-5, 1e-3, n)
    a = confidence(pnls, n_boot=2000, n_perm=2000)
    b = confidence(pnls, n_boot=2000, n_perm=2000)
    assert a is not None and a == b, "same input+seed must be reproducible"
    lo, hi = a["sharpe_ci"]
    assert lo < hi
    # The CI must bracket the point-estimate Sharpe it is a CI *for*.
    point = float(np.mean(pnls) / np.std(pnls, ddof=1))
    assert lo <= point <= hi
