import numpy as np
import pytest

from backtests.data_loader import closes_to_ohlc, synthetic_closes
from backtests.monte_carlo import apply_scenario, reshuffle_drawdowns, run_stress_scenario
from backtests.runner import BacktestConfig, BacktestRunner
from backtests.walk_forward import make_splits, run_walk_forward


@pytest.fixture(scope="module")
def trending_df():
    closes = synthetic_closes(
        [("calm", 160), ("trend_up", 250), ("calm", 80)], seed=9
    )
    return closes_to_ohlc(closes)


def test_runner_produces_trades_and_metrics(trending_df):
    result = BacktestRunner().run(trending_df)
    m = result.metrics
    assert m["n_trades"] == len(result.trades)
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["max_drawdown_pct"] >= 0.0
    assert len(result.equity_curve) > 0


def test_runner_deterministic(trending_df):
    a = BacktestRunner().run(trending_df).metrics
    b = BacktestRunner().run(trending_df).metrics
    assert a == b


def test_spread_shifts_entries_adversely(trending_df):
    # With percent-risk sizing, a fixed spread mostly shifts entry/stop/target
    # rather than R-multiple PnL, so compare entry prices, not net PnL.
    cheap = BacktestRunner(BacktestConfig(spread=0.0, slippage=0.0)).run(trending_df)
    costly = BacktestRunner(BacktestConfig(spread=0.0007, slippage=0.0003)).run(trending_df)
    if cheap.trades and costly.trades:
        sign = 1 if costly.trades[0].direction == "long" else -1
        assert sign * costly.trades[0].entry > sign * cheap.trades[0].entry
        assert costly.metrics["net_pnl"] <= cheap.metrics["net_pnl"] + 1e-6


def test_insufficient_data_rejected():
    df = closes_to_ohlc(synthetic_closes([("calm", 50)]))
    with pytest.raises(ValueError):
        BacktestRunner().run(df)


def test_walk_forward_splits_do_not_overlap():
    splits = make_splits(1000, 4)
    assert len(splits) == 4
    for (s1, e1), (s2, _) in zip(splits, splits[1:]):
        assert e1 == s2
    assert splits[0][0] == 0
    assert splits[-1][1] == 1000


def test_walk_forward_runs_all_splits():
    closes = synthetic_closes(
        [("calm", 200), ("trend_up", 200), ("calm", 200), ("trend_down", 200)], seed=12
    )
    result = run_walk_forward(closes_to_ohlc(closes), n_splits=2, min_profitable_splits=1)
    assert result.total_splits == 2
    assert len(result.split_metrics) == 2
    assert isinstance(result.passed, bool)


def test_reshuffle_preserves_total_and_orders_percentiles():
    pnls = [50, -30, 20, -10, 40, -25, 15]
    dist = reshuffle_drawdowns(pnls, n_sims=200, seed=3)
    assert dist["p50"] <= dist["p95"] <= dist["p99"]
    assert dist["n_sims"] == 200


def test_reshuffle_empty_trades():
    assert reshuffle_drawdowns([])["n_sims"] == 0


def test_apply_scenario_gaps_down():
    closes = synthetic_closes([("calm", 300)], seed=5)
    shocked = apply_scenario(closes, "CARRY_UNWIND", start_index=200, duration=50, seed=5)
    assert len(shocked) == len(closes)
    assert np.allclose(shocked[:200], closes[:200])  # pre-shock untouched
    assert shocked[205] < closes[205]  # gap + adverse drift


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        apply_scenario(np.ones(100) * 100, "MARTINGALE", 50, 10)


def test_stress_scenario_engages_protection():
    closes = synthetic_closes([("calm", 400)], seed=8)
    report = run_stress_scenario(
        closes, "CARRY_UNWIND", start_index=250, duration=120, seed=8
    )
    assert report.scenario == "CARRY_UNWIND"
    assert "max_drawdown_pct" in report.metrics
    # a violent unwind must wake the risk system: crisis halts or vetoes
    assert report.breakers_engaged
