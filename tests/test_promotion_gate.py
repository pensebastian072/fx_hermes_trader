"""Promotion gate checks against explicit gate configs."""

from hermes.promotion_gate import evaluate_backtest_gates, evaluate_walk_forward_gates

GATES = {
    "min_profit_factor": 1.20,
    "max_backtest_drawdown": 0.08,
    "min_walk_forward_splits": 3,
    "require_positive_oos": True,
}

GOOD_METRICS = {"profit_factor": 1.5, "max_drawdown_pct": 0.04, "n_trades": 40}


def test_passing_backtest():
    result = evaluate_backtest_gates(GOOD_METRICS, GATES)
    assert result.passed and not result.failures


def test_low_profit_factor_fails():
    result = evaluate_backtest_gates({**GOOD_METRICS, "profit_factor": 1.0}, GATES)
    assert not result.passed
    assert any("profit_factor" in f for f in result.failures)


def test_deep_drawdown_fails():
    result = evaluate_backtest_gates({**GOOD_METRICS, "max_drawdown_pct": 0.12}, GATES)
    assert not result.passed
    assert any("max_drawdown" in f for f in result.failures)


def test_no_trades_fails():
    result = evaluate_backtest_gates({**GOOD_METRICS, "n_trades": 0}, GATES)
    assert not result.passed


def test_walk_forward_split_count_and_oos():
    ok = [{"net_pnl": 10.0}, {"net_pnl": -2.0}, {"net_pnl": 5.0}]
    assert evaluate_walk_forward_gates(ok, GATES).passed

    too_few = evaluate_walk_forward_gates(ok[:2], GATES)
    assert not too_few.passed

    negative = evaluate_walk_forward_gates(
        [{"net_pnl": -10.0}, {"net_pnl": 1.0}, {"net_pnl": 2.0}], GATES
    )
    assert not negative.passed
