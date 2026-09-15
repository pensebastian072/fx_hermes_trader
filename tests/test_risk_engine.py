from datetime import datetime, timezone

import pytest

from risk.risk_engine import PortfolioState, RiskDecision, RiskEngine

NOW = datetime(2026, 6, 11, 14, 30, tzinfo=timezone.utc)

CONFIG = {
    "starting_capital": 10000,
    "base_risk_per_trade": 0.0025,
    "max_risk_per_trade": 0.005,
    "max_daily_loss": 0.01,
    "max_weekly_loss": 0.03,
    "max_open_trades": 4,
    "max_trades_per_day": 6,
    "max_trades_per_pair_per_day": 2,
    "min_reward_risk": 1.5,
    "close_only": False,
    "event_blackout_calendar": [],
}


def engine(**overrides) -> RiskEngine:
    return RiskEngine({**CONFIG, **overrides})


def test_clean_state_allowed():
    decision = engine().evaluate("USDJPY", NOW, PortfolioState())
    assert decision.allowed
    assert decision.reasons == []


def test_non_paper_mode_blocked():
    eng = RiskEngine(CONFIG, mode="live")
    decision = eng.evaluate("USDJPY", NOW, PortfolioState())
    assert not decision.allowed


def test_close_only_blocks():
    decision = engine(close_only=True).evaluate("USDJPY", NOW, PortfolioState())
    assert not decision.allowed
    assert any("close-only" in r for r in decision.reasons)


def test_max_daily_loss_blocks():
    state = PortfolioState(daily_pnl_pct=-0.011)
    decision = engine().evaluate("USDJPY", NOW, state)
    assert not decision.allowed
    assert any("daily loss" in r for r in decision.reasons)


def test_max_weekly_loss_blocks():
    state = PortfolioState(weekly_pnl_pct=-0.031)
    decision = engine().evaluate("USDJPY", NOW, state)
    assert not decision.allowed
    assert any("weekly loss" in r for r in decision.reasons)


def test_max_open_trades_blocks():
    state = PortfolioState(open_trades=4)
    decision = engine().evaluate("USDJPY", NOW, state)
    assert not decision.allowed


def test_max_trades_per_day_blocks():
    state = PortfolioState(trades_today=6)
    decision = engine().evaluate("USDJPY", NOW, state)
    assert not decision.allowed


def test_max_trades_per_pair_blocks():
    state = PortfolioState(trades_today_for_pair=2)
    decision = engine().evaluate("USDJPY", NOW, state)
    assert not decision.allowed


def test_event_blackout_blocks():
    calendar = [
        {"name": "US CPI", "start": "2026-06-11T14:00:00+00:00", "end": "2026-06-11T15:00:00+00:00"}
    ]
    decision = engine(event_blackout_calendar=calendar).evaluate("USDJPY", NOW, PortfolioState())
    assert not decision.allowed
    assert any("US CPI" in r for r in decision.reasons)


def test_outside_blackout_allowed():
    calendar = [
        {"name": "US CPI", "start": "2026-06-11T16:00:00+00:00", "end": "2026-06-11T17:00:00+00:00"}
    ]
    decision = engine(event_blackout_calendar=calendar).evaluate("USDJPY", NOW, PortfolioState())
    assert decision.allowed


def test_position_size_formula():
    # 10000 * 0.0025 / 0.50 = 50 units
    assert engine().position_size(10000, 0.0025, 0.50) == pytest.approx(50.0)


def test_position_size_caps_risk():
    capped = engine().position_size(10000, 0.05, 0.50)  # asks 5%, capped at 0.5%
    assert capped == pytest.approx(10000 * 0.005 / 0.50)


def test_position_size_rejects_bad_stop():
    with pytest.raises(ValueError):
        engine().position_size(10000, 0.0025, 0)


def test_reward_risk_floor():
    eng = engine()
    assert eng.check_reward_risk(entry=100.0, stop=99.0, target=102.0)  # 2R
    assert not eng.check_reward_risk(entry=100.0, stop=99.0, target=101.0)  # 1R
