from datetime import datetime, timezone

import pytest

from risk.risk_engine import PortfolioState, RiskEngine

NOW = datetime(2026, 6, 11, 14, 30, tzinfo=timezone.utc)

CONFIG = {
    "regime_scaling": {
        "enabled": True,
        "crisis_regime": "HIGH_VOL_CRISIS",
        "scale_start_probability": 0.5,
        "halt_probability": 0.8,
    }
}


def test_full_size_below_scale_start():
    eng = RiskEngine(CONFIG)
    assert eng.regime_size_multiplier(0.0) == 1.0
    assert eng.regime_size_multiplier(0.5) == 1.0


def test_linear_ramp_between_thresholds():
    eng = RiskEngine(CONFIG)
    assert eng.regime_size_multiplier(0.65) == pytest.approx(0.5)
    assert eng.regime_size_multiplier(0.725) == pytest.approx(0.25)


def test_zero_size_at_halt():
    eng = RiskEngine(CONFIG)
    assert eng.regime_size_multiplier(0.8) == 0.0
    assert eng.regime_size_multiplier(0.99) == 0.0


def test_halt_vetoes_new_trades():
    eng = RiskEngine(CONFIG)
    decision = eng.evaluate("USDJPY", NOW, PortfolioState(crisis_probability=0.85))
    assert not decision.allowed
    assert any("crisis regime halt" in r for r in decision.reasons)


def test_below_halt_allowed():
    eng = RiskEngine(CONFIG)
    decision = eng.evaluate("USDJPY", NOW, PortfolioState(crisis_probability=0.7))
    assert decision.allowed


def test_disabled_scaling_is_neutral():
    eng = RiskEngine({"regime_scaling": {"enabled": False}})
    assert eng.regime_size_multiplier(0.99) == 1.0
    decision = eng.evaluate("USDJPY", NOW, PortfolioState(crisis_probability=0.99))
    assert decision.allowed
