import numpy as np
import pytest

from backtests.data_loader import synthetic_closes
from engines.ensemble import (
    CapitalPreservationSpecialist,
    MeanReversionSpecialist,
    RegimeGatedEnsemble,
    TrendSpecialist,
)
from engines.regime_detection.hmm_regime import CRISIS, RANGE, TREND


def test_trend_specialist_sign():
    up = synthetic_closes([("trend_up", 200)], seed=1)
    down = synthetic_closes([("trend_down", 200)], seed=1)
    spec = TrendSpecialist()
    assert spec.score(up) > 20
    assert spec.score(down) < -20


def test_mean_reversion_fades_extremes():
    base = np.full(60, 100.0)
    stretched_up = np.concatenate([base, np.linspace(100, 106, 10)])
    assert MeanReversionSpecialist().score(stretched_up) < -30  # fade the spike


def test_crisis_specialist_always_flat():
    crash = synthetic_closes([("crisis", 100)], seed=2)
    assert CapitalPreservationSpecialist().score(crash) == 0.0


def test_ensemble_is_probability_weighted():
    closes = synthetic_closes([("trend_up", 200)], seed=1)
    ens = RegimeGatedEnsemble()
    pure_trend = ens.score(closes, {TREND: 1.0, RANGE: 0.0, CRISIS: 0.0})
    pure_crisis = ens.score(closes, {TREND: 0.0, RANGE: 0.0, CRISIS: 1.0})
    half = ens.score(closes, {TREND: 0.5, RANGE: 0.0, CRISIS: 0.5})
    assert pure_crisis.final_score == 0.0
    assert half.final_score == pytest.approx(pure_trend.final_score * 0.5)


def test_ensemble_clips_to_bounds():
    closes = synthetic_closes([("trend_up", 400)], seed=4)
    score = RegimeGatedEnsemble().score(closes, {TREND: 1.0})
    assert -100 <= score.final_score <= 100
    assert "TREND" in score.reason
