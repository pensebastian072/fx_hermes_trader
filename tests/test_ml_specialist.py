"""ML specialist wrapper: feature builder, dataset, edge gate, fallback."""

import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from backtests.data_loader import synthetic_closes
from engines.ensemble import MeanReversionSpecialist, TrendSpecialist
from engines.ml_specialist import (
    FEATURE_WINDOW,
    FEATURES_VERSION,
    MIN_BARS,
    MLSpecialist,
    TREND_MODEL,
    build_dataset,
    build_feature_row,
    load_ensemble,
    models_dir,
)
from engines.regime_detection.hmm_regime import RANGE, TREND


@pytest.fixture
def closes():
    return synthetic_closes([("calm", 150), ("trend_up", 150)])


def _write_model(meta_overrides: dict, X, y) -> None:
    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    model = RandomForestClassifier(n_estimators=5, random_state=7).fit(X, y)
    joblib.dump(model, out / TREND_MODEL)
    meta = {
        "features_version": FEATURES_VERSION,
        "oos_accuracy": 0.60,
        "baseline_accuracy": 0.52,
        **meta_overrides,
    }
    (out / TREND_MODEL).with_suffix(".meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_feature_row_shape_and_short_history(closes):
    row = build_feature_row(closes)
    assert row is not None and row.shape == (13,)
    assert np.all(np.isfinite(row))
    assert build_feature_row(closes[: MIN_BARS - 1]) is None


def test_feature_row_window_capped(closes):
    # identical result whether the caller holds 300 or exactly FEATURE_WINDOW bars
    full = build_feature_row(closes)
    capped = build_feature_row(closes[-FEATURE_WINDOW:])
    np.testing.assert_allclose(full, capped)


def test_build_dataset_no_lookahead(closes):
    X, y = build_dataset(closes, horizon=5)
    assert len(X) == len(y) == len(closes) - MIN_BARS - 5
    assert set(np.unique(y)) <= {0, 1}
    # last label uses closes[-1]; truncating the future must not change earlier rows
    X2, y2 = build_dataset(closes[:-1], horizon=5)
    np.testing.assert_allclose(X[: len(X2)], X2)


def test_market_returns_changes_correlation_feature(closes):
    rng = np.random.default_rng(3)
    own_returns = np.diff(np.log(closes))
    # market that mirrors this pair's returns exactly -> correlation feature ~ 1
    market_returns = np.concatenate([[0.0], own_returns])
    row_corr = build_feature_row(closes, market_returns)
    row_neutral = build_feature_row(closes)
    assert row_corr[-1] == pytest.approx(1.0, abs=1e-6)
    assert row_neutral[-1] == 0.0
    assert -1.0 <= row_corr[-1] <= 1.0
    # uncorrelated noise market -> correlation feature differs from the perfect case
    noisy_market = np.concatenate([[0.0], rng.normal(0, 0.001, len(own_returns))])
    row_noisy = build_feature_row(closes, noisy_market)
    assert row_noisy[-1] != row_corr[-1]


def test_build_dataset_market_returns_length_mismatch(closes):
    with pytest.raises(ValueError):
        build_dataset(closes, market_returns=np.zeros(len(closes) - 1))


def test_build_dataset_with_market_returns(closes):
    market_returns = np.zeros(len(closes))
    X, y = build_dataset(closes, market_returns=market_returns, horizon=5)
    assert X.shape[1] == 13
    assert len(X) == len(y) == len(closes) - MIN_BARS - 5


def test_missing_model_scores_zero(data_dir):
    spec = MLSpecialist("trend_rf", TREND_MODEL)
    assert not spec.available
    assert spec.score(np.linspace(100, 110, 100)) == 0.0


def test_edge_gate_blocks_below_baseline_model(data_dir, closes):
    X, y = build_dataset(closes)
    _write_model({"oos_accuracy": 0.50, "baseline_accuracy": 0.52}, X, y)
    assert not MLSpecialist("trend_rf", TREND_MODEL).available


def test_stale_features_version_blocks_model(data_dir, closes):
    X, y = build_dataset(closes)
    _write_model({"features_version": FEATURES_VERSION - 1}, X, y)
    assert not MLSpecialist("trend_rf", TREND_MODEL).available


def test_usable_model_loads_and_scores_in_range(data_dir, closes):
    X, y = build_dataset(closes)
    _write_model({}, X, y)
    spec = MLSpecialist("trend_rf", TREND_MODEL)
    assert spec.available
    score = spec.score(closes)
    assert -100.0 <= score <= 100.0


def test_load_ensemble_falls_back_to_rules(data_dir):
    ensemble = load_ensemble()
    assert isinstance(ensemble.specialists[TREND], TrendSpecialist)
    assert isinstance(ensemble.specialists[RANGE], MeanReversionSpecialist)


def test_load_ensemble_uses_trained_model_when_usable(data_dir, closes):
    X, y = build_dataset(closes)
    _write_model({}, X, y)
    ensemble = load_ensemble()
    assert isinstance(ensemble.specialists[TREND], MLSpecialist)
    assert isinstance(ensemble.specialists[RANGE], MeanReversionSpecialist)
