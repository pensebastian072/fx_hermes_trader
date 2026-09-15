import numpy as np
import pytest

from backtests.data_loader import synthetic_closes
from engines.regime_detection.hmm_regime import (
    CRISIS,
    HMMRegimeDetector,
    build_features,
    load_snapshot,
    save_snapshot,
)


@pytest.fixture(scope="module")
def fitted():
    closes = synthetic_closes([("calm", 300), ("crisis", 120), ("calm", 100)], seed=11)
    return HMMRegimeDetector(seed=7).fit(closes), closes


def test_build_features_shape():
    closes = synthetic_closes([("calm", 100)])
    feats = build_features(closes)
    assert feats.shape[1] == 2
    assert len(feats) == len(closes) - 1 - 20  # returns minus vol warmup


def test_filtered_rows_are_distributions(fitted):
    detector, closes = fitted
    probs = detector.filtered_probabilities(closes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0).all()


def test_crisis_detected_in_high_vol_segment(fitted):
    detector, _ = fitted
    calm = synthetic_closes([("calm", 250)], seed=3)
    crisis = synthetic_closes([("calm", 150), ("crisis", 120)], seed=3)
    p_calm = detector.regime_probabilities(calm)[CRISIS]
    p_crisis = detector.regime_probabilities(crisis)[CRISIS]
    assert p_crisis > p_calm
    assert p_crisis > 0.5


def test_deterministic_with_seed():
    closes = synthetic_closes([("calm", 200), ("crisis", 80)], seed=5)
    a = HMMRegimeDetector(seed=7).fit(closes).regime_probabilities(closes)
    b = HMMRegimeDetector(seed=7).fit(closes).regime_probabilities(closes)
    assert a == b


def test_snapshot_roundtrip(tmp_path, fitted):
    detector, closes = fitted
    snap = detector.snapshot(closes, "USDJPY")
    save_snapshot(snap, tmp_path)
    loaded = load_snapshot(tmp_path)
    assert loaded is not None
    assert loaded.symbol == "USDJPY"
    assert loaded.probabilities == pytest.approx(snap.probabilities)
    assert abs(sum(loaded.probabilities.values()) - 1.0) < 0.01


def test_missing_snapshot_returns_none(tmp_path):
    assert load_snapshot(tmp_path) is None


def test_vol_guard_floors_crisis_when_model_is_calm_blind():
    # Fit on calm data ONLY - the HMM has never seen a crisis state.
    calm = synthetic_closes([("calm", 150)], seed=8)
    detector = HMMRegimeDetector(seed=7).fit(calm)
    shocked = synthetic_closes([("calm", 250), ("crisis", 100)], seed=8)
    probs = detector.regime_probabilities(shocked)
    assert probs[CRISIS] > 0.8  # deterministic guard engaged
    assert abs(sum(probs.values()) - 1.0) < 1e-6
