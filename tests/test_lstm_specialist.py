"""LSTM trend specialist: edge gate, fallback, load_ensemble priority."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backtests.data_loader import synthetic_closes
from engines.ensemble import TrendSpecialist
from engines.lstm_specialist import LSTMSpecialist, LSTMTrendNet, TREND_LSTM_MODEL
from engines.ml_specialist import (
    FEATURES_VERSION,
    MLSpecialist,
    N_FEATURES,
    TREND_MODEL,
    build_lstm_sequences,
    load_ensemble,
    models_dir,
)
from engines.regime_detection.hmm_regime import TREND

SEQ_LEN = 10
HIDDEN_SIZE = 4


@pytest.fixture
def closes():
    return synthetic_closes([("calm", 150), ("trend_up", 150)])


def _write_lstm_model(meta_overrides: dict) -> None:
    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    net = LSTMTrendNet(input_size=N_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=1)
    torch.save(net.state_dict(), out / TREND_LSTM_MODEL)
    meta = {
        "features_version": FEATURES_VERSION,
        "horizon": 5,
        "seq_len": SEQ_LEN,
        "input_size": N_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 1,
        "feature_mean": [0.0] * N_FEATURES,
        "feature_std": [1.0] * N_FEATURES,
        "oos_accuracy": 0.60,
        "baseline_accuracy": 0.52,
        **meta_overrides,
    }
    (out / TREND_LSTM_MODEL).with_suffix(".meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_build_lstm_sequences_shape(closes):
    X, y = build_lstm_sequences(closes, horizon=5, seq_len=SEQ_LEN)
    assert X.ndim == 3 and X.shape[1:] == (SEQ_LEN, N_FEATURES)
    assert len(X) == len(y)
    assert set(np.unique(y)) <= {0, 1}


def test_missing_lstm_model_unavailable(data_dir):
    spec = LSTMSpecialist()
    assert not spec.available
    assert spec.score(np.linspace(100, 110, 100)) == 0.0


def test_edge_gate_blocks_below_baseline_lstm(data_dir):
    _write_lstm_model({"oos_accuracy": 0.50, "baseline_accuracy": 0.52})
    assert not LSTMSpecialist().available


def test_stale_features_version_blocks_lstm(data_dir):
    _write_lstm_model({"features_version": FEATURES_VERSION - 1})
    assert not LSTMSpecialist().available


def test_usable_lstm_scores_in_range(data_dir, closes):
    _write_lstm_model({})
    spec = LSTMSpecialist()
    assert spec.available
    score = spec.score(closes)
    assert -100.0 <= score <= 100.0


def test_short_history_scores_zero(data_dir, closes):
    _write_lstm_model({"seq_len": SEQ_LEN})
    spec = LSTMSpecialist()
    # not enough bars for even MIN_BARS-1 + seq_len rows
    assert spec.score(closes[:60]) == 0.0


def test_load_ensemble_prefers_lstm_over_rf(data_dir, closes):
    from engines.ml_specialist import build_dataset
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    _write_lstm_model({})
    X, y = build_dataset(closes)
    out = models_dir()
    rf = RandomForestClassifier(n_estimators=5, random_state=7).fit(X, y)
    joblib.dump(rf, out / TREND_MODEL)
    (out / TREND_MODEL).with_suffix(".meta.json").write_text(
        json.dumps({"features_version": FEATURES_VERSION, "oos_accuracy": 0.6, "baseline_accuracy": 0.52}),
        encoding="utf-8",
    )

    ensemble = load_ensemble()
    assert isinstance(ensemble.specialists[TREND], LSTMSpecialist)


def test_load_ensemble_falls_back_to_rules_without_torch_models(data_dir):
    ensemble = load_ensemble()
    assert isinstance(ensemble.specialists[TREND], TrendSpecialist)
