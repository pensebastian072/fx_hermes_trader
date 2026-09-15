# tests/

pytest suite. Run from repo root with `pytest` (127 tests).

- `conftest.py` - fixtures: `data_dir` redirects all logs to a temp dir via `FXHT_DATA_DIR` so tests never touch real `data/`; `client` (FastAPI TestClient); `valid_payload`
- coverage: alert schema, webhook pipeline, paper broker, risk engine + regime scaling, HMM regime detector, ensemble, features, ML specialist, LSTM specialist, economic calendar, promotion gate, embeddings, sentiment, supervisor, backtests

`test_lstm_specialist.py` uses `pytest.importorskip("torch")`, so the suite still passes on a box without torch installed.
