import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point all logs at a temp dir so tests never touch real data/."""
    monkeypatch.setenv("FXHT_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(data_dir):
    from app.api.main import create_app

    return TestClient(create_app())


@pytest.fixture
def valid_payload():
    return {
        "source": "tradingview",
        "symbol": "USDJPY",
        "timeframe": "15m",
        "price": 156.10,
        "strategy": "fx_basket_v1",
        "signal": "short_candidate",
        "timestamp": "2026-06-11T14:30:00-04:00",
        "alert_id": "test-usdjpy-001",
    }
