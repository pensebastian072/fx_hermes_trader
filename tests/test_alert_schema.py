import pytest
from pydantic import ValidationError

from app.schemas.alerts import TradingViewAlert

BASE = {
    "source": "tradingview",
    "symbol": "USDJPY",
    "timeframe": "15m",
    "price": 156.10,
    "strategy": "fx_basket_v1",
    "signal": "short_candidate",
    "timestamp": "2026-06-11T14:30:00-04:00",
}


def test_valid_alert_parses():
    alert = TradingViewAlert(**BASE, alert_id="abc-1")
    assert alert.symbol == "USDJPY"
    assert alert.price == pytest.approx(156.10)
    assert alert.dedupe_key() == "abc-1"


def test_symbol_normalization():
    alert = TradingViewAlert(**{**BASE, "symbol": "OANDA:eur/usd"})
    assert alert.symbol == "EURUSD"


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", ""),
        ("symbol", "EU1"),
        ("price", -1.0),
        ("price", 0),
        ("signal", "  "),
        ("timestamp", "not-a-date"),
    ],
)
def test_invalid_fields_rejected(field, value):
    with pytest.raises(ValidationError):
        TradingViewAlert(**{**BASE, field: value})


def test_missing_required_field_rejected():
    payload = dict(BASE)
    del payload["symbol"]
    with pytest.raises(ValidationError):
        TradingViewAlert(**payload)


def test_dedupe_key_without_alert_id_is_stable():
    a = TradingViewAlert(**BASE)
    b = TradingViewAlert(**BASE)
    c = TradingViewAlert(**{**BASE, "price": 156.11})
    assert a.dedupe_key() == b.dedupe_key()
    assert a.dedupe_key() != c.dedupe_key()
