import json

import pytest

from execution.paper_broker import PaperBroker


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_broker(tmp_path) -> PaperBroker:
    return PaperBroker(tmp_path, starting_capital=10000.0)


def test_create_order_logs_order_and_fill(tmp_path):
    broker = make_broker(tmp_path)
    order = broker.create_order(
        symbol="USDJPY", direction="short", entry=156.10, stop=156.57,
        target=155.16, size=50.0, risk_percent=0.0025,
    )
    assert order.mode == "paper"
    assert order.status == "open"
    assert len(broker.open_positions()) == 1

    orders = read_lines(tmp_path / "orders.jsonl")
    fills = read_lines(tmp_path / "fills.jsonl")
    assert len(orders) == 1
    assert len(fills) == 1
    assert fills[0]["kind"] == "open"
    assert fills[0]["price"] == pytest.approx(156.10)


def test_invalid_direction_rejected(tmp_path):
    broker = make_broker(tmp_path)
    with pytest.raises(ValueError):
        broker.create_order(
            symbol="USDJPY", direction="buy", entry=156.10, stop=156.57,
            target=155.16, size=50.0, risk_percent=0.0025,
        )


def test_close_position_realizes_pnl(tmp_path):
    broker = make_broker(tmp_path)
    order = broker.create_order(
        symbol="EURUSD", direction="long", entry=1.0800, stop=1.0768,
        target=1.0865, size=1000.0, risk_percent=0.0025,
    )
    pnl = broker.close_position(order.order_id, price=1.0850)
    assert pnl == pytest.approx(5.0)
    assert broker.equity == pytest.approx(10005.0)
    assert broker.open_positions() == []

    fills = read_lines(tmp_path / "fills.jsonl")
    assert fills[-1]["kind"] == "close"
    assert fills[-1]["realized_pnl"] == pytest.approx(5.0)


def test_short_pnl_math(tmp_path):
    broker = make_broker(tmp_path)
    order = broker.create_order(
        symbol="USDJPY", direction="short", entry=156.10, stop=156.57,
        target=155.16, size=100.0, risk_percent=0.0025,
    )
    pnl = broker.close_position(order.order_id, price=155.60)
    assert pnl == pytest.approx(50.0)


def test_state_rebuilt_from_logs(tmp_path):
    broker = make_broker(tmp_path)
    o1 = broker.create_order(
        symbol="USDJPY", direction="long", entry=156.10, stop=155.63,
        target=157.04, size=50.0, risk_percent=0.0025,
    )
    broker.create_order(
        symbol="EURUSD", direction="long", entry=1.0800, stop=1.0768,
        target=1.0865, size=1000.0, risk_percent=0.0025,
    )
    broker.close_position(o1.order_id, price=156.60)

    rebuilt = make_broker(tmp_path)
    assert len(rebuilt.open_positions()) == 1
    assert rebuilt.open_positions()[0].symbol == "EURUSD"
    assert rebuilt.equity == pytest.approx(broker.equity)
    assert rebuilt.trades_today() == 2
    assert rebuilt.trades_today("USDJPY") == 1


def test_daily_pnl_pct(tmp_path):
    broker = make_broker(tmp_path)
    order = broker.create_order(
        symbol="USDJPY", direction="long", entry=156.10, stop=155.63,
        target=157.04, size=100.0, risk_percent=0.0025,
    )
    broker.close_position(order.order_id, price=155.10)  # -100
    assert broker.daily_pnl_pct() == pytest.approx(-0.01)
    assert broker.weekly_pnl_pct() == pytest.approx(-0.01)


def test_close_unknown_order_rejected(tmp_path):
    broker = make_broker(tmp_path)
    with pytest.raises(ValueError):
        broker.close_position("does-not-exist", price=1.0)
