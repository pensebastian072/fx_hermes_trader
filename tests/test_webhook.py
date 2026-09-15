import json


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "paper"
    assert body["live_trading"] is False


def test_valid_alert_accepted_and_logged(client, data_dir, valid_payload):
    resp = client.post("/webhook/tradingview", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["decision"] == "paper_order_created"
    assert body["final_score"] == -75
    assert body["mode"] == "paper"
    assert body["live_trading"] is False

    alerts = read_lines(data_dir / "logs" / "alerts.jsonl")
    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "USDJPY"
    assert alerts[0]["dedupe_key"] == "test-usdjpy-001"

    traces = read_lines(data_dir / "logs" / "decision_traces.jsonl")
    assert len(traces) == 1
    assert traces[0]["decision"] == "accept"
    assert traces[0]["mode"] == "paper"

    orders = read_lines(data_dir / "logs" / "orders.jsonl")
    assert len(orders) == 1
    assert orders[0]["mode"] == "paper"
    assert orders[0]["direction"] == "short"


def test_invalid_alert_rejected_and_logged(client, data_dir, valid_payload):
    bad = {**valid_payload, "price": "not-a-number"}
    resp = client.post("/webhook/tradingview", json=bad)
    assert resp.status_code == 422
    assert resp.json()["status"] == "rejected"

    rejected = read_lines(data_dir / "logs" / "rejected_alerts.jsonl")
    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"] == "schema validation failed"
    assert not (data_dir / "logs" / "alerts.jsonl").exists()


def test_duplicate_alert_blocked(client, data_dir, valid_payload):
    first = client.post("/webhook/tradingview", json=valid_payload)
    assert first.json()["status"] == "accepted"

    second = client.post("/webhook/tradingview", json=valid_payload)
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["decision"] == "blocked_duplicate"

    alerts = read_lines(data_dir / "logs" / "alerts.jsonl")
    assert len(alerts) == 1
    orders = read_lines(data_dir / "logs" / "orders.jsonl")
    assert len(orders) == 1


def test_symbol_not_in_watchlist_rejected(client, data_dir, valid_payload):
    payload = {**valid_payload, "symbol": "EURNOK", "alert_id": "test-eurnok-001"}
    resp = client.post("/webhook/tradingview", json=payload)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "symbol_not_in_watchlist"
    rejected = read_lines(data_dir / "logs" / "rejected_alerts.jsonl")
    assert len(rejected) == 1
    assert not (data_dir / "logs" / "orders.jsonl").exists()


def test_low_score_signal_creates_no_trade(client, data_dir, valid_payload):
    payload = {**valid_payload, "signal": "watch", "alert_id": "test-watch-001"}
    resp = client.post("/webhook/tradingview", json=payload)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "no_trade"
    traces = read_lines(data_dir / "logs" / "decision_traces.jsonl")
    assert traces[0]["decision"] == "reject"
    assert not (data_dir / "logs" / "orders.jsonl").exists()


def test_pair_daily_limit_blocks_third_trade(client, data_dir, valid_payload):
    # max_trades_per_pair_per_day = 2
    for i in range(3):
        payload = {**valid_payload, "alert_id": f"test-usdjpy-{i}"}
        resp = client.post("/webhook/tradingview", json=payload)
        assert resp.status_code == 200
    assert resp.json()["decision"] == "rejected_by_risk"
    assert any("USDJPY" in r for r in resp.json()["reasons"])
    orders = read_lines(data_dir / "logs" / "orders.jsonl")
    assert len(orders) == 2
