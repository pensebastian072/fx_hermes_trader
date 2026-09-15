# CLAUDE_CODE_START_PROMPT.md

You are Claude Code working inside VS Code on a Windows desktop.

We are building a local-first, paper-traded FX research and operator system called `fx_hermes_trader`.

Read `PROJECT_PLAN.md` first and follow it.

## Project Mission

Build a serious FX paper-trading research harness using:

- Python
- FastAPI
- Pydantic
- Pandas/Polars/PyArrow
- SQLite/DuckDB or JSONL/parquet artifacts
- Streamlit for the first dashboard
- pytest for tests
- TradingView-style webhook alerts
- deterministic risk controls
- local Hermes/Llama 7B–8B model integration later

This is **paper trading only** at first.

Do not build live broker execution in the first milestone.

## Critical Safety Rules

You must not:

- add live trading execution
- add real broker order placement
- disable risk controls
- remove stop losses
- increase risk limits without explicit human approval
- store API keys in code
- store API keys in prompts
- put secrets in frontend code
- average down losers
- create martingale/grid logic
- optimize for profit only
- use lookahead bias
- create repainting TradingView logic
- hide rejected trades
- let the LLM directly place unrestricted orders

The risk engine has final veto power.

The LLM/Hermes layer is a research/review layer, not the direct live trader.

## Build Milestone 1

Create the repository skeleton and a working local app.

Target structure:

```text
fx_hermes_trader/
  README.md
  PROJECT_PLAN.md
  CLAUDE_CODE_START_PROMPT.md
  .env.example
  requirements.txt
  pyproject.toml

  app/
    api/
      main.py
      routes/
        health.py
        webhooks.py
        signals.py
    services/
      artifact_service.py
      signal_service.py
      report_service.py
    schemas/
      alerts.py
      signals.py
      trades.py
      decisions.py

  configs/
    active/
      autonomy.yaml
      risk.yaml
      watchlist.yaml
      scoring.yaml
    proposed/
    archived/

  data/
    raw/
    processed/
    features/
    logs/
    screenshots/
    artifacts/

  engines/
    scoring.py
    features.py
    regime.py
    basket_mean_reversion/
    triangle_regime/
    trend_following/
    session_breakout/
    macro_sentiment/
    chart_patterns/

  risk/
    risk_engine.py
    breakers.py
    event_blackout.py
    exposure.py

  execution/
    paper_broker.py
    broker_base.py
    oanda_adapter_stub.py

  hermes/
    supervisor.py
    journal_agent.py
    promotion_gate.py
    prompts/
      system_prompt.md
      daily_review_prompt.md
      experiment_prompt.md
    memory/
      trade_lessons.jsonl
      strategy_notes.md

  backtests/
    runner.py
    walk_forward.py
    monte_carlo.py
    reports/

  ui/
    streamlit_app.py

  reports/
    daily/
    weekly/
    experiments/

  tests/
    test_alert_schema.py
    test_webhook.py
    test_risk_engine.py
    test_paper_broker.py
```

## Milestone 1 Functional Requirements

Implement:

1. FastAPI app with:
   - `GET /health`
   - `POST /webhook/tradingview`
   - basic error handling

2. TradingView alert schema:
   - source
   - symbol
   - timeframe
   - price
   - strategy
   - signal
   - timestamp
   - optional alert_id

3. Append-only artifact logging:
   - valid alerts to `data/logs/alerts.jsonl`
   - rejected alerts to `data/logs/rejected_alerts.jsonl`
   - decision traces to `data/logs/decision_traces.jsonl`

4. Duplicate protection:
   - if `alert_id` exists, block duplicate
   - if no `alert_id`, generate hash from symbol/timeframe/price/signal/timestamp

5. Config loading:
   - `configs/active/risk.yaml`
   - `configs/active/watchlist.yaml`
   - `configs/active/autonomy.yaml`
   - `configs/active/scoring.yaml`

6. Basic scoring stub:
   - returns `final_score`
   - returns `direction`
   - returns reason
   - no real strategy needed yet

7. Risk engine:
   - max daily loss
   - max weekly loss
   - max open trades
   - max trades per pair per day
   - paper mode default
   - close-only mode
   - event blackout stub

8. Paper broker:
   - create simulated order
   - store open position
   - write order to `data/logs/orders.jsonl`
   - write fills to `data/logs/fills.jsonl`
   - no real broker execution

9. Streamlit dashboard:
   - show latest alerts
   - show latest decisions
   - show open paper positions
   - show risk config
   - show system mode: PAPER
   - show warning that live trading is disabled

10. Tests:
   - alert schema validation
   - invalid alert rejection
   - webhook writes JSONL
   - duplicate alert blocked
   - risk engine blocks when limits exceeded
   - paper broker creates simulated order

## Development Style

Work incrementally.

Before coding, show a short implementation plan.

After coding, summarize:

- files created
- files changed
- how to run
- tests run
- any limitations

Use simple code first.

Avoid premature complexity.

Do not build a full agent framework before the webhook, logs, risk engine, paper broker, and dashboard are working.

## Local Run Commands

The project should run with:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
streamlit run ui/streamlit_app.py
pytest
```

## TradingView Test Payload

Use this test JSON:

```json
{
  "source": "tradingview",
  "symbol": "USDJPY",
  "timeframe": "15m",
  "price": 156.10,
  "strategy": "fx_basket_v1",
  "signal": "short_candidate",
  "timestamp": "2026-06-11T14:30:00-04:00",
  "alert_id": "test-usdjpy-001"
}
```

## Expected Webhook Behavior

When a valid alert arrives:

1. Validate the payload.
2. Check duplicate status.
3. Save alert to JSONL.
4. Run scoring stub.
5. Run risk check.
6. Create decision trace.
7. If paper mode and risk allows, create simulated paper order.
8. Return structured JSON response.

Example response:

```json
{
  "status": "accepted",
  "symbol": "USDJPY",
  "decision": "paper_order_created",
  "final_score": -76,
  "mode": "paper",
  "live_trading": false
}
```

## Final Rule

Build the safe skeleton first.

No live trading.

No hidden risk changes.

No secrets.

Everything logged.

Everything testable.
