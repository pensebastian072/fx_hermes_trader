# FX Hermes Trader

Local-first, **paper-trading only** FX research harness.

- FastAPI webhook receiver for TradingView-style alerts
- Pydantic validation of every payload
- Append-only JSONL artifact logs (alerts, decisions, orders, fills)
- Deterministic risk engine with veto power
- Paper broker simulator (no real broker, no API keys)
- Streamlit operator dashboard
- pytest test suite

> **Safety:** This repository contains **no live trading execution**. The default
> and only supported mode in Milestone 1 is `paper`. No API keys are stored
> anywhere. The LLM/Hermes layer is a research/review layer and cannot place
> live trades.

## Quick start (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.api.main:app --reload
```

Run the dashboard (second terminal):

```bash
streamlit run ui/streamlit_app.py
```

Run tests:

```bash
pytest
```

## Test the webhook

```bash
curl -X POST http://127.0.0.1:8000/webhook/tradingview ^
  -H "Content-Type: application/json" ^
  -d "{\"source\":\"tradingview\",\"symbol\":\"USDJPY\",\"timeframe\":\"15m\",\"price\":156.10,\"strategy\":\"fx_basket_v1\",\"signal\":\"short_candidate\",\"timestamp\":\"2026-06-11T14:30:00-04:00\",\"alert_id\":\"test-usdjpy-001\"}"
```

Expected response:

```json
{
  "status": "accepted",
  "symbol": "USDJPY",
  "decision": "paper_order_created",
  "final_score": -75,
  "mode": "paper",
  "live_trading": false
}
```

## v2 research tools

HMM regime snapshot (feeds risk-engine position scaling + crisis halt):

```bash
python -m engines.regime_detection.run_regime --demo
python -m engines.regime_detection.run_regime --csv data/raw/usdjpy_1h.csv --symbol USDJPY
```

Central-bank sentiment (multi-agent debate over local transcripts; bias only):

```bash
# drop .txt transcripts into data/raw/central_bank/ first
python -m hermes.macro_sentiment.run_sentiment --currency USD --inflation-above-target
python -m hermes.macro_sentiment.run_sentiment --currency USD --ollama   # use local LLM
```

Backtests / stress testing (Python API):

```python
from backtests.data_loader import load_ohlc
from backtests.runner import BacktestRunner
from backtests.walk_forward import run_walk_forward
from backtests.monte_carlo import reshuffle_drawdowns, run_stress_scenario

df = load_ohlc("data/raw/usdjpy_1h.csv")
result = BacktestRunner().run(df)
wf = run_walk_forward(df, n_splits=4, min_profitable_splits=3)
mc = reshuffle_drawdowns(result.trade_pnls)
stress = run_stress_scenario(df["close"].to_numpy(), "CARRY_UNWIND", 500, 120)
```

## v3 tools (data, models, calendar, supervisor)

Download real daily FX history (Yahoo Finance, back to 2003, no API key):

```bash
python -m backtests.download_data                 # all watchlist pairs
python -m backtests.download_data --interval 1h   # intraday (~2 years max)
```

Train ML specialists (RandomForest trend / calibrated SVM range). A model
that does not beat the majority-class baseline out-of-sample is saved but
never loaded - the rule-based specialists stay active:

```bash
python -m backtests.train_specialists
```

## v4 tools (richer features, LSTM)

13-feature v2 vector (`engines/ml_specialist.py`): returns, momentum,
z-score, EMA21/55 + EMA100/200 separation, RSI, MACD histogram, Bollinger %B,
short/long vol ratio, and a cross-pair correlation feature (each pair scored
against the mean return of the other watchlist pairs).

Train the LSTM trend specialist (GPU if available; needs torch). Same
out-of-sample edge gate - `load_ensemble()` tiers LSTM > RandomForest >
rule-based for the TREND slot, using the first that passes its gate:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
python -m backtests.train_lstm                  # all pairs, 20 epochs
python -m backtests.train_lstm --epochs 30 --pairs usdjpy eurusd
```

> Note: on this dataset RF, SVM, and LSTM all land at/below the majority
> baseline for 5-bar daily-FX direction, so the edge gate keeps every trained
> model off and the deterministic specialists stay active. The plumbing is
> in place for when a model with real edge appears.

Generate the event-blackout calendar (NFP by rule, FOMC/CPI/ECB from
`configs/active/economic_calendar.yaml`; windows from autonomy.yaml):

```bash
python -m risk.economic_calendar --days 90
```

Hermes daily supervisor - refreshes artifacts, backtests recent history,
checks promotion gates, reads today's logs, and writes a review with up to 3
advisory suggestions (`reports/hermes/<date>.md`). Uses a local Ollama model
when available, deterministic fallback otherwise. Suggestions are text;
nothing executes them:

```bash
python -m hermes.supervisor --refresh-data
```

## Layout

See `PROJECT_PLAN.md` for the full plan. Each package has its own README with
file-by-file detail:

| Path | Purpose |
|---|---|
| [`app/`](app/README.md) | FastAPI app, routes, services, Pydantic schemas |
| [`engines/`](engines/README.md) | Scoring, features, regime detection, trained RF/SVM/LSTM specialists |
| [`backtests/`](backtests/README.md) | Data download, replay engine, walk-forward, Monte Carlo, model training |
| [`risk/`](risk/README.md) | Deterministic risk engine (final veto), economic calendar |
| [`execution/`](execution/README.md) | Paper broker simulator (live adapter intentionally stubbed) |
| [`hermes/`](hermes/README.md) | Advisory supervisor + macro sentiment (reads logs, never trades) |
| [`configs/`](configs/README.md) | active / proposed / archived YAML config |
| [`ui/`](ui/README.md) | Read-only Streamlit dashboard |
| [`tests/`](tests/README.md) | pytest suite (127 tests) |
| `data/logs/` | Append-only JSONL artifacts (gitignored) |

## Scope

Milestone 1: webhook, validation, dedupe, JSONL logging, risk engine, paper
broker, decision traces, dashboard, tests.

v2 (PROJECT_PLAN.md section 29): HMM regime detector with forward-filtered
(no-lookahead) probabilities, regime-gated ensemble scoring, crisis-probability
position scaling and halt in the risk engine, multi-agent RAG central-bank
sentiment (local, deterministic fallback, bias-only output), backtest runner
with spread/slippage, walk-forward validation, Monte Carlo reshuffling, and
structural macro stress scenarios (RATE_SHOCK / CARRY_UNWIND / VOL_EXPLOSION).

v3 (PROJECT_PLAN.md section 30): real daily history from yfinance back to
2003, trained RandomForest/SVM specialists behind an out-of-sample edge gate,
generated economic-calendar blackouts wired into the risk engine, local
embedding retrieval with TF-IDF fallback, and the Hermes supervisor with a
bounded self-improvement loop (reads logs + its own notes, writes reviews and
suggestions; never edits configs or risk limits).

v4 (2026-06): richer 13-feature v2 vector (RSI, MACD histogram, Bollinger %B,
long EMA cross, vol ratio, cross-pair correlation), LSTM trend specialist with
a CUDA training pipeline, and a tiered `load_ensemble()` (LSTM > RF >
rule-based) - all behind the same out-of-sample edge gate.

Not implemented: Cloudflare tunnel for real TradingView webhooks (human login
step), live execution (requires explicit human approval and is out of scope).
