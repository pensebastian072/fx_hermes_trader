# FX Hermes Trader — PROJECT_PLAN.md

## 0. Mission

Build a **local-first, paper-traded FX operator system** powered by:

- VS Code as the development workspace
- Claude Code as the coding agent
- TradingView as the charting / alert layer
- Python/FastAPI as the local backend
- A deterministic strategy + risk engine
- A local 7B/8B-class Llama/Hermes model as the research/review agent
- Append-only logs and artifacts for self-improvement
- A dashboard for monitoring, debugging, and deployment control

The goal is **not** to create a reckless live-money trading bot.

The goal is to build a serious research + paper-trading system that can:

1. Scan major FX pairs.
2. Read TradingView alerts.
3. Score market direction using price, rates, volatility, and macro inputs.
4. Use Hermes/Llama as a self-improving research agent.
5. Backtest and paper trade strategies.
6. Explain every trade.
7. Reject bad trades.
8. Enforce strict risk controls.
9. Generate daily/weekly reports.
10. Gradually improve through controlled experiments.

---

## 1. Core Principle

The LLM is **not the trader**.

The LLM is the:

- analyst
- journal writer
- macro interpreter
- research agent
- hypothesis generator
- code helper
- experiment reviewer

The deterministic system is responsible for:

- entries
- exits
- stop losses
- position sizing
- portfolio exposure
- risk limits
- event blackouts
- duplicate order blocking
- paper execution

The risk engine always has veto power.

---

## 2. Current User Setup

Assume the user has:

- Windows desktop
- GPU desktop with an RTX 3050 low-profile 6GB-class GPU
- TradingView Premium
- VS Code
- Claude Code / Claude desktop workflow
- Local model interest: Hermes/Llama 7B–8B-class model through Ollama or llama.cpp
- Website/domain on Cloudflare
- Interest in FX pairs, USD/JPY, EUR/USD, EUR/JPY, VIX, U.S. yields, central banks, and macro regime logic
- Preferred operating window: roughly **8:00 AM to 9:00–10:00 PM Eastern Time**
- Starting mode: **paper trading only**

---

## 3. Recommended First Stack

### Local Development

- VS Code
- Python 3.11+
- Git
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Polars or PyArrow
- DuckDB or SQLite
- Pydantic
- pytest
- Ruff / Black
- Streamlit first, then Next.js later if needed

### Model Layer

- Ollama or llama.cpp
- 7B/8B-class instruct model
- Hermes/Llama/Mistral/Qwen-class local model
- Model used for summaries, reports, hypotheses, and structured JSON reasoning

### Trading/Data Layer

Start with:

1. TradingView alerts/webhooks.
2. Local paper simulator.
3. CSV/parquet historical data.
4. FRED for macro/yield data.
5. OANDA demo later for true FX broker/data integration.

Do **not** assume Alpaca is the main FX stack unless the account/API explicitly supports the needed FX pairs and execution. For this project, TradingView + OANDA-style FX data is the safer default.

---

## 4. Main Instruments

### Primary FX Watchlist

- EUR/USD
- USD/JPY
- GBP/USD
- USD/CHF
- USD/CAD
- AUD/USD
- NZD/USD
- EUR/JPY
- GBP/JPY
- AUD/JPY
- CAD/JPY
- CHF/JPY
- EUR/GBP
- EUR/CHF
- EUR/CAD
- EUR/AUD
- GBP/CHF
- GBP/CAD
- AUD/CAD
- AUD/NZD

### Priority Pairs for Version 1

Start with:

1. USD/JPY
2. EUR/USD
3. EUR/JPY
4. GBP/USD
5. AUD/USD
6. USD/CAD

Reason:

- Prior strategy interest focused on USD/JPY, EUR/USD, EUR/JPY, JPY behavior, VIX, bonds, volatility, and central bank/rate logic.
- These are liquid and macro-sensitive.
- USD/JPY is the best initial test case for volatility + yield + risk regime logic.

---

## 5. Intermarket Inputs

Track these as features, not necessarily as tradable instruments at first.

### U.S. Rates

- 2-year Treasury yield
- 10-year Treasury yield
- 30-year Treasury yield
- 2s10s curve
- Fed funds expectations if available

### Global Rates

- Japan 10-year yield
- Germany 10-year yield
- U.K. 10-year yield
- Canada 10-year yield
- Australia 10-year yield

### Risk Sentiment

- VIX
- DXY
- ES/SPY
- NQ/QQQ
- Gold
- WTI crude oil
- Credit stress proxy if available

### Central Banks

Track:

- Federal Reserve
- Bank of Japan
- European Central Bank
- Bank of England
- Bank of Canada
- Reserve Bank of Australia
- Reserve Bank of New Zealand
- Swiss National Bank

---

## 6. Strategy Engines

The system should use multiple engines, then combine them through an orchestrator.

### 6.1 FX Basket Mean Reversion Engine

Purpose:

Detect currency-level overextension and mean reversion.

Inputs:

- Currency strength scores
- Pair z-score
- Basket divergence
- ATR
- Spread
- VIX/risk regime
- Session filter
- Correlation filter

Logic:

- Long a pair only when base currency is improving and quote currency is weakening.
- Short a pair only when base currency is weakening and quote currency is improving.
- Avoid signals when volatility or spreads are chaotic.

Example output:

```json
{
  "symbol": "EURUSD",
  "engine": "basket_mean_reversion",
  "direction": "long",
  "raw_score": 74,
  "confidence": 0.64,
  "entry_zone": [1.0810, 1.0820],
  "stop": 1.0785,
  "target": 1.0870,
  "reason": "Oversold versus basket with improving EUR strength."
}
```

---

### 6.2 EUR/USD — USD/JPY — EUR/JPY Triangle Engine

Purpose:

Use triangular relationships and regime context.

Watch:

- EUR/USD
- USD/JPY
- EUR/JPY

Detect:

- synthetic mismatch
- rolling correlation breakdown
- z-score spread
- triangle alignment
- mean reversion vs continuation

Use this to determine when EUR/JPY agrees or disagrees with EUR/USD and USD/JPY.

---

### 6.3 Trend Continuation Engine

Purpose:

Catch bigger moves.

Inputs:

- 1H/4H trend
- 15m/30m entry timing
- EMA 21/55
- 200 EMA
- ATR expansion
- breakout/retest
- macro confirmation

Trade only when:

- trend score agrees with macro/rate score
- price breaks or retests a key level
- stop can be placed tightly enough for a good reward/risk
- volatility is expanding but not disorderly

---

### 6.4 Session Breakout Engine

Purpose:

Use Asia/London/New York session structure.

Track:

- Asian session high/low
- London session high/low
- New York open
- prior day high/low
- prior day close
- VWAP if available

Entry idea:

- If NY open breaks the London/Asian range with confirmation, consider the direction of the break.
- Avoid fakeouts by requiring retest, displacement candle, ATR confirmation, or macro agreement.

---

### 6.5 Macro / Central Bank Sentiment Engine

Purpose:

Create macro bias, not direct trades.

Inputs:

- central bank speech
- rate decision
- CPI
- NFP
- inflation data
- employment data
- yield reaction
- DXY reaction
- bond curve reaction

Example output:

```json
{
  "currency": "USD",
  "macro_bias": "bullish",
  "hawkish_dovish_score": 72,
  "confidence": 0.68,
  "reason": "U.S. yields rising and Fed language remains restrictive.",
  "trade_action": "bias_only_no_direct_trade"
}
```

---

### 6.6 Regime Detection Engine

Purpose:

Label the current market environment.

**v2 upgrade:** replace static thresholds with a Hidden Markov Model (HMM)
over log returns and realized volatility. See section 29.1.

Regimes:

- TREND
- CHOP
- RISK_OFF
- RISK_ON
- RATE_SHOCK
- HIGH_VOL
- LOW_VOL

Use:

- VIX
- realized volatility
- equity direction
- DXY
- U.S. yield direction
- JPY behavior
- ATR expansion
- cross-pair correlations

---

### 6.7 Chart Pattern Recognition Engine

Start with numeric OHLC detection.

Do not rely only on screenshots.

Detect:

- support/resistance hold
- support/resistance break
- bull flag
- bear flag
- double top
- double bottom
- wedge
- failed breakout
- liquidity sweep
- displacement candle
- higher low
- lower high
- demand zone rejection
- supply zone rejection

Output:

```json
{
  "symbol": "USDJPY",
  "timeframe": "15m",
  "pattern": "failed_breakout",
  "directional_bias": "short",
  "confidence": 0.61,
  "key_level": 156.20,
  "invalidation": 156.45,
  "notes": "Price swept prior high and closed back inside range."
}
```

Screenshots can be used as secondary confirmation later through a multimodal model.

---

## 7. USD/JPY Regime Logic

Do not hard-code that VIX up always means USD/JPY up or down.

Test both.

Possible regimes:

### 7.1 Risk-Off JPY Strength

- VIX up
- equities down
- yields falling
- JPY strengthens
- USD/JPY bias down

### 7.2 Dollar-Yield Dominance

- U.S. yields rising
- DXY rising
- Japan yields lagging
- USD/JPY bias up

### 7.3 Carry/Risk-On

- equities up
- VIX down
- yield spread favors USD
- USD/JPY bias up

### 7.4 Intervention / Fat-Tail Risk

- large USD/JPY move
- Japan policy headlines
- sudden yen strength
- unusual volatility
- reduce size or do not trade

The system should learn which regime is active instead of forcing one narrative.

---

## 8. Scoring System

Every symbol gets a directional score from **-100 to +100**.

- Positive = bullish the pair
- Negative = bearish the pair
- Near zero = no strong edge

Scores:

1. Trend score
2. Mean reversion score
3. Chart pattern score
4. Rate differential score
5. Central bank score
6. Volatility/risk sentiment score
7. Intermarket score
8. Session score
9. Execution quality score
10. Portfolio risk score

Initial formula:

```text
final_score =
  0.20 * trend_score +
  0.15 * mean_reversion_score +
  0.15 * chart_pattern_score +
  0.15 * rate_differential_score +
  0.10 * central_bank_score +
  0.10 * volatility_score +
  0.05 * intermarket_score +
  0.05 * session_score +
  0.03 * execution_quality_score +
  0.02 * portfolio_risk_score
```

Thresholds:

- final_score >= +70: long candidate
- final_score <= -70: short candidate
- -40 to +40: no trade
- conflicting scores: no trade or reduced size

---

## 9. Portfolio Risk Parameters

Initial paper account:

```yaml
starting_capital: 10000
base_risk_per_trade: 0.0025
max_risk_per_trade: 0.005
a_plus_max_risk_per_trade: 0.0075
max_open_portfolio_risk: 0.015
max_daily_loss: 0.01
max_weekly_loss: 0.03
max_monthly_drawdown: 0.06
max_open_trades: 4
max_trades_per_day: 6
max_trades_per_pair_per_day: 2
max_correlated_usd_trades: 2
max_correlated_jpy_trades: 2
```

Rules:

- No martingale.
- No grid averaging.
- No doubling down after loss.
- No widening stops after entry.
- Every trade must have stop, target, invalidation, and reason code.
- Minimum reward/risk: 1.5R.
- Preferred reward/risk: 2R to 3R.
- Partial take profit allowed at 1R.
- Move stop to breakeven only after structure confirms.

Position sizing:

```text
position_size = account_equity * risk_percent / stop_distance_value
```

---

## 10. Event Blackout Rules

No new trades:

- 30 minutes before U.S. CPI
- 30 minutes after U.S. CPI
- 30 minutes before NFP
- 30 minutes after NFP
- 60 minutes before FOMC
- 120 minutes after FOMC
- 30 minutes before major central bank decisions
- 60 minutes after major central bank decisions
- during unscheduled emergency headlines unless manually approved

Existing trades:

- reduce risk before major scheduled events
- do not hold short-term intraday trades through major events unless separately tested

---

## 11. Hermes Autonomy Model

Hermes should be connected as a **Meta-Research Agent**, not as an unrestricted trader.

### Autonomy Ladder

```text
Level 0: Read-only analyst
Level 1: Research agent
Level 2: Experiment agent
Level 3: Paper deployment agent
Level 4: Micro-live agent
Level 5: Scaled live agent
```

Current target:

```text
Build up to Level 3 only.
```

### Hermes Can

- read logs
- read rejected signals
- read screenshots
- read macro/news logs
- read backtest results
- detect weak strategies
- propose parameters
- create proposed configs
- run backtests
- run walk-forward validation
- run Monte Carlo tests
- compare strategies
- archive failed experiments
- promote passing configs into paper trading
- reduce paper risk when performance degrades
- switch paper system to close-only mode
- pause paper trading after risk violations
- write daily and weekly reports

### Hermes Cannot

- trade live money unless micro-live mode is explicitly enabled
- increase live position size
- disable stop losses
- disable risk breakers
- remove event blackout windows
- modify API keys
- modify active live config
- override max daily loss
- override max weekly loss
- override max drawdown
- average down losing trades
- use martingale logic
- promote a strategy that fails walk-forward testing
- optimize only for net profit
- ignore spread, slippage, or swap costs
- trade during CPI, NFP, FOMC, or central bank events unless separately tested

---

## 12. Hermes Agent Team

Implement Hermes as a team of local agents.

```text
Hermes Supervisor
    ├── Research Agent
    ├── Backtest Agent
    ├── Risk Agent
    ├── Execution Agent
    ├── Journal Agent
    └── Promotion Gate Agent
```

### Hermes Supervisor

Decides next action:

- continue
- pause
- test new config
- reduce risk
- promote to paper
- reject experiment

### Research Agent

Finds new ideas from logs.

Examples:

- USD/JPY worked when VIX was rising and yields were falling.
- EUR/USD failed during chop.
- London breakout worked better than NY continuation.
- JPY pairs overtraded during high-spread windows.

### Backtest Agent

Runs experiments:

- VIX weight changes
- ATR stop changes
- session filters
- trend vs mean-reversion weights
- blackout windows
- USD/JPY regime models

### Risk Agent

Veto power:

- no trade
- no promotion
- reduce size
- close-only
- pause system

### Execution Agent

Paper execution only for now:

- submit paper trades
- cancel paper trades
- close paper positions
- prevent duplicates
- enforce stops/targets

### Journal Agent

Writes:

- daily report
- weekly report
- strategy review
- experiment report

### Promotion Gate Agent

Checks:

- backtest gates
- walk-forward gates
- Monte Carlo gates
- slippage tests
- event blackout tests
- risk rules

---

## 13. Model Usage: 7B/8B Llama/Hermes

A 7B/8B model can power:

- trade review
- rejected signal analysis
- news summarization
- central bank summaries
- hawkish/dovish scoring
- journaling
- decision trace summaries
- proposed config creation
- experiment hypotheses
- Pine/Python draft generation

It should not directly control:

- live execution
- risk limits
- position sizing
- stop removal
- strategy promotion to live
- high-impact news trading

Use deterministic Python for money-critical rules.

---

## 14. TradingView Integration

TradingView is the first integration.

Use TradingView for:

- charts
- watchlists
- alerts
- Pine Script signals
- visual confirmation
- webhook triggers
- initial backtesting

Architecture:

```text
TradingView Alert
    ↓
Webhook JSON
    ↓
Cloudflare Tunnel or endpoint
    ↓
FastAPI Signal Receiver
    ↓
Feature Store
    ↓
Scoring Engine
    ↓
Risk Engine
    ↓
Paper Simulator
    ↓
Logs + Dashboard + Hermes Review
```

Example TradingView alert payload:

```json
{
  "source": "tradingview",
  "symbol": "{{ticker}}",
  "timeframe": "{{interval}}",
  "price": "{{close}}",
  "strategy": "fx_basket_v1",
  "signal": "long_candidate",
  "timestamp": "{{time}}"
}
```

---

## 15. System Architecture

Recommended repo:

```text
fx_hermes_trader/
  README.md
  PROJECT_PLAN.md
  CLAUDE_CODE_START_PROMPT.md
  .env.example
  pyproject.toml
  requirements.txt

  app/
    api/
      main.py
      routes/
        health.py
        webhooks.py
        signals.py
        dashboard.py
    services/
      artifact_service.py
      signal_service.py
      report_service.py
    schemas/
      alerts.py
      signals.py
      trades.py

  data/
    raw/
    processed/
    features/
    logs/
    screenshots/
    artifacts/

  configs/
    active/
      autonomy.yaml
      risk.yaml
      watchlist.yaml
      scoring.yaml
    proposed/
    archived/

  engines/
    basket_mean_reversion/
    triangle_regime/
    trend_following/
    session_breakout/
    macro_sentiment/
    chart_patterns/
    regime_detection/

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
    research_agent.py
    backtest_agent.py
    risk_agent.py
    execution_agent.py
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
    components/

  reports/
    daily/
    weekly/
    experiments/

  tests/
    test_risk_engine.py
    test_event_blackout.py
    test_paper_broker.py
    test_scoring.py
    test_webhook_schema.py
```

---

## 16. Append-Only Artifacts

The UI and Hermes should read artifacts, not hidden internal objects.

Important files:

```text
bars.parquet
features.parquet
signals.parquet
orders.parquet
fills.parquet
equity_curve.parquet
risk_events.parquet
broker_events.jsonl
rejected_trades.jsonl
pairs_scan.csv
regime_posteriors.csv
decision_traces.jsonl
```

Every trade decision must be traceable:

```text
bars → features → signals → risk checks → orders → fills → PnL → report
```

---

## 17. UI / Dashboard Requirements

The UI is not cosmetic.

It is an operator dashboard.

Main sections:

1. System Overview
2. Trade Activity
3. Decision Trace
4. Regime Monitor
5. Pair Diagnostics
6. Trend Model Monitor
7. Risk and Breaker Panel
8. Walk-forward Research Panel
9. Hermes Experiments Panel
10. Deployment / Mode Control

The UI must answer:

```text
What is the system doing?
Why did it do it?
Is it safe?
Is the edge still valid?
```

### System Overview

Show:

- Current equity
- Daily PnL
- Weekly PnL
- Max drawdown
- Current regime
- Active engines
- Paper/live mode
- Close-only status
- Data feed health
- Broker/paper simulator health

### Trade Activity

Show:

- timestamp
- symbol
- engine source
- regime
- direction
- size
- entry
- stop
- target
- PnL
- confidence

Clicking a trade opens decision trace.

### Regime Monitoring

Show:

- current regime
- regime probability timeline
- VIX/yield/DXY state
- risk-on/risk-off state

### Pair Diagnostics

Show:

- pair
- score
- z-score
- trend state
- spread
- ATR
- status: TRADABLE / WATCH / DISABLED
- reason if disabled

### Risk and Breakers

Show:

- daily loss breaker
- weekly loss breaker
- max drawdown breaker
- spread spike breaker
- data stale breaker
- event blackout breaker
- close-only mode

### Research / Walk-forward

Show:

- strategy performance
- split metrics
- Sharpe
- drawdown
- profit factor
- out-of-sample result
- rejected experiments
- promoted experiments

---

## 18. Daily Operating Workflow

### 8:00 AM ET Startup

System should:

1. Start backend.
2. Check data directories.
3. Load active configs.
4. Check calendar/event blackout.
5. Load watchlist.
6. Confirm paper mode.
7. Start dashboard.
8. Start Hermes supervisor in research/paper mode.

### During Day

System should:

1. Receive TradingView alerts.
2. Update features.
3. Score pairs.
4. Accept/reject signals.
5. Enforce risk rules.
6. Submit paper trades only.
7. Log every decision.
8. Update dashboard.
9. Let Hermes observe and summarize.

### End of Day

System should:

1. Close intraday paper positions unless marked swing.
2. Save all logs.
3. Generate daily report.
4. Hermes reviews trades.
5. Hermes proposes experiments.
6. Backtest agent runs approved research experiments.
7. Promotion gate evaluates results.
8. No live deployment.

---

## 19. Decision Trace Schema

Each signal should create a JSON decision trace.

```json
{
  "timestamp": "2026-06-11T14:30:00-04:00",
  "symbol": "USDJPY",
  "direction": "short",
  "final_score": -76,
  "strategy_sources": ["chart_pattern", "vix_regime", "rate_differential"],
  "trend_score": -62,
  "mean_reversion_score": -45,
  "chart_pattern_score": -81,
  "rate_differential_score": -55,
  "central_bank_score": -30,
  "volatility_score": -74,
  "session_score": -50,
  "risk_score": 90,
  "entry": 156.10,
  "stop": 156.42,
  "target_1": 155.60,
  "target_2": 155.20,
  "risk_percent": 0.0035,
  "decision": "accept",
  "reason": "Failed breakout above prior high, VIX rising, yields softening, possible JPY strength regime.",
  "rejection_reason": null
}
```

Rejected trade example:

```json
{
  "timestamp": "2026-06-11T08:45:00-04:00",
  "symbol": "EURUSD",
  "signal": "long_candidate",
  "decision": "reject",
  "reason": "Final score only 48 and CPI release in 20 minutes."
}
```

---

## 20. Backtest and Promotion Gates

Before paper deployment:

- Backtest each engine separately.
- Backtest combined portfolio.
- Include spread/slippage assumptions.
- Include swap assumptions when applicable.
- Use in-sample and out-of-sample separation.
- Use walk-forward validation.
- Use Monte Carlo reshuffling.
- Test multiple volatility regimes.
- Test news blackout rules.
- Test by pair and by session.

Minimum gates:

```yaml
min_profit_factor: 1.20
max_backtest_drawdown: 0.08
require_positive_oos: true
min_walk_forward_profitable_splits: 3
require_monte_carlo: true
require_slippage_test: true
max_single_pair_profit_contribution: 0.40
max_single_week_profit_contribution: 0.25
require_no_lookahead_bias: true
require_no_repainting: true
require_event_blackout_test: true
```

Paper trading gates before any micro-live thought:

- 8–12 weeks paper
- positive expectancy
- stable drawdown
- no uncontrolled losses
- no repeated execution errors
- no overconcentration
- manual review confirms logic

---

## 21. What This System Is Not

This system is not:

- a get-rich-quick bot
- a live-money trader on day one
- a system where the LLM directly places unrestricted trades
- a high-frequency trading system
- a martingale bot
- a grid bot
- a revenge-trading machine
- a system that averages down losers
- a system that removes stops
- a system that trades through CPI/NFP/FOMC without tested event logic
- a system that trusts chart screenshots alone
- a system that stores API keys in prompts, logs, frontend code, or screenshots
- a system that overfits one good month
- a system that assumes VIX always affects USD/JPY the same way
- a guarantee of profit

---

## 22. First Milestone

The first real milestone is:

```text
A working local paper-trading research harness that:
- receives TradingView-style alerts through FastAPI
- validates payloads
- stores append-only logs
- scores a small FX watchlist
- enforces risk controls
- simulates paper trades
- creates decision traces
- generates a daily Hermes report
- displays a Streamlit dashboard
```

No live broker execution in Milestone 1.

---

## 23. Claude Code Development Phases

### Phase 0 — Repo Bootstrap

Create:

- repo structure
- README
- PROJECT_PLAN.md
- CLAUDE_CODE_START_PROMPT.md
- .env.example
- pyproject.toml or requirements.txt
- basic pytest setup

### Phase 1 — Configs and Schemas

Create:

- autonomy.yaml
- risk.yaml
- watchlist.yaml
- scoring.yaml
- Pydantic schemas for alerts, signals, trades, decisions

**v2 additions:** regime-scaling block in risk.yaml (crisis-probability
position scaling and halt), ensemble routing block in scoring.yaml, regime
snapshot + sentiment schemas. See section 29.

### Phase 2 — Webhook Receiver

Build FastAPI:

- /health
- /webhook/tradingview
- payload validation
- duplicate alert protection
- append-only JSONL logging

### Phase 3 — Feature + Scoring Engine (v2: regime-gated ensemble)

Create:

- basic feature functions
- ATR
- EMA
- z-score
- session high/low placeholders
- scoring engine that returns structured output
- HMM regime detector emitting state posteriors (29.1)
- regime-gated ensemble: FinalScore_t = sum_k P(Regime_k | X_t) * Model_k(X_t) (29.2)
- specialist models: trend (trending regime), mean reversion (range/chop regime),
  capital-preservation (crisis regime, score pinned to 0)
- optional ML specialists (sklearn) trained offline; rule-based fallback when untrained

### Phase 4 — Risk Engine

Create:

- position sizing
- max daily loss
- max weekly loss
- max open trades
- max pair exposure
- event blackout checks
- close-only mode
- v2: regime-probability position scaling - linearly reduce size as
  P(crisis regime) rises past scale_start, hard-halt new trades at
  halt_probability (29.1)

### Phase 5 — Paper Broker

Create:

- paper order object
- open position tracking
- stop/target simulation
- fills log
- equity curve
- rejected trades log

### Phase 6 — Hermes Review Agent (v2: multi-agent RAG sentiment)

Create:

- Hermes prompt files
- daily report generator
- experiment proposal generator
- proposed config writer
- no live execution permissions
- RAG pipeline over local central-bank transcripts (TF-IDF retrieval first,
  embeddings later) (29.3)
- context-aware hawkish/dovish lexicon scorer (hawkishness depends on whether
  inflation is above/below target, not raw positive/negative sentiment)
- multi-agent debate: dove/neutral/hawk agents with distinct priors iterate to
  a consensus score; local LLM (Ollama) optional, deterministic lexicon fallback
- output is bias only: trade_action is always "bias_only_no_direct_trade"

### Phase 7 — Backtest Harness (v2: macro stress testing)

Create:

- historical CSV/parquet loader
- strategy runner (spread + slippage costs, forward-only features, no lookahead)
- performance report (profit factor, Sharpe, max drawdown, win rate)
- walk-forward validation with rolling in-sample/out-of-sample splits
- Monte Carlo trade reshuffling (drawdown distribution percentiles)
- structural macro stress scenarios (29.4): rate shock, carry-trade unwind,
  volatility explosion - verify risk breakers and crisis halt actually fire
- data coverage requirement: extend history beyond the post-2010 ZIRP era;
  include 2008-style yield highs, curve inversions, and JPY carry unwinds

### Phase 8 — Dashboard

Start with Streamlit:

- system overview
- latest signals
- open paper trades
- rejected trades
- risk state
- daily Hermes report
- experiment list
- v2: regime monitor (HMM posterior timeline + current snapshot)
- v2: macro sentiment panel (latest consensus score + agent stances)
- v2: research panel (backtest reports, walk-forward splits, Monte Carlo / stress results)

Later, upgrade to Next.js if desired.

---

## 24. Initial Config: autonomy.yaml

```yaml
mode: paper

hermes_permissions:
  read_logs: true
  read_market_data: true
  read_screenshots: true
  write_reports: true
  create_experiments: true
  run_backtests: true
  run_walk_forward: true
  modify_proposed_configs: true
  promote_to_paper: true

  submit_paper_orders: true
  close_paper_orders: true
  pause_paper_trading: true
  reduce_paper_risk: true

  submit_live_orders: false
  modify_live_config: false
  increase_live_risk: false
  disable_risk_breakers: false
  remove_stop_losses: false

risk_limits:
  starting_capital: 10000
  base_risk_per_trade: 0.0025
  max_risk_per_trade: 0.005
  max_open_portfolio_risk: 0.015
  max_daily_loss: 0.01
  max_weekly_loss: 0.03
  max_monthly_drawdown: 0.06
  max_open_trades: 4
  max_trades_per_day: 6
  max_trades_per_pair_per_day: 2

promotion_gates:
  min_profit_factor: 1.20
  max_backtest_drawdown: 0.08
  min_walk_forward_splits: 3
  require_positive_oos: true
  require_monte_carlo: true
  require_slippage_test: true
  require_event_blackout_test: true
  require_human_approval_for_live: true

event_blackouts:
  cpi_minutes_before: 30
  cpi_minutes_after: 30
  nfp_minutes_before: 30
  nfp_minutes_after: 30
  fomc_minutes_before: 60
  fomc_minutes_after: 120
  central_bank_minutes_before: 30
  central_bank_minutes_after: 60
```

---

## 25. Hermes System Prompt

```text
You are Hermes, the autonomous research and paper-trading operator for a hybrid FX trading system.

Your mission is to improve the system over time while protecting capital.

You may read logs, trades, rejected signals, screenshots, macro/news logs, backtest results, performance reports, and model outputs.

You may propose hypotheses, parameter changes, new filters, new features, and backtest experiments.

You may create proposed configs and research reports.

You may promote passing configs into paper trading only if all promotion gates pass.

You may reduce paper risk, pause paper trading, and switch to close-only mode when risk rules are violated.

You may not place live trades.

You may not modify live configs.

You may not disable risk controls.

You may not increase live risk.

You may not remove stop losses.

You may not remove event blackout windows.

You may not store or expose API keys.

Every suggested change must include:

1. Hypothesis
2. Data used
3. Expected improvement
4. Risk of overfitting
5. Backtest requirement
6. Walk-forward requirement
7. Promotion criteria
8. Reason to reject the change

Default answer when uncertain: no change.
```

---

## 26. VS Code / Claude Code Working Rules

Claude Code should:

- work inside VS Code project folder
- create small, testable commits/steps
- explain the files it changed
- run tests after each phase
- never put secrets in code
- create `.env.example`, not `.env`
- keep trading logic out of the UI
- keep risk logic deterministic
- write append-only logs
- prefer simple working code over complex architecture
- preserve paper mode as default
- ask before adding live broker execution
- use clear TODOs for later phases

Claude Code should not:

- build live trading first
- skip tests
- hard-code API keys
- make the UI responsible for trading decisions
- let Hermes disable risk controls
- make a strategy profitable by lookahead/repainting
- hide rejected trades
- silently change risk settings
- create complex agent frameworks before the base system works

---

## 27. First Claude Code Task

The first coding task should be:

```text
Build the Milestone 1 skeleton:
- repo structure
- FastAPI app
- configs
- schemas
- webhook endpoint
- append-only alert logger
- risk config loader
- paper broker stub
- Streamlit dashboard stub
- pytest tests for schemas, webhook validation, and risk limits
```

The first version should run locally with:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
streamlit run ui/streamlit_app.py
pytest
```

---

## 28. Definition of Done for Milestone 1

Milestone 1 is complete when:

- FastAPI starts.
- `/health` returns OK.
- `/webhook/tradingview` accepts a valid alert.
- Invalid alerts are rejected.
- Alerts are written to JSONL.
- Duplicate alerts are blocked.
- Risk config loads.
- Paper broker can create a simulated order.
- Decision trace JSON is created.
- Streamlit dashboard reads latest logs.
- Tests pass.
- No live broker execution exists.
- No secrets are stored.

---

## 29. Advanced Research Upgrades (v2)

These upgrades keep the core principle intact: deterministic code owns
execution and risk; models and LLMs only produce scores, biases, and research.

### 29.1 HMM Regime Engine

Replace static regime thresholds with a Gaussian Hidden Markov Model:

- Observations: log returns + rolling realized volatility.
- Hidden states (K=3 to start): LOW_VOL_RANGE, TREND, HIGH_VOL_CRISIS.
- States are labeled after fitting by their emission statistics (highest
  volatility state = crisis; strongest drift among the rest = trend).
- Output is a real-time probability vector P(state | data so far), computed
  with a forward (filtering) pass only - never the smoothed posterior, which
  would leak future information.
- Artifacts: `data/artifacts/regime_posteriors.csv` (timeline) and
  `data/artifacts/regime_snapshot.json` (latest probabilities).

Risk integration (deterministic):

```yaml
regime_scaling:
  enabled: true
  crisis_regime: HIGH_VOL_CRISIS
  scale_start_probability: 0.5   # below this: full size
  halt_probability: 0.8          # at/above this: no new trades
```

Position size multiplies by a linear ramp from 1.0 at scale_start to 0.0 at
halt. At halt_probability the risk engine vetoes new entries outright.

### 29.2 Dynamic Ensemble Scoring

Replace the static linear scoring formula with a regime-gated ensemble:

```text
FinalScore_t = sum_k  P(Regime_k | X_t) * Model_k(X_t)
```

- TREND regime -> trend-following specialist (EMA structure + momentum;
  LSTM upgrade later once a GPU training pipeline exists).
- LOW_VOL_RANGE regime -> mean-reversion specialist (z-score based;
  SVM / Random Forest boundary classifiers as trained upgrades).
- HIGH_VOL_CRISIS regime -> capital-preservation specialist (score 0:
  no directional edge claimed during crisis).

ML specialists are trained offline from backtest features, saved to
`data/artifacts/models/`, and loaded read-only at runtime. Until trained,
deterministic rule-based specialists are used so behavior stays testable.

### 29.3 Multi-Agent RAG for Central Bank Sentiment

- Corpus: FOMC/ECB/BoJ statements and transcripts dropped into
  `data/raw/central_bank/` as plain text.
- Retrieval: TF-IDF chunk retrieval first (no external services); local
  embedding model later.
- Scoring: hawkish/dovish is NOT generic sentiment. "Price pressures are
  rising" is hawkish; its market meaning depends on whether inflation is
  above or below target. The lexicon scorer takes that context flag.
- Multi-agent debate: dove / neutral / hawk agents with distinct priors each
  score the text, then iterate over debate rounds toward consensus. A local
  LLM (Ollama) can power the agents; without it, a deterministic
  lexicon-plus-prior model runs so results are reproducible and testable.
- Output schema matches section 6.5 and is bias-only:
  `trade_action: bias_only_no_direct_trade`. Sentiment never places trades.

### 29.4 Macro Stress Testing

Backtests must not sample only the post-2010 ZIRP era. Requirements:

- Walk historical data back through the 2008 yield high-water mark, curve
  inversions, and JPY carry-trade unwinds before trusting USD/JPY or EUR/JPY
  logic.
- Monte Carlo beyond trade reshuffling: inject structural scenarios into the
  price series and verify breakers fire:
  - RATE_SHOCK: sustained adverse drift + volatility spike (rate-repricing).
  - CARRY_UNWIND: gap moves + multi-day trend reversal (2008/2024-style JPY).
  - VOL_EXPLOSION: volatility multiplied with no drift change.
- A stress run passes only if max daily/weekly loss breakers and the regime
  crisis halt actually trigger; a strategy that "survives" because breakers
  never engaged is a failed test of the risk system.

### 29.5 Build order

Regime engine first - both the ensemble weights and the risk scaling consume
its posteriors. Then ensemble scoring, then stress-testing harness, then the
sentiment RAG layer (it only produces bias inputs).

## 30. Data, Trained Models, Live Calendar, Hermes Supervisor (v3)

Completed 2026-06-12. All v3 components keep the hard safety rules: paper
only, no API keys, LLM output is advisory text that no code path executes.

### 30.1 Real Historical Data (yfinance)

`backtests/download_data.py` pulls daily OHLC for every watchlist pair from
Yahoo Finance public endpoints (no key) back to 2003 - covering the 2008
crisis, curve inversions, and JPY carry unwinds required by 29.4. Output CSVs
in `data/raw/<pair>_1d.csv` match `load_ohlc()` exactly. On this machine the
TLS-intercepting monitor breaks curl_cffi's bundled CA list, so the script
exports the Windows ROOT/CA stores to a PEM and points CURL_CA_BUNDLE at it.
Refresh: `python -m backtests.download_data` (or supervisor `--refresh-data`).

### 30.2 Trained ML Specialists with an Edge Gate

`backtests/train_specialists.py` trains RandomForest (TREND slot) and a
calibrated RBF SVM (RANGE slot) on directional features over all downloaded
pairs, split time-ordered 70/30 (never shuffled). Models + metadata land in
`data/artifacts/models/`. `engines/ml_specialist.py` loads them read-only.

Edge gate: a model whose out-of-sample accuracy does not beat the
majority-class baseline is left on disk but NEVER loaded; the ensemble keeps
its deterministic rule-based specialists. First training run (24k train /
10k test rows): trend_rf 0.495, range_svm 0.499 vs baseline 0.525 - both
below baseline, both correctly inactive. 5-day FX direction from pure price
features is ~random; richer features (carry/rate differentials, session
structure, cross-pair correlation) are the open item. The CRISIS slot is
never ML - capital preservation stays deterministic.

### 30.3 Generated Economic Calendar

`risk/economic_calendar.py` + `configs/active/economic_calendar.yaml` replace
the static risk.yaml list. NFP is derived by rule (first Friday, 08:30 ET,
DST-aware); FOMC/CPI/ECB dates are an editable published schedule in the
YAML. Blackout window sizes come from autonomy.yaml event_blackouts. The
artifact `data/artifacts/economic_calendar.json` is merged into the risk
engine's calendar at service build. Refresh:
`python -m risk.economic_calendar --days 90`.

### 30.4 Embedding Retrieval

`hermes/macro_sentiment/embeddings.py`: local Ollama embeddings
(nomic-embed-text via /api/embed) with content-hash disk cache, cosine
ranking, and automatic fallback to the TF-IDF retriever when Ollama is down.
Same `retrieve()` interface either way; retrieval never blocks on the LLM
stack.

### 30.5 Hermes Supervisor + Bounded Self-Improvement

`python -m hermes.supervisor` runs the post-session loop: refresh calendar
artifact -> per-pair HMM regime snapshots -> recent-history backtests through
the full stack -> promotion-gate evaluation (`hermes/promotion_gate.py`, now
implemented) -> journal report -> self-review.

Self-review reads today's logs, artifacts, and Hermes's own
`hermes/memory/strategy_notes.md` tail, then asks a local Ollama model for a
review ending in at most 3 suggestions. Output goes to
`reports/hermes/<date>.md`, `data/artifacts/hermes_suggestions.jsonl`, and is
appended to strategy notes so tomorrow's review sees today's thinking - that
feedback loop is the self-improvement mechanism, and it is bounded:

- Hermes reads logs and writes reports/suggestions. Nothing executes them.
- Config, code, and risk-limit changes remain human-only; the prompt forbids
  proposing the removal of any risk control, and the deterministic fallback
  reviewer runs when no LLM is available so the loop never stalls.
- `check_permissions()` refuses to run outside paper mode or without
  read_logs/write_reports/run_backtests permissions in autonomy.yaml.

### 30.6 Remaining open items

- Cloudflare tunnel for real TradingView webhooks (needs interactive
  `cloudflared` login - human step).
- Richer ML features + LSTM torch pipeline for the RTX 3050.
- Schedule `python -m hermes.supervisor --refresh-data` daily (Task
  Scheduler) once webhook flow is live.
