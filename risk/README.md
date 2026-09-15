# risk/

Deterministic risk engine. Final veto power over every trade. **No LLM input reaches this package** - all limits come from `configs/active/risk.yaml`.

- `risk_engine.py` - `RiskEngine`: per-trade/daily/weekly limits, position sizing, crisis-probability position scaling and halt, close-only mode
- `economic_calendar.py` - `python -m risk.economic_calendar --days N` - generates event-blackout windows from `configs/active/economic_calendar.yaml` (NFP by first-Friday rule, FOMC/CPI/central-bank decisions as explicit dates); window sizes from `autonomy.yaml`
- `event_blackout.py` - blackout-window checks (Milestone-1 static calendar from `risk.yaml`; superseded by `economic_calendar.py`)
- `breakers.py` - circuit breakers (daily/weekly loss + close-only already enforced in `risk_engine.py`; remainder planned)
- `exposure.py` - portfolio exposure / correlation limits (planned)

The risk engine has the final say: a trade the ensemble wants can be vetoed here, never the reverse.
