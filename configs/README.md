# configs/

YAML configuration. The running system reads only `active/`.

- `active/` - live config loaded at startup
  - `risk.yaml` - all risk-engine limits (the engine reads nothing else)
  - `autonomy.yaml` - mode (`paper`; app refuses to start if not), event-blackout windows, promotion gates, `submit_live_orders: false`
  - `watchlist.yaml` - traded pairs
  - `scoring.yaml` - signal->score mapping, thresholds, stop/target geometry
  - `economic_calendar.yaml` - FOMC/CPI/ECB event dates for blackout generation
- `proposed/` - candidate configs awaiting evaluation (never auto-promoted)
- `archived/` - retired configs

**Safety:** `app/deps.py` refuses to start if `autonomy.yaml` mode != `paper`. Promotion from `proposed/` to `active/` is a human action, never automated.
