# app/

FastAPI application: webhook receiver, validation, and service wiring.

- `api/main.py` - app factory (`uvicorn app.api.main:app --reload`)
- `deps.py` - service wiring, built once at startup from `configs/active/`
- `paths.py` - filesystem locations; `FXHT_DATA_DIR` overrides the data dir (used by tests)
- `schemas/` - Pydantic models: alerts, decisions, regime, sentiment, signals, trades
- `services/`
  - `artifact_service.py` - append-only JSONL helpers (records are never mutated/deleted)
  - `config_service.py` - YAML loading for `configs/active/`
  - `signal_service.py` - alert pipeline: validate -> dedupe -> log -> score -> risk check -> paper order -> decision trace
