# execution/

Order execution. **Paper only - no live trading anywhere in this package.**

- `broker_base.py` - abstract broker interface
- `paper_broker.py` - `PaperBroker`: simulates market orders with immediate fills at the requested entry price. All activity persisted append-only to `orders.jsonl`/`fills.jsonl`; state is rebuilt from those logs on startup, so the JSONL files are the source of truth.
- `oanda_adapter_stub.py` - intentionally non-functional live-broker stub. Marks where a demo-account adapter would plug in later. Stores no API keys. Live execution is out of scope and requires explicit human approval (`autonomy.yaml: submit_live_orders: false`).
