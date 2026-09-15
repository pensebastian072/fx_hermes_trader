Review today's paper-trading activity.

Inputs you will receive:
- alerts.jsonl entries for today
- decision_traces.jsonl entries for today
- orders.jsonl / fills.jsonl entries for today
- current risk config

Produce a daily review with:
1. What the system did today (counts, symbols, directions).
2. Why trades were accepted or rejected (cite decision traces).
3. Any risk-engine vetoes and whether they were correct.
4. One observation worth investigating (or "none").
5. No config changes unless evidence is strong; default is no change.

You may not propose disabling any risk control.
