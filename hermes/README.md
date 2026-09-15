# hermes/

The research/review layer. **Advisory only - Hermes never trades, never edits configs or risk limits.** It reads logs and its own notes, and writes reports + text suggestions.

- `supervisor.py` - `python -m hermes.supervisor [--refresh-data]` - daily supervisor: refreshes artifacts, backtests recent history, checks promotion gates, reads today's logs, writes a review with up to 3 advisory suggestions to `reports/hermes/<date>.md`. Uses a local Ollama model when available, deterministic fallback otherwise.
- `journal_agent.py` - deterministic daily report generator (pure-Python summary of the day's logs, no LLM)
- `promotion_gate.py` - evaluates the gates a strategy config must pass before promotion to paper (from `autonomy.yaml promotion_gates`). Evaluates and reports only; never edits configs. Live promotion always requires human approval by policy.
- `macro_sentiment/` - multi-agent RAG central-bank sentiment over local transcripts (bias only, never a trade signal)
  - `run_sentiment.py` - `python -m hermes.macro_sentiment.run_sentiment --currency USD [--ollama]`
  - `debate.py` / `lexicon.py` / `retrieval.py` / `embeddings.py` / `ollama_client.py` - debate orchestration, lexicon fallback, local embedding retrieval with TF-IDF fallback, Ollama client
- `prompts/` - prompt files
- `memory/` - Hermes's own strategy notes

By construction the deterministic risk engine has final veto over anything the LLM layer might suggest; suggestions are text and nothing executes them.
