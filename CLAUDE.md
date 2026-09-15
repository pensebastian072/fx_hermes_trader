# FX Hermes Trader — agent guide

Local-first **paper-only** FX research harness. FastAPI webhook receiver → Pydantic
validation → deterministic risk engine (final veto) → paper broker simulator.
Append-only JSONL artifacts (alerts, decisions, orders, fills). Streamlit operator
dashboard. Hermes (LLM) is a **supervisor/review layer** — it writes
`hermes_suggestions.jsonl` (flag-file pattern) and can never place a trade.

Build history + milestones: **`PROJECT_PLAN.md`** (read it first). Note:
`CLAUDE_CODE_START_PROMPT.md` at root is the original v1 bootstrap prompt — historical,
not current instructions.

Skills: load `paper-trading-guardrails` before editing webhook/risk/execution/Hermes/UI
code, `quant-research-gate` before model/backtest work, `win-quant-env` before
installs/git/PS. Before committing guarded-path changes, run
`node ~/.claude/hooks/quant-review-scope.js fx_hermes_trader` and, on
`DECISION: REVIEW`, the `quant-reviewer` subagent.

## Hard rules

1. Paper only. `configs/active/autonomy.yaml` mode stays `paper` — `app/deps.py`
   refuses to start otherwise. No live broker, no live API keys, ever.
2. Hermes/LLM never on the hot path. Flag-file only, fail-safe: stale or missing file
   → no effect on decisions.
3. Deterministic risk engine has final veto. Nothing bypasses it.
4. Dashboard/API bind 127.0.0.1 only. Never expose or tunnel.
5. ML stays shadow until it clears the research gate (PBO/Deflated Sharpe).

## gpu_meanrev research rules (learned the hard way, 2026-08-06)

The 1-minute FX study lives in `gpu_meanrev/`. Read `journal/B01_B02_POSTMORTEM.md`
before proposing a B03 — it is the record of what B01 and B02 could not answer.

1. **Report breakeven cost next to profit factor.** B02's gross edge was 0.406 bps
   against an assumed 1.0 bp round-trip cost: the cost assumption was 2.5x the whole
   signal, and nobody noticed for two days because no scorecard carried the comparison.
   Anything under ~0.5 bps gross is untradable here regardless of its statistics.
2. **Hand the gate day-aggregated PnL, not per-trade rows.** Trades overlap and run
   across correlated pairs; per-trade `n` overstates the sample and flatters the
   Deflated Sharpe. B02's headline combo is t=1.12 on daily PnL — insignificant before
   any deflation at all.
3. **Trade records must carry the entry side**, not just the exit. `run_fold` records
   `entry_bar`, `entry_ts`, `side`, `entry_z`, `bars_held`. `entry_ts` is stored, never
   derived: `entry_bar` is fold-local and the panel index is an irregular union of
   observed minutes. B01/B02 stored exit-only and their entry-hour attribution is
   permanently unrecoverable.
4. **Check provenance before trusting a checkpoint.** `loader.sanitize_closes`' global
   band filter (f877bec, 2026-08-05 21:37) is the only thing catching sustained
   wrong-scale segments. Trade files written before it are contaminated wherever their
   universe meets 2000-2007 — B01's are. Run
   `python -m gpu_meanrev.analysis.data_provenance` rather than assuming.
5. **Never build two full panels concurrently.** Two jobs drove free memory to 0.9GB and
   stalled both (2026-08-05). Bound both ends of `build_panel` before the union/ffill.
6. **Any diagnostics path that reimplements the loader must re-assert the holdout clip**
   — duplicating `build_panel` duplicates away the one guard it exists to enforce.

## Verification commands

Use `.venv\Scripts\python` — bare `python` is the Store stub on this box.

- Tests: `.venv\Scripts\python -m pytest`
- API: `.venv\Scripts\python -m uvicorn app.api.main:app --reload` (127.0.0.1:8000)
- Dashboard: `.venv\Scripts\python -m streamlit run ui/streamlit_app.py`
- Smoke webhook: POST to `http://127.0.0.1:8000/webhook/tradingview` (payload example
  in `README.md`)

## Environment

- Python 3.11 in `.venv`; `.venv\pip.ini` already persists the truststore flag for the
  TLS-intercepting proxy.
- Windows 10, local-timezone day boundaries, no POSIX-only syscalls.
- Norton AV may transiently lock `.git/objects` — retry `git add`/`commit`.


### GPU numeric mode (`gpu.tune_backend()`, added 2026-09-01)

`get_device()` now enables TF32 on CUDA. Measured on this box's RTX 3050 idle at
full clock, 4096x4096 matmul, median of 3: fp32 1.43 -> TF32 2.62 TFLOPS (**1.84x**).
Only `matmul.allow_tf32` actually changes: torch already defaults
`cudnn.allow_tf32` to True, so conv/RNN layers were always running TF32 --
do **not** read this change as "the conv results moved".

| env var | default | effect |
| --- | --- | --- |
| `GPU_TF32` | on | `0` restores the previous precision setting. NOT bit-exact reproduction -- that also needs `cudnn.benchmark` pinned, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the same torch/CUDA/driver build. |
| `GPU_MEM_FRACTION` | 0.92 | `0` lifts the per-process VRAM cap. The cap exists so an oversized allocation raises `OutOfMemoryError` instead of silently spilling into system RAM through the Windows driver's sysmem fallback and running 10-50x slower. It caps against **total** VRAM, not free -- with ollama holding ~5.9 of 6 GB the driver still binds first. |

`confidence()` stamps the returned state under `"tuning"`, so a recorded number
says which numeric mode produced it. Keep that stamp: without it an old row and a
new one are indistinguishable in the ledger.

**Do not re-run a battery merely to compare TF32 against non-TF32.**
`registry.log_trial` appends to the ledger and `n_trials` is cumulative, so a
curiosity re-run permanently tightens the deflated-Sharpe threshold for that
family. Check precision on a scratch script, never through the harness.
