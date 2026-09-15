"""Hermes supervisor - daily research/review loop (PROJECT_PLAN.md section 12).

Run AFTER the trading session. Pipeline:
  1. refresh the economic-calendar blackout artifact
  2. refresh regime snapshots from the latest daily bars
  3. replay recent history through the backtester per pair, check promotion gates
  4. generate the deterministic journal report
  5. self-review: read today's logs + artifacts + Hermes's own past notes,
     ask the local LLM for observations, write a review with suggestions

Self-improvement is bounded by autonomy.yaml hermes_permissions:
  - Hermes READS logs, market data, reports, and its own strategy notes;
  - Hermes WRITES reports and suggestions (reports/hermes/, strategy notes);
  - Hermes NEVER edits configs, code, or risk limits. Suggestions are inert
    text until a human reviews them; the promotion gate and the deterministic
    risk engine are not bypassable from here. No live trading exists.

Usage:
  python -m hermes.supervisor                 (full daily run)
  python -m hermes.supervisor --refresh-data  (download fresh bars first)
  python -m hermes.supervisor --skip-backtests --model qwen2.5:7b
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.paths import PROJECT_ROOT, data_dir, logs_dir
from app.services.artifact_service import read_jsonl
from app.services.config_service import load_config
from backtests.data_loader import load_ohlc
from backtests.runner import BacktestConfig, BacktestRunner, save_report
from engines.ml_specialist import RANGE_MODEL, TREND_MODEL, load_ensemble, models_dir
from engines.regime_detection.hmm_regime import HMMRegimeDetector, save_snapshot
from hermes.journal_agent import generate_daily_report
from hermes.macro_sentiment.ollama_client import DEFAULT_URL, OllamaClient
from hermes.promotion_gate import evaluate_backtest_gates
from risk.economic_calendar import write_artifact as write_calendar_artifact

RECENT_BARS = 1500  # ~6 years of daily bars per backtest replay
PREFERRED_MODELS = ["hermes3:8b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
PROMPTS_DIR = Path(__file__).parent / "prompts"
MEMORY_DIR = Path(__file__).parent / "memory"
STRATEGY_NOTES_TAIL_CHARS = 4000


def check_permissions() -> dict:
    autonomy = load_config("autonomy")
    if autonomy.get("mode") != "paper":
        raise RuntimeError("supervisor refuses to run: autonomy mode is not 'paper'")
    perms = autonomy.get("hermes_permissions") or {}
    for needed in ("read_logs", "write_reports", "run_backtests"):
        if not perms.get(needed):
            raise RuntimeError(f"hermes permission '{needed}' is disabled in autonomy.yaml")
    return autonomy


def collect_today_logs() -> dict:
    logs = logs_dir()
    today = datetime.now(timezone.utc).date().isoformat()

    def today_only(rows: list[dict]) -> list[dict]:
        return [r for r in rows if str(r.get("timestamp", "")).startswith(today)]

    traces = today_only(read_jsonl(logs / "decision_traces.jsonl"))
    vetoes = [t for t in traces if t.get("risk_checks_failed")]
    return {
        "date": today,
        "alerts": len(today_only(read_jsonl(logs / "alerts.jsonl"))),
        "rejected_alerts": len(today_only(read_jsonl(logs / "rejected_alerts.jsonl"))),
        "decision_traces": len(traces),
        "accepted": len([t for t in traces if t.get("decision") == "accept"]),
        "risk_vetoes": len(vetoes),
        "veto_reasons": [
            f"{t.get('symbol')}: {t.get('rejection_reason')}" for t in vetoes
        ][:20],
        "paper_orders": len(today_only(read_jsonl(logs / "orders.jsonl"))),
    }


def refresh_regime_snapshots(pairs: list[str]) -> dict[str, dict]:
    """Fit per-pair HMMs on full daily history; persist the primary snapshot."""
    artifacts = data_dir() / "artifacts"
    out: dict[str, dict] = {}
    for i, pair in enumerate(pairs):
        csv = data_dir() / "raw" / f"{pair.lower()}_1d.csv"
        if not csv.exists():
            continue
        closes = load_ohlc(csv)["close"].to_numpy()
        detector = HMMRegimeDetector().fit(closes[:-250] if len(closes) > 600 else closes)
        snapshot = detector.snapshot(closes, pair)
        if i == 0:
            save_snapshot(snapshot, artifacts)  # webhook pipeline reads this one
        out[pair] = {
            "dominant": snapshot.dominant_regime,
            "crisis_probability": round(snapshot.crisis_probability, 3),
        }
    return out


def run_recent_backtests(pairs: list[str]) -> dict[str, dict]:
    """Replay recent history per pair through the full stack + gate check."""
    reports_dir = data_dir() / "artifacts" / "backtests"
    results: dict[str, dict] = {}
    for pair in pairs:
        csv = data_dir() / "raw" / f"{pair.lower()}_1d.csv"
        if not csv.exists():
            continue
        df = load_ohlc(csv).tail(RECENT_BARS).reset_index(drop=True)
        runner = BacktestRunner(BacktestConfig(), ensemble=load_ensemble())
        result = runner.run(df)
        save_report(result, f"{pair.lower()}_recent", reports_dir)
        gate = evaluate_backtest_gates(result.metrics)
        results[pair] = {
            "metrics": {
                k: result.metrics[k]
                for k in ("n_trades", "net_pnl", "win_rate", "profit_factor",
                          "max_drawdown_pct", "sharpe", "return_pct")
            },
            "gates_passed": gate.passed,
            "gate_failures": gate.failures,
        }
    return results


def model_status() -> dict:
    out = {}
    for f in (TREND_MODEL, RANGE_MODEL):
        meta_path = (models_dir() / f).with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out[f] = {
                "oos_accuracy": meta.get("oos_accuracy"),
                "baseline_accuracy": meta.get("baseline_accuracy"),
                "active": float(meta.get("oos_accuracy", 0))
                > float(meta.get("baseline_accuracy", 0.5)),
            }
        else:
            out[f] = {"active": False, "oos_accuracy": None}
    return out


def deterministic_observations(
    logs: dict, regimes: dict, backtests: dict, models: dict
) -> list[str]:
    """Rule-based fallback review - works with no LLM running."""
    obs: list[str] = []
    if logs["alerts"] == 0:
        obs.append(
            "No alerts received today - webhook source idle or not connected "
            "(Cloudflare tunnel for TradingView is still an open item)."
        )
    if logs["decision_traces"] and logs["risk_vetoes"] == logs["decision_traces"]:
        obs.append("Every decision today was vetoed - check veto reasons for a stuck breaker.")
    for pair, r in regimes.items():
        if r["crisis_probability"] >= 0.5:
            obs.append(
                f"{pair}: elevated crisis probability {r['crisis_probability']} - "
                "position scaling is reducing size."
            )
    for pair, b in backtests.items():
        if not b["gates_passed"]:
            obs.append(f"{pair}: recent-history backtest fails promotion gates: "
                       + "; ".join(b["gate_failures"]))
    inactive = [name for name, m in models.items() if not m.get("active")]
    if inactive:
        obs.append(
            f"ML specialists not in use ({', '.join(inactive)}): out-of-sample accuracy "
            "does not beat the majority-class baseline. Suggest richer features "
            "(carry/rate differentials, session structure, cross-pair correlation) "
            "before retraining; rule-based specialists remain in effect."
        )
    return obs or ["No anomalies detected by deterministic checks."]


def pick_model(requested: str | None, base_url: str = DEFAULT_URL) -> str | None:
    """Requested model if available, else first preferred model Ollama has."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return None
    if not available:
        return None
    if requested:
        return requested if requested in available else None
    for name in PREFERRED_MODELS:
        if name in available:
            return name
    return available[0]


def build_review_prompt(payload: dict) -> str:
    system = (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")
    review = (PROMPTS_DIR / "daily_review_prompt.md").read_text(encoding="utf-8")
    notes_path = MEMORY_DIR / "strategy_notes.md"
    notes_tail = ""
    if notes_path.exists():
        notes_tail = notes_path.read_text(encoding="utf-8")[-STRATEGY_NOTES_TAIL_CHARS:]
    return (
        f"{system}\n\n{review}\n\n"
        "## Today's data (JSON)\n```json\n"
        f"{json.dumps(payload, indent=2, default=str)}\n```\n\n"
        "## Your own recent strategy notes (memory)\n"
        f"{notes_tail or '(none yet)'}\n\n"
        "Respond with the daily review. End with a section titled "
        "'## Suggestions' containing at most 3 bullet points; each suggestion "
        "must be something a human could evaluate, never a config change you "
        "apply yourself. You may not propose disabling any risk control."
    )


def write_review(
    payload: dict, llm_text: str | None, fallback_obs: list[str], model: str | None
) -> Path:
    today = payload["logs"]["date"]
    reports_dir = PROJECT_ROOT / "reports" / "hermes"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Hermes Daily Review - {today}",
        "",
        f"Mode: PAPER. Generated by hermes.supervisor; suggestions are advisory only.",
        f"Reviewer: {'LLM ' + model if llm_text else 'deterministic fallback (no LLM available)'}",
        "",
        "## Snapshot",
        "```json",
        json.dumps(payload, indent=2, default=str),
        "```",
        "",
    ]
    if llm_text:
        lines += ["## LLM review", "", llm_text.strip(), ""]
    lines += ["## Deterministic observations", ""]
    lines += [f"- {o}" for o in fallback_obs]

    path = reports_dir / f"{today}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Append to Hermes's own memory so tomorrow's review sees today's thinking.
    notes_path = MEMORY_DIR / "strategy_notes.md"
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(f"\n### {today} (supervisor)\n")
        for o in fallback_obs:
            f.write(f"- {o}\n")
        if llm_text:
            f.write(f"- LLM review written to reports/hermes/{today}.md\n")

    # Machine-readable suggestion trail for the dashboard / later analysis.
    suggestions_log = data_dir() / "artifacts" / "hermes_suggestions.jsonl"
    suggestions_log.parent.mkdir(parents=True, exist_ok=True)
    with suggestions_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "date": today,
            "source": f"llm:{model}" if llm_text else "deterministic",
            "observations": fallback_obs,
            "llm_review_path": str(path) if llm_text else None,
        }) + "\n")
    return path


def run_daily(refresh_data: bool = False, skip_backtests: bool = False,
              model: str | None = None) -> Path:
    check_permissions()
    pairs = list(load_config("watchlist").get("pairs", []))

    if refresh_data:
        from backtests.download_data import _trust_windows_certs, download_pair

        _trust_windows_certs()
        for pair in pairs:
            try:
                download_pair(pair)
            except Exception as exc:
                print(f"data refresh failed for {pair}: {exc}")

    write_calendar_artifact()
    regimes = refresh_regime_snapshots(pairs)
    backtests = {} if skip_backtests else run_recent_backtests(pairs)
    models = model_status()
    logs = collect_today_logs()
    generate_daily_report()

    payload = {
        "logs": logs,
        "regimes": regimes,
        "recent_backtests": backtests,
        "ml_models": models,
    }
    fallback_obs = deterministic_observations(logs, regimes, backtests, models)

    llm_text = None
    chosen = pick_model(model)
    if chosen:
        llm_text = OllamaClient(model=chosen).generate(build_review_prompt(payload))

    path = write_review(payload, llm_text, fallback_obs, chosen)
    print(f"Hermes review written: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--skip-backtests", action="store_true")
    parser.add_argument("--model", help="Ollama model name (default: first available)")
    args = parser.parse_args()
    run_daily(
        refresh_data=args.refresh_data,
        skip_backtests=args.skip_backtests,
        model=args.model,
    )


if __name__ == "__main__":
    main()
