"""Deterministic daily report generator.

Milestone 1: pure-Python summary of the day's logs (no LLM). A local
Hermes/Llama model will later turn these numbers into narrative reviews -
it will only ever read logs and write reports, never trade.

Usage: python -m hermes.journal_agent
"""

from datetime import datetime, timezone
from pathlib import Path

from app.paths import PROJECT_ROOT, logs_dir
from app.services.artifact_service import read_jsonl


def generate_daily_report(reports_dir: Path | None = None) -> Path:
    logs = logs_dir()
    reports_dir = reports_dir or (PROJECT_ROOT / "reports" / "daily")
    reports_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date().isoformat()
    alerts = read_jsonl(logs / "alerts.jsonl")
    rejected = read_jsonl(logs / "rejected_alerts.jsonl")
    traces = read_jsonl(logs / "decision_traces.jsonl")
    orders = read_jsonl(logs / "orders.jsonl")

    accepted = [t for t in traces if t.get("decision") == "accept"]
    risk_rejected = [t for t in traces if t.get("risk_checks_failed")]

    lines = [
        f"# Daily Report - {today}",
        "",
        f"Mode: PAPER (live trading disabled)",
        "",
        f"- Alerts received: {len(alerts)}",
        f"- Alerts rejected (validation/duplicate/watchlist): {len(rejected)}",
        f"- Decision traces: {len(traces)}",
        f"- Trades accepted: {len(accepted)}",
        f"- Trades vetoed by risk engine: {len(risk_rejected)}",
        f"- Paper orders created: {len(orders)}",
        "",
        "## Risk vetoes",
    ]
    if risk_rejected:
        for t in risk_rejected:
            lines.append(f"- {t.get('symbol')}: {t.get('rejection_reason')}")
    else:
        lines.append("- none")

    path = reports_dir / f"{today}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Report written: {generate_daily_report()}")
