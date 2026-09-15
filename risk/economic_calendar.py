"""Generate event blackout windows from the economic calendar schedule.

Replaces the Milestone-1 static list in risk.yaml: events live in
configs/active/economic_calendar.yaml (NFP by first-Friday rule, FOMC/CPI/
central-bank decisions as explicit dates), and window sizes are derived from
autonomy.yaml event_blackouts minutes-before/after - exactly as the plan's
"later milestones" note in risk/event_blackout.py prescribed.

The generator is deterministic and offline (no API keys, no scraping). The
output artifact data/artifacts/economic_calendar.json is merged with any
static risk.yaml entries when services are built.

Usage:
  python -m risk.economic_calendar --days 90
"""

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.paths import data_dir
from app.services.config_service import load_config

ARTIFACT_NAME = "economic_calendar.json"


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _event_utc(day: date, time_local: str, tz_name: str) -> datetime:
    hour, minute = (int(x) for x in time_local.split(":"))
    local = datetime.combine(day, time(hour, minute), tzinfo=ZoneInfo(tz_name))
    return local.astimezone(timezone.utc)


def _window_minutes(kind: str, blackout_cfg: dict) -> tuple[int, int]:
    before = blackout_cfg.get(f"{kind}_minutes_before", 30)
    after = blackout_cfg.get(f"{kind}_minutes_after", 60)
    return int(before), int(after)


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def generate_events(
    start: date, end: date, calendar_cfg: dict | None = None
) -> list[dict]:
    """All scheduled events in [start, end] as {name, kind, event_time_utc}."""
    cfg = calendar_cfg if calendar_cfg is not None else load_config("economic_calendar")
    events: list[dict] = []

    nfp = (cfg.get("rules") or {}).get("nfp") or {}
    if nfp.get("enabled"):
        for year, month in _months_between(start, end):
            day = first_friday(year, month)
            if start <= day <= end:
                events.append(
                    {
                        "name": nfp.get("name", "US Nonfarm Payrolls"),
                        "kind": nfp.get("kind", "nfp"),
                        "event_time_utc": _event_utc(
                            day,
                            nfp.get("time_local", "08:30"),
                            nfp.get("timezone", "America/New_York"),
                        ),
                    }
                )

    for entry in cfg.get("explicit_events") or []:
        day = entry["date"]
        if isinstance(day, str):
            day = date.fromisoformat(day)
        if start <= day <= end:
            events.append(
                {
                    "name": entry["name"],
                    "kind": entry.get("kind", "central_bank"),
                    "event_time_utc": _event_utc(
                        day,
                        entry.get("time_local", "12:00"),
                        entry.get("timezone", "America/New_York"),
                    ),
                }
            )

    events.sort(key=lambda e: e["event_time_utc"])
    return events


def build_blackout_calendar(
    start: date, end: date, blackout_cfg: dict | None = None
) -> list[dict]:
    """Blackout windows compatible with risk.event_blackout.is_blackout."""
    cfg = blackout_cfg if blackout_cfg is not None else (
        load_config("autonomy").get("event_blackouts") or {}
    )
    windows = []
    for event in generate_events(start, end):
        before, after = _window_minutes(event["kind"], cfg)
        t = event["event_time_utc"]
        windows.append(
            {
                "name": event["name"],
                "start": (t - timedelta(minutes=before)).isoformat(),
                "end": (t + timedelta(minutes=after)).isoformat(),
            }
        )
    return windows


def write_artifact(days: int = 90, artifacts_dir: Path | None = None) -> Path:
    today = datetime.now(timezone.utc).date()
    windows = build_blackout_calendar(today, today + timedelta(days=days))
    artifacts_dir = artifacts_dir or (data_dir() / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / ARTIFACT_NAME
    path.write_text(json.dumps(windows, indent=2), encoding="utf-8")
    return path


def load_artifact(artifacts_dir: Path | None = None) -> list[dict]:
    path = (artifacts_dir or (data_dir() / "artifacts")) / ARTIFACT_NAME
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    path = write_artifact(days=args.days)
    windows = load_artifact()
    print(f"{len(windows)} blackout windows -> {path}")
    for w in windows[:10]:
        print(f"  {w['start']} .. {w['end']}  {w['name']}")


if __name__ == "__main__":
    main()
