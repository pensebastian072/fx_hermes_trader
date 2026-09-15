"""Event blackout checks.

Milestone 1: a static calendar of blackout windows loaded from risk.yaml.
Later milestones replace this with a real economic-calendar feed that derives
windows from the autonomy.yaml minutes-before/after settings.
"""

from datetime import datetime


def is_blackout(
    ts: datetime, calendar: list[dict] | None
) -> tuple[bool, str | None]:
    """Return (True, event_name) if ts falls inside any blackout window."""
    for event in calendar or []:
        start = _parse(event["start"], ts)
        end = _parse(event["end"], ts)
        if start <= ts <= end:
            return True, str(event.get("name", "unnamed_event"))
    return False, None


def _parse(value: str | datetime, reference: datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    # Naive calendar entries inherit the alert's timezone so comparison is valid.
    if dt.tzinfo is None and reference.tzinfo is not None:
        dt = dt.replace(tzinfo=reference.tzinfo)
    return dt
