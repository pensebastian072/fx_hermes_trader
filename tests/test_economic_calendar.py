"""Economic calendar generation -> blackout windows -> risk engine veto."""

from datetime import date, datetime, timezone

from risk.economic_calendar import (
    build_blackout_calendar,
    first_friday,
    generate_events,
    load_artifact,
    write_artifact,
)
from risk.event_blackout import is_blackout

CAL_CFG = {
    "rules": {
        "nfp": {
            "enabled": True,
            "kind": "nfp",
            "name": "US Nonfarm Payrolls",
            "time_local": "08:30",
            "timezone": "America/New_York",
        }
    },
    "explicit_events": [
        {
            "name": "FOMC rate decision",
            "kind": "fomc",
            "date": "2026-06-17",
            "time_local": "14:00",
            "timezone": "America/New_York",
        }
    ],
}

BLACKOUT_CFG = {
    "nfp_minutes_before": 30,
    "nfp_minutes_after": 30,
    "fomc_minutes_before": 60,
    "fomc_minutes_after": 120,
}


def test_first_friday():
    assert first_friday(2026, 6) == date(2026, 6, 5)
    assert first_friday(2026, 7) == date(2026, 7, 3)
    assert first_friday(2026, 1) == date(2026, 1, 2)  # Jan 1 is a Thursday


def test_generate_events_dst_aware():
    events = generate_events(date(2026, 1, 1), date(2026, 7, 31), CAL_CFG)
    nfp = {e["event_time_utc"].date(): e["event_time_utc"] for e in events
           if e["kind"] == "nfp"}
    # January NFP: 8:30 EST = 13:30 UTC; July NFP: 8:30 EDT = 12:30 UTC
    assert nfp[date(2026, 1, 2)].hour == 13
    assert nfp[date(2026, 7, 3)].hour == 12


def test_blackout_windows_use_kind_minutes():
    windows = build_blackout_calendar(
        date(2026, 6, 15), date(2026, 6, 20), BLACKOUT_CFG
    )
    # uses the real configs/active/economic_calendar.yaml for events
    fomc = [w for w in windows if "FOMC" in w["name"]]
    assert fomc, "expected the 2026-06-17 FOMC decision in the window"
    start = datetime.fromisoformat(fomc[0]["start"])
    end = datetime.fromisoformat(fomc[0]["end"])
    assert (end - start).total_seconds() == (60 + 120) * 60


def test_artifact_roundtrip_and_risk_engine_veto(data_dir):
    write_artifact(days=60)
    calendar = load_artifact()
    assert calendar, "expected upcoming events within 60 days"
    inside = datetime.fromisoformat(calendar[0]["start"]) + (
        datetime.fromisoformat(calendar[0]["end"])
        - datetime.fromisoformat(calendar[0]["start"])
    ) / 2
    blackout, name = is_blackout(inside, calendar)
    assert blackout and name == calendar[0]["name"]
    outside = datetime(2099, 1, 1, tzinfo=timezone.utc)
    assert is_blackout(outside, calendar) == (False, None)
