"""Hermes supervisor: permissions, log collection, review writing, fallbacks.

The LLM is never required: every test runs with Ollama absent (pick_model
monkeypatched or httpx pointed at a dead port).
"""

import json
from datetime import datetime, timezone

import pytest

import hermes.supervisor as supervisor
from app.paths import logs_dir
from app.services.artifact_service import append_jsonl


def test_check_permissions_passes_with_active_config():
    autonomy = supervisor.check_permissions()
    assert autonomy["mode"] == "paper"


def test_check_permissions_rejects_non_paper(monkeypatch):
    monkeypatch.setattr(
        supervisor, "load_config", lambda name: {"mode": "live", "hermes_permissions": {}}
    )
    with pytest.raises(RuntimeError, match="paper"):
        supervisor.check_permissions()


def test_check_permissions_rejects_missing_permission(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "load_config",
        lambda name: {
            "mode": "paper",
            "hermes_permissions": {"read_logs": True, "write_reports": False},
        },
    )
    with pytest.raises(RuntimeError, match="write_reports"):
        supervisor.check_permissions()


def test_collect_today_logs_counts_todays_entries_only(data_dir):
    logs = logs_dir()
    today = datetime.now(timezone.utc).isoformat()
    append_jsonl(logs / "alerts.jsonl", {"timestamp": today, "symbol": "USDJPY"})
    append_jsonl(logs / "alerts.jsonl", {"timestamp": "2020-01-01T00:00:00+00:00"})
    append_jsonl(
        logs / "decision_traces.jsonl",
        {
            "timestamp": today,
            "symbol": "USDJPY",
            "decision": "reject",
            "risk_checks_failed": True,
            "rejection_reason": "max daily loss reached",
        },
    )
    out = supervisor.collect_today_logs()
    assert out["alerts"] == 1
    assert out["risk_vetoes"] == 1
    assert "USDJPY: max daily loss reached" in out["veto_reasons"]


def test_deterministic_observations_flag_known_issues():
    obs = supervisor.deterministic_observations(
        logs={"alerts": 0, "decision_traces": 2, "risk_vetoes": 2},
        regimes={"USDJPY": {"dominant": "HIGH_VOL_CRISIS", "crisis_probability": 0.7}},
        backtests={"EURUSD": {"gates_passed": False, "gate_failures": ["profit_factor 0.9"]}},
        models={"trend_rf.joblib": {"active": False}},
    )
    text = " ".join(obs)
    assert "No alerts" in text
    assert "vetoed" in text
    assert "crisis probability" in text
    assert "promotion gates" in text
    assert "ML specialists not in use" in text


def test_pick_model_none_when_ollama_down(monkeypatch):
    assert supervisor.pick_model(None, base_url="http://127.0.0.1:9") is None


def test_write_review_outputs_report_notes_and_suggestions(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.setattr(supervisor, "MEMORY_DIR", memory)

    payload = {"logs": {"date": "2026-06-12"}, "regimes": {}, "recent_backtests": {}}
    path = supervisor.write_review(payload, None, ["observation one"], None)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "deterministic fallback" in content
    assert "observation one" in content

    notes = (memory / "strategy_notes.md").read_text(encoding="utf-8")
    assert "2026-06-12 (supervisor)" in notes

    suggestions = (data_dir / "artifacts" / "hermes_suggestions.jsonl").read_text(
        encoding="utf-8"
    )
    record = json.loads(suggestions.strip().splitlines()[-1])
    assert record["source"] == "deterministic"
    assert record["observations"] == ["observation one"]


def test_build_review_prompt_forbids_config_changes(monkeypatch, tmp_path):
    payload = {"logs": {"date": "2026-06-12"}}
    prompt = supervisor.build_review_prompt(payload)
    assert "may not propose disabling any risk control" in prompt
    assert "2026-06-12" in prompt
