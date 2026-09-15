"""Pre-registered experiment batteries + the n_trials ledger.

Adapted from alpaca_gpu_lab/src/experiments/registry.py (same discipline,
repointed at fx_hermes_trader/journal/ — this ledger is LOCAL to this repo;
only the gate *math* is shared/imported across repos, never trial-count
bookkeeping):

1. register(...) BEFORE any result exists — writes
   journal/experiments/registered/<battery_id>.json. Refuses to overwrite
   (append-only science — a changed hypothesis is a NEW battery id).
2. Every (params, pair-set) combo actually executed appends one line to
   journal/experiments/ledger.jsonl.
3. n_trials_for(family) = cumulative ledger trials against that family +
   the family's seed — fed to evaluate_gate(..., n_trials=...).
   "fx_1min_meanrev" seeds at 0: genuinely new ground, no prior history of
   1-minute FX price-action reversion testing on this box.
4. Scorecards land in journal/scorecards/, one human-readable row appended
   to journal/RESULTS.md. Honest FAILs are the expected output.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gpu_meanrev import config

REGISTERED = config.EXPERIMENTS / "registered"
LEDGER = config.EXPERIMENTS / "ledger.jsonl"
RESULTS_MD = config.JOURNAL / "RESULTS.md"

FAMILY_SEEDS = {
    "fx_1min_meanrev": 0,  # no prior history — new ground
}


class UnregisteredBattery(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def register(
    battery_id: str,
    hypothesis: str,
    dataset_family: str,
    pairs: list[str],
    param_grid: dict,
    n_trials: int,
) -> Path:
    """Pre-register a battery. Refuses to overwrite an existing registration."""
    REGISTERED.mkdir(parents=True, exist_ok=True)
    path = REGISTERED / f"{battery_id}.json"
    if path.exists():
        raise FileExistsError(
            f"battery {battery_id} already registered — registrations are "
            "append-only; a changed design needs a new battery id"
        )
    rec = {
        "battery_id": battery_id,
        "registered_at": _now(),
        "hypothesis": hypothesis,
        "dataset_family": dataset_family,
        "pairs": pairs,
        "param_grid": param_grid,
        "n_trials": n_trials,
        "source": config.HF_REPO_ID,
    }
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return path


def load_registration(battery_id: str) -> dict:
    path = REGISTERED / f"{battery_id}.json"
    if not path.exists():
        raise UnregisteredBattery(
            f"battery {battery_id} is not registered — call register() with a "
            "written hypothesis BEFORE evaluating"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def log_trial(battery_id: str, dataset_family: str, params: dict, pairs: list[str]) -> None:
    """Append one executed param combo to the ledger (one line per combo, not per pair)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "ts": _now(), "battery_id": battery_id, "dataset_family": dataset_family,
        "params": params, "pairs": pairs,
    })
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_trials_for(dataset_family: str) -> int:
    n = FAMILY_SEEDS.get(dataset_family, 0)
    if LEDGER.exists():
        with LEDGER.open(encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("dataset_family") == dataset_family:
                        n += 1
                except json.JSONDecodeError:
                    continue  # torn line from a killed run — ignore, never crash
    return max(n, 1)


def append_result_row(battery_id: str, verdict: dict, note: str = "") -> None:
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(
            "# Results log\n\n"
            "Every gate verdict, honest FAILs included — that is the system "
            "working, not a bug.\n\n"
            "| date | battery | n | PF | Sharpe | DSR ratio | PBO | n_trials | verdict | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    dsr = verdict.get("deflated_sharpe") or {}
    row = ("| {d} | {b} | {n} | {pf} | {sr} | {ratio} | {pbo} | {nt} | {v} | {note} |\n").format(
        d=_now()[:10], b=battery_id,
        n=verdict.get("n_trades", "-"),
        pf=_fmt(verdict.get("profit_factor")),
        sr=_fmt(verdict.get("sharpe")),
        ratio=_fmt(dsr.get("ratio")),
        pbo=_fmt(verdict.get("pbo")),
        nt=dsr.get("n_trials", "-"),
        v="PASS" if verdict.get("passes") else "FAIL",
        note=note.replace("|", "/"),
    )
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write(row)


def write_scorecard(battery_id: str, payload: dict) -> Path:
    config.SCORECARDS.mkdir(parents=True, exist_ok=True)
    out = config.SCORECARDS / f"{battery_id}_{_now()[:10]}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
