"""Operator dashboard (Milestone 1).

Read-only: this UI displays logs and configs. It contains no trading logic
and cannot place, modify, or close trades.

Run with: streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd
import streamlit as st

from app.paths import data_dir, logs_dir
from app.services.artifact_service import read_jsonl
from app.services.config_service import load_config

st.set_page_config(page_title="FX Hermes Trader", layout="wide")

st.title("FX Hermes Trader — Operator Dashboard")
st.error("MODE: PAPER — live trading is DISABLED and not implemented in this build.")

logs = logs_dir()
alerts = read_jsonl(logs / "alerts.jsonl")
rejected = read_jsonl(logs / "rejected_alerts.jsonl")
traces = read_jsonl(logs / "decision_traces.jsonl")
orders = read_jsonl(logs / "orders.jsonl")
fills = read_jsonl(logs / "fills.jsonl")

risk_config = load_config("risk")
watchlist = load_config("watchlist").get("pairs", [])

starting_capital = float(risk_config.get("starting_capital", 10000))
realized = sum(f.get("realized_pnl", 0.0) for f in fills if f.get("kind") == "close")
closed_ids = {f["order_id"] for f in fills if f.get("kind") == "close"}
open_orders = [o for o in orders if o["order_id"] not in closed_ids]

# --- System overview ---------------------------------------------------------
st.header("System Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity (paper)", f"${starting_capital + realized:,.2f}")
c2.metric("Realized PnL", f"${realized:,.2f}")
c3.metric("Open paper positions", len(open_orders))
c4.metric("Alerts received", len(alerts))
c5.metric("Alerts rejected", len(rejected))
st.caption(f"Watchlist: {', '.join(watchlist)}")

# --- Regime monitor ----------------------------------------------------------
st.header("Regime Monitor (HMM)")
artifacts = data_dir() / "artifacts"
snapshot_path = artifacts / "regime_snapshot.json"
posteriors_path = artifacts / "regime_posteriors.csv"
if snapshot_path.exists():
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    r1, r2, r3 = st.columns(3)
    r1.metric("Dominant regime", snap.get("dominant_regime", "?"))
    r2.metric("P(crisis)", f"{snap.get('crisis_probability', 0):.2f}")
    r3.metric("Symbol / bars", f"{snap.get('symbol', '?')} / {snap.get('n_bars', 0)}")
    if posteriors_path.exists():
        post = pd.read_csv(posteriors_path).set_index("bar")
        st.line_chart(post)
    scaling = risk_config.get("regime_scaling", {})
    if scaling.get("enabled") and snap.get("crisis_probability", 0) >= scaling.get(
        "halt_probability", 0.8
    ):
        st.error("CRISIS HALT ACTIVE — new trades vetoed by risk engine.")
else:
    st.info(
        "No regime snapshot yet. Run: python -m engines.regime_detection.run_regime "
        "--csv <bars.csv> --symbol USDJPY (or --demo)"
    )

# --- Macro sentiment ----------------------------------------------------------
st.header("Central Bank Sentiment (multi-agent debate)")
sentiment_files = sorted(artifacts.glob("sentiment_*.json")) if artifacts.exists() else []
if sentiment_files:
    for f in sentiment_files:
        s = json.loads(f.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{s['currency']} bias", s["macro_bias"])
        c2.metric("Hawk/Dove score", f"{s['hawkish_dovish_score']:+.1f}")
        c3.metric("Confidence", f"{s['confidence']:.2f}")
        c4.metric("Action", s["trade_action"])
        st.caption(s["reason"])
else:
    st.info(
        "No sentiment artifacts. Drop transcripts in data/raw/central_bank/ and run: "
        "python -m hermes.macro_sentiment.run_sentiment --currency USD"
    )

# --- Hermes review / suggestions ----------------------------------------------
st.header("Hermes Review (advisory only)")
st.caption(
    "Suggestions are inert text written by hermes.supervisor; no code path "
    "executes them. Config and risk changes remain human-only."
)
suggestions = read_jsonl(artifacts / "hermes_suggestions.jsonl", limit=10)
if suggestions:
    for s in suggestions[::-1]:
        with st.expander(f"{s.get('date')} — {s.get('source')}"):
            for obs in s.get("observations", []):
                st.markdown(f"- {obs}")
    latest_reviews = sorted((PROJECT_ROOT / "reports" / "hermes").glob("*.md"))
    if latest_reviews:
        st.markdown(latest_reviews[-1].read_text(encoding="utf-8"))
else:
    st.info("No Hermes reviews yet. Run: python -m hermes.supervisor")

# --- Upcoming event blackouts ---------------------------------------------------
st.header("Upcoming Event Blackouts")
calendar_path = artifacts / "economic_calendar.json"
if calendar_path.exists():
    windows = json.loads(calendar_path.read_text(encoding="utf-8"))
    if windows:
        st.dataframe(pd.DataFrame(windows), use_container_width=True)
else:
    st.info("No calendar artifact. Run: python -m risk.economic_calendar")

# --- Research panel -----------------------------------------------------------
st.header("Research / Backtests")
reports_dir = PROJECT_ROOT / "backtests" / "reports"
report_files = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []
supervisor_reports = artifacts / "backtests"
if supervisor_reports.exists():
    report_files += sorted(supervisor_reports.glob("*.json"))
if report_files:
    rows = []
    for f in report_files:
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"name": r.get("name"), **r.get("metrics", {})})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No backtest reports yet (backtests/reports/*.json).")

# --- Latest alerts -----------------------------------------------------------
st.header("Latest Alerts")
if alerts:
    st.dataframe(pd.DataFrame(alerts[-20:][::-1]), use_container_width=True)
else:
    st.info("No alerts yet. POST to /webhook/tradingview to create some.")

# --- Decision traces ---------------------------------------------------------
st.header("Latest Decisions")
if traces:
    df = pd.DataFrame(traces[-20:][::-1])
    cols = [
        c
        for c in [
            "timestamp", "symbol", "signal", "direction", "final_score",
            "decision", "rejection_reason", "entry", "stop", "target", "size",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols], use_container_width=True)
else:
    st.info("No decisions yet.")

# --- Open paper positions ----------------------------------------------------
st.header("Open Paper Positions")
if open_orders:
    st.dataframe(pd.DataFrame(open_orders), use_container_width=True)
else:
    st.info("No open paper positions.")

# --- Rejected alerts ---------------------------------------------------------
st.header("Rejected Alerts")
if rejected:
    st.dataframe(pd.DataFrame(rejected[-20:][::-1]), use_container_width=True)
else:
    st.info("No rejected alerts.")

# --- Risk config -------------------------------------------------------------
st.header("Risk Configuration (read-only)")
st.json(risk_config)
st.caption(
    "Risk limits are enforced by the deterministic risk engine "
    "(risk/risk_engine.py). This dashboard cannot change them."
)
