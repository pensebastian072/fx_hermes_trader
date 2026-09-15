"""Download + cache HistData FX 1-minute parquet files from HF.

Uses hf_hub_download (not the `datasets` lib): it natively resumes partial
downloads and no-ops on an already-cached file by content hash, which
satisfies "cache every pull, never spend it twice" with no extra bookkeeping.
Idempotent at the pair level -- a killed run resumes by re-iterating the pair
list; already-cached pairs are instant no-ops.

--probe pulls exactly PROBE_PAIRS and writes a real date-range/gap/size
report to journal/data/probe_report.json BEFORE any full pull is attempted
(the box's standing "probe before a long run" rule). Full pull only reads the
resolved pair list from configs/research/fx_1m_pairs.yaml -- run
hf_catalog.py first (or pass --pairs) if that file doesn't exist yet.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.data.hf_download --probe
  .venv\\Scripts\\python.exe -m gpu_meanrev.data.hf_download --full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import truststore
truststore.inject_into_ssl()

import pandas as pd

from gpu_meanrev import config


def _download_pair(pair: str) -> dict:
    """Download (or confirm cached) ticks.parquet + gaps.parquet for one pair."""
    from huggingface_hub import hf_hub_download

    out = {"pair": pair, "ticks_path": None, "gaps_path": None, "error": None}
    try:
        out["ticks_path"] = hf_hub_download(
            repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE,
            filename=f"{pair.lower()}/ticks.parquet",
            local_dir=str(config.RAW),
        )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"ticks: {e!r}"
        return out
    try:
        out["gaps_path"] = hf_hub_download(
            repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE,
            filename=f"{pair.lower()}/gaps.parquet",
            local_dir=str(config.RAW),
        )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"gaps: {e!r}"  # ticks still usable, gaps optional-ish
    return out


def _profile_pair(pair: str, ticks_path: str, gaps_path: str | None) -> dict:
    """Real date-range / coverage / gap-density numbers for one pair."""
    df = pd.read_parquet(ticks_path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ts = pd.to_datetime(df[ts_col], utc=True)
    ts = ts.sort_values()

    n_rows = len(ts)
    min_ts, max_ts = ts.iloc[0], ts.iloc[-1]
    dupes = int(ts.duplicated().sum())
    monotonic = bool(ts.is_monotonic_increasing)

    # weekday-hours implied coverage: FX is quiet Sat/most of Sun by design,
    # so denominator is weekday minutes only, not the full calendar span.
    full_range = pd.date_range(min_ts, max_ts, freq="min", tz="UTC")
    weekday_minutes = int((full_range.weekday < 5).sum())
    coverage_pct = round(100.0 * n_rows / weekday_minutes, 2) if weekday_minutes else None

    gap_info: dict = {}
    if gaps_path:
        try:
            gdf = pd.read_parquet(gaps_path)
            gap_info = {
                "n_gaps": len(gdf),
                "columns": list(gdf.columns),
            }
            dur_col = next((c for c in gdf.columns if "dur" in c.lower()), None)
            if dur_col:
                gap_info["largest_gap"] = str(gdf[dur_col].max())
                gap_info["total_gap"] = str(gdf[dur_col].sum())
        except Exception as e:  # noqa: BLE001
            gap_info = {"error": repr(e)}

    return {
        "pair": pair,
        "n_rows": n_rows,
        "min_ts": str(min_ts),
        "max_ts": str(max_ts),
        "duplicate_timestamps": dupes,
        "monotonic": monotonic,
        "weekday_minutes_in_range": weekday_minutes,
        "coverage_pct_weekday": coverage_pct,
        "file_size_mb": round(Path(ticks_path).stat().st_size / 1e6, 2),
        "gaps": gap_info,
    }


def probe(pairs: list[str] | None = None) -> dict:
    pairs = pairs or config.PROBE_PAIRS
    profiles = []
    for pair in pairs:
        dl = _download_pair(pair)
        if dl["error"] and not dl["ticks_path"]:
            profiles.append({"pair": pair, "error": dl["error"]})
            continue
        profiles.append(_profile_pair(pair, dl["ticks_path"], dl["gaps_path"]))

    report = {"probed_pairs": pairs, "profiles": profiles}
    config.DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_REPORTS / "probe_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["_written_to"] = str(out_path)
    return report


def full_pull(pairs: list[str] | None = None) -> dict:
    if pairs is None:
        import yaml
        if not config.RESOLVED_PAIRS_FILE.exists():
            raise FileNotFoundError(
                f"{config.RESOLVED_PAIRS_FILE} missing -- run hf_catalog.py "
                "and write the resolved pair list first, or pass --pairs."
            )
        pairs = yaml.safe_load(config.RESOLVED_PAIRS_FILE.read_text(encoding="utf-8"))["pairs"]

    results = []
    for pair in pairs:
        results.append(_download_pair(pair))
    n_ok = sum(1 for r in results if not r["error"])
    return {"n_pairs": len(pairs), "n_ok": n_ok, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=None)
    args = ap.parse_args()

    if args.probe:
        print(json.dumps(probe(args.pairs), indent=2, default=str))
    elif args.full:
        print(json.dumps(full_pull(args.pairs), indent=2, default=str))
    else:
        ap.print_help()
