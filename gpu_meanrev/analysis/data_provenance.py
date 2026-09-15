"""Which archived batteries were computed before the wrong-scale data fix?

`loader.sanitize_closes` gained its GLOBAL-BAND filter on 2026-08-05 21:37
(commit f877bec). That filter is the only thing that catches a sustained
wrong-scale SEGMENT — the local centred-median filter structurally cannot,
because a long enough bad run drags the local median with it (AUDJPY 2005:
8,413 contiguous minutes printing 0.67 against an ~85 median, none removed).

Every trade file written before that timestamp was therefore produced by a
loader that could not see those bars, and any battery whose universe and
window overlap them is contaminated. That is not hypothetical: B01's regime
table reports 2005 profit factor 111.6 against a Sharpe of 0.039, which is
the AUDJPY defect booking 126x "returns" and nothing else.

This module answers the question mechanically, so the claim is reproducible
instead of resting on a one-off shell command:
  * per pair, how many bars the current sanitizer removes, and in which years
  * per battery, whether its (pairs x years) footprint intersects those years

It only READS. Deciding what to do with a contaminated checkpoint —
recompute, quarantine, delete — is the user's call, not this script's.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.analysis.data_provenance
"""
from __future__ import annotations

import json

import pandas as pd

from gpu_meanrev import config
from gpu_meanrev.batteries.b02_mtf_confluence import MAJORS_7, YEARS as B02_YEARS
from gpu_meanrev.data.loader import _ticks_path, sanitize_closes

OUT_PATH = config.SCORECARDS / "data_provenance.json"

# The fix landed here; anything written earlier used the blind loader.
GLOBAL_BAND_FIX_AT = "2026-08-05 21:37 (commit f877bec)"

BATTERIES = {
    "B01_zscore_reversion": {"pairs": config.PAIRS_CANDIDATE,
                             "years": list(range(2000, 2023))},
    "B02_mtf_confluence": {"pairs": MAJORS_7, "years": B02_YEARS},
}


def bad_bars(pair: str) -> dict:
    """Bars the CURRENT sanitizer rejects, and the years they sit in.

    Mirrors sanitize_closes' own two stages rather than calling it, because
    the report needs the TIMESTAMPS of the rejected bars and the function
    returns only counts.
    """
    df = pd.read_parquet(_ticks_path(pair), columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    s = df.set_index("ts")["close"]

    nonpositive = s <= 0
    pos = s[~nonpositive]
    med = pos.median()
    out_of_band = (pos > med * 4.0) | (pos < med / 4.0)

    flagged = pd.concat([s[nonpositive], pos[out_of_band]]).sort_index()
    _clean, rep = sanitize_closes(s)
    return {
        "pair": pair,
        "rows_in": rep["rows_in"],
        "dropped_nonpositive": rep["dropped_nonpositive"],
        "dropped_out_of_band": rep["dropped_out_of_band"],
        "dropped_off_scale": rep["dropped_off_scale"],
        "dropped_nan": rep["dropped_nan"],
        "bad_years": sorted({int(y) for y in flagged.index.year}),
        "n_bad_bars": int(len(flagged)),
    }


def main() -> dict:
    per_pair = []
    for pair in config.PAIRS_CANDIDATE:
        try:
            per_pair.append(bad_bars(pair))
        except FileNotFoundError:
            continue

    contaminated = {r["pair"]: r for r in per_pair if r["n_bad_bars"]}
    verdicts = {}
    for bid, spec in BATTERIES.items():
        hits = []
        for pair, r in contaminated.items():
            if pair not in spec["pairs"]:
                continue
            overlap = sorted(set(r["bad_years"]) & set(spec["years"]))
            if overlap:
                hits.append({"pair": pair, "years": overlap,
                             "n_bad_bars": r["n_bad_bars"]})
        verdicts[bid] = {
            "clean": not hits,
            "contaminated_by": hits,
            "affected_years": sorted({y for h in hits for y in h["years"]}),
        }

    report = {
        "global_band_fix_at": GLOBAL_BAND_FIX_AT,
        "note": "read-only; deletes and recomputes are the user's call",
        "per_pair": per_pair,
        "battery_verdicts": verdicts,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    print(f"{'pair':<8} {'nonpos':>7} {'out_of_band':>12} {'off_scale':>10}  years")
    for r in rep["per_pair"]:
        if r["n_bad_bars"]:
            print(f"{r['pair']:<8} {r['dropped_nonpositive']:>7} "
                  f"{r['dropped_out_of_band']:>12} {r['dropped_off_scale']:>10}  "
                  f"{r['bad_years']}")
    print()
    for bid, v in rep["battery_verdicts"].items():
        state = "CLEAN" if v["clean"] else f"CONTAMINATED in {v['affected_years']}"
        print(f"{bid}: {state}")
    print(f"\nwrote {OUT_PATH}")
