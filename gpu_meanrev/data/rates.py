"""Short-rate panel per currency, for the B04 cross-sectional carry battery.

Source: FRED, OECD "3-Month or 90-day Rates and Yields: Interbank Rates",
series family IR3TIB01<CC>M156N, monthly, percent per annum. One family across
all eight currencies on purpose — mixing an interbank rate for one currency
with a policy rate for another manufactures a cross-sectional spread out of the
definition difference rather than out of the market.

Coverage probed live 2026-08-07 before this module was written (not assumed):
all 8 series run past 2026, and all cover 180/180 months of the 2008-2022
in-sample window except USD at 179 — see MAX_FFILL_MONTHS for how that single
hole is handled.

The FRED API key is NOT stored in this repo. It is read from $FRED_API_KEY, or
failing that from copper_brain's .env, which is where it already lives.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from gpu_meanrev import config

API = "https://api.stlouisfed.org/fred/series/observations"

# 8 currencies -> the 28 pairs of the resolved universe are exactly the 8-choose-2
# combinations, so every long/short currency leg is directly quotable.
SERIES = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
}
CURRENCIES = list(SERIES)

# One month of the USD series is missing inside the in-sample window. Carrying a
# 3-month rate forward one month is a far smaller distortion than dropping a
# whole rebalance date for every currency, but the limit is 1: two consecutive
# missing months means the series is broken and should fail loudly, not be
# smoothed into looking healthy.
MAX_FFILL_MONTHS = 1

# Publication lag. OECD monthly aggregates are not knowable on the last day of
# the month they describe, so the portfolio formed at the end of month m ranks on
# the rate for month m-1. This is a lookahead guard, NOT a tunable parameter —
# it is deliberately excluded from the B04 grid.
PUBLICATION_LAG_MONTHS = 1

CACHE = config.DATA / "raw" / "fred_short_rates.parquet"
CACHE_REPORT = config.DATA_REPORTS / "fred_short_rates_report.json"

_COPPER_ENV = Path(r"C:\Users\<your-user>\copper_brain\.env")


def _api_key() -> str:
    k = os.environ.get("FRED_API_KEY")
    if k:
        return k.strip()
    if _COPPER_ENV.exists():
        m = re.search(r"^FRED_API_KEY=(.+)$", _COPPER_ENV.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1).strip()
    raise RuntimeError(
        "no FRED API key — set $FRED_API_KEY (it is not stored in this repo)"
    )


def _redact(text: str, key: str) -> str:
    """FRED takes the API key as a URL QUERY PARAM, so it is embedded in every
    error message requests/urllib3 produce — `raise_for_status` on any 4xx, and
    any TLS failure behind this box's intercepting proxy, both print the full
    URL including `api_key=...` to the console. Every exception this module
    raises goes through here first."""
    return text.replace(key, "***") if key else text


def _fetch_one(series_id: str, key: str) -> pd.Series:
    import requests

    try:
        resp = requests.get(
            API,
            params={"series_id": series_id, "api_key": key, "file_type": "json"},
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(_redact(
            f"FRED fetch failed for {series_id}: {e!r}. If this is a certificate "
            "error it is the box's TLS-intercepting proxy — see copper_brain/"
            "copper_brain/certs.py::ensure_ca_bundle for the CA-bundle fix.",
            key,
        )) from None  # `from None`: the original traceback also carries the key
    obs = [o for o in resp.json().get("observations", []) if o["value"] != "."]
    idx = pd.to_datetime([o["date"] for o in obs])
    return pd.Series([float(o["value"]) for o in obs], index=idx, name=series_id)


def fetch_rates(refresh: bool = False) -> pd.DataFrame:
    """(months, 8) percent-per-annum short rates, monthly, cached to parquet."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)

    key = _api_key()
    cols = {ccy: _fetch_one(sid, key) for ccy, sid in SERIES.items()}
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "month"

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    CACHE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_REPORT.write_text(json.dumps({
        "series": SERIES,
        "fetched_rows": len(df),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "missing_per_currency": {c: int(df[c].isna().sum()) for c in df.columns},
    }, indent=2), encoding="utf-8")
    return df


def signal_rates(start: str | None = None, end: str | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Rates as they may be USED: gap-checked, then shifted by the publication lag.

    After this call, row t holds the rate that was already published as of month
    t, so ranking on row t to trade during month t+1 cannot see the future.

    start/end clip BEFORE the gap check, and callers should pass the price
    panel's real range. The check is about holes inside the window being
    traded; the ragged publication edge past the end of the price archive
    (measured 2026-08-07: EUR and GBP unpublished for 2026-02..2026-06) is not a
    data defect and must not fail a run that never touches those months.
    """
    df = fetch_rates(refresh=refresh)
    df = df.resample("MS").last()
    # Clip with a PAD at the start, wide enough to cover the publication lag and
    # the ffill limit. Clipping exactly at `start` and only then shifting would
    # leave the first in-window month NaN, silently costing the battery its first
    # rebalance — the position would be flat for a month for no modelled reason.
    pad = PUBLICATION_LAG_MONTHS + MAX_FFILL_MONTHS + 1
    if start is not None:
        df = df[df.index >= pd.Timestamp(start) - pd.DateOffset(months=pad)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]

    runs = _max_nan_run(df)
    bad = {c: n for c, n in runs.items() if n > MAX_FFILL_MONTHS}
    if bad:
        raise RuntimeError(
            f"short-rate series have gaps longer than {MAX_FFILL_MONTHS} month(s): "
            f"{bad} — fix the source, do not forward-fill across them"
        )
    df = df.ffill(limit=MAX_FFILL_MONTHS)
    return df.shift(PUBLICATION_LAG_MONTHS)


def _max_nan_run(df: pd.DataFrame) -> dict[str, int]:
    """Longest run of consecutive NaN per column, ignoring the leading run before
    a series starts (a series that simply begins later is not a gap)."""
    out = {}
    for c in df.columns:
        s = df[c]
        first = s.first_valid_index()
        s = s if first is None else s.loc[first:]
        isna = s.isna()
        best = run = 0
        for v in isna:
            run = run + 1 if v else 0
            best = max(best, run)
        out[c] = best
    return out


def main() -> None:  # pragma: no cover - operator entry point
    df = fetch_rates(refresh=True)
    print(f"{len(df)} months  {df.index.min().date()} -> {df.index.max().date()}")
    print(df.loc["2015-01-01":"2015-03-01"].round(3))
    print("\nmissing:", {c: int(df[c].isna().sum()) for c in df.columns})
    print("max NaN run:", _max_nan_run(df.resample("MS").last()))
    print(f"cache: {CACHE}")


if __name__ == "__main__":  # pragma: no cover
    main()
