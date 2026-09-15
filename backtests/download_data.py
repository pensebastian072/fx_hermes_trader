"""Download real FX history from Yahoo Finance into data/raw/.

Usage:
  python -m backtests.download_data                    (all watchlist pairs, daily)
  python -m backtests.download_data --pairs USDJPY     (subset)
  python -m backtests.download_data --interval 1h      (intraday; Yahoo caps ~730d)

Daily candles reach back to ~2003 on Yahoo, which covers the 2008 crisis,
curve inversions, and JPY carry unwinds required by PROJECT_PLAN.md 29.4.
Output CSVs match backtests.data_loader.REQUIRED_COLUMNS so load_ohlc() can
read them directly. No API keys: Yahoo's public endpoints only.
"""

import argparse
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.paths import data_dir
from app.services.config_service import load_config

DAILY_START = "2003-01-01"
INTRADAY_MAX_DAYS = 729  # Yahoo rejects 1h requests beyond ~730 days


def _trust_windows_certs() -> None:
    """This machine has a TLS-intercepting monitor; trust the OS cert store.

    truststore covers stdlib ssl, but yfinance fetches via curl_cffi which
    only reads a CA bundle file - so export the Windows ROOT/CA stores
    (interceptor cert included) to a PEM and point CURL_CA_BUNDLE at it.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    import os

    import certifi

    bundle = data_dir() / "artifacts" / "windows_ca_bundle.pem"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    pem_parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    for store in ("ROOT", "CA"):
        for der, enc_type, _trust in ssl.enum_certificates(store):
            if enc_type == "x509_asn":
                pem_parts.append(ssl.DER_cert_to_PEM_cert(der))
    bundle.write_text("\n".join(pem_parts), encoding="utf-8")
    os.environ["CURL_CA_BUNDLE"] = str(bundle)
    os.environ["SSL_CERT_FILE"] = str(bundle)


def yahoo_ticker(pair: str) -> str:
    return f"{pair.upper()}=X"


def download_pair(pair: str, interval: str = "1d", raw_dir: Path | None = None) -> Path:
    import yfinance as yf

    if interval == "1d":
        start = DAILY_START
    else:
        start = (datetime.now(timezone.utc) - timedelta(days=INTRADAY_MAX_DAYS)).strftime(
            "%Y-%m-%d"
        )

    df = yf.download(
        yahoo_ticker(pair),
        start=start,
        interval=interval,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for {pair} ({yahoo_ticker(pair)})")

    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], utc=True),
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
        }
    ).dropna()
    out = out.sort_values("timestamp").reset_index(drop=True)

    raw_dir = raw_dir or (data_dir() / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{pair.lower()}_{interval}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", help="default: watchlist.yaml pairs")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h"])
    args = parser.parse_args()

    _trust_windows_certs()
    pairs = args.pairs or load_config("watchlist").get("pairs", [])
    for pair in pairs:
        try:
            path = download_pair(pair, interval=args.interval)
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            print(f"{pair}: {n} bars -> {path}")
        except Exception as exc:  # keep going; one bad ticker shouldn't kill the batch
            print(f"{pair}: FAILED ({exc})")


if __name__ == "__main__":
    main()
