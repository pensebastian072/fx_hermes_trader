"""Batch regime update: fit HMM on bar data, write snapshot + posterior timeline.

Usage:
  python -m engines.regime_detection.run_regime --csv data/raw/usdjpy_1h.csv --symbol USDJPY
  python -m engines.regime_detection.run_regime --demo            (synthetic data)

The webhook pipeline reads data/artifacts/regime_snapshot.json on every alert;
run this on a schedule (or after each data pull) to keep it fresh.
"""

import argparse
from pathlib import Path

from app.paths import data_dir
from backtests.data_loader import load_ohlc, synthetic_closes
from engines.regime_detection.hmm_regime import (
    HMMRegimeDetector,
    save_posterior_timeline,
    save_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="OHLC csv/parquet file")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--demo", action="store_true", help="use synthetic data")
    args = parser.parse_args()

    if args.demo:
        closes = synthetic_closes([("calm", 200), ("trend_up", 150), ("crisis", 80)])
    elif args.csv:
        closes = load_ohlc(args.csv)["close"].to_numpy()
    else:
        parser.error("provide --csv or --demo")

    detector = HMMRegimeDetector().fit(closes)
    snapshot = detector.snapshot(closes, args.symbol)
    artifacts = data_dir() / "artifacts"
    save_snapshot(snapshot, artifacts)
    save_posterior_timeline(detector, closes, artifacts)
    print(f"Regime snapshot for {args.symbol}: {snapshot.probabilities}")
    print(f"Dominant: {snapshot.dominant_regime}  P(crisis)={snapshot.crisis_probability:.2f}")


if __name__ == "__main__":
    main()
