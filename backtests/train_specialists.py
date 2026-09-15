"""Train ML specialists offline (PROJECT_PLAN.md 29.2).

Usage:
  python -m backtests.train_specialists            (all data/raw/*_1d.csv)
  python -m backtests.train_specialists --pairs usdjpy eurusd

Trains on real downloaded history (run backtests.download_data first):
- trend_rf:   RandomForest on directional features, 5-bar forward horizon
- range_svm:  RBF SVM (probability-calibrated), same features

Time-ordered split: first 70% train, last 30% out-of-sample - never
shuffled, so validation is always on data the model has not seen and that
comes strictly after training. Models are saved to data/artifacts/models/
with a metadata sidecar; engines.ml_specialist refuses to load a model whose
features_version doesn't match the current feature builder.

LSTM trend specialist is deferred until a torch pipeline for the RTX 3050
exists; the RandomForest fills the TREND slot meanwhile.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.paths import data_dir
from backtests.data_loader import load_ohlc
from engines.ml_specialist import (
    FEATURES_VERSION,
    RANGE_MODEL,
    TREND_MODEL,
    build_dataset,
    models_dir,
)

HORIZON = 5
TRAIN_FRACTION = 0.7
SVM_MAX_TRAIN = 6000  # RBF SVC is O(n^2); subsample evenly, keep time order


def _market_returns_for(name: str, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Per-pair log-return series indexed by timestamp, for the cross-pair
    correlation feature: the mean return of every *other* pair on each date."""
    returns_by_pair = {
        pair: pd.Series(np.log(df["close"].to_numpy()), index=df["timestamp"]).diff()
        for pair, df in dfs.items()
    }
    others = pd.DataFrame({p: r for p, r in returns_by_pair.items() if p != name})
    return others.mean(axis=1, skipna=True)


def collect_dataset(pairs: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = data_dir() / "raw"
    files = sorted(raw.glob("*_1d.csv"))
    if pairs:
        wanted = {p.lower() for p in pairs}
        files = [f for f in files if f.stem.removesuffix("_1d") in wanted]
    if not files:
        raise SystemExit(f"no *_1d.csv files in {raw}; run backtests.download_data first")

    # Loaded once for all pairs so each pair's cross-pair correlation feature
    # is computed against the OTHER pairs only (no self-reference).
    dfs = {f.stem.removesuffix("_1d"): load_ohlc(f) for f in files}

    X_train, y_train, X_test, y_test = [], [], [], []
    used = []
    for f in files:
        name = f.stem.removesuffix("_1d")
        df = dfs[name]
        closes = df["close"].to_numpy()
        market_factor = _market_returns_for(name, dfs)
        market_returns = (
            market_factor.reindex(pd.DatetimeIndex(df["timestamp"])).fillna(0.0).to_numpy()
        )
        X, y = build_dataset(closes, market_returns=market_returns, horizon=HORIZON)
        if len(X) == 0:
            continue
        # split each pair on its own timeline so test data is late for every pair
        cut = int(len(X) * TRAIN_FRACTION)
        X_train.append(X[:cut])
        y_train.append(y[:cut])
        X_test.append(X[cut:])
        y_test.append(y[cut:])
        used.append(f.stem)
    return (
        (np.vstack(X_train), np.concatenate(y_train)),
        (np.vstack(X_test), np.concatenate(y_test)),
        used,
    )


def _save(model, name: str, metrics: dict, sources: list[str]) -> Path:
    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    joblib.dump(model, path)
    meta = {
        "features_version": FEATURES_VERSION,
        "horizon": HORIZON,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        **metrics,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*")
    args = parser.parse_args()

    (X_tr, y_tr), (X_te, y_te), sources = collect_dataset(args.pairs)
    baseline = float(max(y_te.mean(), 1 - y_te.mean()))
    print(f"dataset: {len(X_tr)} train / {len(X_te)} test rows from {sources}")
    print(f"majority-class baseline: {baseline:.4f}")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=7, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_acc = float(accuracy_score(y_te, rf.predict(X_te)))
    _save(rf, TREND_MODEL, {"oos_accuracy": rf_acc, "baseline_accuracy": baseline}, sources)
    print(f"trend_rf   oos accuracy: {rf_acc:.4f} ({'USABLE' if rf_acc > baseline else 'below baseline - will not load'})")

    idx = np.linspace(0, len(X_tr) - 1, min(SVM_MAX_TRAIN, len(X_tr))).astype(int)
    svm = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=7), ensemble=False),
    )
    svm.fit(X_tr[idx], y_tr[idx])
    svm_acc = float(accuracy_score(y_te, svm.predict(X_te)))
    _save(svm, RANGE_MODEL, {"oos_accuracy": svm_acc, "baseline_accuracy": baseline}, sources)
    print(f"range_svm  oos accuracy: {svm_acc:.4f} ({'USABLE' if svm_acc > baseline else 'below baseline - will not load'})")


if __name__ == "__main__":
    main()
