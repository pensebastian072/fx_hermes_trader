"""Train the LSTM trend specialist offline (PROJECT_PLAN.md 29.2 / 30).

Usage:
  python -m backtests.train_lstm                  (all data/raw/*_1d.csv)
  python -m backtests.train_lstm --pairs usdjpy eurusd
  python -m backtests.train_lstm --epochs 30

Sequence model: SEQ_LEN bars of the v2 feature vector (engines.ml_specialist
.build_feature_matrix) -> single-layer LSTM -> P(up over the next HORIZON
bars). Same time-ordered 70/30 split and edge gate as train_specialists.py:
a model that doesn't beat the majority-class baseline out-of-sample is saved
to disk but engines.lstm_specialist.LSTMSpecialist refuses to load it.

Trains on GPU if available (torch.cuda), else CPU.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from app.paths import data_dir
from backtests.data_loader import load_ohlc
from backtests.train_specialists import _market_returns_for
from engines.lstm_specialist import LSTMTrendNet, TREND_LSTM_MODEL
from engines.ml_specialist import FEATURES_VERSION, N_FEATURES, build_lstm_sequences, models_dir

HORIZON = 5
SEQ_LEN = 30
TRAIN_FRACTION = 0.7
HIDDEN_SIZE = 32
NUM_LAYERS = 1
BATCH_SIZE = 256


def collect_sequences(pairs: list[str] | None = None):
    raw = data_dir() / "raw"
    files = sorted(raw.glob("*_1d.csv"))
    if pairs:
        wanted = {p.lower() for p in pairs}
        files = [f for f in files if f.stem.removesuffix("_1d") in wanted]
    if not files:
        raise SystemExit(f"no *_1d.csv files in {raw}; run backtests.download_data first")

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
        X, y = build_lstm_sequences(closes, market_returns=market_returns, horizon=HORIZON, seq_len=SEQ_LEN)
        if len(X) == 0:
            continue
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


def _to_loader(X: np.ndarray, y: np.ndarray, device: torch.device, shuffle: bool):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    (X_tr, y_tr), (X_te, y_te), sources = collect_sequences(args.pairs)
    baseline = float(max(y_te.mean(), 1 - y_te.mean()))
    print(f"dataset: {len(X_tr)} train / {len(X_te)} test sequences from {sources}")
    print(f"majority-class baseline: {baseline:.4f}")

    # Standardize features using train-set statistics only.
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_loader = _to_loader(X_tr, y_tr, device, shuffle=True)

    model = LSTMTrendNet(input_size=N_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss / len(X_tr):.4f}")

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        p_up = torch.sigmoid(model(X_te_t)).cpu().numpy()
    preds = (p_up >= 0.5).astype(int)
    oos_acc = float(accuracy_score(y_te, preds))
    print(f"trend_lstm oos accuracy: {oos_acc:.4f} ({'USABLE' if oos_acc > baseline else 'below baseline - will not load'})")

    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / TREND_LSTM_MODEL
    torch.save(model.state_dict(), path)
    meta = {
        "features_version": FEATURES_VERSION,
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
        "input_size": N_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "oos_accuracy": oos_acc,
        "baseline_accuracy": baseline,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
