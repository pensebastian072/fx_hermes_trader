# backtests/

Historical data, replay engine, validation, and offline model training.

- `data_loader.py` - `load_ohlc()` (CSV/parquet) and `synthetic_closes()` (seeded regime-segment generator for tests/stress scenarios)
- `download_data.py` - `python -m backtests.download_data` - daily FX history from Yahoo Finance back to 2003, no API key (writes `data/raw/*_1d.csv`, gitignored)
- `runner.py` - `BacktestRunner`: replays bars through the HMM regime detector + regime-gated ensemble + risk engine + simulated position, with spread/slippage costs (fractions of price)
- `walk_forward.py` - rolling in-sample/out-of-sample splits; re-fits the HMM per split so no OOS segment is scored by a model that saw it
- `monte_carlo.py` - trade-order reshuffling for drawdown distributions + structural macro stress scenarios (RATE_SHOCK / CARRY_UNWIND / VOL_EXPLOSION)
- `train_specialists.py` - `python -m backtests.train_specialists` - trains `trend_rf` (RandomForest) and `range_svm` (calibrated SVM) on the v2 feature set, time-ordered 70/30 split, edge gate vs majority baseline
- `train_lstm.py` - `python -m backtests.train_lstm` - trains the LSTM trend specialist (GPU if available) on 30-bar feature sequences, same edge gate
- `reports/` - JSON output from `save_report()` (gitignored)

No lookahead anywhere in this package: regime probabilities, specialist scores, and feature rows at bar *t* only ever see `closes[:t+1]`.
