# engines/

Scoring, regime detection, and trained specialists.

- `scoring.py` - Milestone-1 stub: maps TradingView signal names to scores + stop/target geometry
- `features.py` - pure technical-indicator functions (EMA, ATR, RSI, MACD histogram, Bollinger %B, z-score, rolling correlation) - no lookahead
- `ensemble.py` - `RegimeGatedEnsemble`: `FinalScore = sum(P(regime) * specialist_score)`; rule-based specialists (`TrendSpecialist`, `MeanReversionSpecialist`, `CapitalPreservationSpecialist`)
- `ml_specialist.py` - `build_feature_row`/`build_dataset`/`build_feature_matrix` (13-feature vector, v2), `MLSpecialist` wrapping RandomForest/SVM models trained by `backtests/train_specialists.py`, `load_ensemble()` factory
- `lstm_specialist.py` - `LSTMSpecialist` + `LSTMTrendNet` (1-layer LSTM over 30-bar feature sequences), trained by `backtests/train_lstm.py`
- `regime_detection/`
  - `hmm_regime.py` - Gaussian HMM over [return, realized vol], labeled TREND/RANGE/CRISIS, forward-filtered (no lookahead), with a deterministic vol-guard floor on crisis probability
  - `run_regime.py` - batch: fit HMM, write snapshot + posterior timeline

**Edge gate:** every trained model (RF/SVM/LSTM) is loaded read-only and only used if its out-of-sample accuracy beats the majority-class baseline (checked via a `.meta.json` sidecar). Otherwise `load_ensemble()` falls back to the next tier, down to the deterministic rule-based specialists - the CRISIS slot is never ML.
