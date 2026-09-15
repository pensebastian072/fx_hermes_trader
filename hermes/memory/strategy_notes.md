# Strategy Notes

Append-only notebook for Hermes observations. Empty at Milestone 1.

### 2026-06-12 (supervisor)
- No alerts received today - webhook source idle or not connected (Cloudflare tunnel for TradingView is still an open item).
- EURUSD: recent-history backtest fails promotion gates: profit_factor 0.00 < required 1.20; max_drawdown 0.314 > allowed 0.080
- GBPUSD: recent-history backtest fails promotion gates: profit_factor 0.00 < required 1.20; max_drawdown 0.323 > allowed 0.080
- AUDUSD: recent-history backtest fails promotion gates: profit_factor 0.00 < required 1.20; max_drawdown 0.727 > allowed 0.080
- USDCAD: recent-history backtest fails promotion gates: profit_factor 0.00 < required 1.20; max_drawdown 0.340 > allowed 0.080
- ML specialists not in use (trend_rf.joblib, range_svm.joblib): out-of-sample accuracy does not beat the majority-class baseline. Suggest richer features (carry/rate differentials, session structure, cross-pair correlation) before retraining; rule-based specialists remain in effect.
- LLM review written to reports/hermes/2026-06-12.md

### 2026-06-12 (supervisor)
- No alerts received today - webhook source idle or not connected (Cloudflare tunnel for TradingView is still an open item).
- AUDUSD: recent-history backtest fails promotion gates: profit_factor 1.04 < required 1.20
- USDCAD: recent-history backtest fails promotion gates: profit_factor 0.88 < required 1.20
- ML specialists not in use (trend_rf.joblib, range_svm.joblib): out-of-sample accuracy does not beat the majority-class baseline. Suggest richer features (carry/rate differentials, session structure, cross-pair correlation) before retraining; rule-based specialists remain in effect.
