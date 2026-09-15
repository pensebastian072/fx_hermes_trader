"""Regime-gated ensemble scoring (PROJECT_PLAN.md 29.2).

FinalScore_t = sum_k P(Regime_k | X_t) * Model_k(X_t)

Specialists are deterministic rule-based models in this milestone. Trained
sklearn specialists can be dropped into data/artifacts/models/ later and
loaded via MLSpecialist; behavior must stay identical in shape (score in
[-100, 100]).
"""

from dataclasses import dataclass, field

import numpy as np

from engines.features import ema, zscore
from engines.regime_detection.hmm_regime import CRISIS, RANGE, TREND


def _clip(score: float) -> float:
    return float(np.clip(score, -100.0, 100.0))


class TrendSpecialist:
    """EMA 21/55 separation + momentum. Positive when trending up."""

    name = "trend_following"

    def score(self, closes: np.ndarray, market_returns: np.ndarray | None = None) -> float:
        closes = np.asarray(closes, dtype=float)
        if len(closes) < 55:
            return 0.0
        fast = ema(closes.tolist(), 21)
        slow = ema(closes.tolist(), 55)
        separation = (fast - slow) / closes[-1]  # fraction of price
        momentum = (closes[-1] - closes[-21]) / closes[-21]
        # 0.5% EMA separation or 1% momentum over 21 bars each saturate at ~100
        return _clip(separation / 0.005 * 60 + momentum / 0.01 * 40)


class MeanReversionSpecialist:
    """Fade z-score extremes over a rolling window."""

    name = "mean_reversion"

    def __init__(self, window: int = 50):
        self.window = window

    def score(self, closes: np.ndarray, market_returns: np.ndarray | None = None) -> float:
        closes = np.asarray(closes, dtype=float)
        if len(closes) < self.window:
            return 0.0
        z = zscore(closes[-self.window :].tolist())
        # z = +2 (stretched up) -> strong short; saturate at |z| = 2.5
        return _clip(-z / 2.5 * 100)


class CapitalPreservationSpecialist:
    """Crisis regime: claim no directional edge."""

    name = "capital_preservation"

    def score(self, closes: np.ndarray, market_returns: np.ndarray | None = None) -> float:
        return 0.0


@dataclass
class EnsembleScore:
    final_score: float
    regime_probabilities: dict[str, float]
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class RegimeGatedEnsemble:
    def __init__(self, specialists: dict[str, object] | None = None):
        self.specialists = specialists or {
            TREND: TrendSpecialist(),
            RANGE: MeanReversionSpecialist(),
            CRISIS: CapitalPreservationSpecialist(),
        }

    def score(
        self,
        closes: np.ndarray,
        regime_probabilities: dict[str, float],
        market_returns: np.ndarray | None = None,
    ) -> EnsembleScore:
        components: dict[str, float] = {}
        final = 0.0
        for regime, specialist in self.specialists.items():
            p = float(regime_probabilities.get(regime, 0.0))
            s = float(specialist.score(closes, market_returns))
            components[regime] = s
            final += p * s
        dominant = max(regime_probabilities, key=regime_probabilities.get, default="?")
        reason = (
            f"Regime-gated ensemble: dominant {dominant} "
            f"(P={regime_probabilities.get(dominant, 0):.2f}); "
            + ", ".join(f"{r}={s:+.0f}" for r, s in components.items())
        )
        return EnsembleScore(
            final_score=_clip(final),
            regime_probabilities=dict(regime_probabilities),
            components=components,
            reason=reason,
        )


# TODO(later): MLSpecialist wrapping a trained sklearn classifier
# (RandomForest/SVM for range boundaries, LSTM for trend once a GPU training
# pipeline exists). Train offline in backtests, save with joblib to
# data/artifacts/models/, load read-only here. Rule-based specialists above
# remain the fallback so the system never depends on an untrained model.
