"""HMM regime detector (PROJECT_PLAN.md 29.1).

Gaussian HMM over [log return, rolling realized volatility]. States are
labeled after fitting from their emission statistics:

- highest-volatility state  -> HIGH_VOL_CRISIS
- strongest |drift| of rest -> TREND
- remaining state           -> LOW_VOL_RANGE

Probabilities exposed to the rest of the system are FILTERED (forward pass
only, P(state_t | observations up to t)). The smoothed posterior from
forward-backward uses future bars and would be lookahead bias.

Deterministic volatility guard: an HMM fitted on calm history cannot
represent a crisis it has never seen - all of its states stay calm and the
filter never leaves them. So crisis probability is additionally floored when
current realized volatility reaches multiples of fit-time volatility
(ratio 2x -> floor starts, 4x -> floor 1.0). The statistical model proposes;
the deterministic guard disposes.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from app.schemas.regime import RegimeSnapshot

CRISIS = "HIGH_VOL_CRISIS"
TREND = "TREND"
RANGE = "LOW_VOL_RANGE"

VOL_WINDOW = 20
VOL_GUARD_LOW_RATIO = 2.0  # current vol / fit vol below this: no floor
VOL_GUARD_HIGH_RATIO = 4.0  # at/above this: crisis probability floored to 1.0


def build_features(closes: np.ndarray) -> np.ndarray:
    """[log_return, rolling realized vol] per bar; first VOL_WINDOW bars dropped."""
    closes = np.asarray(closes, dtype=float)
    if closes.ndim != 1 or len(closes) < VOL_WINDOW + 2:
        raise ValueError(f"need at least {VOL_WINDOW + 2} closes")
    if np.any(closes <= 0):
        raise ValueError("closes must be positive")
    returns = np.diff(np.log(closes))
    vol = np.array(
        [returns[max(0, i - VOL_WINDOW + 1) : i + 1].std() for i in range(len(returns))]
    )
    feats = np.column_stack([returns, vol])
    return feats[VOL_WINDOW:]


class HMMRegimeDetector:
    def __init__(self, n_states: int = 3, seed: int = 7, n_iter: int = 100):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=seed,
        )
        self.state_labels: dict[int, str] = {}
        self.fit_vol_typical = 0.0
        self.fitted = False

    def fit(self, closes: np.ndarray) -> "HMMRegimeDetector":
        feats = build_features(closes)
        self.model.fit(feats)
        self._label_states()
        self.fit_vol_typical = float(np.median(feats[:, 1]))
        self.fitted = True
        return self

    def _label_states(self) -> None:
        means = self.model.means_  # [n_states, 2]: col 0 = return drift, col 1 = vol
        order_by_vol = np.argsort(means[:, 1])
        crisis_state = int(order_by_vol[-1])
        rest = [int(s) for s in order_by_vol[:-1]]
        if len(rest) >= 2:
            trend_state = max(rest, key=lambda s: abs(means[s, 0]))
            self.state_labels = {crisis_state: CRISIS, trend_state: TREND}
            for s in rest:
                if s not in self.state_labels:
                    self.state_labels[s] = RANGE
        elif len(rest) == 1:
            self.state_labels = {crisis_state: CRISIS, rest[0]: RANGE}
        else:
            self.state_labels = {crisis_state: CRISIS}

    def filtered_probabilities(self, closes: np.ndarray) -> np.ndarray:
        """Forward-filtered P(state_t | obs[0..t]) for every bar. No future data."""
        if not self.fitted:
            raise RuntimeError("detector not fitted")
        feats = build_features(closes)
        log_likelihood = self.model._compute_log_likelihood(feats)
        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)

        n, k = log_likelihood.shape
        log_alpha = np.zeros((n, k))
        log_alpha[0] = log_startprob + log_likelihood[0]
        for t in range(1, n):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
                + log_likelihood[t]
            )
        # normalize each row -> filtered distribution
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def regime_probabilities(self, closes: np.ndarray) -> dict[str, float]:
        """Latest filtered probabilities aggregated by regime label,
        with the deterministic volatility-guard floor applied."""
        probs = self.filtered_probabilities(closes)[-1]
        out = {CRISIS: 0.0, TREND: 0.0, RANGE: 0.0}
        for state, p in enumerate(probs):
            out[self.state_labels.get(state, RANGE)] += float(p)

        current_vol = float(build_features(closes)[-1, 1])
        floor = self._crisis_floor(current_vol)
        if floor > out[CRISIS]:
            remaining = 1.0 - out[CRISIS]
            scale = (1.0 - floor) / remaining if remaining > 0 else 0.0
            for label in (TREND, RANGE):
                out[label] *= scale
            out[CRISIS] = floor
        return out

    def _crisis_floor(self, current_vol: float) -> float:
        ratio = current_vol / max(self.fit_vol_typical, 1e-12)
        if ratio <= VOL_GUARD_LOW_RATIO:
            return 0.0
        if ratio >= VOL_GUARD_HIGH_RATIO:
            return 1.0
        return (ratio - VOL_GUARD_LOW_RATIO) / (VOL_GUARD_HIGH_RATIO - VOL_GUARD_LOW_RATIO)

    def snapshot(self, closes: np.ndarray, symbol: str) -> RegimeSnapshot:
        probs = self.regime_probabilities(closes)
        dominant = max(probs, key=probs.get)
        return RegimeSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            probabilities=probs,
            dominant_regime=dominant,
            crisis_probability=probs.get(CRISIS, 0.0),
            n_bars=len(closes),
        )


def save_snapshot(snapshot: RegimeSnapshot, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "regime_snapshot.json"
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_snapshot(artifacts_dir: Path) -> RegimeSnapshot | None:
    path = artifacts_dir / "regime_snapshot.json"
    if not path.exists():
        return None
    return RegimeSnapshot(**json.loads(path.read_text(encoding="utf-8")))


def save_posterior_timeline(
    detector: HMMRegimeDetector, closes: np.ndarray, artifacts_dir: Path
) -> Path:
    """Write per-bar filtered probabilities to regime_posteriors.csv."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    probs = detector.filtered_probabilities(closes)
    labels = [detector.state_labels.get(s, RANGE) for s in range(probs.shape[1])]
    path = artifacts_dir / "regime_posteriors.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("bar," + ",".join(labels) + "\n")
        for i, row in enumerate(probs):
            f.write(f"{i}," + ",".join(f"{p:.6f}" for p in row) + "\n")
    return path
