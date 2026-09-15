"""Walk-forward validation (PROJECT_PLAN.md Phase 7).

Rolling in-sample/out-of-sample splits. The runner re-fits its HMM on each
split's warmup, so every OOS segment is scored by a model that never saw it.
Gate: at least `min_profitable_splits` OOS-profitable splits (autonomy.yaml
promotion_gates.min_walk_forward_splits).
"""

from dataclasses import dataclass, field

import pandas as pd

from backtests.runner import BacktestConfig, BacktestRunner


@dataclass
class WalkForwardResult:
    split_metrics: list[dict] = field(default_factory=list)
    profitable_splits: int = 0
    total_splits: int = 0
    passed: bool = False


def make_splits(n_bars: int, n_splits: int, oos_fraction: float = 0.3) -> list[tuple[int, int]]:
    """Return (start, end) index ranges; consecutive OOS segments never overlap."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    segment = n_bars // n_splits
    splits = []
    for k in range(n_splits):
        start = k * segment
        end = n_bars if k == n_splits - 1 else (k + 1) * segment
        splits.append((start, end))
    return splits


def run_walk_forward(
    df: pd.DataFrame,
    n_splits: int = 4,
    min_profitable_splits: int = 3,
    config: BacktestConfig | None = None,
) -> WalkForwardResult:
    result = WalkForwardResult(total_splits=n_splits)
    for start, end in make_splits(len(df), n_splits):
        segment = df.iloc[start:end].reset_index(drop=True)
        runner = BacktestRunner(config=config)  # fresh detector per split
        bt = runner.run(segment)
        metrics = dict(bt.metrics)
        metrics["split"] = (start, end)
        result.split_metrics.append(metrics)
        if metrics["net_pnl"] > 0:
            result.profitable_splits += 1
    result.passed = result.profitable_splits >= min_profitable_splits
    return result
