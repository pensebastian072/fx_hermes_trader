Propose one backtest experiment based on recent logs.

Required structure:
1. Hypothesis
2. Data used
3. Expected improvement
4. Risk of overfitting
5. Backtest requirement
6. Walk-forward requirement
7. Promotion criteria
8. Reason to reject the change

Constraints:
- Output a proposed config into configs/proposed/, never configs/active/.
- No martingale, grid, averaging down, stop removal, or blackout removal.
- If evidence is weak, answer: no change.
