# Results log

Every gate verdict, honest FAILs included — that is the system working, not a bug.

| date | battery | n | PF | Sharpe | DSR ratio | PBO | n_trials | verdict | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-04 | B01_zscore_reversion_SMOKE | 934 | 1.122 | 0.029 | -0.110 | 0.226 | 4 | FAIL | clustering day_ratio=14.594 |
| 2026-08-04 | B01_scaling_probe | 8 | 95.095 | 1.662 | 0.308 | - | 160 | FAIL | clustering day_ratio=8.0 |
| 2026-08-04 | B01_zscore_reversion_PILOT | 1644 | 1.140 | 0.047 | -0.801 | 0.008 | 160 | FAIL | clustering day_ratio=18.267 |
| 2026-08-06 | B02_mtf_confluence | 17368 | 1.029 | 0.010 | -1.076 | 0.016 | 72 | FAIL | MTF filter no effect; hypothesis rejected |
| 2026-08-06 | B02_mtf_confluence (daily-agg) | 2314 | - | 0.023 | -1.288 | - | 72 | FAIL | same combo, day-aggregated PnL: t=1.12, not significant before deflation |
| 2026-08-07 | B01_zscore_reversion (2008-2022, CLEAN) | 120930 | 0.972 | -0.009 | -5.618 | 0.040 | 160 | FAIL | combo 124; supersedes every earlier B01 row |
| 2026-08-07 | B01_zscore_reversion (daily-agg) | 4549 | 0.909 | -0.029 | -4.679 | 0.000 | 160 | FAIL | t=-1.985, i.e. significantly NEGATIVE net of cost |

## Provenance notes

**The three 2026-08-04 B01 rows are partly invalid (found 2026-08-06).** This does NOT
apply to the 2026-08-07 rows, which are the clean 2008-2022 rerun. Every B01 trade file
that existed on 2026-08-06
predates the `sanitize_closes` global-band fix (f877bec, 2026-08-05 21:37), and the
28-pair universe carries 8,419 wrong-scale bars, all in **2000–2007** (8,413 of them
AUDJPY 2005). B01's regime-table rows for those years are data artifacts — the 2005
PF = 111.6 is the AUDJPY 0.67-vs-85-median defect and must never be quoted. Rows for
2008 and 2015–2022, and the PILOT row above, are clean. The paused full run must have
its 2000–2007 blocks **recomputed, not resumed**.

**B02 is clean and verified so**: zero non-positive or out-of-band bars in the 7 USD
majors over 2015–2022, checked directly.

**Resolved 2026-08-06**: B01 re-scoped to **2008–2022** rather than recomputed back to
2000 (2008-03-30 is the common-coverage start across all 28 pairs anyway). Pre-fix block
files quarantined, not deleted, under
`journal/data/B01_zscore_reversion_checkpoint/QUARANTINE_pre_fix_2000_2008/`. The
2009–2014 gap recomputed under the fixed loader (done 2026-08-07). **Any B01 row that
predates this note covers a different, partly-contaminated period and is superseded.**

**B03 in-sample run 2026-08-07 — INCONCLUSIVE BY CONSTRUCTION. Holdout NOT touched.**

48 combos, 60-minute bars, 7 USD majors, 2015-2022, 363,196 trades, 5.5 min.

The pre-committed kill criterion passes: combo 39 (lookback 96h, z 1.5, hold 168h)
reaches >= 2x assumed cost in **7 of 8 years**. But the pre-registered rules are
**internally inconsistent**, and the inconsistency was only visible after the run:

| combo | params | years >= 2x cost | day-agg Sharpe | DSR |
|---|---|---|---|---|
| 39 | lb 96h, z 1.5, hold 168h | **7 of 8** | 0.0457 | −0.718 |
| 44 | lb 96h, z 2.5, hold 6h | **1 of 8** | 0.1053 | **+2.029** |

`KILL_CRITERION` authorizes proceeding on combo 39; `selection_rule_for_holdout`
("highest in-sample day-aggregated Sharpe") then selects combo **44** — a configuration
that fails the very screen used to justify the run. The selection rule never said
"among kill-passing combos", and that gap is load-bearing.

**No repair is legitimate now.** Restricting selection to kill-passers, or changing the
metric, would be choosing a rule after seeing which combo it favours. B03 is therefore
recorded as INCONCLUSIVE and its holdout shot is **not spent**. A successor battery must
pre-register ONE coherent rule (screen and selection over the same combo set) before
anything else is looked at.

What the run does say about the hypothesis, split verdict:
* Gross edge **does** scale with holding period — every combo clearing 2x cost is a long
  hold (168h).
* Risk-adjusted return goes the **opposite** way — the best day-aggregated Sharpe is the
  *shortest* hold (6h), the reverse of B03's prediction.

Multiplicity caveat: only 2 of 48 combos clear 6+ years (9 more sit at exactly 5). A max
of 7/8 over 48 trials is weak, which is the inflation built into "at least one
configuration". Combo 44's DSR of +2.029 is the first number on this box above the 1.645
bar — **in-sample, ex-post, on a config that failed the cost screen.** A lead, not a
result, and explicitly not a promotion.

Full diagnostics of what the surviving edge is made of — cost breakeven, hour
concentration, stale-price and volatility confounds tested and rejected, execution
latency, effective sample size — in `journal/B01_B02_POSTMORTEM.md`.

## B01 final, 2008–2022 clean (2026-08-07)

Rerun complete: 2009–2014 computed under the fixed loader in 7.03 h, 2008 + 2015–2022
reused (verified clean), pre-2008 dropped. 107.2M trade rows, 15 years, 28 pairs.

Ex-post winner is now **combo 124** (lookback 240, entry_z 3.0, max_hold) — it was combo
152 on the contaminated log, so the bad bars had been steering the selection itself.
**PF > 1 in only 6 of 15 years** (range 0.850–1.040); every single year has a negative
DSR ratio. The 2005 PF = 111.6 artifact is gone and the clean table is flat and boring,
which is the honest picture.

Net of cost the strategy is **significantly negative**, not merely unprofitable:
day-aggregated t = **−1.985**, sign-permutation p = 0.9995, bootstrap Sharpe CI
[−0.015, −0.003] entirely below zero.

### Corrected 2026-08-07 — an earlier version of this section was wrong

It claimed "B01's gross edge is 0.422 bps ... the same ~0.41 bps as B02 ... independently
reproduced". **That was an error and the conclusion does not survive it.** The 0.422 came
from adding the MAJORS round-trip cost (1.0 bp) back uniformly, but B01 charges tiered
costs (`config.COST_PER_SIDE`: majors 0.5 bp/side, crosses 1.5 bp/side) and **74.8% of
combo 124's 120,930 trades are crosses**, which were charged 3.0 bp round-trip. Mean
charged cost is 2.496 bp, not 1.0.

Adding back the cost that was actually charged:

| | n | net bps | gross bps |
|---|---|---|---|
| crosses | 90,475 | −0.905 | — |
| majors | 30,455 | **+0.394** | 1.394 |
| blended | 120,930 | −0.581 | **1.919** |

So B01's gross edge is **1.919 bps blended / 1.394 bps majors-only**, against B02's
0.406 bps. Even on the comparable majors universe they differ by 3.4x. The two are also
different estimands — B02's 0.406 is the pooled control arm across 24 filter-off combos,
B01's is one ex-post-selected combo out of 160.

**There is no "two batteries, same 0.41 bps" finding.** B02's cost-breakeven result
stands on B02 alone, on its own 7-major universe, which is also what B03 was registered
against — that part is unaffected.

**B01's verdict is unchanged**: net of the costs actually charged it is significantly
negative (t = −1.985), and the gate still FAILs.

One thing genuinely worth a look later, stated as an observation and not a result: the
**majors subset is net positive (+0.394 bps/trade)** while the crosses carry the loss.
That is a post-hoc slice of an ex-post combo — it proves nothing and must not be
harvested. If it is worth testing it needs its own pre-registration.
| 2026-08-07 | B03_horizon | 1162 | 1.134 | 0.046 | -0.718 | 0.044 | 48 | FAIL | IN-SAMPLE design pass, NOT evidence; PROCEED to single holdout evaluation |
| 2026-08-07 | B04_xs_carry | 3905 | 1.036 | 0.011 | -2.193 | 0.091 | 288 | FAIL | kill=FAIL best=k2_quarterly_inverse_vol breakeven=78.74x holdout unspent |
