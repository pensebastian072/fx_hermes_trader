# B01 / B02 post-mortem — 2026-08-06

A diagnostics pass over two **already-closed, already-failed** batteries. Nothing here
selects a combo, charges a trial, or touches the 2023+ holdout. B02's verdict (FAIL,
MTF hypothesis rejected) is unchanged by every number below; the point was to find out
what the 0.4 bps gross edge actually consists of, which alternative explanations were
never ruled out, and which of them survive.

Sources: `gpu_meanrev/analysis/b02_diagnostics.py`, `gpu_meanrev/analysis/signal_microstructure.py`.
Scorecards: `B02_mtf_confluence_diagnostics.json`, `B02_signal_microstructure.json`,
`B02_execution_delay.json`.

---

## 1. The cost assumption was never checked against the edge it was charged to

The control arm's gross edge is **0.406 bps** of price per trade. B02 charged 1.0 bp
round-trip (0.5 bp/side, majors). Net therefore turns positive only below:

| round-trip cost | net bps | PF |
|---|---|---|
| 0.0 bp | +0.406 | 1.145 |
| 0.3 bp | +0.106 | 1.036 |
| **0.406 bp (breakeven)** | **0.000** | **1.000** |
| 0.5 bp | −0.094 | 0.968 |
| 1.0 bp (assumed) | −0.594 | 0.813 |
| 2.0 bp | −1.594 | 0.570 |

0.406 bps of price ≈ **0.45 pips round-trip** on a 1.10-handle major. The study's own
cost assumption is **2.5x the entire signal**. No setting in the grid closes that gap —
this is not a marginal FAIL that a better parameter reaches.

## 2. Half the PnL is booked on closes in the hours where the flat cost is least defensible

**Read the causal direction carefully.** B01 and B02 stored the EXIT timestamp only, so
this is a table of when positions *closed*, not when they were opened or when their
spread was paid. The headline combo holds for hours, and half the round-trip cost is
charged at an entry hour this log cannot report. So the table below supports "the PnL is
booked on closes in the thin hours" and **not** "the edge is earned in the thin hours" —
the latter needs the entry timestamps that only runs after 2026-08-06 will have.

Control arm gross bps by exit hour bucket and year:

| bucket | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | share of gross PnL |
|---|---|---|---|---|---|---|---|---|---|
| 21–02 UTC (rollover / Asia) | 0.929 | 0.824 | 0.764 | 0.806 | 0.625 | 0.643 | 0.736 | 0.814 | **47.9%** |
| 13–16 UTC (LDN/NY overlap) | 1.091 | 0.920 | 0.840 | 0.600 | 0.630 | 0.600 | 0.675 | 0.585 | 24.3% |
| 07–12 UTC (London) | 0.754 | 0.637 | 0.321 | 0.412 | 0.378 | 0.191 | 0.260 | 0.284 | 25.9% |
| other | 0.213 | 0.102 | 0.140 | 0.036 | 0.063 | −0.065 | −0.073 | −0.247 | 1.9% |

Shares are of total gross PnL and sum to 100.0%. Two things fall out. Almost half of it
is booked on closes in the 21:00–02:00 UTC window — the daily rollover and the Asia
handover, the thinnest, widest-spread hours of the FX day — and that is the **only
bucket that has not decayed**. The liquid-hour buckets roughly halved over 2015–2022. A
flat 0.5 bp/side across all 24 hours is wrong in the direction that flatters the result,
and the exposure it mis-prices is concentrated, not spread evenly.

Breakeven cost per bucket (2022): rollover/Asia 0.90 pips round-trip, LDN/NY overlap
0.64, London 0.31. Retail rollover spreads on EURUSD are multiples of the first number.

Regenerate: `b02_diagnostics.q2b_hour_bucket_by_year`, written to
`B02_mtf_confluence_diagnostics.json`.

**This repo has no spread data.** The honest statement is the breakeven, not a
simulated net — quoting an hour-varying cost curve we do not have would be inventing
the input that decides the answer.

## 3. Ruled out: the edge is not a stale-price artifact

`build_panel` forward-fills up to 10 minutes. A filled run is flat, which depresses the
rolling std and mechanically inflates the z of the next real tick — the exact shape that
made research_ledger's lead-lag "signal" turn out to be 100% INDA stale-NAV. Neither
battery conditioned on staleness.

Signed forward return at h=960 min, split by the forward-filled fraction of the lookback
window:

Bucket 2 is **cumulative, not disjoint**: `≤2%` contains every zero-fill event
(2022: n=340,859 vs 295,508 at fill=0). Buckets 3 and 4 are disjoint. The conclusion
rests on `fill = 0`, which is a clean isolate.

| year | fill = 0 | ≤2% (incl. 0) | (2,10]% | >10% |
|---|---|---|---|---|
| 2015 | 2.576 | 2.433 | −2.589 | 5.077 |
| 2016 | 1.829 | 2.007 | 2.350 | 5.715 |
| 2017 | 1.075 | 0.789 | 2.881 | 3.450 |
| 2018 | 0.407 | 0.260 | 0.765 | 2.396 |
| 2019 | 0.653 | 0.788 | 0.283 | −8.868 |
| 2020 | −0.691 | −0.532 | −0.019 | −20.635 |
| 2021 | −0.042 | 0.283 | 0.587 | 5.564 |
| 2022 | 1.022 | 0.982 | 0.776 | −2.526 |

The `fill = 0` column tracks the overall edge in every year, so the edge is present in
windows containing no filled bars at all. The `>10%` column swings from +5.7 to −20.6 on
~2k events — it is noise and must not be read as a finding in either direction. Fill
density is low throughout anyway (0.17–1.11% of cells).

**Verdict: not a stale-price artifact.**

## 4. Ruled out: the decay is not falling volatility

| year | 1m realised vol (bps) | ctrl gross bps | **gross / vol** | % cells ffilled |
|---|---|---|---|---|
| 2015 | 2.036 | 0.649 | **0.319** | 0.473 |
| 2016 | 1.999 | 0.533 | **0.266** | 0.524 |
| 2017 | 1.429 | 0.438 | **0.307** | 0.780 |
| 2018 | 1.361 | 0.403 | **0.296** | 0.785 |
| 2019 | 1.184 | 0.362 | **0.306** | 0.173 |
| 2020 | 1.826 | 0.268 | **0.147** | 0.497 |
| 2021 | 1.255 | 0.321 | **0.255** | 1.111 |
| 2022 | 1.974 | 0.269 | **0.136** | 0.701 |

2022 has essentially 2015's volatility and half its vol-normalised edge. corr(gross/vol,
year) = −0.72; corr(gross, %ffill) = −0.23.

**What this establishes: the volatility confound does not explain the decay, and neither
does archive density.** What it does *not* establish is the strength of the decay itself.
n = 8 years with no confidence interval, and gross/vol is essentially flat across
2015–2019 (0.319, 0.266, 0.307, 0.296, 0.306) with the whole decline coming from 2020 and
2022. B02's secondary finding survives its two obvious alternative explanations; it is not
thereby confirmed to the strength the raw correlation suggests.

## 5. Ruled out: zero-latency execution is optimistic but immaterial

Both batteries compute z from a window ending at bar *t* and enter at bar *t*'s own
close. That is a zero-latency fill, never stated as an assumption. Signed forward return
(h=960) with the entry delayed *d* minutes:

| delay | mean bps across 2015–2022 | % of zero-latency |
|---|---|---|
| 0 min | 0.888 | 100% |
| 1 min | 0.853 | 96% |
| 2 min | 0.828 | 93% |
| 5 min | 0.777 | 87% |
| 10 min | 0.726 | 82% |

A minute of latency costs 4% of the signal. Worth fixing in the next battery for
correctness, but it changes no verdict. (2018 and 2021 are the exceptions, losing
~40% and ~30% at 10 min — a thinner edge decays into latency faster.)

## 6. The gate's effective sample was overstated

The gate was handed one observation per trade. Trades overlap in time and run across 7
USD-correlated pairs, so those rows are far from independent, and an inflated *n* makes
the Deflated Sharpe look **better** than it is:

| | n | Sharpe | DSR ratio | t-stat |
|---|---|---|---|---|
| combo 64, per trade | 17,368 | 0.0101 | −1.076 | — |
| combo 64, daily aggregated | 2,314 days | 0.0234 | **−1.288** | **1.12** |

7.5 trades per day for the headline combo. On daily PnL the strategy is not significant
before any deflation is applied at all (t = 1.12). B02 failed on the flattering number,
so the FAIL is robust — but future batteries should hand the gate day-aggregated PnL,
not per-trade rows.

## 7. The pooled decay is smoother than the underlying signal

Control-arm gross bps (24 combos pooled) decays almost monotonically, corr with year
−0.94. The single-combo signal-level edge (lookback 240, z=2, h=960) does not:
2.05, 2.06, 1.10, 0.34, 0.77, **−0.56**, 0.41, 0.94. It is negative in 2020 and flat in
2021 — a regime story (covid trend-following punishing fades), not a slide. The smooth
monotone curve is partly an artifact of averaging 24 combos. Per-fold beats pooled, again.

## 8. Every pair decays — it is not one pair dragging the panel

Control-arm gross bps, pair × year: all 7 majors fall from 2015 to 2022. USDJPY goes
negative (0.297 → −0.121); NZDUSD is the strongest throughout (0.989 → 0.387). There is
no majors/crosses or single-pair split worth harvesting here, consistent with B01's
14/14 per-pair coin flip.

---

## Data-provenance finding: B01's trade log predates the wrong-scale fix

`sanitize_closes`'s **global band** filter (commit f877bec, loader.py written
2026-08-05 21:37) catches sustained wrong-scale segments the local median filter
structurally cannot see. Every B01 trade file on disk was written before that:

* `B01_zscore_reversion_checkpoint/trades/block_*.parquet` — 2026-08-04, years 2000–2008
* `B01_zscore_reversion_checkpoint/trades/year_*.parquet` — 2026-08-05 16:23–18:06
* `B01_zscore_reversion_PILOT_checkpoint/` — 2026-08-04

Contaminated bars in the 28-pair candidate universe, measured 2026-08-06 by
`gpu_meanrev/analysis/data_provenance.py` (regenerable; writes
`journal/scorecards/data_provenance.json`):

| pair | non-positive | out-of-band | years |
|---|---|---|---|
| AUDJPY | 0 | **8,413** | 2005 |
| AUDUSD | 0 | 4 | 2000, 2005 |
| EURUSD | 1 | 1 | 2001 |
| GBPCHF | 0 | 1 | 2007 |

All of it sits in **2000–2007**. That is precisely where B01's regime table shows
2005 PF = **111.6** against a Sharpe of 0.039 — the AUDJPY 0.67-against-an-85-median
defect, booking 126x "returns". So:

* **B01 regime-table rows 2000–2007 are invalid.** The 2005 row is a data artifact, not
  a result, and should never be quoted.
* **B01 rows 2008 and 2015–2022, and the PILOT, are clean** — no contaminated bars fall
  in those years.
* **Resuming the paused B01 run from its checkpoint would splice post-fix data onto a
  pre-fix log.** The 2000–2007 blocks could not simply be resumed.

**Decision taken 2026-08-06: B01 is re-scoped to 2008–2022.** 2008-03-30 is the
common-coverage start across all 28 resolved pairs anyway, so the dropped years were
thin-universe data never comparable with the rest of the panel — and dropping them
removes every contaminated bar from the battery in one step, rather than spending GPU
hours recomputing years that were always the weakest evidence. The 31 pre-fix block
files are **quarantined, not deleted**, under
`journal/data/B01_zscore_reversion_checkpoint/QUARANTINE_pre_fix_2000_2008/` with a
README explaining why they must not be read. The 2009–2014 gap is being recomputed under
the fixed loader; 2008 and 2015–2022 year files are reused unchanged (verified clean).

Schema note: years computed before 2026-08-06 carry no entry-side columns, years after it
do. A concat yields nulls for the older years — expected, and not worth a recompute.

**B02 is clean and verified so.** Its universe (7 USD majors) over its window
(2015–2022) contains **zero** non-positive or out-of-band bars, so B02's numbers are
unaffected by which loader version was resident when it ran. Checked directly rather
than assumed.

---

## What this changes for the next pre-registration

1. **A 1-minute fade is not the trade.** In 2022 the signed edge rises from 0.025 bps at
   1 min to 0.935 bps at 960 min (not strictly monotonic — it dips at h=480), and across
   2015–2022 the shape is mixed: 2018 falls away by h=960, 2020 turns negative. What is
   consistent is the **absence of a snap-back** — no horizon shows the edge peaking and
   decaying, which is what a mean-reverting move is supposed to look like. B01/B02 fixed
   `max_hold = 4 x lookback` by assertion; nothing in the data supports a short hold.
   Either hold for hours (and then 1-minute bars are the wrong research frequency), or
   find the sub-hour structure that actually snaps back. Note this measurement uses combo
   64's parameters — B02's *ex-post* best — so it diagnoses that result and cannot be
   carried forward as a finding without a fresh pre-registration.
2. **Cost realism is the binding constraint, not signal discovery.** Anything whose gross
   edge is under ~0.5 bps is untradable at this box's achievable spreads regardless of
   its statistics. Future batteries should report breakeven cost next to PF, and should
   report edge **excluding** 21:00–02:00 UTC as the honest baseline.
3. **Hand the gate day-aggregated PnL**, not per-trade rows.
4. **Instrumentation is fixed** (2026-08-06): `run_fold` now records `entry_bar`,
   `entry_ts`, `side`, `entry_z` and `bars_held` alongside the exit. B01/B02 stored
   exit-only, which is why entry-hour, long/short and holding-period attribution above had
   to be reconstructed at signal level instead of read off the trade log. `entry_ts` is
   stored rather than derived because `entry_bar` is fold-local and the panel index is an
   irregular union of observed minutes — `ts - bars_held` minutes is not the entry time.
   The **first thing to run on the next battery's log is the entry-hour version of §2**,
   which is the claim this pass could not make.

## What was decided off the back of this (2026-08-06)

**B03 is pre-registered and deliberately not implemented yet**
(`gpu_meanrev/batteries/b03_horizon.py`, registration in
`journal/experiments/registered/B03_horizon.json`). It tests the no-snap-back horizon
finding — item 1 of "What this changes for the next pre-registration" below, not §5,
which is execution latency — directly: 48
combos on **60-minute** bars, lookbacks and holds in hours, same 7 USD majors, per-year
2015–2022. Three things in it exist because of this post-mortem:

* **A pre-committed kill criterion.** If the best configuration's gross edge is under 2x
  its assumed round-trip cost in 6 of 8 years, B03 is a FAIL and no grid expansion
  follows. B01 and B02 both spent days searching inside a hole that cost realism had
  already closed; this makes that outcome terminal instead of an invitation to widen.
* **A smaller grid than either predecessor** (48 vs 160 and 72). Every trial raises the
  Deflated Sharpe bar, and the diagnosis is that the constraint is cost, not search.
* **An explicit statement that the in-sample years are design-only.** The horizon
  observation was made on B02's in-sample data using its ex-post best combo, so testing
  it over 2015–2022 is testing a hypothesis against the data that generated it. The only
  confirmatory test is one evaluation of one frozen configuration on the locked 2023+
  holdout. There is no second holdout.

## Reproducing every number above

```
.venv\Scripts\python.exe -m gpu_meanrev.analysis.b02_diagnostics          # SS1, 2, 6, 7, 8
.venv\Scripts\python.exe -m gpu_meanrev.analysis.signal_microstructure --full   # SS3, 4
.venv\Scripts\python.exe -m gpu_meanrev.analysis.signal_microstructure --delays # SS5
.venv\Scripts\python.exe -m gpu_meanrev.analysis.data_provenance          # provenance
```

All four are read-only over committed trade logs and the local archive, cost no GPU, and
touch no holdout. Total runtime ~15 minutes, dominated by panel builds.
