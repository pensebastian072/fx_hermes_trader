# gpu_meanrev handoff — 1-minute FX mean-reversion study

Written 2026-08-07. Scope: the `gpu_meanrev/` module only (GPU/CUDA mean-reversion
research on 1-minute FX bars). Nothing here touches the live paper-trading harness in
`app/`, `engines/` or `hermes/` — this module bypasses all of it and is a pure backtest
path. Everything is SHADOW; nothing has been promoted and nothing sizes money.

Commits: `2257c41` (B01 rerun + post-mortem + B03 registration), `ae7743e` (B03 runner).

---

## Where the study stands in one paragraph

Three pre-registered batteries have run. **B01 and B02 are honest FAILs. B03 is
inconclusive by construction and its holdout shot is unspent.** The reason the first two
failed is not a weak search — it is that the signal is real and far too small to pay for
its own execution. B02's gross edge is 0.406 bps against the 1.0 bp round-trip cost the
study itself assumed. Three alternative explanations for the result were tested and
rejected. The single most useful thing a successor can do is stop searching for a better
parameter and fix the cost model.

## Verdicts

| battery | universe / period | grid | headline | verdict |
|---|---|---|---|---|
| **B01** z-score / Bollinger | 28 pairs, 2008–2022 | 160 | per-trade DSR **−5.618**, PBO 0.040, n=120,930 | **FAIL** |
| | | | day-agg (n=4,549) DSR −4.679, **t = −1.985** | significantly *negative* net of cost |
| **B02** MTF confluence | 7 USD majors, 2015–2022 | 72 | DSR **−1.076**, PBO 0.016, n=17,368 | **FAIL, hypothesis rejected** |
| | | | day-agg (n=2,314) DSR −1.288, t = 1.12 | not significant before deflation |
| **B03** horizon, hourly bars | 7 USD majors, 2015–2022 | 48 | kill criterion passed, rules incoherent | **INCONCLUSIVE** |

B01's PF > 1 in only **6 of 15 years**; every year has a negative DSR. B02's tightest
confluence arm beat its control in 1 of 8 years — the MTF hypothesis is dead.

## The finding that actually matters

**B02's control-arm gross edge is 0.406 bps of price. Breakeven round-trip cost is
therefore 0.406 bps ≈ 0.45 pips all-in.** The study charged 1.0 bp. Its own cost
assumption is 2.5× the entire signal, and no setting in the grid closes that gap.

Corollaries a successor should treat as constraints, not opinions:

* Anything whose gross edge is under ~0.5 bps is untradable at this box's achievable
  spreads regardless of its statistics.
* **Report breakeven cost next to profit factor, always.** Nobody did for two days, which
  is the only reason this took a post-mortem to surface.
* **48% of B02's gross PnL is booked on closes in 21:00–02:00 UTC** — rollover and the
  Asia handover, the widest-spread hours — and that bucket is the only one that has not
  decayed. Stated as *closes in*, not *earned in*: the logs stored exit times only. An
  honest baseline excludes those hours.

## Alternative explanations tested and REJECTED

Do not re-run these; they are settled and reproducible.

| hypothesis | result |
|---|---|
| The edge is a stale-price / forward-fill artifact | **Rejected.** Edge lives in the zero-fill bucket in every year. |
| The 2015→2022 decay is just falling volatility | **Rejected.** Vol-normalised edge still halves; 2022 has 2015's vol. |
| Zero-latency fills carried it | **Rejected.** 1 min of delay costs 4% of the signal, 10 min costs 18%. |

Two structural facts also established:

* **No reversion half-life.** The signed payoff never peaks and decays out to 960 min —
  there is nothing to snap back into. `max_hold = 4 × lookback` was asserted, never
  measured.
* **The gate's effective sample was overstated.** Per-trade `n` flattered the Deflated
  Sharpe in both batteries. **Hand the gate day-aggregated PnL.**

## B03: read this before touching it

B03 tested whether a longer holding horizon amortises the cost. It produced a **split
verdict**: gross edge *does* scale with holding period (every combo clearing 2× cost is a
168h hold), but risk-adjusted return runs the *opposite* way (the best day-aggregated
Sharpe is the shortest hold, 6h).

It is inconclusive for a procedural reason, not a numerical one. The two pre-registered
rules disagree about which configuration the run authorizes:

| combo | params | years ≥ 2× cost | day-agg Sharpe | DSR |
|---|---|---|---|---|
| 39 | lb 96h, z 1.5, hold 168h | **7 of 8** ✓ | 0.0457 | −0.718 |
| 44 | lb 96h, z 2.5, hold 6h | **1 of 8** ✗ | 0.1053 | **+2.029** |

`KILL_CRITERION` authorizes proceeding on combo 39. `selection_rule_for_holdout`
("highest in-sample day-aggregated Sharpe") then selects combo 44 — which fails the very
screen used to justify the run. The selection rule never scoped itself to kill-passing
combos.

**Hard constraints for whoever picks this up:**

1. **The 2023+ holdout was NOT read. Its single shot is unspent.** Keep it that way until
   a coherent design exists.
2. **Do not repair B03 in place.** Restricting selection to kill-passers, or switching the
   metric, means choosing a rule after seeing which combo it favours. That is the exact
   p-hacking move the apparatus exists to prevent. A changed design is a **new battery
   id** — the registry refuses to overwrite a registration for this reason.
3. **Combo 44's DSR of +2.029 is the first number on this box above the 1.645 bar. It is
   in-sample, ex-post, and on a config that failed the cost screen.** A lead, not a
   result. Do not promote it, do not quote it as evidence.
4. Multiplicity: only 2 of 48 combos clear 6+ years, with 9 more at exactly 5. A max of
   7/8 across 48 trials is weak.

**The methodology lesson, which generalises past this repo:** a pre-registration needs
*one coherent rule*. A screen on metric A plus a selection rule on metric B is an
incoherence you can only discover after the run, and by then every fix is p-hacking.

## Data provenance — a trap that already bit once

`loader.sanitize_closes`' **global-band** filter (commit `f877bec`, 2026-08-05 21:37) is
the only stage that catches a sustained wrong-scale price *segment*; the local
centred-median filter structurally cannot, because a long enough bad run drags the local
median with it. Every B01 trade file written before that timestamp was contaminated.

The archive contains 8,419 bad bars across the 28-pair universe, **all in 2000–2007** —
8,413 of them AUDJPY in 2005 printing ~0.67 against an ~85 median, which booked 126×
"returns" and produced the old regime table's 2005 profit factor of **111.6**.

Resolution taken: B01 was **re-scoped to 2008–2022** (2008-03-30 is the common-coverage
start across all 28 pairs anyway), the pre-fix blocks **quarantined not deleted** under
`journal/data/B01_zscore_reversion_checkpoint/QUARANTINE_pre_fix_2000_2008/`, and
2009–2014 recomputed under the fixed loader (7.03 h). B02 verified **clean** — zero bad
bars in its universe and window.

Notable side effect: the clean rerun's ex-post winner moved from combo 152 to **124**, so
the contaminated bars had been steering the *selection*, not merely inflating one year.

**Always run `python -m gpu_meanrev.analysis.data_provenance` before trusting a
checkpoint** rather than assuming its vintage.

## A correction already in the record — do not re-quote the old number

An earlier draft of `RESULTS.md` claimed B01's gross edge was **0.422 bps**, "the same
~0.41 bps as B02, independently reproduced". **That was wrong by 4.5×.** B01 charges
tiered costs (majors 0.5 bp/side, crosses 1.5 bp/side) and **74.8% of combo 124's trades
are crosses** paying 3.0 bp round-trip; mean charged cost is 2.496 bp. Adding back the
cost actually charged gives **1.919 bps blended / 1.394 bps majors-only** against B02's
0.406.

There is **no "two batteries agree" finding**. B02's cost result stands on B02's 7-major
universe alone. B01's FAIL is unaffected either way.

Generalised rule: when adding a cost back to recover gross, add the cost that was
**actually charged per row**, never a single tier's constant.

One observation logged but explicitly *not* harvested: B01's majors subset is net
positive (+0.394 bps/trade) while the crosses carry the loss. That is a post-hoc slice of
an ex-post combo. If it is worth testing, it needs its own pre-registration.

## Infrastructure notes (all learned the hard way)

* **Never build two full panels concurrently.** Two jobs drove free memory to 0.9 GB and
  wedged both. Bound *both* ends of `build_panel` before the union/ffill.
* **`regime_table` streams.** `pd.concat` over B01's 107.2M rows cannot complete on this
  box; it now ranks combos over a two-column pass, then reads back only the winner.
* **`bootstrap_ci` chunks.** The one-shot `(n_boot × n)` resample OOM'd the 6 GB GPU at
  n=120,930. Chunked results are statistically equivalent but not bit-identical to
  pre-chunking scorecards.
* **`run_fold` records entry-side fields** (`entry_bar`, `entry_ts`, `side`, `entry_z`,
  `bars_held`). B01/B02 stored exit-only, which is why the whole post-mortem had to
  reconstruct attribution at signal level. `entry_ts` is stored, never derived —
  `entry_bar` is fold-local and the panel index is an irregular union of minutes.
* **Never size a long run from a GPU-contended sample.** An early 83 h estimate was wrong
  by ~20×.
* Rolling windows use `avg_pool1d` on x and x² in **float64**; `.unfold()` is
  O(n·window) = 425 GB at real scale, and float32 eats the O(1e-8) variance signal.
* B01 year files written before 2026-08-06 have no entry-side columns; later ones do. A
  concat yields nulls for the older years. Expected, not worth a recompute.

## Commands

```
# diagnostics — all read-only, no GPU, no holdout, ~15 min total
.venv\Scripts\python.exe -m gpu_meanrev.analysis.b02_diagnostics
.venv\Scripts\python.exe -m gpu_meanrev.analysis.signal_microstructure --full
.venv\Scripts\python.exe -m gpu_meanrev.analysis.signal_microstructure --delays
.venv\Scripts\python.exe -m gpu_meanrev.analysis.data_provenance

# batteries
.venv\Scripts\python.exe -m gpu_meanrev.batteries.b01_zscore_reversion --years --from-year 2008 --to-year 2022
.venv\Scripts\python.exe -m gpu_meanrev.batteries.b03_horizon --smoke   # 2022 only, publishes nothing
.venv\Scripts\python.exe -m gpu_meanrev.batteries.b03_horizon --full    # 2015-2022, ~5.5 min

# reporting
.venv\Scripts\python.exe -m gpu_meanrev.reporting.regime_table B01_zscore_reversion

.venv\Scripts\python.exe -m pytest        # 170 tests
```

Runtimes measured on this box: B01 ≈ 51 min/year (28 pairs × 160 combos, 1-min bars);
B03 ≈ 41 s/year (7 pairs × 48 combos, hourly bars).

## Standing rules for this module

1. **Holdout is 2023-01-01.** Reading it needs `unlock_holdout=True` in code *and*
   `FX_MEANREV_HOLDOUT_UNLOCK=yes`. It is meant to be flipped once, at the very end. It
   has never been flipped.
2. Everything ships SHADOW until PBO < 0.5 **and** Deflated Sharpe > 0. Nothing here has
   cleared it.
3. The gate is imported from `macro_gpu_lab` and never re-ported.
4. A changed design is a **new battery id**, never an edited registration.
5. Report failures as failures. Three of three batteries have failed honestly, and that
   is the system working.

## Suggested next step

Fix the cost model before running anything else. Every result so far is decided by a
number the repo cannot currently check — there is no spread data in this archive, only
mid/bid closes. Until an hour-varying, year-varying spread series exists for the 7
majors, another battery just adds trials to the deflation without addressing the binding
constraint.

If a successor battery is wanted anyway, it should be a **B04** with one coherent
pre-registered rule (screen and selection over the same combo set), and it should carry
B03's split verdict forward as the thing to explain: gross edge scales with holding
period, risk-adjusted return does not.
