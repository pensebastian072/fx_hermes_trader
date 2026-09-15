"""GPU-batched per-bar mean-reversion state machine, one fold at a time.

Position/PnL is path-dependent (an open trade's exit depends on whether it
already exited on a prior bar) -- that state machine can't be expressed as a
single vectorized op across time, only across the *batch* dimension. This
runs one Python-level loop over bars WITHIN one fold (folds sized in months,
not the full multi-year history at once), where each iteration updates
(batch,)-shaped state tensors via vectorized torch ops reading that bar's
signal slice across every (pair, combo) simultaneously.

Batch dimension = n_pairs x n_combos. At full grid size (28 x 160 = 4480)
this does NOT fit a single fold's tensors in the RTX 3050's 6GB alongside
everything else running on the box -- callers must chunk the combo list
(see batteries/b01_zscore_reversion.py) and run this function once per
chunk, accumulating results. This function itself does no chunking; it
trusts the caller to size `combos` sensibly (that's what Stage 5's smoke
test is for).
"""
from __future__ import annotations

import torch

from gpu_meanrev.gpu import get_device
from gpu_meanrev.signals.mean_reversion import Combo, session_mask


def run_fold(
    prices: torch.Tensor,                    # (n_pairs, n_bars) float, this fold's bars
    z_by_window: dict[int, torch.Tensor],      # lookback_min -> (n_pairs, n_bars)
    bb_by_window: dict[int, torch.Tensor],       # lookback_min -> (n_pairs, n_bars)
    fold_index,                                    # pandas DatetimeIndex for this fold (for session_mask)
    combos: list[Combo],
    pairs: list[str],
    cost_per_side: torch.Tensor,               # (n_pairs,) fraction of price, per side
    device=None,
    collect_trades: bool = True,
    extra_entry_gate: torch.Tensor | None = None,   # (batch, n_bars) bool, or None
) -> tuple[torch.Tensor, list[dict]]:
    """Returns ((n_pairs, n_combos) total PnL, list of per-trade records).

    Each trade record: {"bar": int, "ts": Timestamp, "pair": str,
    "combo_idx": int, "pnl": float, "entry_bar": int, "entry_ts": Timestamp,
    "side": int, "entry_z": float, "bars_held": int} — needed for real (not
    fold-aggregated) gate evaluation: pbo_cscv/deflated_sharpe want per-trade
    PnL in chronological order, not one point per fold (10 CSCV groups x 4
    needs >=40 observations; a handful of folds is nowhere near that).

    `bar`/`ts` are the EXIT. B01 and B02 stored ONLY those, which made
    entry-hour, long/short and holding-period attribution impossible after
    the fact — the whole point of the 2026-08-06 diagnostics pass was blocked
    on it (see analysis/b02_diagnostics.py `gaps`).

    `entry_ts` is stored explicitly and is NOT redundant with the other
    fields: `entry_bar` is FOLD-LOCAL (callers invoke this once per block, so
    bar 0 means something different every call) and the panel index is the
    irregular union of observed minutes, so `ts - bars_held` minutes is not
    the entry time either. Without it, entry hour — the reason these fields
    exist — stays unrecoverable. Cost is one extra index lookup per exit.

    `extra_entry_gate` is an optional additional ENTRY permission mask,
    ANDed with the session mask (it never forces an exit — an already-open
    position runs its normal exit rule). B02 uses it to carry the
    multi-timeframe confluence filter. Shape must be (n_pairs*n_combos,
    n_bars) in the same pair-major order as everything else here.
    """
    device = device or get_device()
    n_pairs, n_bars = prices.shape
    n_combos = len(combos)
    batch = n_pairs * n_combos

    # (n_pairs, n_combos, n_bars) by selecting each combo's lookback window,
    # then flatten to (batch, n_bars). repeat_interleave keeps pair-major
    # ordering consistent with cost_per_side/prices below.
    z_stack = torch.stack([z_by_window[c.lookback_min] for c in combos], dim=1)
    bb_stack = torch.stack([bb_by_window[c.lookback_min] for c in combos], dim=1)
    z_flat = z_stack.reshape(batch, n_bars).to(device)
    bb_flat = bb_stack.reshape(batch, n_bars).to(device)

    entry_z = torch.tensor([c.entry_z for c in combos], device=device, dtype=torch.float32).repeat(n_pairs)
    max_hold = torch.tensor([c.max_hold_bars for c in combos], device=device, dtype=torch.long).repeat(n_pairs)
    # "fixed_hold" (B03) is the same mechanism as "max_hold" (B01/B02) — exit
    # after N bars — differing only in that N is stated outright rather than
    # derived as 4x the lookback. Sharing the code path keeps the tested state
    # machine authoritative instead of forking it per battery.
    is_max_hold_exit = torch.tensor(
        [c.exit_rule in ("max_hold", "fixed_hold") for c in combos], device=device
    ).repeat(n_pairs)
    needs_bb_confirm = torch.tensor([c.signal_family == "zscore_bollinger_confirm" for c in combos], device=device).repeat(n_pairs)

    sess_by_combo = torch.stack([session_mask(fold_index, c.session) for c in combos], dim=0).to(device)  # (n_combos, n_bars)
    sess_flat = sess_by_combo.repeat(n_pairs, 1)  # (batch, n_bars) -- pair-major, matches repeat() above
    if extra_entry_gate is not None:
        if tuple(extra_entry_gate.shape) != (batch, n_bars):
            raise ValueError(
                f"extra_entry_gate must be ({batch}, {n_bars}), got {tuple(extra_entry_gate.shape)}"
            )
        sess_flat = sess_flat & extra_entry_gate.to(device)

    cps = cost_per_side.to(device).repeat_interleave(n_combos)  # (batch,)
    px = prices.repeat_interleave(n_combos, dim=0).to(device)   # (batch, n_bars)

    position = torch.zeros(batch, device=device)
    entry_price = torch.zeros(batch, device=device)
    bars_held = torch.zeros(batch, dtype=torch.long, device=device)
    pnl = torch.zeros(batch, device=device)
    # Entry-side attribution carried alongside the position state.
    entry_bar = torch.full((batch,), -1, dtype=torch.long, device=device)
    entry_z_at = torch.zeros(batch, device=device)
    trades: list[dict] = []

    for t in range(n_bars):
        z_t = z_flat[:, t]
        bb_t = bb_flat[:, t]
        px_t = px[:, t]
        sess_t = sess_flat[:, t]

        # A NaN check alone is NOT enough: the source archive contained finite
        # but impossible prices (a NEGATIVE EURUSD print, AUDJPY at 0.67 against
        # an ~85 median). Those passed every NaN guard and booked fake profit of
        # up to 9.1e11 on a single trade. Non-positive prices are rejected here
        # as a second line of defence behind loader.sanitize_closes.
        invalid = torch.isnan(z_t) | torch.isnan(px_t) | (px_t <= 0)
        z_safe = torch.where(invalid, torch.zeros_like(z_t), z_t)

        open_mask = position != 0

        # -- exit check for currently-open positions --
        inner_exit = (~is_max_hold_exit) & (z_safe.abs() < 0.5)
        hold_exit = is_max_hold_exit & (bars_held >= max_hold)
        exit_now = open_mask & (inner_exit | hold_exit) & (~invalid)

        # entry_price is > 0 for any real open position (entries only happen on
        # a valid bar). clamp_min here is a divide-by-zero guard ONLY -- it must
        # never be reached for an open position, and previously it silently
        # turned a bad entry price into an astronomical return instead of
        # rejecting the trade.
        safe_entry = torch.where(entry_price > 0, entry_price, torch.ones_like(entry_price))
        realized = torch.where(
            exit_now,
            position * (px_t - entry_price) / safe_entry - 2.0 * cps,
            torch.zeros_like(pnl),
        )
        pnl = pnl + realized
        if collect_trades:
            fired = exit_now.nonzero(as_tuple=True)[0]
            if fired.numel():
                ts = fold_index[t]
                fired_list = fired.tolist()
                pnl_list = realized[fired].tolist()
                # Read entry-side state BEFORE the reset below clears it. One
                # stacked device->host transfer rather than five: at 11.5M
                # trades the per-exit sync count is not free.
                attrib = torch.stack([
                    entry_bar[fired].to(realized.dtype),
                    position[fired],
                    entry_z_at[fired],
                    bars_held[fired].to(realized.dtype),
                ], dim=1).tolist()
                for slot, val, (eb, sd, ez, hd) in zip(fired_list, pnl_list, attrib):
                    pair_idx, combo_idx = divmod(slot, n_combos)
                    trades.append({"bar": t, "ts": ts, "pair": pairs[pair_idx],
                                   "combo_idx": combo_idx, "pnl": val,
                                   "entry_bar": int(eb), "entry_ts": fold_index[int(eb)],
                                   "side": int(sd), "entry_z": ez,
                                   "bars_held": int(hd)})
        position = torch.where(exit_now, torch.zeros_like(position), position)
        bars_held = torch.where(exit_now, torch.zeros_like(bars_held), bars_held)

        # -- entry check for flat slots (re-entry same bar after an exit above is allowed) --
        flat_slots = (position == 0) & sess_t & (~invalid)
        long_trigger = z_safe < -entry_z
        short_trigger = z_safe > entry_z
        bb_ok_long = (~needs_bb_confirm) | (bb_t < 0)
        bb_ok_short = (~needs_bb_confirm) | (bb_t > 1)
        enter_long = flat_slots & long_trigger & bb_ok_long
        enter_short = flat_slots & short_trigger & bb_ok_short

        position = torch.where(enter_long, torch.ones_like(position), position)
        position = torch.where(enter_short, -torch.ones_like(position), position)
        entered = enter_long | enter_short
        entry_price = torch.where(entered, px_t, entry_price)
        entry_bar = torch.where(entered, torch.full_like(entry_bar, t), entry_bar)
        entry_z_at = torch.where(entered, z_safe, entry_z_at)

        still_open_or_new = (position != 0)
        bars_held = torch.where(still_open_or_new, bars_held + 1, bars_held)

    return pnl.reshape(n_pairs, n_combos).cpu(), trades
