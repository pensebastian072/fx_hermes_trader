"""GPU-accelerated confidence for an OOS PnL series.

Adapted near-verbatim from alpaca_gpu_lab/src/backtest_ci.py (itself from
macro_gpu_lab/macro_gpu_lab/backtest.py) — generic resampling helper, not
gate math. The point-estimate gate (PBO/DSR via gate.py) stays authoritative;
this adds a bootstrap Sharpe CI and a sign-permutation p-value so a "pass"
isn't taken on a single point estimate.

Fail-safe: no torch/CUDA -> CPU tensors; too little PnL -> None. Never raises.
"""
from __future__ import annotations

import numpy as np

from gpu_meanrev import config
from gpu_meanrev.gpu import get_device, tune_backend

N_BOOT = 10000
N_PERM = 10000
CI = (2.5, 97.5)


def confidence(pnls, n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict | None:
    """Bootstrap Sharpe CI + permutation p-value(mean>0) on GPU when available."""
    arr = np.asarray(pnls, dtype=np.float64)
    if arr.size < 8:
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> skip, gate still stands
        return None

    device = get_device()
    g = torch.Generator(device=device)
    g.manual_seed(config.SEED)
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    n = t.numel()

    # Both stages materialise a (rows x n) matrix. At B01's full 2008-2022
    # scale that is 10000 x 120930 = 4.5GB of float32 per stage, which does not
    # fit the RTX 3050's 6GB — the original one-shot version OOM'd the moment a
    # battery outgrew its pilot. Chunk the ROW dimension so peak memory is
    # bounded by CHUNK_ELEMS regardless of n.
    #
    # Peak is ~0.4-0.65GB per stage at this setting, NOT the 128MB a float32-only
    # reading suggests: `idx`/`signs` come back from torch.randint as int64
    # (256MB at 32M elems), the gather adds a float32 copy (128MB), and
    # `* 2 - 1` allocates another int64 temporary.
    #
    # Reproducibility: chunking changes the draw ORDER, not the distribution
    # (rows are iid however the generator stream is partitioned). But
    # rows_per_chunk depends on n, so a CI computed by the pre-chunking code —
    # or at a different n — will not reproduce bit-for-bit. Statistically
    # equivalent, not numerically identical.
    CHUNK_ELEMS = 32_000_000
    rows_per_chunk = max(1, min(n_boot, CHUNK_ELEMS // max(n, 1)))

    # -- bootstrap: resample-with-replacement -> Sharpe distribution --
    sharpe_parts = []
    done = 0
    while done < n_boot:
        rows = min(rows_per_chunk, n_boot - done)
        idx = torch.randint(0, n, (rows, n), generator=g, device=device)
        samp = t[idx]
        mean = samp.mean(dim=1)
        std = samp.std(dim=1, unbiased=True)
        sharpe_parts.append(torch.where(std > 0, mean / std, torch.zeros_like(mean)).cpu())
        del idx, samp, mean, std
        done += rows
    sharpe = torch.cat(sharpe_parts)
    lo, hi = np.percentile(sharpe.numpy(), CI)
    sharpe_pos = float((sharpe > 0).float().mean().item())

    # -- sign-permutation test: p(mean >= observed | random +/- signs) --
    obs_mean = float(t.mean().item())
    perm_parts = []
    done = 0
    while done < n_perm:
        rows = min(rows_per_chunk, n_perm - done)
        signs = torch.randint(0, 2, (rows, n), generator=g, device=device) * 2 - 1
        perm_parts.append((t.unsqueeze(0) * signs).mean(dim=1).cpu())
        del signs
        done += rows
    perm_means = torch.cat(perm_parts)
    p_value = float((perm_means >= obs_mean).float().mean().item())

    return {
        "device": str(device),
        "tuning": tune_backend(),
        "n_boot": n_boot,
        "n_perm": n_perm,
        "sharpe_ci": [round(float(lo), 4), round(float(hi), 4)],
        "sharpe_prob_positive": round(sharpe_pos, 4),
        "mean": round(obs_mean, 6),
        "perm_p_value": round(p_value, 4),
    }
