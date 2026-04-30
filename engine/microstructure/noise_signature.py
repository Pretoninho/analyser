"""
engine/microstructure/noise_signature.py

Microstructure-noise and volatility-signature utilities.

Implements reusable tools to:
- compute realized variance across sampling grids
- estimate microstructure noise variance from lag-1 autocovariance
- compute volatility signature plots
- compute two-scale realized variance (TSRV) denoiser
- derive optimal sampling step from noise/signal trade-off (2/3 law)
- build calendar / tick / volume clocks

All functions are vectorized for backtest workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12


def log_returns(price: pd.Series) -> pd.Series:
    """Compute log-returns with NaN-safe handling."""
    p = pd.to_numeric(price, errors="coerce").ffill().bfill()
    r = np.log(p / p.shift(1)).fillna(0.0)
    return r.astype("float64")


def realized_variance(price: pd.Series, step: int = 1) -> float:
    """
    Realized variance on a sub-sampled grid.

    RV(step) = sum_i [log(P_{i}) - log(P_{i-step})]^2 over i=step,2*step,...
    """
    if step <= 0:
        raise ValueError("step must be >= 1")

    p = pd.to_numeric(price, errors="coerce").ffill().bfill().to_numpy(dtype="float64")
    idx = np.arange(0, len(p), step)
    if len(idx) < 2:
        return 0.0

    lp = np.log(p[idx])
    r = np.diff(lp)
    return float(np.sum(r * r))


def realized_variance_series(price: pd.Series, steps: list[int]) -> pd.DataFrame:
    """Compute RV for multiple sampling steps."""
    rows = []
    for s in steps:
        rows.append({"step": int(s), "rv": realized_variance(price, step=int(s))})
    out = pd.DataFrame(rows)
    out["step"] = out["step"].astype("int64")
    out["rv"] = out["rv"].astype("float64")
    return out.sort_values("step").reset_index(drop=True)


def estimate_noise_variance_from_lag1(price: pd.Series) -> float:
    """
    Estimate microstructure noise variance from lag-1 autocovariance.

    Under bid-ask bounce stylization:
        Cov(r_t, r_{t-1}) ~= -eta^2
    => eta^2 ~= -Cov(r_t, r_{t-1})
    """
    r = log_returns(price)
    r1 = r.shift(1).fillna(0.0)
    cov1 = float(((r - r.mean()) * (r1 - r1.mean())).mean())
    eta2 = max(-cov1, 0.0)
    return float(eta2)


def estimate_integrated_variance_proxy(price: pd.Series, coarse_step: int = 60) -> float:
    """
    Coarse-grid IV proxy to reduce noise contamination.

    Uses RV on a sparse step as a practical IV approximation.
    """
    return float(realized_variance(price, step=max(1, int(coarse_step))))


def optimal_step_two_thirds_law(
    noise_var: float,
    signal_var: float,
    n_obs: int,
    min_step: int = 1,
) -> int:
    """
    Approximate optimal sampling step from 2/3 power law.

    Heuristic form:
        step* ~ ((noise_var / signal_var)^(2/3)) * n_obs^(1/3)

    Returns integer step >= min_step.
    """
    nv = max(float(noise_var), _EPS)
    sv = max(float(signal_var), _EPS)
    n = max(int(n_obs), 2)

    step = ((nv / sv) ** (2.0 / 3.0)) * (n ** (1.0 / 3.0))
    return int(max(min_step, round(step)))


def two_scale_realized_variance(price: pd.Series, k: int | None = None) -> float:
    """
    Two-scale realized variance (TSRV) style estimator.

    Simplified implementation:
      RV_fast = RV(step=1)
      RV_slow = average_{j=0..k-1} RV of sub-grids j mod k
      TSRV = RV_slow - (k/n) * RV_fast

    This removes first-order noise inflation from ultra-HF sampling.
    """
    p = pd.to_numeric(price, errors="coerce").ffill().bfill().to_numpy(dtype="float64")
    n = len(p)
    if n < 3:
        return 0.0

    if k is None:
        k = int(max(2, np.sqrt(n)))
    k = max(2, min(int(k), n // 2 if n >= 4 else 2))

    rv_fast = realized_variance(pd.Series(p), step=1)

    rv_sub = []
    lp = np.log(p)
    for j in range(k):
        idx = np.arange(j, n, k)
        if len(idx) >= 2:
            r = np.diff(lp[idx])
            rv_sub.append(float(np.sum(r * r)))

    if len(rv_sub) == 0:
        return float(rv_fast)

    rv_slow = float(np.mean(rv_sub))
    tsrv = rv_slow - (k / max(n, 1)) * rv_fast
    return float(max(tsrv, 0.0))


def volatility_signature(
    price: pd.Series,
    steps: list[int],
    annualization_factor: float = 1.0,
    compute_tsrv: bool = True,
) -> pd.DataFrame:
    """
    Build volatility signature table over multiple sampling steps.

    Returns columns:
      step, rv, rv_ann, noise_penalty, tsrv_ref
    """
    sig = realized_variance_series(price, steps)
    sig["rv_ann"] = sig["rv"] * float(annualization_factor)

    # Noise penalty approximation from lag-1 covariance model:
    # E[RV_m] ~= IV + 2*m*eta^2  where m ~ n/step
    eta2 = estimate_noise_variance_from_lag1(price)
    n = len(price)
    m = (n / sig["step"].clip(lower=1)).astype("float64")
    sig["noise_penalty"] = 2.0 * m * eta2

    if compute_tsrv:
        tsrv_ref = two_scale_realized_variance(price)
        sig["tsrv_ref"] = tsrv_ref
    else:
        sig["tsrv_ref"] = np.nan

    return sig


def build_tick_clock(df: pd.DataFrame, chunk_size: int = 50) -> pd.DataFrame:
    """
    Build tick-time aggregation index.

    Every chunk_size events -> one clock bucket.
    """
    out = df.copy()
    n = len(out)
    bucket = np.arange(n) // max(1, int(chunk_size))
    out["tick_bucket"] = bucket.astype("int64")
    return out


def build_volume_clock(
    df: pd.DataFrame,
    volume_col: str = "volume",
    target_volume: float = 1_000.0,
) -> pd.DataFrame:
    """
    Build business-time (volume clock) buckets.

    New bucket starts each time cumulative traded volume exceeds target_volume.
    """
    out = df.copy()
    vol = pd.to_numeric(out[volume_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")

    target = max(float(target_volume), _EPS)
    csum = np.cumsum(vol)
    bucket = np.floor(csum / target).astype("int64")
    out["volume_bucket"] = bucket
    return out


def aggregate_price_by_bucket(
    df: pd.DataFrame,
    price_col: str,
    bucket_col: str,
) -> pd.Series:
    """
    Aggregate to one price per bucket (last price in each bucket).
    """
    p = pd.to_numeric(df[price_col], errors="coerce").ffill().bfill()
    b = pd.to_numeric(df[bucket_col], errors="coerce").fillna(0).astype("int64")

    tmp = pd.DataFrame({"p": p, "b": b})
    agg = tmp.groupby("b", sort=True)["p"].last()
    return agg.astype("float64")
