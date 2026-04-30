from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.noise_signature import (
    aggregate_price_by_bucket,
    build_tick_clock,
    build_volume_clock,
    estimate_integrated_variance_proxy,
    estimate_noise_variance_from_lag1,
    optimal_step_two_thirds_law,
    two_scale_realized_variance,
    volatility_signature,
)


def build_noisy_price(n: int = 20000, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Latent efficient log-price (Brownian-like)
    latent_ret = rng.normal(0.0, 0.00025, n)
    log_p_star = np.log(100.0) + np.cumsum(latent_ret)

    # Microstructure noise with bid-ask bounce pattern
    c = 0.0008  # half spread in log-price approximation
    q = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    # enforce stronger alternation to induce lag-1 negative autocovariance
    q[1::2] *= -1.0

    log_p_obs = log_p_star + c * q
    p_obs = np.exp(log_p_obs)

    # synthetic event data
    vol = rng.lognormal(mean=1.3, sigma=0.5, size=n)
    ts = pd.date_range("2026-03-01", periods=n, freq="ms", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": p_obs,
            "volume": vol,
        }
    )


def main() -> None:
    df = build_noisy_price()

    # Signature on raw event-time price
    steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    sig = volatility_signature(df["price"], steps=steps)

    eta2 = estimate_noise_variance_from_lag1(df["price"])
    iv_proxy = estimate_integrated_variance_proxy(df["price"], coarse_step=200)
    step_opt = optimal_step_two_thirds_law(
        noise_var=eta2,
        signal_var=max(iv_proxy, 1e-12),
        n_obs=len(df),
    )
    tsrv = two_scale_realized_variance(df["price"])

    print("Noise diagnostics:")
    print(
        {
            "noise_var_eta2": round(float(eta2), 10),
            "iv_proxy": round(float(iv_proxy), 10),
            "step_opt_2_3": int(step_opt),
            "tsrv": round(float(tsrv), 10),
        }
    )

    print("\nVolatility signature:")
    print(sig.to_string(index=False))

    # Compare calendar vs tick vs volume clocks (same module utilities)
    tick_df = build_tick_clock(df, chunk_size=50)
    vol_df = build_volume_clock(df, volume_col="volume", target_volume=150.0)

    p_tick = aggregate_price_by_bucket(tick_df, price_col="price", bucket_col="tick_bucket")
    p_vol = aggregate_price_by_bucket(vol_df, price_col="price", bucket_col="volume_bucket")

    rv_tick = float(np.sum(np.diff(np.log(p_tick.values)) ** 2)) if len(p_tick) > 1 else 0.0
    rv_vol = float(np.sum(np.diff(np.log(p_vol.values)) ** 2)) if len(p_vol) > 1 else 0.0

    print("\nClock comparison:")
    print(
        {
            "n_raw": int(len(df)),
            "n_tick_buckets": int(len(p_tick)),
            "n_volume_buckets": int(len(p_vol)),
            "rv_tick_clock": round(rv_tick, 10),
            "rv_volume_clock": round(rv_vol, 10),
        }
    )

    print("\nOK", sig.shape)


if __name__ == "__main__":
    main()
