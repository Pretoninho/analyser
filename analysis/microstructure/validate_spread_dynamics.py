from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.spread_dynamics import (
    compute_spread_dynamics_features,
    compute_trade_spread_tca,
    queue_fill_probability,
)


def build_synthetic_lob(n: int = 4000, levels: int = 6, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-04-01", periods=n, freq="ms", tz="UTC")

    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.005, n))
    spread = np.maximum(0.02 + rng.normal(0.0, 0.002, n), 0.01)

    # high-vol regimes: spread widening
    shock = (np.arange(n) > 1200) & (np.arange(n) < 1550)
    spread[shock] *= 2.8

    df = pd.DataFrame({"timestamp": ts})

    for k in range(1, levels + 1):
        tick = 0.01 * k
        df[f"bid_px_{k}"] = mid - spread / 2 - tick
        df[f"ask_px_{k}"] = mid + spread / 2 + tick

        bsz = rng.gamma(shape=2.0, scale=25.0, size=n)
        asz = rng.gamma(shape=2.0, scale=25.0, size=n)

        # asymmetry and depth shifts
        bsz[900:1200] *= 1.7
        asz[1800:2200] *= 1.8

        df[f"bid_sz_{k}"] = bsz
        df[f"ask_sz_{k}"] = asz

    return df


def build_synthetic_trades(df_lob: pd.DataFrame, seed: int = 78) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df_lob)

    dirn = np.where(rng.random(n) > 0.5, 1, -1)
    mid = 0.5 * (df_lob["bid_px_1"].to_numpy() + df_lob["ask_px_1"].to_numpy())
    spread = (df_lob["ask_px_1"] - df_lob["bid_px_1"]).to_numpy()

    # executions around quotes, with occasional adverse prints
    trade = mid + dirn * 0.5 * spread + rng.normal(0.0, 0.002, n)
    future_mid = mid + rng.normal(0.0, 0.01, n)

    return pd.DataFrame(
        {
            "trade_price": trade,
            "direction": dirn,
            "mid_at_trade": mid,
            "future_mid_tau": future_mid,
        }
    )


def main() -> None:
    lob = build_synthetic_lob()
    feat = compute_spread_dynamics_features(lob, tick_size=0.01, depth_levels=5)

    trades = build_synthetic_trades(lob)
    tca = compute_trade_spread_tca(
        trade_price=trades["trade_price"],
        direction=trades["direction"],
        mid_at_trade=trades["mid_at_trade"],
        future_mid_tau=trades["future_mid_tau"],
    )

    lam = pd.Series(np.abs(np.random.normal(1.2, 0.4, len(lob))))
    q_pos = pd.Series(np.abs(np.random.normal(120, 55, len(lob))))
    theta = pd.Series(np.abs(np.random.normal(0.8, 0.2, len(lob))))
    pfill = queue_fill_probability(lam, q_pos, theta, horizon_sec=0.5)

    print("Spread features:")
    print(
        feat[
            [
                "spread_quoted",
                "spread_rel_bps",
                "spread_tick_ratio",
                "microprice_delta_bps",
                "dom_depth_imbalance",
            ]
        ]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nTick regime counts:")
    print(feat["tick_regime"].value_counts())

    print("\nTCA spread decomposition (bps):")
    print(tca.describe().round(6).to_string())

    print("\nQueue fill probability:")
    print(pfill.describe().round(6).to_string())

    print("\nOK", feat.shape, tca.shape)


if __name__ == "__main__":
    main()
