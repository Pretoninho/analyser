"""
Quick validation for aggressive execution and market-impact features.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.microstructure.aggressive_execution import (
    compute_aggressive_execution_features,
    optimal_slicing_schedule,
    simulate_market_sweep,
)


def make_synthetic_lob(n: int = 3500, levels: int = 6, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = np.cumsum(np.maximum(rng.normal(1.0, 0.05, size=n), 0.2))
    mid = 25000.0 + np.cumsum(rng.normal(0.0, 0.18, size=n))
    spread = np.maximum(0.5, 1.0 + rng.normal(0.0, 0.08, size=n))

    shock = rng.binomial(1, 0.04, size=n).astype(bool)
    spread[shock] *= rng.uniform(2.0, 5.0, size=shock.sum())

    df = pd.DataFrame({"ts": ts})
    df["aggr_order_qty"] = np.maximum(0.5, rng.lognormal(mean=1.1, sigma=0.55, size=n))
    df.loc[shock, "aggr_order_qty"] *= 2.5
    df["aggr_side"] = np.where(rng.random(n) > 0.5, "buy", "sell")
    df["adv"] = np.maximum(10000.0, rng.lognormal(mean=10.8, sigma=0.25, size=n))
    df["sigma_daily"] = np.maximum(0.005, rng.lognormal(mean=-4.1, sigma=0.25, size=n))

    for level in range(1, levels + 1):
        tick = 0.25 * level
        df[f"bid_px_{level}"] = mid - spread / 2 - tick
        df[f"ask_px_{level}"] = mid + spread / 2 + tick

        bid_sz = rng.gamma(shape=2.2, scale=6.0, size=n)
        ask_sz = rng.gamma(shape=2.1, scale=6.3, size=n)

        bid_sz[shock] *= rng.uniform(0.20, 0.55, size=shock.sum())
        ask_sz[shock] *= rng.uniform(0.20, 0.55, size=shock.sum())

        df[f"bid_sz_{level}"] = np.maximum(0.1, bid_sz)
        df[f"ask_sz_{level}"] = np.maximum(0.1, ask_sz)

    return df


def validate_pipeline() -> None:
    df = make_synthetic_lob()
    feat = compute_aggressive_execution_features(df, tick_size=0.25)

    required_cols = [
        "exec_vwap",
        "exec_levels_used",
        "exec_slippage_bps",
        "implementation_shortfall_bps",
        "impact_sqrt_bps",
        "impact_temp_bps",
        "impact_perm_bps",
        "impact_total_bps",
        "taker_cost_flag",
    ]
    for c in required_cols:
        assert c in feat.columns, f"missing column: {c}"

    row = df.iloc[1000]
    sim = simulate_market_sweep(
        side="buy",
        quantity=8.0,
        bid_prices=row[[f"bid_px_{k}" for k in range(1, 7)]].to_numpy(dtype="float64"),
        bid_sizes=row[[f"bid_sz_{k}" for k in range(1, 7)]].to_numpy(dtype="float64"),
        ask_prices=row[[f"ask_px_{k}" for k in range(1, 7)]].to_numpy(dtype="float64"),
        ask_sizes=row[[f"ask_sz_{k}" for k in range(1, 7)]].to_numpy(dtype="float64"),
    )

    schedule = optimal_slicing_schedule(parent_qty=24.0, n_child=6, curve="front_loaded")

    assert np.isfinite(sim["vwap_exec"])
    assert sim["levels_used"] >= 1
    assert np.isclose(schedule.sum(), 24.0)
    assert np.all(schedule > 0)
    assert feat["exec_levels_used"].ge(0).all()
    assert feat["impact_total_bps"].ge(0).all()

    print("AGGRESSIVE EXECUTION FEATURES SHAPE:", feat.shape)
    print("Sweep sample:", {k: round(float(v), 6) for k, v in sim.items()})
    print("Front-loaded schedule:", np.round(schedule, 6).tolist())
    print("Implementation shortfall mean:", round(float(feat["implementation_shortfall_bps"].mean()), 6))
    print("Square-root impact mean:", round(float(feat["impact_sqrt_bps"].mean()), 6))
    print("Almgren total impact mean:", round(float(feat["impact_total_bps"].mean()), 6))
    print("Taker cost ratio:", round(float(feat["taker_cost_flag"].mean()), 6))
    print("OK", feat.shape)


if __name__ == "__main__":
    validate_pipeline()