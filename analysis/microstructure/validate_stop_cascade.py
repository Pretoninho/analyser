from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.stop_cascade import (
    compute_stop_cascade_features,
    robust_stop_plan,
    stop_limit_fill_probability_sell,
    stop_market_vwap_sell,
)


def build_synthetic_lob(n: int = 3000, levels: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = pd.date_range("2026-02-01", periods=n, freq="ms", tz="UTC")
    mid = 50000 + np.cumsum(rng.normal(0, 0.35, n))

    df = pd.DataFrame({"timestamp": ts})

    base_spread = np.maximum(0.5 + rng.normal(0, 0.05, n), 0.2)

    # Inject shock window with spread widening + depth evaporation
    shock = (np.arange(n) > 1300) & (np.arange(n) < 1650)
    base_spread[shock] *= 4.0

    for k in range(1, levels + 1):
        tick = 0.1 * k
        df[f"bid_px_{k}"] = mid - base_spread / 2 - tick
        df[f"ask_px_{k}"] = mid + base_spread / 2 + tick

        bid_sz = rng.gamma(shape=2.2, scale=9.0, size=n)
        ask_sz = rng.gamma(shape=2.2, scale=9.0, size=n)

        # Quote fading on bid side during shock
        bid_sz[shock] *= 0.12
        ask_sz[shock] *= 0.45

        df[f"bid_sz_{k}"] = bid_sz
        df[f"ask_sz_{k}"] = ask_sz

    # Simulated clustered stop triggers
    trig = np.zeros(n, dtype="int8")
    trig[1320:1360] = 1
    trig[1400:1480] = 1
    trig[1550:1600] = 1

    est_size = np.ones(n) * 3.0
    est_size[shock] = 8.0

    df["stop_trigger_event"] = trig
    df["est_stop_size"] = est_size
    return df


def main() -> None:
    df = build_synthetic_lob()
    out = compute_stop_cascade_features(df, depth_levels=3)

    cols = [
        "spread_z",
        "top_bid_depth",
        "vacuum_down_score",
        "cascade_lambda",
        "convex_impact",
        "cascade_risk_score",
        "cascade_risk_flag",
    ]

    print(out[cols].describe().round(5).to_string())

    print("\nCascade risk counts:")
    print(out["cascade_risk_flag"].value_counts().sort_index())

    # Check sweep slippage on a shock snapshot
    i = 1450
    bid_px = out.loc[i, [f"bid_px_{k}" for k in range(1, 7)]].to_numpy(dtype="float64")
    bid_sz = out.loc[i, [f"bid_sz_{k}" for k in range(1, 7)]].to_numpy(dtype="float64")

    trigger = float(out.loc[i, "bid_px_1"])
    sim = stop_market_vwap_sell(order_size=12.0, bid_prices=bid_px, bid_sizes=bid_sz)
    slippage = trigger - sim["vwap_exec"] if np.isfinite(sim["vwap_exec"]) else np.nan

    # Dynamic robust plan sample
    mid_i = 0.5 * (out.loc[i, "bid_px_1"] + out.loc[i, "ask_px_1"])
    plan = robust_stop_plan(
        trigger_price=trigger,
        current_mid=mid_i,
        sigma_inst=0.0018,
        side="sell",
        spread_z=float(out.loc[i, "spread_z"]),
        parent_qty=12.0,
        child_qty=2.0,
    )

    fill_prob = stop_limit_fill_probability_sell(
        limit_price=float(plan["dynamic_limit_price"]),
        bid_prices=bid_px,
        bid_sizes=bid_sz,
        order_size=12.0,
    )

    print("\nShock snapshot slippage check:")
    print({
        "trigger": round(trigger, 6),
        "vwap_exec": round(float(sim["vwap_exec"]), 6),
        "slippage_abs": round(float(slippage), 6),
        "levels_used": int(sim["levels_used"]),
        "filled_qty": round(float(sim["filled_qty"]), 6),
    })

    print("\nRobust stop plan:")
    print(plan)
    print("stop_limit_immediate_fill_ratio", round(float(fill_prob), 6))

    print("\nOK", out.shape)


if __name__ == "__main__":
    main()
