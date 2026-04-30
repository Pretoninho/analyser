"""
Quick validation for liquidity void and extreme microvolatility features.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.microstructure.liquidity_void import compute_liquidity_void_features


def make_synthetic_lob(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = np.cumsum(np.maximum(rng.normal(1.0, 0.05, size=n), 0.2))
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=n))

    shock_idx = rng.choice(np.arange(300, n - 300), size=20, replace=False)
    shock = np.zeros(n, dtype="float64")
    shock[shock_idx] = rng.normal(0.0, 0.35, size=len(shock_idx))
    mid = mid + np.cumsum(shock)

    base_spread = np.maximum(0.01, 0.02 + rng.lognormal(mean=-4.1, sigma=0.45, size=n))
    spread = base_spread.copy()
    spread[shock_idx] *= rng.uniform(4.0, 12.0, size=len(shock_idx))

    bid_px_1 = mid - 0.5 * spread
    ask_px_1 = mid + 0.5 * spread

    bid_sz_1 = np.maximum(0.1, rng.gamma(2.2, 4.0, size=n))
    ask_sz_1 = np.maximum(0.1, rng.gamma(2.1, 4.2, size=n))

    bid_sz_1[shock_idx] *= rng.uniform(0.02, 0.20, size=len(shock_idx))
    ask_sz_1[shock_idx] *= rng.uniform(0.02, 0.20, size=len(shock_idx))

    tick = 0.01
    data: dict[str, np.ndarray] = {
        "ts": ts,
        "bid_px_1": bid_px_1,
        "ask_px_1": ask_px_1,
        "bid_sz_1": bid_sz_1,
        "ask_sz_1": ask_sz_1,
    }

    for level in range(2, 6):
        data[f"bid_px_{level}"] = bid_px_1 - (level - 1) * tick
        data[f"ask_px_{level}"] = ask_px_1 + (level - 1) * tick

        depth_mult_bid = rng.uniform(0.8, 1.5, size=n)
        depth_mult_ask = rng.uniform(0.8, 1.5, size=n)
        data[f"bid_sz_{level}"] = np.maximum(0.1, bid_sz_1 * depth_mult_bid)
        data[f"ask_sz_{level}"] = np.maximum(0.1, ask_sz_1 * depth_mult_ask)

    cancel_vol = rng.poisson(1.2, size=n).astype("float64")
    new_limit_vol = rng.poisson(1.5, size=n).astype("float64")
    sweep_vol = rng.poisson(0.4, size=n).astype("float64")

    cancel_vol[shock_idx] += rng.poisson(10.0, size=len(shock_idx))
    sweep_vol[shock_idx] += rng.poisson(8.0, size=len(shock_idx))
    new_limit_vol[shock_idx] *= rng.uniform(0.1, 0.4, size=len(shock_idx))

    data["cancel_vol"] = cancel_vol
    data["new_limit_vol"] = new_limit_vol
    data["sweep_vol"] = sweep_vol

    return pd.DataFrame(data)


def validate_pipeline() -> None:
    df = make_synthetic_lob()
    feat = compute_liquidity_void_features(df, tick_size=0.01, depth_levels=5)

    required_cols = [
        "liquidity_void_score",
        "void_duration_sec",
        "cancel_arrival_ratio",
        "replenishment_rate",
        "impact_coeff",
        "tail_index_hill",
        "pot_expected_shortfall",
        "protective_quote_regime",
    ]
    for c in required_cols:
        assert c in feat.columns, f"missing column: {c}"

    assert feat["liquidity_void_score"].ge(0).all()
    assert feat["depth_depletion_ratio"].between(0, 1).all()
    assert feat["protective_quote_regime"].isin(["normal", "widen", "pause"]).all()

    print("LIQUIDITY VOID FEATURES SHAPE:", feat.shape)
    print("Void score mean:", round(float(feat["liquidity_void_score"].mean()), 6))
    print("Void flag ratio:", round(float(feat["void_flag"].mean()), 6))
    print("Tail index mean:", round(float(feat["tail_index_hill"].mean()), 6))
    print("Expected shortfall mean:", round(float(feat["pot_expected_shortfall"].mean()), 6))
    print("Quote regimes:", feat["protective_quote_regime"].value_counts().to_dict())
    print("OK", feat.shape)


if __name__ == "__main__":
    validate_pipeline()