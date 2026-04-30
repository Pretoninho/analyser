from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.glosten_milgrom import sequential_gm_quotes


def build_synthetic_order_sign(n: int = 4000, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)

    # Base random signs
    s = np.where(rng.random(n) > 0.5, 1, -1).astype("int8")

    # Inject directional toxic bursts
    s[900:1250] = 1
    s[2200:2500] = -1
    s[3100:3400] = 1

    idx = pd.RangeIndex(n)
    return pd.Series(s, index=idx, name="order_sign")


def main() -> None:
    signs = build_synthetic_order_sign()

    out = sequential_gm_quotes(
        order_sign=signs,
        v_high=101.0,
        v_low=99.0,
        delta0=0.5,
        processing_bps=0.02,
        inventory_bps=0.03,
    )

    cols = [
        "mu_hat",
        "delta_prior",
        "delta_post",
        "bid_info",
        "ask_info",
        "spread_info",
        "spread_total",
        "toxic_regime",
    ]

    print(out[cols].describe().round(6).to_string())

    print("\nToxic regime counts:")
    print(out["toxic_regime"].value_counts().sort_index())

    print("\nSpread means by regime:")
    print(out.groupby("toxic_regime")[["spread_info", "spread_total", "mu_hat"]].mean().round(6).to_string())

    print("\nTail sample:")
    print(out[cols].tail(8).to_string(index=False))

    print("\nOK", out.shape)


if __name__ == "__main__":
    main()
