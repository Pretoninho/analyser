from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.orderbook_imbalance import (
    compute_obi_features_from_lob,
    compute_trade_imbalance,
    merge_lob_trade_features_asof,
)


def build_synthetic_lob(n: int = 2000, levels: int = 5, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    base = 70000.0
    spread = 0.5

    ts = pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC")
    mid = base + np.cumsum(np.random.normal(0, 0.2, n))

    lob = pd.DataFrame({"timestamp": ts})
    for k in range(1, levels + 1):
        tick = 0.1 * k
        lob[f"bid_px_{k}"] = mid - spread / 2 - tick
        lob[f"ask_px_{k}"] = mid + spread / 2 + tick

        bid_sz = np.random.gamma(shape=2.0, scale=8.0, size=n)
        ask_sz = np.random.gamma(shape=2.0, scale=8.0, size=n)

        # Inject asymmetric regimes to test detection sensitivity.
        if k <= 2:
            bid_sz[500:800] *= 1.8
            ask_sz[1200:1450] *= 2.0

        lob[f"bid_sz_{k}"] = bid_sz
        lob[f"ask_sz_{k}"] = ask_sz

    return lob


def build_synthetic_trades(ts: pd.Series, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed + 1)
    n = len(ts)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "qty": np.random.lognormal(mean=1.0, sigma=0.5, size=n),
            "side": np.where(np.random.rand(n) > 0.5, "buy", "sell"),
        }
    )


def main() -> None:
    lob = build_synthetic_lob()
    lobf = compute_obi_features_from_lob(lob, depth=5, decay=0.08, tick_size=0.1)

    trades = build_synthetic_trades(lob["timestamp"])
    trf = compute_trade_imbalance(trades)

    merged = merge_lob_trade_features_asof(lobf, trf, tolerance="1s")

    cols = [
        "obi_norm",
        "obi_vel",
        "microprice_divergence_bps",
        "void_up_flag",
        "void_down_flag",
        "obi_buy_pressure",
        "obi_sell_pressure",
    ]
    print(merged[cols].describe().round(4).to_string())

    print("\nflags counts:")
    print("void_up_flag", int(merged["void_up_flag"].sum()))
    print("void_down_flag", int(merged["void_down_flag"].sum()))
    print("obi_buy_pressure", int(merged["obi_buy_pressure"].sum()))
    print("obi_sell_pressure", int(merged["obi_sell_pressure"].sum()))

    print("\ntrade imbalance:")
    print(merged[["tfi_raw", "tfi_norm"]].describe().round(4).to_string())

    print(f"\nOK {merged.shape}")


if __name__ == "__main__":
    main()
