from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.binance import load_binance_1m
from engine.microstructure.orderflow import compute_orderflow_features, ofi_regime


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate orderflow features on Binance 1m data.")
    parser.add_argument("--rows", type=int, default=44_640, help="Number of last rows to test.")
    args = parser.parse_args()

    df = load_binance_1m()
    df = df.iloc[-args.rows :].copy().reset_index(drop=True)

    df = compute_orderflow_features(df)
    df["ofi_regime"] = ofi_regime(df)

    cols = [
        "close",
        "ofi_norm",
        "beta_proxy",
        "hawkes_lambda",
        "depth_proxy",
        "sigma_ewma",
        "lri",
        "ofi_regime",
    ]

    print(df[cols].tail(8).to_string())
    print("\n--- Stats ---")
    print(df[cols].describe().round(5).to_string())
    print("\nofi_regime counts:")
    print(df["ofi_regime"].value_counts().sort_index())
    print(f"\nOK - shape: {df.shape}")


if __name__ == "__main__":
    main()
