from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import EdgeBuildConfig, build_deribit_edge_frame


def _print_latest_summary(df: pd.DataFrame, context: dict) -> None:
    latest = df.iloc[-1]

    print("\n=== Deribit Futures Edge Analysis ===")
    print(f"asset={context['asset']} timeframe={context['timeframe']} bars={context['bars']} days={context['days']}")

    ts = latest.get("timestamp")
    close = latest.get("close")
    edge_total = latest.get("edge_total")
    funding_ann = latest.get("funding_annualized")
    rv = latest.get("realized_vol_annual")

    if pd.notna(ts):
        print(f"latest_ts={ts}")
    if pd.notna(close):
        print(f"close={float(close):,.2f}")
    if pd.notna(funding_ann):
        print(f"funding_annualized={float(funding_ann):+.4f}")
    if pd.notna(rv):
        print(f"realized_vol_annual={float(rv):.4f}")
    if pd.notna(edge_total):
        print(f"edge_total={float(edge_total):.4f}")

    print("\nTop edge components (latest bar):")
    edge_cols = [
        "edge_funding_reversion",
        "edge_carry_momentum",
        "edge_carry_stress",
        "edge_mark_dislocation",
        "edge_options_vol_premium",
        "edge_skew_panic",
        "edge_term_structure_kink",
    ]
    parts = []
    for col in edge_cols:
        val = float(latest.get(col, 0.0) or 0.0)
        parts.append((val, col))
    for value, name in sorted(parts, reverse=True):
        print(f"  {name:<28} {value:.4f}")


def run_deribit_futures_analysis(
    asset: str = "BTC",
    timeframe: str = "1h",
    days: int = 60,
    output_csv: str | None = "db/deribit_futures_edges.csv",
) -> tuple[pd.DataFrame, dict]:
    cfg = EdgeBuildConfig(asset=asset.upper(), timeframe=timeframe, days=days)
    df, context = build_deribit_edge_frame(cfg)

    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\nSaved edge frame: {path}")

    _print_latest_summary(df, context)
    return df, context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Deribit futures edge analysis.")
    parser.add_argument("--asset", type=str, default="BTC", help="Asset symbol supported by Deribit module.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d).")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days to fetch.")
    parser.add_argument("--output", type=str, default="db/deribit_futures_edges.csv", help="CSV output path.")
    args = parser.parse_args()

    run_deribit_futures_analysis(
        asset=args.asset,
        timeframe=args.timeframe,
        days=args.days,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
