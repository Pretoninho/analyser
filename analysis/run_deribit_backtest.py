from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from analysis.deribit_futures.backtest import BacktestConfig, run_edge_backtest, _print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Deribit futures edge backtest.")
    parser.add_argument("--asset",     type=str,   default="BTC")
    parser.add_argument("--timeframe", type=str,   default="1h")
    parser.add_argument("--days",      type=int,   default=90)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output",    type=str,   default="db/deribit_backtest.csv")
    args = parser.parse_args()

    cfg = BacktestConfig(
        asset=args.asset,
        timeframe=args.timeframe,
        days=args.days,
        threshold=args.threshold,
    )
    results_df, context = run_edge_backtest(cfg)
    _print_results(results_df, context)

    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out, index=False)
        print(f"\nSaved backtest results: {out}")


if __name__ == "__main__":
    main()
