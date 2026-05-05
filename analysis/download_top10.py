"""
download_top10.py — Télécharge l'historique 1m des top 10 paires USDT-M Binance.

Chaque paire → data_binance/<symbol>_1m.parquet
Les fichiers mensuels CSV bruts sont mis en cache dans data_binance/raw/

Usage :
    python download_top10.py                    # top 10 depuis 2020-01
    python download_top10.py --start 2022 1     # depuis janvier 2022
    python download_top10.py --symbols SOLUSDT AVAXUSDT  # paires spécifiques
    python download_top10.py --skip BTCUSDT ETHUSDT      # sauter BTC et ETH déjà téléchargés
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Top 10 paires USDT-M Binance Futures (mars 2026)
TOP10 = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   nargs=2, type=int, metavar=("YEAR", "MONTH"), default=[2020, 1])
    parser.add_argument("--end",     nargs=2, type=int, metavar=("YEAR", "MONTH"), default=None)
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Paires spécifiques (défaut: top 10)")
    parser.add_argument("--skip",    nargs="+", default=[],
                        help="Paires à sauter (ex: BTCUSDT si déjà téléchargé)")
    args = parser.parse_args()

    symbols = [s.upper() for s in (args.symbols or TOP10)]
    skip    = {s.upper() for s in args.skip}
    start   = tuple(args.start)
    end     = tuple(args.end) if args.end else None

    targets = [s for s in symbols if s not in skip]

    print(f"\n{'='*60}")
    print(f"  BATCH DOWNLOAD — {len(targets)} paire(s)")
    print(f"  Période : {start[0]}-{start[1]:02d} → {end or 'now'}")
    print(f"  Paires : {', '.join(targets)}")
    if skip:
        print(f"  Ignorées : {', '.join(skip)}")
    print(f"{'='*60}\n")

    from data.binance import download_binance_1m

    results = {}
    for i, sym in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] ── {sym} ──")
        t0 = time.time()
        try:
            path = download_binance_1m(start=start, end=end, symbol=sym)
            size_mb = path.stat().st_size / 1_048_576
            elapsed = time.time() - t0
            results[sym] = f"OK  ({size_mb:.0f} MB, {elapsed:.0f}s)"
        except Exception as e:
            results[sym] = f"ERREUR: {e}"
            print(f"  [!] {sym} échoué: {e}")

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ")
    print(f"{'='*60}")
    for sym, status in results.items():
        print(f"  {sym:<12} {status}")
    print()


if __name__ == "__main__":
    main()
