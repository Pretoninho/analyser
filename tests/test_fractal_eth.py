"""
test_fractal_eth.py — Test Fractal Detection sur ETH/USDT

Charge M15, Daily, Weekly depuis Binance via ccxt
et fait tourner les 3 détecteurs (STRICT, MODÉRÉ, FRÉQUENT).

Usage :
    python test_fractal_eth.py
    python test_fractal_eth.py --symbol ETH/USDT
    python test_fractal_eth.py --symbol BTC/USDT
"""

import sys
import asyncio
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies" / "fractal"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH/USDT", help="Symbole ccxt (ex: ETH/USDT, BTC/USDT)")
    args = parser.parse_args()

    symbol = args.symbol
    print(f"\n{'='*60}")
    print(f"  FRACTAL DETECTION TEST — {symbol}")
    print(f"{'='*60}\n")

    # ── Chargement des données ──────────────────────────────────
    print("[1/4] Chargement données Binance via ccxt...")
    from main import BinanceDataLoader
    loader = BinanceDataLoader(symbol=symbol)
    df_m15, df_daily, df_weekly = loader.load_all_timeframes()

    if df_m15.empty or df_daily.empty or df_weekly.empty:
        print("ERREUR: Impossible de charger les données.")
        sys.exit(1)

    print(f"  M15    : {len(df_m15)} bougies  "
          f"({df_m15['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_m15['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  Daily  : {len(df_daily)} bougies  "
          f"({df_daily['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_daily['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  Weekly : {len(df_weekly)} bougies  "
          f"({df_weekly['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_weekly['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")

    # ── Détecteurs individuels ──────────────────────────────────
    print(f"\n[2/4] Exécution des détecteurs...")

    from detector_strict import FractalDetectorStrict
    from detector_modere import FractalDetectorModere
    from detector_frequent import FractalDetectorFrequent

    strict   = FractalDetectorStrict()
    modere   = FractalDetectorModere()
    frequent = FractalDetectorFrequent()

    sig_strict   = strict.detect(df_m15, df_daily, df_weekly)
    sig_modere   = modere.detect(df_m15, df_daily)
    sig_frequent = frequent.detect(df_m15)

    print(f"  STRICT   : {len(sig_strict)} signal(s)")
    print(f"  MODÉRÉ   : {len(sig_modere)} signal(s)")
    print(f"  FRÉQUENT : {len(sig_frequent)} signal(s)")

    # ── Détail des signaux ──────────────────────────────────────
    all_signals = [
        ("STRICT",   sig_strict),
        ("MODÉRÉ",   sig_modere),
        ("FRÉQUENT", sig_frequent),
    ]

    total = sum(len(s) for _, s in all_signals)
    if total == 0:
        print("\n  Aucun signal détecté sur les données actuelles.")
        print("  (Normal si le marché ne montre pas de fractale ICT active)")
    else:
        print(f"\n[3/4] Détail des signaux ({total} total) :")
        for setup_name, signals in all_signals:
            if not signals:
                continue
            print(f"\n  ── {setup_name} ──")
            for sig in signals:
                ts   = sig.get("timestamp", "?")
                pat  = sig.get("pattern", "?")
                kz   = sig.get("kz", "?")
                px   = sig.get("entry_price", 0)
                conf = sig.get("confidence", 0)
                print(f"    [{ts}] {pat}  KZ={kz}  entry=${px:,.2f}  conf={conf:.3f}")

    # ── Orchestrateur complet (sans Discord) ───────────────────
    print(f"\n[4/4] Orchestrateur complet (sans Discord)...")

    async def run_orch():
        from orchestrator import FractalOrchestrator
        orch = FractalOrchestrator(discord_webhook_url=None)
        signals = await orch.detect_and_notify(
            df_m15=df_m15,
            daily=df_daily,
            weekly=df_weekly,
            active_setups=["STRICT", "MODÉRÉ", "FRÉQUENT"],
        )
        return signals, orch.get_signals_summary()

    orch_signals, summary = asyncio.run(run_orch())
    print(f"  Signaux émis via orchestrateur : {len(orch_signals)}")
    print(f"  Résumé : {summary}")

    print(f"\n{'='*60}")
    print(f"  TEST TERMINÉ — {symbol}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
