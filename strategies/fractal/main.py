"""
Main Runner for Fractal Detection System
Loads Binance data and runs the orchestrator with API
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import json
from typing import Tuple
import os
from dotenv import load_dotenv

from orchestrator import FractalOrchestrator

load_dotenv()


class BinanceDataLoader:
    def __init__(self, symbol: str = "BTC/USDT"):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        self.symbol = symbol

    def fetch_ohlcv(self, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Charge les données OHLCV depuis Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').reset_index()
            return df
        except Exception as e:
            print(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()

    def load_live_timeframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Charge une fenêtre réduite pour la détection live :
        - M15 : 4 jours (J-2 → J+2) soit ~384 bougies — couvre J courant + J-1 + marge
        - Daily : 14 jours (2 semaines) — suffisant pour Inside Day J et J-1
        - Weekly : 8 semaines — suffit pour Inside Week
        """
        print(f"Loading live {self.symbol} data from Binance...")

        df_m15 = self.fetch_ohlcv('15m', limit=384)   # ~4 jours
        print(f"✓ M15: {len(df_m15)} candles")

        df_daily = self.fetch_ohlcv('1d', limit=14)   # 2 semaines
        print(f"✓ Daily: {len(df_daily)} candles")

        df_weekly = self.fetch_ohlcv('1w', limit=8)   # 8 semaines
        print(f"✓ Weekly: {len(df_weekly)} candles")

        return df_m15, df_daily, df_weekly

    def load_all_timeframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Charge l'historique complet pour backtesting (non utilisé en live)."""
        print(f"Loading full history {self.symbol} data from Binance...")
        df_m15 = self.fetch_ohlcv('15m', limit=672)
        df_daily = self.fetch_ohlcv('1d', limit=365)
        df_weekly = self.fetch_ohlcv('1w', limit=260)
        return df_m15, df_daily, df_weekly


async def run_detection_cycle(orchestrator: FractalOrchestrator,
                              loader: BinanceDataLoader,
                              active_setups: list = None):
    """Exécute un cycle complet de détection"""
    df_m15, df_daily, df_weekly = loader.load_live_timeframes()

    if df_m15.empty or df_daily.empty or df_weekly.empty:
        print("Error: Could not load data from Binance")
        return

    print(f"\n{'='*50}")
    print(f"Detection cycle: {datetime.utcnow().isoformat()}")
    print(f"{'='*50}")

    signals = await orchestrator.detect_and_notify(
        df_m15=df_m15,
        daily=df_daily,
        weekly=df_weekly,
        active_setups=active_setups or ['STRICT', 'MODÉRÉ', 'FRÉQUENT']
    )

    summary = orchestrator.get_signals_summary()
    print(f"\n📊 Detection Summary:")
    print(f"   Total signals: {summary.get('total', 0)}")
    for setup, count in summary.get('by_setup', {}).items():
        print(f"   [{setup}]: {count}")

    return signals


async def start_api_server():
    """Démarre le serveur API FastAPI"""
    import uvicorn
    config = uvicorn.Config(
        app="api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Point d'entrée principal"""
    # Configuration
    SYMBOL = os.getenv('BINANCE_SYMBOL', 'BTC/USDT')
    DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK')
    DETECTION_INTERVAL = int(os.getenv('DETECTION_INTERVAL', '3600'))  # 1 heure par défaut
    ACTIVE_SETUPS = os.getenv('ACTIVE_SETUPS', 'STRICT,MODÉRÉ,FRÉQUENT').split(',')

    print("🚀 Starting Fractal Detection System")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Active setups: {ACTIVE_SETUPS}")
    print(f"   Detection interval: {DETECTION_INTERVAL}s")

    # Initialiser l'orchestrator
    orchestrator = FractalOrchestrator(discord_webhook_url=DISCORD_WEBHOOK)

    # Initialiser le chargeur de données
    loader = BinanceDataLoader(symbol=SYMBOL)

    # Tâches asynchrones
    tasks = []

    # Tâche 1: Boucle de détection périodique
    async def detection_loop():
        while True:
            try:
                await run_detection_cycle(
                    orchestrator=orchestrator,
                    loader=loader,
                    active_setups=ACTIVE_SETUPS
                )
            except Exception as e:
                print(f"Error in detection cycle: {e}")

            await asyncio.sleep(DETECTION_INTERVAL)

    # Tâche 2: Serveur API (optionnel - commenter pour désactiver)
    # tasks.append(start_api_server())
    tasks.append(detection_loop())

    # Exécuter toutes les tâches
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
