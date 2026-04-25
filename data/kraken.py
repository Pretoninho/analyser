"""
data/kraken.py — Collecte des prix OHLCV depuis Kraken.

Kraken est choisi car :
- Pas de restriction géographique (contrairement à Binance)
- API publique sans clé pour les données de marché
- OHLCV complet : open, high, low, close, vwap, volume
- Limite généreuse : 720 bougies par appel
"""

import requests
import pandas as pd
from datetime import datetime
from config import ASSETS, SOURCES, FETCH


# Mapping timeframe → intervalle Kraken (en minutes)
KRAKEN_INTERVALS = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
    "1w":  10080,
}


def _fetch_raw(symbol: str, interval_min: int, since: int = None) -> list:
    """
    Appel brut à l'API Kraken OHLC.
    Retourne la liste de bougies brutes.
    """
    params = {
        "pair":     symbol,
        "interval": interval_min,
    }
    if since:
        params["since"] = since

    url = f"{SOURCES['kraken_base_url']}/OHLC"

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[kraken] Erreur réseau : {e}")
        return []

    if data.get("error"):
        print(f"[kraken] Erreur API : {data['error']}")
        return []

    # La clé du résultat est le nom de la paire (ex: XXBTZUSD)
    result = data.get("result", {})
    pair_key = next((k for k in result if k != "last"), None)
    if not pair_key:
        print("[kraken] Aucune donnée dans la réponse.")
        return []

    return result[pair_key]


def fetch_ohlcv(asset: str, timeframe: str, limit: int = None) -> pd.DataFrame:
    """
    Récupère les bougies OHLCV depuis Kraken pour un actif donné.

    Args:
        asset:     Clé de l'actif dans config.ASSETS (ex: "BTC")
        timeframe: Timeframe souhaité (ex: "1h", "1d")
        limit:     Nombre de bougies à conserver (None = toutes)

    Returns:
        DataFrame avec colonnes : timestamp, open, high, low, close, volume, vwap
    """
    if asset not in ASSETS:
        raise ValueError(f"Actif inconnu : {asset}. Disponibles : {list(ASSETS.keys())}")

    if timeframe not in KRAKEN_INTERVALS:
        raise ValueError(f"Timeframe inconnu : {timeframe}. Disponibles : {list(KRAKEN_INTERVALS.keys())}")

    symbol       = ASSETS[asset]["kraken"]
    interval_min = KRAKEN_INTERVALS[timeframe]
    limit        = limit or FETCH["ohlcv_limit"]

    print(f"[kraken] Fetch {asset} {timeframe} ({symbol})...")

    raw = _fetch_raw(symbol, interval_min)
    if not raw:
        return pd.DataFrame()

    # Kraken retourne : [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
    ])

    # Typage
    df["timestamp"] = df["timestamp"].astype(int)
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)

    # Exclure la dernière bougie (potentiellement incomplète)
    df = df.iloc[:-1]

    # Garder les N dernières
    if limit and len(df) > limit:
        df = df.tail(limit)

    df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]].reset_index(drop=True)

    print(f"[kraken] {len(df)} bougies recuperees : "
          f"{datetime.utcfromtimestamp(df['timestamp'].iloc[0]).strftime('%Y-%m-%d %H:%M')} -> "
          f"{datetime.utcfromtimestamp(df['timestamp'].iloc[-1]).strftime('%Y-%m-%d %H:%M')} UTC")

    return df


