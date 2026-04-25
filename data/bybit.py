"""
data/bybit.py — Open Interest et Funding Rate depuis Bybit Perpetuals.

Bybit est choisi pour :
- API publique sans clé, sans restriction géographique
- Open Interest historique (résolution minimale : 5min)
- Funding Rate historique (intervalles de 8h)
- Données sur BTCUSDT perpetual (corrélées au spot Kraken)
"""

import requests
import pandas as pd
from config import SOURCES

BYBIT_SYMBOLS = {
    "BTC": "BTCUSDT",
}


def fetch_open_interest(asset: str, limit: int = 200) -> pd.DataFrame:
    """
    Récupère l'historique de l'Open Interest depuis Bybit.

    Résolution : 5min (minimum disponible sur Bybit V5).
    Retourne un DataFrame avec colonnes : ts (unix seconds), open_interest (en BTC).
    """
    symbol = BYBIT_SYMBOLS.get(asset)
    if not symbol:
        print(f"[bybit] Actif inconnu : {asset}")
        return pd.DataFrame()

    url = f"{SOURCES['bybit_base_url']}/open-interest"
    params = {
        "category":     "linear",
        "symbol":       symbol,
        "intervalTime": "5min",
        "limit":        limit,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[bybit] Erreur réseau OI : {e}")
        return pd.DataFrame()

    if data.get("retCode") != 0:
        print(f"[bybit] Erreur API OI : {data.get('retMsg')}")
        return pd.DataFrame()

    items = data.get("result", {}).get("list", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items, columns=["openInterest", "timestamp"])
    df["ts"]            = df["timestamp"].astype(int) // 1000   # ms -> secondes
    df["open_interest"] = df["openInterest"].astype(float)
    df = df[["ts", "open_interest"]].sort_values("ts").reset_index(drop=True)

    print(f"[bybit] OI : {len(df)} points — {asset} "
          f"({pd.to_datetime(df['ts'].iloc[0], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} -> "
          f"{pd.to_datetime(df['ts'].iloc[-1], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC)")
    return df


def fetch_funding_rate(asset: str, limit: int = 200) -> pd.DataFrame:
    """
    Récupère l'historique du Funding Rate depuis Bybit.

    Intervalles de 8h (funding payments : 00:00, 08:00, 16:00 UTC).
    Retourne un DataFrame avec colonnes : ts (unix seconds), funding_rate (float, ex: 0.0001 = 0.01%).
    """
    symbol = BYBIT_SYMBOLS.get(asset)
    if not symbol:
        print(f"[bybit] Actif inconnu : {asset}")
        return pd.DataFrame()

    url = f"{SOURCES['bybit_base_url']}/funding/history"
    params = {
        "category": "linear",
        "symbol":   symbol,
        "limit":    limit,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[bybit] Erreur réseau funding : {e}")
        return pd.DataFrame()

    if data.get("retCode") != 0:
        print(f"[bybit] Erreur API funding : {data.get('retMsg')}")
        return pd.DataFrame()

    items = data.get("result", {}).get("list", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["ts"]           = df["fundingRateTimestamp"].astype(int) // 1000  # ms -> secondes
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)

    print(f"[bybit] Funding rate : {len(df)} points — {asset} "
          f"({pd.to_datetime(df['ts'].iloc[0], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} -> "
          f"{pd.to_datetime(df['ts'].iloc[-1], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC)")
    return df
