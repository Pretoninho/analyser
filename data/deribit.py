"""
data/deribit.py — OHLCV, Funding Rate et Open Interest depuis Deribit BTC-PERPETUAL.
"""

import time
import requests
import pandas as pd
from config import SOURCES

# Deribit retourne au maximum ~5 000 bougies par appel
_DERIBIT_MAX_CANDLES = 5000

DERIBIT_INSTRUMENTS = {
    "BTC": "BTC-PERPETUAL",
}

DERIBIT_RESOLUTIONS = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "4h":  "240",
    "1d":  "1D",
}

_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def fetch_ohlcv(asset: str, timeframe: str = "1m", limit: int = 720) -> pd.DataFrame:
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    resolution = DERIBIT_RESOLUTIONS.get(timeframe)
    if not instrument or not resolution:
        print(f"[deribit] Actif ou timeframe inconnu : {asset} {timeframe}")
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - limit * _INTERVAL_SECONDS[timeframe] * 1000

    url = f"{SOURCES['deribit_base_url']}/get_tradingview_chart_data"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
        "resolution":      resolution,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau OHLCV : {e}")
        return pd.DataFrame()

    result = data.get("result", {})
    if result.get("status") != "ok" or not result.get("ticks"):
        print(f"[deribit] Reponse OHLCV invalide : {data.get('error', result.get('status'))}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "ts":     [t // 1000 for t in result["ticks"]],
        "open":   result["open"],
        "high":   result["high"],
        "low":    result["low"],
        "close":  result["close"],
        "volume": result["volume"],
    })

    df = df.iloc[:-1]  # exclure la derniere bougie (potentiellement incomplete)

    if limit and len(df) > limit:
        df = df.tail(limit)

    df = df.reset_index(drop=True)

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] OHLCV {asset} {timeframe} : {len(df)} bougies {first} -> {last} UTC")
    return df


def fetch_funding_rate(asset: str, limit: int = 200) -> pd.DataFrame:
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        print(f"[deribit] Actif inconnu : {asset}")
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - limit * 8 * 3600 * 1000  # funding toutes les 8h

    url = f"{SOURCES['deribit_base_url']}/get_funding_rate_history"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau funding : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API funding : {data['error']}")
        return pd.DataFrame()

    items = data.get("result", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["ts"]           = df["timestamp"].astype(int) // 1000
    df["funding_rate"] = df["interest_8h"].astype(float)
    df = df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] Funding {asset} : {len(df)} points {first} -> {last} UTC")
    return df


def fetch_ohlcv_historical(asset: str, timeframe: str = "1m", days: int = 30) -> pd.DataFrame:
    """
    Recupere l'historique complet sur N jours en paginant les appels Deribit.

    Deribit retourne ~5 000 bougies max par appel.
    Pour 30 jours de 1min : 43 200 bougies -> ~9 appels pagines.

    Args:
        asset     : "BTC"
        timeframe : "1m", "5m", "1h", etc.
        days      : nombre de jours a remonter (defaut 30)

    Returns:
        DataFrame consolide : ts, open, high, low, close, volume (trié ASC, sans doublons)
    """
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    resolution = DERIBIT_RESOLUTIONS.get(timeframe)
    if not instrument or not resolution:
        print(f"[deribit] Actif ou timeframe inconnu : {asset} {timeframe}")
        return pd.DataFrame()

    interval_ms  = _INTERVAL_SECONDS[timeframe] * 1000
    chunk_ms     = _DERIBIT_MAX_CANDLES * interval_ms
    end_ms       = int(time.time() * 1000)
    start_ms     = end_ms - days * 86400 * 1000

    url    = f"{SOURCES['deribit_base_url']}/get_tradingview_chart_data"
    chunks = []
    cursor = start_ms

    n_chunks = max(1, (end_ms - start_ms) // chunk_ms)
    print(f"[deribit] Backfill {asset} {timeframe} sur {days} jours (~{n_chunks} appels)...")

    try:
        while cursor < end_ms:
            chunk_end = min(cursor + chunk_ms, end_ms)
            done_pct  = (cursor - start_ms) / (end_ms - start_ms) * 100
            print(f"  {done_pct:5.1f}% — {pd.to_datetime(cursor, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')} UTC", end="\r")

            params = {
                "instrument_name": instrument,
                "start_timestamp": cursor,
                "end_timestamp":   chunk_end,
                "resolution":      resolution,
            }
            try:
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                print(f"\n[deribit] Erreur reseau (chunk) : {e}")
                time.sleep(2)
                continue

            result = data.get("result", {})
            if result.get("status") == "ok" and result.get("ticks"):
                chunks.append(pd.DataFrame({
                    "ts":     [t // 1000 for t in result["ticks"]],
                    "open":   result["open"],
                    "high":   result["high"],
                    "low":    result["low"],
                    "close":  result["close"],
                    "volume": result["volume"],
                }))

            cursor = chunk_end
            time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n[deribit] Interruption — {len(chunks)} chunks recuperes, sauvegarde partielle...")

    if not chunks:
        print("[deribit] Aucune donnee historique recuperee.")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df.iloc[:-1]  # exclure la derniere bougie incomplete

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] Historique {asset} {timeframe} : {len(df):,} bougies {first} -> {last} UTC")
    return df


def fetch_funding_rate_historical(asset: str, days: int = 30) -> pd.DataFrame:
    """Recupere l'historique du funding rate sur N jours (intervalles 8h)."""
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    url = f"{SOURCES['deribit_base_url']}/get_funding_rate_history"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau funding historique : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API funding : {data['error']}")
        return pd.DataFrame()

    items = data.get("result", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["ts"]           = df["timestamp"].astype(int) // 1000
    df["funding_rate"] = df["interest_8h"].astype(float)
    df = df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    print(f"[deribit] Funding historique {asset} : {len(df)} points sur {days} jours")
    return df


def fetch_open_interest(asset: str) -> pd.DataFrame:
    """
    Snapshot courant de l'OI depuis Deribit (pas d'historique REST disponible).
    Retourne un DataFrame avec un seul point (ts, open_interest).
    Le collector forward-fille cette valeur sur toute la grille 1min.
    """
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        print(f"[deribit] Actif inconnu : {asset}")
        return pd.DataFrame()

    url = f"{SOURCES['deribit_base_url']}/get_book_summary_by_instrument"
    params = {"instrument_name": instrument}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau OI : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API OI : {data['error']}")
        return pd.DataFrame()

    result = data.get("result", [])
    if not result:
        return pd.DataFrame()

    item = result[0]
    oi   = float(item.get("open_interest", 0))
    ts   = int(item.get("creation_timestamp", time.time() * 1000)) // 1000

    print(f"[deribit] OI {asset} : {oi:,.0f} BTC @ {pd.to_datetime(ts, unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC")
    return pd.DataFrame([{"ts": ts, "open_interest": oi}])
