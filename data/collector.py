"""
data/collector.py — Collecte et fusion des donnees de marche BTC 1min.

Sources :
    OHLCV      -> Deribit BTC-PERPETUAL (resolution 1min, jusqu'a 720 bougies)
    OI         -> Deribit BTC-PERPETUAL (snapshot courant, forward-fille sur grille 1min)
    Funding    -> Deribit BTC-PERPETUAL (intervalles 8h, forward-fille sur grille 1min)

Sortie : table market_1m dans prices.db
"""

import pandas as pd
from data.deribit  import (
    fetch_ohlcv, fetch_open_interest, fetch_funding_rate,
    fetch_ohlcv_historical, fetch_funding_rate_historical,
)
from data.storage  import save_market_1m, load_market_1m


def collect_btc_1m(limit: int = 720) -> pd.DataFrame:
    """
    Collecte, fusionne et persiste les donnees BTC 1min.

    Args:
        limit : nombre de bougies 1min a recuperer depuis Deribit (max 720)

    Returns:
        DataFrame : ts, open, high, low, close, volume, open_interest, funding_rate, timestamp (UTC)
    """
    ohlcv = fetch_ohlcv("BTC", "1m", limit=limit)
    if ohlcv.empty:
        print("[collector] Echec : aucune donnee OHLCV Deribit.")
        return pd.DataFrame()

    df = ohlcv.set_index("ts")

    # OI : snapshot courant forward-fille sur toute la grille
    oi = fetch_open_interest("BTC")
    if not oi.empty:
        oi_series = oi.set_index("ts")["open_interest"]
        df["open_interest"] = oi_series.reindex(df.index, method="ffill")
        if df["open_interest"].isna().all():
            df["open_interest"] = float(oi["open_interest"].iloc[-1])
    else:
        print("[collector] OI non disponible.")
        df["open_interest"] = None

    # Funding : intervalles 8h forward-filles sur grille 1min
    fr = fetch_funding_rate("BTC", limit=200)
    if not fr.empty:
        fr_series = fr.set_index("ts")["funding_rate"]
        df["funding_rate"] = fr_series.reindex(df.index, method="ffill")
    else:
        print("[collector] Funding rate non disponible.")
        df["funding_rate"] = None

    df = df.reset_index()

    save_market_1m("BTC", df)

    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    _print_summary(df)
    return df


def backfill_btc_1m(days: int = 30) -> pd.DataFrame:
    """
    Remplit la DB avec l'historique BTC 1min sur N jours depuis Deribit.

    Strategie :
        - OHLCV : pagination automatique (~9 appels pour 30 jours)
        - Funding : historique complet sur la meme periode
        - OI : snapshot courant forward-fille (pas d'historique REST Deribit)
        - INSERT OR IGNORE : safe, ne duplique pas les bougies existantes

    Args:
        days : nombre de jours a remonter (defaut 30, max ~90 en pratique)

    Returns:
        DataFrame complet insere en DB
    """
    print(f"\n[collector] Backfill {days} jours de donnees BTC 1min...")

    ohlcv = fetch_ohlcv_historical("BTC", "1m", days=days)
    if ohlcv.empty:
        print("[collector] Echec backfill : aucune donnee OHLCV.")
        return pd.DataFrame()

    df = ohlcv.set_index("ts")

    # Funding historique
    fr = fetch_funding_rate_historical("BTC", days=days)
    if not fr.empty:
        fr_series = fr.set_index("ts")["funding_rate"]
        df["funding_rate"] = fr_series.reindex(df.index, method="ffill")
    else:
        df["funding_rate"] = None

    # OI : snapshot courant seulement (forward-fille sur tout l'historique)
    oi = fetch_open_interest("BTC")
    if not oi.empty:
        df["open_interest"] = float(oi["open_interest"].iloc[-1])
    else:
        df["open_interest"] = None

    df = df.reset_index()
    save_market_1m("BTC", df)

    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    print(f"[collector] Backfill termine : {len(df):,} bougies inserees en DB.\n")
    return df


def load_latest_btc_1m(limit: int = 500) -> pd.DataFrame:
    return load_market_1m("BTC", limit=limit)


def _print_summary(df: pd.DataFrame):
    first   = df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
    last    = df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    oi_ok   = df["open_interest"].notna().sum()
    fr_ok   = df["funding_rate"].notna().sum()
    oi_last = df["open_interest"].iloc[-1]
    fr_last = df["funding_rate"].iloc[-1]
    oi_str  = f"{oi_last:,.0f} BTC" if pd.notna(oi_last) else "N/A"
    fr_str  = f"{fr_last:.6f}"      if pd.notna(fr_last) else "N/A"
    print(
        f"\n[collector] BTC 1min | {len(df)} bougies | {first} -> {last} UTC\n"
        f"  close   : ${df['close'].iloc[-1]:,.2f}\n"
        f"  OI      : {oi_str}  ({oi_ok}/{len(df)} bougies renseignees)\n"
        f"  funding : {fr_str}  ({fr_ok}/{len(df)} bougies renseignees)\n"
    )
