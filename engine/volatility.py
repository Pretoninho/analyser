"""
engine/volatility.py — Calcul des indicateurs de volatilité et de structure.

Indicateurs :
    - Rendements logarithmiques
    - Volatilité réalisée (rolling)
    - Volatilité annualisée
    - ATR (Average True Range)
    - Z-score des rendements
    - Rendement sur N périodes
"""

import numpy as np
import pandas as pd
from config import ENGINE


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les rendements logarithmiques sur la colonne 'close'.
    log_return = ln(close_t / close_t-1)
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_realized_vol(df: pd.DataFrame,
                          window_short: int = None,
                          window_long: int = None) -> pd.DataFrame:
    """
    Calcule la volatilité réalisée rolling (écart-type des log returns).

    - vol_short : rolling sur window_short périodes
    - vol_long  : rolling sur window_long périodes
    - vol_annualized : annualisée depuis vol_short (×√252 pour daily, ×√8760 pour hourly)
    """
    df = df.copy()
    w_short = window_short or ENGINE["vol_window_short"]
    w_long  = window_long  or ENGINE["vol_window_long"]

    if "log_return" not in df.columns:
        df = compute_log_returns(df)

    df["vol_short"] = df["log_return"].rolling(w_short).std()
    df["vol_long"]  = df["log_return"].rolling(w_long).std()

    # Facteur d'annualisation selon la fréquence détectée
    if len(df) > 1:
        median_diff = df["timestamp"].diff().median()
        if hasattr(median_diff, "total_seconds"):
            seconds = median_diff.total_seconds()
        else:
            seconds = float(median_diff) / 1e9  # nanoseconds → seconds
        periods_per_year = (365 * 24 * 3600) / seconds if seconds > 0 else 252
    else:
        periods_per_year = 252

    df["vol_annualized"] = df["vol_short"] * np.sqrt(periods_per_year) * 100  # en %
    df["vol_realized"]   = df["vol_short"]

    return df


def compute_atr(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """
    Calcule l'ATR (Average True Range).
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    df = df.copy()
    period = period or ENGINE["atr_period"]

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"]     = tr.rolling(period).mean()
    df["atr_pct"] = (df["atr"] / df["close"]) * 100  # ATR en % du prix

    return df


def compute_zscore(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Calcule le z-score des log returns sur une fenêtre glissante.
    z = (rendement - mu) / sigma
    Indique dans quel tier de sigma se situe le mouvement actuel.
    """
    df = df.copy()
    window = window or ENGINE["zscore_window"]

    if "log_return" not in df.columns:
        df = compute_log_returns(df)

    rolling_mean = df["log_return"].rolling(window).mean()
    rolling_std  = df["log_return"].rolling(window).std()

    df["zscore"] = (df["log_return"] - rolling_mean) / rolling_std
    return df


def compute_vol_zscore(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Z-score de la volatilité réalisée courte (vol of vol).
    Détecte les spikes de volatilité anormaux.
    """
    df = df.copy()
    if "vol_short" not in df.columns:
        df = compute_realized_vol(df)

    mu  = df["vol_short"].rolling(window).mean()
    sig = df["vol_short"].rolling(window).std()
    df["vol_zscore"] = (df["vol_short"] - mu) / sig
    return df


def compute_volume_zscore(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Z-score du volume de trading.
    Un pic de volume accompagné d'un spike de vol signale une liquidation.
    """
    df = df.copy()
    mu  = df["volume"].rolling(window).mean()
    sig = df["volume"].rolling(window).std()
    df["volume_zscore"] = (df["volume"] - mu) / sig
    return df


def compute_price_zscore(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Z-score du niveau de prix (style Bollinger Band).
    z = (close - MA_N) / std_N
    Mesure si le prix est anormalement haut ou bas par rapport à sa moyenne.
    Positif = sur-acheté, Négatif = sur-vendu.
    """
    df = df.copy()
    mu  = df["close"].rolling(window).mean()
    sig = df["close"].rolling(window).std()
    df["price_zscore"] = (df["close"] - mu) / sig
    return df


def compute_returns_n(df: pd.DataFrame,
                      periods: list = [1, 5, 20]) -> pd.DataFrame:
    """
    Calcule les rendements sur N périodes (en %).
    Ex: periods=[1,5,20] → return_1, return_5, return_20
    """
    df = df.copy()
    for n in periods:
        df[f"return_{n}"] = df["close"].pct_change(n) * 100
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : applique tous les indicateurs dans l'ordre.
    Point d'entrée principal pour le moteur.
    """
    df = compute_log_returns(df)
    df = compute_realized_vol(df)
    df = compute_atr(df)
    df = compute_zscore(df)
    df = compute_vol_zscore(df)
    df = compute_volume_zscore(df)
    df = compute_price_zscore(df)
    df = compute_returns_n(df)
    return df