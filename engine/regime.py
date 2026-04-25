"""
engine/regime.py — Détection et classification des régimes de marché.

Régimes :
    LOW   → volatilité sous le percentile 33%
    MID   → volatilité entre percentile 33% et 66%
    HIGH  → volatilité au-dessus du percentile 66%

Le sigma de référence pour le z-score est ajusté selon le régime.
"""

import numpy as np
import pandas as pd
from config import ENGINE


def classify_regime(df: pd.DataFrame,
                    vol_col: str = "vol_short",
                    low_pct: int = None,
                    high_pct: int = None) -> pd.DataFrame:
    """
    Classifie chaque période en régime LOW / MID / HIGH.

    Le percentile est calculé sur toute la série disponible —
    c'est la distribution historique qui définit les seuils.

    Args:
        df:      DataFrame avec colonne de volatilité
        vol_col: Colonne de volatilité à utiliser
        low_pct: Percentile bas (défaut : config ENGINE)
        high_pct:Percentile haut (défaut : config ENGINE)

    Returns:
        DataFrame avec colonnes : regime, regime_score, vol_pct_rank
    """
    df = df.copy()
    low_pct  = low_pct  or ENGINE["regime_low_pct"]
    high_pct = high_pct or ENGINE["regime_high_pct"]

    if vol_col not in df.columns:
        raise ValueError(f"Colonne '{vol_col}' absente du DataFrame.")

    vol = df[vol_col].dropna()
    if vol.empty:
        df["regime"]       = None
        df["regime_score"] = None
        df["vol_pct_rank"] = None
        return df

    p_low  = np.percentile(vol, low_pct)
    p_high = np.percentile(vol, high_pct)

    def _classify(v):
        if pd.isna(v):
            return None
        if v <= p_low:
            return "LOW"
        if v <= p_high:
            return "MID"
        return "HIGH"

    df["regime"] = df[vol_col].apply(_classify)

    # Score numérique : 0=LOW, 1=MID, 2=HIGH (utile pour les graphiques)
    score_map = {"LOW": 0, "MID": 1, "HIGH": 2}
    df["regime_score"] = df["regime"].map(score_map)

    # Rang percentile de la vol actuelle (0-100)
    df["vol_pct_rank"] = df[vol_col].rank(pct=True) * 100

    return df


def get_regime_stats(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques de rendement par régime.

    Utile pour l'analyse d'edge :
    - Rendement moyen par régime
    - Win rate par régime
    - Z-score moyen par régime

    Returns:
        Dict avec stats par régime : LOW, MID, HIGH
    """
    if "regime" not in df.columns or "log_return" not in df.columns:
        return {}

    stats = {}
    for regime in ["LOW", "MID", "HIGH"]:
        subset = df[df["regime"] == regime]["log_return"].dropna()
        if subset.empty:
            stats[regime] = {}
            continue

        stats[regime] = {
            "count":       len(subset),
            "mean_return": round(subset.mean() * 100, 4),
            "median_return": round(subset.median() * 100, 4),
            "std":         round(subset.std() * 100, 4),
            "win_rate":    round((subset > 0).mean() * 100, 2),
            "best":        round(subset.max() * 100, 4),
            "worst":       round(subset.min() * 100, 4),
            "sharpe_approx": round(
                subset.mean() / subset.std() * (252 ** 0.5)
                if subset.std() > 0 else 0, 2
            ),
        }

    return stats


def get_current_regime(df: pd.DataFrame) -> dict:
    """
    Retourne le régime actuel (dernière ligne du DataFrame).

    Returns:
        Dict avec regime, vol_actuelle, vol_pct_rank, zscore, seuils
    """
    if df.empty or "regime" not in df.columns:
        return {}

    last = df.dropna(subset=["regime"]).iloc[-1]

    vol_col = "vol_short"
    vol_series = df[vol_col].dropna()
    p_low  = np.percentile(vol_series, ENGINE["regime_low_pct"])
    p_high = np.percentile(vol_series, ENGINE["regime_high_pct"])

    return {
        "regime":       last.get("regime"),
        "regime_score": last.get("regime_score"),
        "vol_current":  round(float(last.get(vol_col, 0) or 0) * 100, 4),
        "vol_annualized": round(float(last.get("vol_annualized", 0) or 0), 2),
        "vol_pct_rank": round(float(last.get("vol_pct_rank", 0) or 0), 1),
        "zscore":       round(float(last.get("zscore", 0) or 0), 3),
        "atr_pct":      round(float(last.get("atr_pct", 0) or 0), 4),
        "threshold_low":  round(p_low * 100, 4),
        "threshold_high": round(p_high * 100, 4),
        "price":        round(float(last.get("close", 0)), 2),
        "timestamp":    last.get("timestamp"),
    }