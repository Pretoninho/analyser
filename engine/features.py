"""
engine/features.py — Feature engineering pour l'agent Deep RL.

9 features normalisées par bougie 5min :
  0. close_pct   : rendement bougie        [-1, 1]
  1. body_pct    : direction + force body  [-1, 1]
  2. upper_wick  : mèche supérieure        [ 0, 1]
  3. lower_wick  : mèche inférieure        [ 0, 1]
  4. vol_ratio   : volume relatif          [ 0, 1]
  5. delta       : pression achat approx   [ 0, 1]
  6. funding     : funding rate normalisé  [-1, 1]
  7. session     : session FOREX encodée   [ 0, 1]
  8. position    : position courante       {-1, 0, 1}  ← ajouté par l'env

Les 8 premières sont précalculées ; la 9e (position) est dynamique.
"""

import numpy as np
import pandas as pd
from engine.state import aggregate_5m, label_session

# ── Constantes ─────────────────────────────────────────────────

N_FEATURES  = 9          # 8 marché + 1 position
SEQ_LEN     = 24         # 2h de bougies 5min (référence, utilisé si besoin)

_CLOSE_CLIP = 0.03       # ±3% max par bougie
_VOL_CLIP   = 5.0        # volume relatif max 5×
_FUND_CLIP  = 0.005      # ±0.5% funding rate

FEATURE_COLS = [
    "f_close_pct",
    "f_body_pct",
    "f_upper_wick",
    "f_lower_wick",
    "f_vol_ratio",
    "f_delta",
    "f_funding",
    "f_session",
]   # 8 colonnes — position ajoutée dynamiquement


# ── Pipeline ───────────────────────────────────────────────────

def compute_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège 1min → 5min et calcule les 8 features marché normalisées.

    Compatible avec les données Deribit (funding_rate présent) et Binance
    (taker_buy_vol présent, funding_rate = 0).

    Args:
        df_1m : DataFrame 1min avec colonnes open/high/low/close/volume
                + optionnellement taker_buy_vol et funding_rate.

    Returns:
        DataFrame 5min avec colonnes ts, timestamp, session, FEATURE_COLS.
    """
    df = aggregate_5m(df_1m)
    df["session"] = label_session(df["timestamp"])

    close  = df["close"]
    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    candle_range = (high - low).clip(lower=1e-9)
    body         = close - open_

    # 0. close_pct — rendement normalisé
    close_pct = close.pct_change().fillna(0).clip(-_CLOSE_CLIP, _CLOSE_CLIP)
    df["f_close_pct"] = (close_pct / _CLOSE_CLIP).astype("float32")

    # 1. body_pct — direction et force de la bougie
    df["f_body_pct"] = (body / candle_range).clip(-1, 1).astype("float32")

    # 2. upper_wick — absorption vendeurs en haut
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1))
    df["f_upper_wick"] = (upper / candle_range).clip(0, 1).astype("float32")

    # 3. lower_wick — absorption acheteurs en bas
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - low)
    df["f_lower_wick"] = (lower / candle_range).clip(0, 1).astype("float32")

    # 4. vol_ratio — volume relatif (détecte l'intérêt inhabituel)
    avg_vol = volume.rolling(20, min_periods=1).mean().clip(lower=1e-9)
    vol_ratio = (volume / avg_vol).clip(0, _VOL_CLIP)
    df["f_vol_ratio"] = (vol_ratio / _VOL_CLIP).astype("float32")

    # 5. delta — pression achat approximée
    # Si taker_buy_vol disponible (Binance) : ratio réel.
    # Sinon : approximation par close location.
    if "taker_buy_vol" in df_1m.columns:
        taker_5m = _aggregate_taker(df_1m)
        taker_5m = taker_5m.reindex(df.index, fill_value=0.5)
        df["f_delta"] = taker_5m.clip(0, 1).astype("float32")
    else:
        df["f_delta"] = ((close - low) / candle_range).clip(0, 1).astype("float32")

    # 6. funding rate normalisé
    if "funding_rate" in df.columns:
        funding = df["funding_rate"].fillna(0).clip(-_FUND_CLIP, _FUND_CLIP)
        df["f_funding"] = (funding / _FUND_CLIP).astype("float32")
    else:
        df["f_funding"] = 0.0

    # 7. session encodée [0, 1]
    df["f_session"] = (df["session"].astype(float) / 3.0).astype("float32")

    return df.reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────

def _aggregate_taker(df_1m: pd.DataFrame) -> pd.Series:
    """
    Agrège taker_buy_vol de 1min en 5min et le divise par le volume total
    pour obtenir un ratio [0, 1] (fraction d'achat agressif).
    """
    df = df_1m.copy()
    df["ts_5m"] = (df["ts"] // 300) * 300

    agg = df.groupby("ts_5m").agg(
        taker_buy = ("taker_buy_vol", "sum"),
        total_vol = ("volume",        "sum"),
    )
    ratio = agg["taker_buy"] / agg["total_vol"].clip(lower=1e-9)
    return ratio.rename("f_delta")


def obs_from_row(row: pd.Series, position: int) -> np.ndarray:
    """
    Construit le vecteur d'observation (N_FEATURES,) depuis une ligne du DataFrame
    de features et la position courante.

    Utilisé par DeepTradingEnv._get_obs() et par l'inférence live.
    """
    feat = row[FEATURE_COLS].values.astype(np.float32)
    pos  = np.array([float(position)], dtype=np.float32)
    return np.concatenate([feat, pos])
