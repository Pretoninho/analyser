# ── TA Strategy — identification des triggers + calcul des outcomes ──────────
#
# Principe d'efficacité :
#   Les triggers (session + 2-bar reversal) et les outcomes (win/loss)
#   sont calculés UNE SEULE FOIS avec ATR fixe (TP_SL_ATR=14).
#   Le sweep externe ajoute ensuite les features paramétriques à chaque trade.
#   Cela évite de relancer la simulation 108 fois.
#
# Trigger :
#   LONG  — 2 bougies rouges consécutives puis bougie verte (dans session)
#   SHORT — 2 bougies vertes consécutives puis bougie rouge (dans session)
#
# Entry  : open de la bougie suivante
# Exit   : premier hit TP ou SL sur high/low des bougies suivantes
# Timeout: MAX_BARS bougies sans hit → trade exclu des stats

import numpy as np
import pandas as pd
from strategies.ta.config import (
    SESSIONS_UTC, TP_MULT, SL_MULT, TP_SL_ATR, MAX_BARS,
)
from strategies.ta.features import _atr


def _session_mask(index: pd.DatetimeIndex, sessions: list) -> np.ndarray:
    """Masque booléen : True si l'heure UTC tombe dans une session."""
    h = index.hour
    mask = np.zeros(len(index), dtype=bool)
    for start, end in sessions:
        mask |= (h >= start) & (h < end)
    return mask


def build_trades(df15: pd.DataFrame) -> pd.DataFrame:
    """
    Identifie tous les triggers sur df15 et calcule l'outcome de chaque trade.

    Retourne un DataFrame de trades avec colonnes :
      entry_idx   : position entière dans df15 (index du bar d'entrée)
      entry_time  : timestamp UTC du bar d'entrée
      direction   : 'LONG' | 'SHORT'
      entry_price : open du bar d'entrée
      atr_at_entry: ATR_14 au moment du signal (bar i)
      tp          : niveau TP
      sl          : niveau SL
      outcome     : 'win' | 'loss'
      n_bars      : nombre de bougies avant résolution
    """
    # ATR fixe pour TP/SL
    atr = _atr(df15["high"], df15["low"], df15["close"], TP_SL_ATR)

    # Direction de la bougie : +1 verte, -1 rouge, 0 doji
    body = np.sign(df15["close"].values - df15["open"].values)

    # Masque session
    session_mask = _session_mask(df15.index, SESSIONS_UTC)

    highs  = df15["high"].values
    lows   = df15["low"].values
    opens  = df15["open"].values
    atr_v  = atr.values

    records = []

    n = len(df15)
    for i in range(2, n - 1):
        if not session_mask[i]:
            continue

        b0, b1, b2 = body[i], body[i - 1], body[i - 2]

        long_trigger  = (b0 > 0) and (b1 < 0) and (b2 < 0)
        short_trigger = (b0 < 0) and (b1 > 0) and (b2 > 0)

        if not (long_trigger or short_trigger):
            continue

        atr_i = atr_v[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        direction   = "LONG" if long_trigger else "SHORT"
        entry_price = opens[i + 1]
        tp = entry_price + TP_MULT * atr_i if direction == "LONG" else entry_price - TP_MULT * atr_i
        sl = entry_price - SL_MULT * atr_i if direction == "LONG" else entry_price + SL_MULT * atr_i

        outcome = None
        n_bars  = 0
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            h, lo = highs[j], lows[j]
            n_bars = j - i
            if direction == "LONG":
                if lo <= sl:
                    outcome = "loss"; break
                if h  >= tp:
                    outcome = "win";  break
            else:
                if h  >= sl:
                    outcome = "loss"; break
                if lo <= tp:
                    outcome = "win";  break

        if outcome is None:
            continue  # timeout — exclu

        records.append({
            "entry_idx":   i + 1,
            "entry_time":  df15.index[i + 1],
            "direction":   direction,
            "entry_price": entry_price,
            "atr_at_entry": atr_i,
            "tp":          tp,
            "sl":          sl,
            "outcome":     outcome,
            "n_bars":      n_bars,
        })

    return pd.DataFrame(records)
