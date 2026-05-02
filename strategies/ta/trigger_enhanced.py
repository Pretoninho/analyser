# ── Enhanced Trigger — Multi-structure alignment + ATR filter ──
#
# Remplace 2-bar reversal simple par :
#   1. 4H structure alignment (swing + EMA slope)
#   2. ATR compression filter (ignore si expansion)
#   3. 15m 2-bar reversal MAIS confirmé seulement si aligné

import numpy as np
import pandas as pd
from strategies.ta.config import SESSIONS_UTC, TP_MULT, SL_MULT, TP_SL_ATR, MAX_BARS
from strategies.ta.features import _atr


def _session_mask(index: pd.DatetimeIndex, sessions: list) -> np.ndarray:
    """Masque booléen : True si l'heure UTC tombe dans une session."""
    h = index.hour
    mask = np.zeros(len(index), dtype=bool)
    for start, end in sessions:
        mask |= (h >= start) & (h < end)
    return mask


def build_trades_enhanced(df15: pd.DataFrame, atr_expansion_threshold: float = 1.2) -> pd.DataFrame:
    """
    Trigger amélioré multi-structure.

    Filtres:
      1. ATR compression/neutral ONLY (ignore expansion > 1.2 threshold)
      2. EMA slope alignment (4H) — reversal contre le trend
      3. Swing state check (4H) — structure cohérente
      4. Session mask (London/NY)
      5. 2-bar reversal standard (15m)

    Retourne DataFrame de trades (même format que build_trades).
    """
    # Agrégation 4H
    df4h = df15.resample("4h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    if len(df4h) < 2:
        return pd.DataFrame()

    # Features 4H
    from strategies.ta.features import _ema, _swing_state_4h
    ema_4h = _ema(df4h["close"], 50)  # EMA50 for structure
    ema_slope = np.sign(ema_4h.diff()).fillna(0).astype("int8")
    swing_4h = _swing_state_4h(df4h["close"], n=3)

    # Forward-fill vers 15m
    ema_slope_15m = ema_slope.reindex(df15.index, method="ffill").fillna(0).astype("int8")
    swing_15m = swing_4h.reindex(df15.index, method="ffill").fillna(0).astype("int8")

    # ATR filter
    atr = _atr(df15["high"], df15["low"], df15["close"], TP_SL_ATR)
    atr_ma = atr.rolling(20).mean()
    atr_ratio = atr / atr_ma.replace(0.0, np.nan)

    # Direction de la bougie
    body = np.sign(df15["close"].values - df15["open"].values)

    # Masque session
    session_mask = _session_mask(df15.index, SESSIONS_UTC)

    highs  = df15["high"].values
    lows   = df15["low"].values
    opens  = df15["open"].values
    atr_v  = atr.values
    atr_r  = atr_ratio.values

    records = []

    n = len(df15)
    for i in range(2, n - 1):
        if not session_mask[i]:
            continue

        # ── Filter 1: ATR expansion rejection ─────────────────────────────
        if not np.isnan(atr_r[i]) and atr_r[i] > atr_expansion_threshold:
            continue

        b0, b1, b2 = body[i], body[i - 1], body[i - 2]

        long_trigger  = (b0 > 0) and (b1 < 0) and (b2 < 0)
        short_trigger = (b0 < 0) and (b1 > 0) and (b2 > 0)

        if not (long_trigger or short_trigger):
            continue

        # ── Filter 2: EMA slope alignment (contrarian) ──────────────────────
        # LONG quand downtrend (ema_slope=-1) ou flat (0), SHORT quand uptrend (ema_slope=+1)
        ema_sl = ema_slope_15m.iloc[i] if i < len(ema_slope_15m) else 0
        swing_st = swing_15m.iloc[i] if i < len(swing_15m) else 0

        if long_trigger:
            # LONG reversal : meilleur si dans downtrend ou flat
            if ema_sl > 0:  # Dans uptrend = contre-intuitive, filtre
                continue
        else:  # short_trigger
            # SHORT reversal : meilleur si dans uptrend ou flat
            if ema_sl < 0:  # Dans downtrend = contre-intuitive, filtre
                continue

        # ── Filter 3: Swing state confirmation ──────────────────────────────
        # Si swing_st définit une direction, reversal doit l'opposer légèrement
        # mais on ne rejette pas si swing=0 (mixte)
        if swing_st != 0:
            if long_trigger and swing_st > 0:  # LONG dans uptrend swing
                continue
            if short_trigger and swing_st < 0:  # SHORT dans downtrend swing
                continue

        # ── Check ATR validity ───────────────────────────────────────────────
        atr_i = atr_v[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # ── Standard trade simulation ────────────────────────────────────────
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
            continue  # timeout

        records.append({
            "entry_idx":   i + 1,
            "entry_time":  df15.index[i + 1],
            "direction":   direction,
            "entry_price": entry_price,
            "atr_at_entry": atr_i,
            "atr_ratio_at_entry": atr_r[i] if not np.isnan(atr_r[i]) else np.nan,
            "ema_slope_4h": int(ema_sl),
            "swing_4h": int(swing_st),
            "tp":          tp,
            "sl":          sl,
            "outcome":     outcome,
            "n_bars":      n_bars,
        })

    return pd.DataFrame(records)
