"""
engine/state.py — Calcul des etats pour l'agent RL Pi*.

Dimensions :
    SESSION (4)         : ASIA_DEAD / LONDON / OVERLAP / NY
    VOLATILITY (5)      : LOW / RISING / HIGH / EXTREME / FALLING
    PRICE_STRUCTURE (7) : RANGE_MID / UPTREND / DOWNTREND / BREAKOUT / REJECTION
                          / RANGE_TOP / RANGE_BOTTOM

Encodage : state_id = SESSION*35 + VOLATILITY*7 + PRICE_STRUCTURE  (0-139)

UPTREND/DOWNTREND : structure par swing highs/lows sur 1h (non bruité)
RANGE_TOP/BOTTOM  : percentile du prix dans le range récent (actionnable)
RANGE_MID         : milieu de range → FLAT
BREAKOUT/REJECTION: signaux price action 5min (override)

Entree  : DataFrame market_1m (depuis storage)
Sortie  : DataFrame 5min avec colonnes session, volatility, price_structure, state_id
"""

import numpy as np
import pandas as pd
from enum import IntEnum


# ── Enums ──────────────────────────────────────────────────────

class Session(IntEnum):
    ASIA_DEAD = 0   # 22:00-07:00 UTC (Tokyo + dead zone)
    LONDON    = 1   # 07:00-13:00 UTC (London open complet)
    OVERLAP   = 2   # 13:00-16:00 UTC (London/NY overlap)
    NY        = 3   # 16:00-22:00 UTC (NY pur)


class Volatility(IntEnum):
    LOW     = 0
    RISING  = 1
    HIGH    = 2
    EXTREME = 3
    FALLING = 4


class PriceStructure(IntEnum):
    RANGE_MID    = 0  # Range, prix au milieu → FLAT
    UPTREND      = 1  # Swing HH+HL sur 1h
    DOWNTREND    = 2  # Swing LH+LL sur 1h
    BREAKOUT     = 3  # Cassure range + volume (5min override)
    REJECTION    = 4  # Mèche longue (5min override)
    RANGE_TOP    = 5  # Range + prix dans top 25% → SHORT mean-reversion
    RANGE_BOTTOM = 6  # Range + prix dans bottom 25% → LONG mean-reversion


class HTFBias(IntEnum):
    BULL    = 0   # EMA20 > EMA50 * (1 + threshold)
    NEUTRAL = 1   # EMA20 et EMA50 proches
    BEAR    = 2   # EMA20 < EMA50 * (1 - threshold)


# ── Agregation 1min -> 5min ─────────────────────────────────────

def aggregate_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Agregation OHLCV + OI + funding de 1min en 5min."""
    df = df_1m.copy()
    df["ts_5m"] = (df["ts"] // 300) * 300

    agg = df.groupby("ts_5m").agg(
        open          = ("open",          "first"),
        high          = ("high",          "max"),
        low           = ("low",           "min"),
        close         = ("close",         "last"),
        volume        = ("volume",        "sum"),
        open_interest = ("open_interest", "last"),
        funding_rate  = ("funding_rate",  "last"),
    ).reset_index().rename(columns={"ts_5m": "ts"})

    agg["timestamp"] = pd.to_datetime(agg["ts"], unit="s", utc=True)
    return agg.sort_values("ts").reset_index(drop=True)


# ── Agregation 1min -> 1h ──────────────────────────────────────

def aggregate_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Agregation OHLCV de 1min en 1h."""
    df = df_1m.copy()
    df["ts_1h"] = (df["ts"] // 3600) * 3600
    agg = df.groupby("ts_1h").agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum"),
    ).reset_index().rename(columns={"ts_1h": "ts"})
    agg["timestamp"] = pd.to_datetime(agg["ts"], unit="s", utc=True)
    return agg.sort_values("ts").reset_index(drop=True)


# ── HTF Bias ────────────────────────────────────────────────────

def compute_htf_bias(df_1m: pd.DataFrame,
                     ema_fast: int   = 20,
                     ema_slow: int   = 50,
                     threshold: float = 0.002) -> pd.Series:
    """
    Calcule le biais directionnel HTF (EMA 20/50 sur 1h).

    Retourne un pd.Series indexé par ts (unix 1h) avec des valeurs HTFBias.
    Le biais de la bougie 1h N est celui de la bougie N-1 (pas de look-ahead).

    BULL    : EMA20 > EMA50 * (1 + threshold)
    BEAR    : EMA20 < EMA50 * (1 - threshold)
    NEUTRAL : entre les deux
    """
    df_1h = aggregate_1h(df_1m)
    close = df_1h["close"]

    ema20 = close.ewm(span=ema_fast, adjust=False).mean()
    ema50 = close.ewm(span=ema_slow, adjust=False).mean()

    bias = pd.Series(int(HTFBias.NEUTRAL), index=df_1h.index, dtype=int)
    bias[ema20 > ema50 * (1 + threshold)] = int(HTFBias.BULL)
    bias[ema20 < ema50 * (1 - threshold)] = int(HTFBias.BEAR)

    # Décalage d'une bougie : on utilise la 1h précédente pour éviter le look-ahead
    df_1h["htf_bias"] = bias.shift(1).fillna(int(HTFBias.NEUTRAL)).astype(int).values
    return df_1h.set_index("ts")["htf_bias"]


def apply_htf_mask(mask: np.ndarray, htf_bias: int) -> np.ndarray:
    """
    Applique le biais HTF sur un masque d'actions (N_ACTIONS,).

    BULL (0)    → SHORT (2) masqué — ne pas shorter une tendance haussière
    BEAR (2)    → LONG  (1) masqué — ne pas longer une tendance baissière
    NEUTRAL (1) → aucune contrainte
    FLAT (0) toujours disponible.
    """
    result = mask.copy()
    if htf_bias == int(HTFBias.BULL):
        result[2] = False
    elif htf_bias == int(HTFBias.BEAR):
        result[1] = False
    result[0] = True
    return result


# ── Labeling ────────────────────────────────────────────────────

def label_session(ts_utc: pd.Series) -> pd.Series:
    hour   = ts_utc.dt.hour
    result = pd.Series(int(Session.ASIA_DEAD), index=ts_utc.index, dtype=int)
    result[(hour >= 7)  & (hour < 13)] = int(Session.LONDON)
    result[(hour >= 13) & (hour < 16)] = int(Session.OVERLAP)
    result[(hour >= 16) & (hour < 22)] = int(Session.NY)
    # ASIA_DEAD : 22:00-07:00 (défaut déjà posé)
    return result


def label_volatility(df: pd.DataFrame, atr_period: int = 14, window: int = 100) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr      = tr.rolling(atr_period).mean()
    atr_norm = atr / close.clip(lower=1e-9)

    p33 = atr_norm.rolling(window, min_periods=20).quantile(0.33)
    p66 = atr_norm.rolling(window, min_periods=20).quantile(0.66)
    p90 = atr_norm.rolling(window, min_periods=20).quantile(0.90)

    atr_delta = atr_norm - atr_norm.shift(3)

    result = pd.Series(int(Volatility.LOW), index=df.index, dtype=int)
    is_extreme = atr_norm > p90
    is_high    = (atr_norm > p66) & ~is_extreme
    is_rising  = (atr_norm >= p33) & (atr_norm < p66) & (atr_delta > 0)
    is_falling = (atr_norm >= p33) & (atr_norm < p66) & (atr_delta <= 0)

    result[is_falling] = int(Volatility.FALLING)
    result[is_rising]  = int(Volatility.RISING)
    result[is_high]    = int(Volatility.HIGH)
    result[is_extreme] = int(Volatility.EXTREME)
    return result


def _swing_structure_1h(df_1m: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    Détecte la structure directionnelle par swing highs/lows sur 1h.

    Compare le rolling max/min des `lookback` dernières bougies 1h vs la période
    précédente de même durée. Si les deux progressent → UPTREND, régressent →
    DOWNTREND, sinon → RANGE_MID.

    Retourne une pd.Series indexée par ts 1h (unix), sans look-ahead (shift 1).
    """
    df_1h = aggregate_1h(df_1m)
    high  = df_1h["high"]
    low   = df_1h["low"]

    recent_high = high.rolling(lookback, min_periods=lookback).max()
    recent_low  = low.rolling(lookback, min_periods=lookback).min()
    prev_high   = recent_high.shift(lookback)
    prev_low    = recent_low.shift(lookback)

    result = pd.Series(int(PriceStructure.RANGE_MID), index=df_1h.index, dtype=int)
    result[(recent_high > prev_high) & (recent_low > prev_low)] = int(PriceStructure.UPTREND)
    result[(recent_high < prev_high) & (recent_low < prev_low)] = int(PriceStructure.DOWNTREND)

    df_1h["swing"] = result.shift(1).fillna(int(PriceStructure.RANGE_MID)).astype(int).values
    return df_1h.set_index("ts")["swing"]


def label_price_structure(df: pd.DataFrame, df_1m: pd.DataFrame,
                           window: int = 20) -> pd.Series:
    """
    Classifie la structure de prix en 7 états.

    UPTREND / DOWNTREND : swing highs/lows sur 1h (signal structurel stable)
    RANGE_TOP / RANGE_BOTTOM : percentile dans le range récent (mean-reversion)
    RANGE_MID : milieu de range → pas de signal directionnel
    BREAKOUT / REJECTION : patterns 5min, override sur tout
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # ── Structure 1h (UPTREND / DOWNTREND / RANGE) ─────────────
    swing_map = _swing_structure_1h(df_1m)
    ts_1h     = (df["ts"] // 3600) * 3600
    htf_struct = ts_1h.map(swing_map).fillna(int(PriceStructure.RANGE_MID)).astype(int)

    # ── Position dans le range (pour les bougies en RANGE) ─────
    range_high = high.shift(1).rolling(window, min_periods=5).max()
    range_low  = low.shift(1).rolling(window, min_periods=5).min()
    range_span = (range_high - range_low).clip(lower=1e-9)
    pct_in_range = (close - range_low) / range_span

    # ── Breakout 5min ───────────────────────────────────────────
    avg_vol  = volume.rolling(window).mean()
    breakout = (
        ((close > range_high) | (close < range_low)) &
        (volume > avg_vol * 1.5)
    )

    # ── Rejection 5min (mèche longue) ──────────────────────────
    candle_range = (high - low).clip(lower=1e-9)
    body         = (df["open"] - close).abs()
    rejection    = (
        (body / candle_range < 0.3) &
        (candle_range > candle_range.rolling(window).mean())
    )

    # ── Composition finale ──────────────────────────────────────
    result = htf_struct.copy()

    # Sous-états RANGE selon position dans le range
    is_range = (htf_struct == int(PriceStructure.RANGE_MID))
    result[is_range & (pct_in_range >= 0.75)] = int(PriceStructure.RANGE_TOP)
    result[is_range & (pct_in_range <= 0.25)] = int(PriceStructure.RANGE_BOTTOM)

    # Override 5min (priorité maximale)
    result[breakout]  = int(PriceStructure.BREAKOUT)
    result[rejection] = int(PriceStructure.REJECTION)

    return result


# ── Pipeline complet ────────────────────────────────────────────

def compute_states(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Agregation 1min -> 5min + labeling des 3 dimensions d'etat + biais HTF.

    Retourne un DataFrame 5min avec colonnes :
        session, volatility, price_structure, state_id (0-139), htf_bias
    """
    df = aggregate_5m(df_1m)

    df["session"]         = label_session(df["timestamp"])
    df["volatility"]      = label_volatility(df)
    df["price_structure"] = label_price_structure(df, df_1m)

    # state_id : SESSION*35 + VOLATILITY*7 + PRICE_STRUCTURE  (0-139)
    df["state_id"] = (
        df["session"].astype(int)         * 35 +
        df["volatility"].astype(int)      * 7  +
        df["price_structure"].astype(int)
    )

    # Biais HTF : EMA 20/50 sur 1h, aligné sur chaque bougie 5m
    htf_map       = compute_htf_bias(df_1m)
    df["ts_1h"]   = (df["ts"] // 3600) * 3600
    df["htf_bias"] = df["ts_1h"].map(htf_map).fillna(int(HTFBias.NEUTRAL)).astype(int)
    df.drop(columns=["ts_1h"], inplace=True)

    return df
