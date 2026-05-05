# ── TA Strategy — calcul des features ───────────────────────────────────────
# Implémentations manuelles numpy/pandas. Aucune dépendance TA externe.

import numpy as np
import pandas as pd
from pathlib import Path
from strategies.ta.config import (
    DATA_DIR, SYMBOL,
    RSI_BINS, RSI_LABELS,
    STOCH_BINS, STOCH_LABELS,
    ATR_BINS, ATR_LABELS,
    REGIME_EMA_LEN, REGIME_SLOPE_DAYS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_15m(data_dir: Path = DATA_DIR, symbol: str = SYMBOL) -> pd.DataFrame:
    """
    Lit tous les CSVs 1m Binance, concatène et rééchantillonne en 15m.
    Colonnes résultantes : open, high, low, close, volume (DatetimeIndex UTC).
    """
    cols = ["ts", "open", "high", "low", "close", "volume",
            "close_ts", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]

    frames = sorted(data_dir.glob(f"{symbol}-1m-*.csv"))
    if not frames:
        raise FileNotFoundError(f"Aucun fichier 1m trouvé dans {data_dir}")

    parts = []
    for f in frames:
        # Certains fichiers ont un header (ex: 2024+), d'autres non
        with open(f, "r") as fh:
            first = fh.readline().strip()
        has_header = not first.split(",")[0].lstrip("-").isdigit()

        df = pd.read_csv(
            f,
            header=0 if has_header else None,
            names=None if has_header else cols,
            usecols=[0, 1, 2, 3, 4, 5],
            dtype="float64",
        )
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
        df["ts"] = df["ts"].astype("int64")
        parts.append(df)

    raw = pd.concat(parts, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
    raw = raw.set_index("ts").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    df15 = raw.resample("15min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    return df15


# ─────────────────────────────────────────────────────────────────────────────
# Indicateurs de base
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series,
             k_period: int, smooth_k: int) -> pd.Series:
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    rang    = (highest - lowest).replace(0.0, np.nan)
    k_raw   = 100.0 * (close - lowest) / rang
    return k_raw.rolling(smooth_k).mean()


def _regime_daily(df15: pd.DataFrame,
                  ema_len: int = 200,
                  slope_days: int = 5) -> pd.Series:
    """
    Regime macro base sur EMA200 daily, forward-fille sur 15m.
      'bull'  : close > EMA200_1D ET slope positive
      'bear'  : close < EMA200_1D ET slope negative
      'range' : tous les autres cas (EMA plate ou price de l'autre cote mais sans momentum)
    """
    df1d = df15.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    ema200    = _ema(df1d["close"], ema_len)
    # Slope : variation moyenne de l'EMA sur slope_days jours
    slope     = ema200.diff(slope_days)
    above     = df1d["close"] > ema200
    slope_up  = slope > 0
    slope_dn  = slope < 0

    regime_1d = pd.Series("range", index=df1d.index, dtype=object)
    regime_1d[above  & slope_up]  = "bull"
    regime_1d[(~above) & slope_dn] = "bear"

    # Forward-fill vers 15m (+ bfill pour les premières bougies du jour avant le 1er close 1D)
    return regime_1d.reindex(df15.index, method="ffill").bfill()


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    tp      = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv     = tp * df["volume"]
    date_key = df.index.normalize()
    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    return cum_tpv / cum_vol.replace(0.0, np.nan)


def _swing_state_4h(close_4h: pd.Series, n: int = 3) -> pd.Series:
    """
    Détermine la structure de marché sur n bougies 4H :
      +1 = uptrend (HH + HL)
      -1 = downtrend (LH + LL)
       0 = mixte
    """
    roll_high = close_4h.rolling(n).max()
    roll_low  = close_4h.rolling(n).min()
    ph        = close_4h.shift(1).rolling(n).max()
    pl        = close_4h.shift(1).rolling(n).min()

    up   = (roll_high > ph) & (roll_low > pl)
    down = (roll_high < ph) & (roll_low < pl)

    state = pd.Series(0, index=close_4h.index, dtype="int8")
    state[up]   =  1
    state[down] = -1
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des features pour un combo de paramètres
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    df15: pd.DataFrame,
    ema_len:  int,
    rsi_len:  int,
    stoch_k_period: int,
    stoch_smooth_k: int,
    stoch_d_period: int,   # conservé pour extension future (non utilisé en state)
    atr_len:  int,
) -> pd.DataFrame:
    """
    Calcule et discrétise tous les features sur df15.

    Retourne df15 enrichi des colonnes :
      ema_state   : +1 (above) / -1 (below)
      ema_slope   : +1 / 0 / -1
      swing       : +1 / 0 / -1
      rsi_state   : 'oversold' | 'weak' | 'strong' | 'overbought'
      stoch_state : idem
      atr_state   : 'compression' | 'neutral' | 'expansion'
      vwap_state  : +1 (above) / -1 (below)
      regime      : 'bull' | 'bear' | 'range'
    """
    feat = df15.copy()

    # ── Agrégation 4H ────────────────────────────────────────────────────────
    df4h = df15.resample("4h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    ema_4h    = _ema(df4h["close"], ema_len)
    ema_st_4h = (df4h["close"] > ema_4h).astype("int8").replace({0: -1})
    slope_4h  = np.sign(ema_4h.diff()).fillna(0).astype("int8")
    swing_4h  = _swing_state_4h(df4h["close"])

    # Forward-fill vers 15m
    feat["ema_state"] = ema_st_4h.reindex(feat.index, method="ffill")
    feat["ema_slope"] = slope_4h.reindex(feat.index,  method="ffill")
    feat["swing"]     = swing_4h.reindex(feat.index,  method="ffill")

    # ── 15m ──────────────────────────────────────────────────────────────────
    feat["_rsi"]   = _rsi(feat["close"], rsi_len)
    feat["_stoch"] = _stoch_k(feat["high"], feat["low"], feat["close"],
                               stoch_k_period, stoch_smooth_k)
    feat["_atr"]   = _atr(feat["high"], feat["low"], feat["close"], atr_len)
    atr_ma         = feat["_atr"].rolling(20).mean()
    # fillna(1.0) : avant les 20 premiers bars, on considère l'ATR ratio = 1 (neutre)
    feat["_atr_ratio"] = (feat["_atr"] / atr_ma.replace(0.0, np.nan)).fillna(1.0)
    feat["_vwap"]  = _vwap_daily(feat)

    # ── Discrétisation ───────────────────────────────────────────────────────
    feat["rsi_state"] = pd.cut(
        feat["_rsi"], bins=RSI_BINS, labels=RSI_LABELS, right=False
    ).astype("str")

    feat["stoch_state"] = pd.cut(
        feat["_stoch"], bins=STOCH_BINS, labels=STOCH_LABELS, right=False
    ).astype("str")

    feat["atr_state"] = pd.cut(
        feat["_atr_ratio"], bins=ATR_BINS, labels=ATR_LABELS, right=False
    ).astype("str")

    feat["vwap_state"] = np.where(feat["close"] > feat["_vwap"], 1, -1)

    # ── Regime macro (EMA200 daily) ───────────────────────────────────────────
    feat["regime"] = _regime_daily(feat, ema_len=REGIME_EMA_LEN,
                                   slope_days=REGIME_SLOPE_DAYS)

    # Nettoyage colonnes intermédiaires
    feat.drop(columns=["_rsi", "_stoch", "_atr", "_atr_ratio", "_vwap"], inplace=True)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Colonnes d'état utilisées dans le sweep
# ─────────────────────────────────────────────────────────────────────────────
STATE_COLS = ["regime", "ema_state", "ema_slope", "swing", "rsi_state",
             "stoch_state", "atr_state", "vwap_state"]
