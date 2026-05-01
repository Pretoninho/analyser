"""
live_runner.py — Moteur de détection TA en temps réel.

Récupère les données Binance via API publique, calcule les 108 combos
de features sur la dernière bougie 15m fermée, et vérifie les matches
contre les configs stables IS+OOS.

Usage autonome :
    python strategies/ta/live_runner.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests
import numpy as np
import pandas as pd
from itertools import product

from strategies.ta.config import (
    EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS,
    SESSIONS_UTC, RESULTS_DIR, SYMBOL,
    RSI_BINS, RSI_LABELS, STOCH_BINS, STOCH_LABELS,
    ATR_BINS, ATR_LABELS,
    REGIME_EMA_LEN, REGIME_SLOPE_DAYS,
)
from strategies.ta.features import _ema, _rsi, _atr, _stoch_k, _vwap_daily, _swing_state_4h

# ── Constantes ────────────────────────────────────────────────────────────────
BINANCE_URL    = "https://api.binance.com/api/v3/klines"
LIVE_WR_DROP   = -0.05   # drop max autorisé IS → OOS
LIVE_MIN_N_OOS = 5       # trades OOS minimum
LIVE_EXCL_REGIME = {"range"}   # régimes exclus (trop fragiles)

# Combo de référence pour afficher l'état "résumé" dans le dashboard
REF_PARAMS = "EMA50_RSI14_SK14SS3SD3_ATR14"


# ─────────────────────────────────────────────────────────────────────────────
# Fetch Binance
# ─────────────────────────────────────────────────────────────────────────────

def _parse_klines(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "qv", "trades", "tbbv", "tbqv", "ignore",
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")
    return df.set_index("ts")[["open", "high", "low", "close", "volume"]]


def fetch_klines(interval: str, limit: int, symbol: str = SYMBOL) -> pd.DataFrame:
    """Récupère limit bougies Binance (API publique, pas de clé nécessaire)."""
    resp = requests.get(
        BINANCE_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return _parse_klines(resp.json())


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des configs stables
# ─────────────────────────────────────────────────────────────────────────────

def load_live_configs() -> dict:
    """
    Charge les configs stables IS+OOS depuis sweep_IS_vs_OOS.csv.

    Retourne un dict :
      clé   = (params, direction, regime, ema_state, ema_slope,
                swing, rsi_state, stoch_state, atr_state, vwap_state)
      valeur = {n_IS, wr_IS, n_OOS, wr_OOS, exp_R_OOS}
    """
    path = RESULTS_DIR / "sweep_IS_vs_OOS.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df["wr_drop"] = df["wr_OOS"] - df["wr_IS"]

    filt = df[
        (df["wr_drop"] >= LIVE_WR_DROP) &
        (df["n_OOS"] >= LIVE_MIN_N_OOS) &
        (~df["regime"].isin(LIVE_EXCL_REGIME))
    ]

    configs = {}
    for _, row in filt.iterrows():
        key = (
            row["params"], row["direction"],
            row["regime"],
            int(row["ema_state"]), int(row["ema_slope"]),
            int(row["swing"]),
            str(row["rsi_state"]), str(row["stoch_state"]), str(row["atr_state"]),
            int(row["vwap_state"]),
        )
        configs[key] = {
            "n_IS":     int(row["n_IS"]),
            "wr_IS":    round(float(row["wr_IS"]),    4),
            "n_OOS":    int(row["n_OOS"]),
            "wr_OOS":   round(float(row["wr_OOS"]),   4),
            "exp_R_OOS": round(float(row["exp_R_OOS"]), 4),
        }
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Calcul de l'état courant pour un combo de paramètres
# ─────────────────────────────────────────────────────────────────────────────

def _compute_state_last(
    df15: pd.DataFrame,
    df4h: pd.DataFrame,
    df1d: pd.DataFrame,
    ema_len: int,
    rsi_len: int,
    stoch_k_period: int,
    stoch_smooth_k: int,
    atr_len: int,
) -> dict:
    """
    Calcule les features discrétisées de la DERNIÈRE bougie 15m fermée.
    Utilise df4h pour l'EMA et le swing, df1d pour le régime.
    """
    # ── EMA + swing sur 4H ───────────────────────────────────────────────────
    ema_4h      = _ema(df4h["close"], ema_len)
    last_close4 = df4h["close"].iloc[-1]
    last_ema4   = ema_4h.iloc[-1]
    ema_state   = 1 if last_close4 > last_ema4 else -1

    slope_val   = ema_4h.diff().iloc[-1]
    ema_slope   = int(np.sign(slope_val)) if not np.isnan(slope_val) else 0

    swing_4h    = _swing_state_4h(df4h["close"])
    swing       = int(swing_4h.iloc[-1])

    # ── RSI / Stoch / ATR sur 15m ────────────────────────────────────────────
    rsi_s     = _rsi(df15["close"], rsi_len)
    stoch_s   = _stoch_k(df15["high"], df15["low"], df15["close"],
                         stoch_k_period, stoch_smooth_k)
    atr_s     = _atr(df15["high"], df15["low"], df15["close"], atr_len)
    atr_ma_s  = atr_s.rolling(20).mean()

    rsi_val   = float(rsi_s.iloc[-1])
    stoch_val = float(stoch_s.iloc[-1])
    atr_val   = float(atr_s.iloc[-1])
    atr_ma    = float(atr_ma_s.iloc[-1])
    atr_ratio = atr_val / atr_ma if atr_ma > 0 else 1.0

    # ── VWAP sur 15m ─────────────────────────────────────────────────────────
    vwap_s    = _vwap_daily(df15)
    vwap_val  = float(vwap_s.iloc[-1])
    vwap_state = 1 if df15["close"].iloc[-1] > vwap_val else -1

    # ── Discrétisation ───────────────────────────────────────────────────────
    rsi_state   = str(pd.cut([rsi_val],   bins=RSI_BINS,   labels=RSI_LABELS,   right=False)[0])
    stoch_state = str(pd.cut([stoch_val], bins=STOCH_BINS, labels=STOCH_LABELS, right=False)[0])
    atr_state   = str(pd.cut([atr_ratio], bins=ATR_BINS,   labels=ATR_LABELS,   right=False)[0])

    # ── Régime macro (EMA200 daily) ───────────────────────────────────────────
    ema200_1d = _ema(df1d["close"], REGIME_EMA_LEN)
    slope_1d  = float(ema200_1d.diff(REGIME_SLOPE_DAYS).iloc[-1])
    above_1d  = df1d["close"].iloc[-1] > ema200_1d.iloc[-1]
    if above_1d and slope_1d > 0:
        regime = "bull"
    elif (not above_1d) and slope_1d < 0:
        regime = "bear"
    else:
        regime = "range"

    return {
        "regime":     regime,
        "ema_state":  ema_state,
        "ema_slope":  ema_slope,
        "swing":      swing,
        "rsi_state":  rsi_state,
        "stoch_state": stoch_state,
        "atr_state":  atr_state,
        "vwap_state": vwap_state,
        # Valeurs brutes pour affichage
        "_rsi":       round(rsi_val,   2) if not np.isnan(rsi_val)   else None,
        "_stoch":     round(stoch_val, 2) if not np.isnan(stoch_val) else None,
        "_atr":       round(atr_val,   4) if not np.isnan(atr_val)   else None,
        "_atr_ratio": round(atr_ratio, 3),
        "_vwap":      round(vwap_val,  2) if not np.isnan(vwap_val)  else None,
        "_ema_4h":    round(float(last_ema4), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Détection du trigger
# ─────────────────────────────────────────────────────────────────────────────

def _detect_trigger(df15_closed: pd.DataFrame) -> str | None:
    """2-bar reversal sur les 3 dernières bougies fermées."""
    if len(df15_closed) < 3:
        return None
    body = np.sign(df15_closed["close"].values - df15_closed["open"].values)
    b0, b1, b2 = body[-1], body[-2], body[-3]
    if b0 > 0 and b1 < 0 and b2 < 0:
        return "LONG"
    if b0 < 0 and b1 > 0 and b2 > 0:
        return "SHORT"
    return None


def _in_session(dt: pd.Timestamp) -> bool:
    h = dt.hour
    for start, end in SESSIONS_UTC:
        if start <= h < end:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def scan(symbol: str = SYMBOL) -> dict:
    """
    Récupère les données live Binance, calcule les 108 combos de features
    sur la dernière bougie 15m fermée, et retourne les matches LIVE_CONFIGS.

    Retourne un dict contenant :
      timestamp       : heure UTC du scan
      last_bar_time   : heure de la dernière bougie 15m fermée
      last_price      : dernier close 15m
      in_session      : bool
      trigger         : 'LONG' | 'SHORT' | None
      n_matches       : nombre de configs qui matchent
      matches         : liste de dicts triés par exp_R_OOS décroissant
      current_state   : état courant (combo de référence EMA50_RSI14_SK14...)
      df15            : DataFrame 15m pour le chart (200 bougies)
      df4h            : DataFrame 4H
      df1d            : DataFrame 1D
    """
    now = pd.Timestamp.now("UTC")

    # ── Fetch données multi-TF ────────────────────────────────────────────────
    # 15m : 200 bougies (~50h) pour chart + features 15m avec warmup
    # 4H  : 600 bougies (~100 jours) pour EMA + swing 4H
    # 1D  : 600 bougies (~1.6 ans) pour régime EMA200 daily (warmup suffisant)
    df15 = fetch_klines("15m", 200, symbol)
    df4h = fetch_klines("4h",  600, symbol)
    df1d = fetch_klines("1d",  600, symbol)

    # Exclure la dernière bougie 15m (en cours de formation)
    df15_closed = df15.iloc[:-1]

    trigger    = _detect_trigger(df15_closed)
    in_session = _in_session(df15_closed.index[-1])
    active_trigger = trigger if in_session else None

    # ── Scan des 108 combos ───────────────────────────────────────────────────
    live_configs = load_live_configs()
    matches      = []
    all_states   = {}   # params_label -> state dict

    for ema_len, rsi_len, (sk, ssk, sd), atr_len in product(
        EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS
    ):
        params_label = f"EMA{ema_len}_RSI{rsi_len}_SK{sk}SS{ssk}SD{sd}_ATR{atr_len}"

        state = _compute_state_last(
            df15_closed, df4h, df1d,
            ema_len, rsi_len, sk, ssk, atr_len,
        )
        all_states[params_label] = state

        if not active_trigger:
            continue

        key = (
            params_label, active_trigger,
            state["regime"],
            state["ema_state"], state["ema_slope"],
            state["swing"],
            state["rsi_state"], state["stoch_state"], state["atr_state"],
            state["vwap_state"],
        )
        if key in live_configs:
            meta = live_configs[key]
            matches.append({
                "params":     params_label,
                "direction":  active_trigger,
                "regime":     state["regime"],
                "ema_state":  state["ema_state"],
                "ema_slope":  state["ema_slope"],
                "swing":      state["swing"],
                "rsi_state":  state["rsi_state"],
                "stoch_state": state["stoch_state"],
                "atr_state":  state["atr_state"],
                "vwap_state": state["vwap_state"],
                **meta,
            })

    matches.sort(key=lambda x: x["exp_R_OOS"], reverse=True)

    return {
        "timestamp":     now.isoformat(),
        "last_bar_time": df15_closed.index[-1].isoformat(),
        "last_price":    float(df15_closed["close"].iloc[-1]),
        "in_session":    in_session,
        "trigger":       active_trigger,
        "trigger_raw":   trigger,   # trigger sans filtre session (pour affichage)
        "n_matches":     len(matches),
        "matches":       matches,
        "current_state": all_states.get(REF_PARAMS, {}),
        "all_states":    all_states,
        "df15":          df15,          # inclut la bougie courante (pour le chart)
        "df15_closed":   df15_closed,
        "df4h":          df4h,
        "df1d":          df1d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Scanning live BTC/USDT...")
    result = scan()
    print(f"Timestamp    : {result['timestamp']}")
    print(f"Last bar     : {result['last_bar_time']}")
    print(f"Last price   : ${result['last_price']:,.2f}")
    print(f"In session   : {result['in_session']}")
    print(f"Trigger      : {result['trigger_raw']} "
          f"({'ACTIF' if result['trigger'] else 'hors session'})")
    cs = result["current_state"]
    print(f"Regime       : {cs.get('regime', 'N/A')}")
    print(f"EMA state    : {cs.get('ema_state', 'N/A')} "
          f"(slope {cs.get('ema_slope', 'N/A')})")
    print(f"Swing        : {cs.get('swing', 'N/A')}")
    print(f"RSI ({cs.get('_rsi', '?')})  : {cs.get('rsi_state', 'N/A')}")
    print(f"Stoch ({cs.get('_stoch', '?')}) : {cs.get('stoch_state', 'N/A')}")
    print(f"ATR ratio ({cs.get('_atr_ratio', '?')}) : {cs.get('atr_state', 'N/A')}")
    print(f"VWAP state   : {cs.get('vwap_state', 'N/A')}")
    print(f"\nMatches      : {result['n_matches']}")
    if result["matches"]:
        print("\nTop 5 matches :")
        for m in result["matches"][:5]:
            print(f"  {m['params']:38s}  {m['direction']:5s}  "
                  f"WR_OOS={m['wr_OOS']:.1%}  "
                  f"exp={m['exp_R_OOS']:.2f}R  "
                  f"(n={m['n_OOS']})")
