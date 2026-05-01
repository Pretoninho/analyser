"""
signal_logger.py — Log des signaux TA live + résolution automatique des trades.

Deux fonctions principales :
  log_signal(result)     : appel au moment du signal → écrit dans ta_signals.csv
  resolve_pending()      : appel toutes les heures → vérifie TP/SL sur les signaux non résolus

CSV : db/ta_signals.csv
Colonnes :
  signal_id, timestamp, direction, entry_price, tp, sl, atr_at_entry,
  regime, ema_state, swing, rsi_state, stoch_state, atr_state, vwap_state,
  n_matches, top_params, top_wr_oos, top_exp_oos,
  outcome (pending/win/loss/timeout), exit_price, exit_time, n_bars, r_realized
"""

import sys
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from strategies.ta.config import TP_MULT, SL_MULT, TP_SL_ATR, MAX_BARS
from strategies.ta.features import _atr
from strategies.ta.live_runner import fetch_klines

SIGNALS_CSV = ROOT / "db" / "ta_signals.csv"

_COLS = [
    "signal_id", "timestamp", "direction", "entry_price", "tp", "sl",
    "atr_at_entry", "regime", "ema_state", "swing",
    "rsi_state", "stoch_state", "atr_state", "vwap_state",
    "n_matches", "top_params", "top_wr_oos", "top_exp_oos",
    "outcome", "exit_price", "exit_time", "n_bars", "r_realized",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    if SIGNALS_CSV.exists():
        return pd.read_csv(SIGNALS_CSV, dtype={"signal_id": str})
    return pd.DataFrame(columns=_COLS)


def _save(df: pd.DataFrame) -> None:
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SIGNALS_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Log d'un nouveau signal
# ─────────────────────────────────────────────────────────────────────────────

def log_signal(result: dict) -> str:
    """
    Enregistre un signal TA dans ta_signals.csv.
    result = dict retourné par live_runner.scan()

    Retourne le signal_id.
    """
    direction   = result["trigger"]
    last_price  = result["last_price"]
    cs          = result["current_state"]
    matches     = result["matches"]
    last_bar    = result["last_bar_time"]

    if not direction or not matches:
        return ""

    # Calcul TP/SL à partir de l'ATR14 de la dernière bougie
    df15 = result["df15_closed"]
    atr_series = _atr(df15["high"], df15["low"], df15["close"], TP_SL_ATR)
    atr_val    = float(atr_series.iloc[-1])

    if direction == "LONG":
        tp = last_price + TP_MULT * atr_val
        sl = last_price - SL_MULT * atr_val
    else:
        tp = last_price - TP_MULT * atr_val
        sl = last_price + SL_MULT * atr_val

    top = matches[0]
    signal_id = str(uuid.uuid4())[:8]

    row = {
        "signal_id":   signal_id,
        "timestamp":   last_bar,
        "direction":   direction,
        "entry_price": round(last_price, 2),
        "tp":          round(tp, 2),
        "sl":          round(sl, 2),
        "atr_at_entry": round(atr_val, 4),
        "regime":      cs.get("regime", ""),
        "ema_state":   cs.get("ema_state", ""),
        "swing":       cs.get("swing", ""),
        "rsi_state":   cs.get("rsi_state", ""),
        "stoch_state": cs.get("stoch_state", ""),
        "atr_state":   cs.get("atr_state", ""),
        "vwap_state":  cs.get("vwap_state", ""),
        "n_matches":   len(matches),
        "top_params":  top["params"],
        "top_wr_oos":  round(top["wr_OOS"], 4),
        "top_exp_oos": round(top["exp_R_OOS"], 4),
        "outcome":     "pending",
        "exit_price":  None,
        "exit_time":   None,
        "n_bars":      None,
        "r_realized":  None,
    }

    df = _load()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    print(f"[signal_logger] Signal logged: {signal_id} | {direction} @ {last_price:.2f} "
          f"TP={tp:.2f} SL={sl:.2f}", flush=True)
    return signal_id


# ─────────────────────────────────────────────────────────────────────────────
# Résolution des trades pendants
# ─────────────────────────────────────────────────────────────────────────────

def resolve_pending(symbol: str = "BTCUSDT") -> int:
    """
    Vérifie les signaux en état 'pending' et calcule l'outcome
    en comparant TP/SL aux bougies 15m depuis le signal.

    Retourne le nombre de trades résolus.
    """
    df = _load()
    pending = df[df["outcome"] == "pending"]
    if pending.empty:
        return 0

    # Fetch les 300 dernières bougies 15m (couvre MAX_BARS=48 + marge)
    try:
        df15 = fetch_klines("15m", 300, symbol)
    except Exception as e:
        print(f"[signal_logger] Fetch failed: {e}", flush=True)
        return 0

    resolved = 0
    for idx, sig in pending.iterrows():
        entry_time = pd.Timestamp(sig["timestamp"], tz="UTC")
        direction  = sig["direction"]
        tp         = float(sig["tp"])
        sl         = float(sig["sl"])
        entry_price = float(sig["entry_price"])

        # Bougies après l'entrée
        future = df15[df15.index > entry_time]
        if future.empty:
            continue

        outcome    = None
        exit_price = None
        exit_time  = None
        n_bars     = None

        for bar_n, (bar_time, bar) in enumerate(future.iterrows(), 1):
            hit_tp = (bar["high"] >= tp) if direction == "LONG" else (bar["low"] <= tp)
            hit_sl = (bar["low"] <= sl)  if direction == "LONG" else (bar["high"] >= sl)

            if hit_tp and hit_sl:
                # Les deux touchés dans la même bougie → SL en premier (conservateur)
                outcome    = "loss"
                exit_price = sl
                exit_time  = bar_time.isoformat()
                n_bars     = bar_n
                break
            elif hit_tp:
                outcome    = "win"
                exit_price = tp
                exit_time  = bar_time.isoformat()
                n_bars     = bar_n
                break
            elif hit_sl:
                outcome    = "loss"
                exit_price = sl
                exit_time  = bar_time.isoformat()
                n_bars     = bar_n
                break

            if bar_n >= MAX_BARS:
                outcome    = "timeout"
                exit_price = float(bar["close"])
                exit_time  = bar_time.isoformat()
                n_bars     = bar_n
                break

        if outcome is None:
            # Pas encore assez de bougies → reste pending
            continue

        # Calcul R réalisé
        atr = float(sig["atr_at_entry"])
        if atr > 0:
            pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
            r_realized = round(pnl / atr, 4)
        else:
            r_realized = None

        df.at[idx, "outcome"]    = outcome
        df.at[idx, "exit_price"] = round(exit_price, 2)
        df.at[idx, "exit_time"]  = exit_time
        df.at[idx, "n_bars"]     = n_bars
        df.at[idx, "r_realized"] = r_realized
        resolved += 1

        print(f"[signal_logger] Resolved {sig['signal_id']}: {outcome} "
              f"@ {exit_price:.2f} ({r_realized:+.2f}R) in {n_bars} bars", flush=True)

    if resolved:
        _save(df)

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Stats live
# ─────────────────────────────────────────────────────────────────────────────

def live_stats() -> dict:
    """
    Calcule les statistiques sur les signaux résolus.
    Retourne un dict avec WR, exp_R, n par régime et direction.
    """
    df = _load()
    resolved = df[df["outcome"].isin(["win", "loss"])]

    if resolved.empty:
        return {"n_total": 0, "n_pending": len(df[df["outcome"] == "pending"]),
                "wr": None, "exp_R": None, "by_regime": {}, "by_direction": {}}

    wr    = (resolved["outcome"] == "win").mean()
    exp_R = resolved["r_realized"].astype(float).mean()

    by_regime = {}
    for reg, grp in resolved.groupby("regime"):
        wr_r = (grp["outcome"] == "win").mean()
        exp_r = grp["r_realized"].astype(float).mean()
        by_regime[reg] = {"n": len(grp), "wr": round(wr_r, 4), "exp_R": round(exp_r, 4)}

    by_dir = {}
    for d, grp in resolved.groupby("direction"):
        wr_d = (grp["outcome"] == "win").mean()
        exp_d = grp["r_realized"].astype(float).mean()
        by_dir[d] = {"n": len(grp), "wr": round(wr_d, 4), "exp_R": round(exp_d, 4)}

    return {
        "n_total":     len(df),
        "n_pending":   len(df[df["outcome"] == "pending"]),
        "n_resolved":  len(resolved),
        "n_timeout":   len(df[df["outcome"] == "timeout"]),
        "wr":          round(float(wr), 4),
        "exp_R":       round(float(exp_R), 4),
        "by_regime":   by_regime,
        "by_direction": by_dir,
    }


if __name__ == "__main__":
    print("Resolving pending signals...")
    n = resolve_pending()
    print(f"{n} signal(s) resolved.")
    stats = live_stats()
    print(f"\nStats live TA:")
    print(f"  Total    : {stats['n_total']}")
    print(f"  Pending  : {stats['n_pending']}")
    print(f"  Resolved : {stats.get('n_resolved', 0)}")
    if stats["wr"] is not None:
        print(f"  WR       : {stats['wr']:.1%}")
        print(f"  Exp R    : {stats['exp_R']:+.3f}R")
