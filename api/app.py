"""
api/app.py -- FastAPI backend Pi*.

Endpoints :
  GET /api/daily/{date}            rapport journalier (live + shadow)
  GET /api/trades                  historique filtrable
  GET /api/candles/{date}/{mac}    OHLC Binance pour vue detail trade
  GET /api/performance             metriques agregees
  GET /api/qtable                  Q-table etats + valeurs

Lancement local :
  uvicorn api.app:app --reload --port 8000
"""

import os
import sys
import csv
import pickle
import requests
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.stats_state import MACROS, decode
from engine.q_agent import QAgent
from pi_config import LIVE_MACROS, SHADOW_MACROS

ET_TZ     = pytz.timezone("America/New_York")
DB_DIR    = ROOT / "db"
LIVE_CSV  = DB_DIR / "live_trades.csv"
SHADOW_CSV = DB_DIR / "shadow_trades.csv"

MAC_NAMES = {
    1: "08:50", 2: "09:50", 3: "10:50",
    4: "11:50", 5: "12:50", 6: "13:50", 7: "14:50",
}
LC_LABELS = {0: "NO_RAID",  1: "RAID_HIGH", 2: "RAID_LOW"}
PC_LABELS = {0: "NEUTRAL",  1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_LABELS = {0: "NO_SWEEP", 1: "SWEEP_HIGH", 2: "SWEEP_LOW"}
AC_LABELS = {0: "FLAT",     1: "LONG",       2: "SHORT"}

app = FastAPI(title="Pi* API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    """Charge et fusionne live_trades.csv + shadow_trades.csv."""
    frames = []

    for path, trade_type in [(LIVE_CSV, "live"), (SHADOW_CSV, "shadow")]:
        if path.exists():
            df = pd.read_csv(path)
            df["type"] = trade_type
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Nettoyage types
    for col in ("mac_idx", "mc", "dc", "lc", "sc", "pc", "state", "action", "n_candles"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("q_val", "entry_px", "tp_px", "sl_px", "pnl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["would_trade"] = df["would_trade"].astype(str).str.lower() == "true"
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df.sort_values("date", ascending=False).reset_index(drop=True)


def load_qtable():
    path = DB_DIR / "stats_agent.pkl"
    if not path.exists():
        return None
    try:
        return QAgent.load(str(path))
    except Exception as e:
        print(f"[api] Impossible de charger stats_agent.pkl : {e}")
        return None


def fetch_binance_candles(date_et: date, mac_idx: int) -> list:
    """
    Recupere les bougies 1m autour d'une macro depuis Binance.
    Fenetre : 45 min avant mac_start → 120 min apres.
    """
    mac_start, _ = MACROS[mac_idx]

    # Calcul timestamp UTC de debut (45 min avant la macro)
    ts_day  = pd.Timestamp(date_et).tz_localize(ET_TZ)
    offset  = (mac_start - 45) * 60  # secondes
    ts_from = ts_day + pd.Timedelta(seconds=offset)
    ts_from_ms = int(ts_from.timestamp() * 1000)

    base = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
    resp = requests.get(
        f"{base}/api/v3/klines",
        params={
            "symbol":    "BTCUSDT",
            "interval":  "1m",
            "startTime": ts_from_ms,
            "limit":     165,   # 45 + 120 = 165 bougies
        },
        timeout=15,
    )
    resp.raise_for_status()

    candles = []
    for row in resp.json():
        ts_s = int(row[0]) // 1000  # Unix seconds pour TradingView
        candles.append({
            "time":  ts_s,
            "open":  float(row[1]),
            "high":  float(row[2]),
            "low":   float(row[3]),
            "close": float(row[4]),
        })
    return candles


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Calcule WR, Return, Sharpe, PF, MaxDD sur un DataFrame de trades."""
    df = trades_df[trades_df["would_trade"] == True].copy()
    if df.empty:
        return {"n": 0, "wr": None, "total_return": None,
                "sharpe": None, "profit_factor": None, "max_dd": None}

    df = df.dropna(subset=["pnl"])
    n       = len(df)
    wins    = df[df["pnl"] > 0]
    losses  = df[df["pnl"] < 0]
    wr      = len(wins) / n if n > 0 else 0
    ret     = float(df["pnl"].sum())
    avg     = float(df["pnl"].mean())
    std     = float(df["pnl"].std()) if n > 1 else 0
    sharpe  = avg / std * (252 ** 0.5) if std > 0 else 0
    gross_p = float(wins["pnl"].sum())
    gross_l = abs(float(losses["pnl"].sum()))
    pf      = gross_p / gross_l if gross_l > 0 else None

    # Max drawdown
    cumret  = df["pnl"].cumsum()
    peak    = cumret.cummax()
    dd      = (cumret - peak)
    max_dd  = float(dd.min())

    return {
        "n": n, "wr": round(wr, 4),
        "total_return": round(ret, 6),
        "avg_trade": round(avg, 6),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(pf, 4) if pf else None,
        "max_dd": round(max_dd, 6),
    }


# ── Routes ────────────────────────────────────────────────────────

@app.get("/api/daily/{date_str}")
def get_daily(date_str: str):
    """Rapport journalier : toutes les macros d'une date."""
    try:
        day = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(400, "Format date invalide. Utiliser YYYY-MM-DD.")

    df = load_trades()
    if df.empty:
        return {"date": date_str, "macros": [], "summary": {}}

    day_df = df[df["date"] == day]

    macros = []
    for mac_idx in sorted(MACROS.keys()):
        row = day_df[day_df["mac_idx"] == mac_idx]
        if row.empty:
            macros.append({
                "mac_idx":  mac_idx,
                "mac_name": MAC_NAMES[mac_idx],
                "type":     "live" if mac_idx in LIVE_MACROS else "shadow",
                "status":   "no_data",
            })
            continue

        r = row.iloc[0]
        entry = {
            "mac_idx":    int(r["mac_idx"]),
            "mac_name":   MAC_NAMES[mac_idx],
            "type":       str(r["type"]),
            "lc":         int(r["lc"]) if not np.isnan(r["lc"]) else -1,
            "sc":         int(r["sc"]) if not np.isnan(r["sc"]) else -1,
            "pc":         int(r["pc"]) if not np.isnan(r["pc"]) else -1,
            "lc_label":   LC_LABELS.get(int(r["lc"]), "?") if not np.isnan(r["lc"]) else "?",
            "sc_label":   SC_LABELS.get(int(r["sc"]), "?") if not np.isnan(r["sc"]) else "?",
            "pc_label":   PC_LABELS.get(int(r["pc"]), "?") if not np.isnan(r["pc"]) else "?",
            "action":     int(r["action"]) if not np.isnan(r["action"]) else 0,
            "direction":  AC_LABELS.get(int(r["action"]), "FLAT") if not np.isnan(r["action"]) else "FLAT",
            "q_val":      float(r["q_val"]) if pd.notna(r["q_val"]) else 0.0,
            "would_trade": bool(r["would_trade"]),
            "flat_reason": str(r.get("flat_reason", "")),
            "entry_px":   float(r["entry_px"]) if pd.notna(r.get("entry_px")) else None,
            "tp_px":      float(r["tp_px"])    if pd.notna(r.get("tp_px"))    else None,
            "sl_px":      float(r["sl_px"])    if pd.notna(r.get("sl_px"))    else None,
            "pnl":        float(r["pnl"])      if pd.notna(r.get("pnl"))      else None,
            "exit_reason": str(r.get("exit_reason", "")),
            "n_candles":  int(r["n_candles"])  if pd.notna(r.get("n_candles")) else 0,
        }
        macros.append(entry)

    # Summary
    traded = [m for m in macros if m.get("would_trade")]
    live_pnl   = sum(m["pnl"] for m in traded if m["type"] == "live"   and m["pnl"] is not None)
    shadow_pnl = sum(m["pnl"] for m in traded if m["type"] == "shadow" and m["pnl"] is not None)

    return {
        "date": date_str,
        "macros": macros,
        "summary": {
            "live_trades":  len([m for m in traded if m["type"] == "live"]),
            "live_pnl":     round(live_pnl, 6),
            "shadow_trades": len([m for m in traded if m["type"] == "shadow"]),
            "shadow_pnl":   round(shadow_pnl, 6),
        },
    }


@app.get("/api/trades")
def get_trades(
    trade_type:  Optional[str] = Query(None, alias="type"),
    mac_idx:     Optional[int] = None,
    exit_reason: Optional[str] = None,
    from_date:   Optional[str] = Query(None, alias="from"),
    to_date:     Optional[str] = Query(None, alias="to"),
    limit:       int = 200,
):
    """Historique des trades avec filtres optionnels."""
    df = load_trades()
    if df.empty:
        return {"trades": [], "total": 0, "metrics": {}}

    df = df[df["would_trade"] == True]

    if trade_type:
        df = df[df["type"] == trade_type]
    if mac_idx is not None:
        df = df[df["mac_idx"] == mac_idx]
    if exit_reason:
        df = df[df["exit_reason"].str.upper() == exit_reason.upper()]
    if from_date:
        df = df[df["date"] >= date.fromisoformat(from_date)]
    if to_date:
        df = df[df["date"] <= date.fromisoformat(to_date)]

    metrics = compute_metrics(df)
    df = df.head(limit)

    trades = []
    for _, r in df.iterrows():
        trades.append({
            "date":         str(r["date"]),
            "mac_idx":      int(r["mac_idx"]),
            "mac_name":     MAC_NAMES.get(int(r["mac_idx"]), "?"),
            "type":         str(r["type"]),
            "direction":    AC_LABELS.get(int(r["action"]), "FLAT"),
            "lc_label":     LC_LABELS.get(int(r["lc"]), "?") if pd.notna(r["lc"]) else "?",
            "sc_label":     SC_LABELS.get(int(r["sc"]), "?") if pd.notna(r["sc"]) else "?",
            "pc_label":     PC_LABELS.get(int(r["pc"]), "?") if pd.notna(r["pc"]) else "?",
            "q_val":        float(r["q_val"]) if pd.notna(r["q_val"]) else 0.0,
            "entry_px":     float(r["entry_px"]) if pd.notna(r.get("entry_px")) else None,
            "tp_px":        float(r["tp_px"])    if pd.notna(r.get("tp_px"))    else None,
            "sl_px":        float(r["sl_px"])    if pd.notna(r.get("sl_px"))    else None,
            "pnl":          float(r["pnl"])      if pd.notna(r.get("pnl"))      else None,
            "exit_reason":  str(r.get("exit_reason", "")),
            "n_candles":    int(r["n_candles"])  if pd.notna(r.get("n_candles")) else 0,
        })

    return {"trades": trades, "total": len(trades), "metrics": metrics}


@app.get("/api/candles/{date_str}/{mac_idx}")
def get_candles(date_str: str, mac_idx: int):
    """
    OHLC 1m autour d'une macro pour TradingView Lightweight Charts.
    Retourne aussi entry/TP/SL/pre_start/mac_start si trade trouve.
    """
    try:
        day = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(400, "Format date invalide.")

    if mac_idx not in MACROS:
        raise HTTPException(400, f"mac_idx {mac_idx} inconnu.")

    # Candles Binance
    try:
        candles = fetch_binance_candles(day, mac_idx)
    except Exception as e:
        raise HTTPException(502, f"Erreur Binance : {e}")

    # Infos trade si disponible
    df    = load_trades()
    trade = None
    if not df.empty:
        row = df[(df["date"] == day) & (df["mac_idx"] == mac_idx) & (df["would_trade"] == True)]
        if not row.empty:
            r = row.iloc[0]
            trade = {
                "direction":   AC_LABELS.get(int(r["action"]), "FLAT"),
                "entry_px":    float(r["entry_px"]) if pd.notna(r.get("entry_px")) else None,
                "tp_px":       float(r["tp_px"])    if pd.notna(r.get("tp_px"))    else None,
                "sl_px":       float(r["sl_px"])    if pd.notna(r.get("sl_px"))    else None,
                "pnl":         float(r["pnl"])      if pd.notna(r.get("pnl"))      else None,
                "exit_reason": str(r.get("exit_reason", "")),
                "n_candles":   int(r["n_candles"])  if pd.notna(r.get("n_candles")) else 0,
            }

    mac_start, _ = MACROS[mac_idx]
    return {
        "date":       date_str,
        "mac_idx":    mac_idx,
        "mac_name":   MAC_NAMES[mac_idx],
        "mac_start":  mac_start,
        "pre_start":  mac_start - 20,
        "candles":    candles,
        "trade":      trade,
    }


@app.get("/api/performance")
def get_performance(
    trade_type: Optional[str] = Query(None, alias="type"),
):
    """Metriques agregees globales + par macro + par mois."""
    df = load_trades()
    if df.empty:
        return {"overall": {}, "by_macro": [], "by_month": [], "pnl_curve": []}

    if trade_type:
        df = df[df["type"] == trade_type]

    # Global
    overall = compute_metrics(df)

    # Par macro
    by_macro = []
    for mac_idx in sorted(MACROS.keys()):
        sub = df[df["mac_idx"] == mac_idx]
        m   = compute_metrics(sub)
        m["mac_idx"]  = mac_idx
        m["mac_name"] = MAC_NAMES[mac_idx]
        m["type"]     = "live" if mac_idx in LIVE_MACROS else "shadow"
        by_macro.append(m)

    # Par mois
    df2 = df[df["would_trade"] == True].dropna(subset=["pnl"]).copy()
    df2["month"] = pd.to_datetime(df2["date"]).dt.to_period("M").astype(str)
    by_month = []
    for month, grp in df2.groupby("month"):
        by_month.append({
            "month":        month,
            "n":            len(grp),
            "total_return": round(float(grp["pnl"].sum()), 6),
            "wr":           round(len(grp[grp["pnl"] > 0]) / len(grp), 4),
        })

    # Courbe P&L cumulee
    df2 = df2.sort_values("date")
    df2["cumulative_pnl"] = df2["pnl"].cumsum()
    pnl_curve = [
        {"date": str(r["date"]), "pnl": round(float(r["cumulative_pnl"]), 6)}
        for _, r in df2.iterrows()
    ]

    return {
        "overall":   overall,
        "by_macro":  by_macro,
        "by_month":  by_month,
        "pnl_curve": pnl_curve,
    }


@app.get("/api/qtable")
def get_qtable(mac_idx: Optional[int] = None):
    """
    Etats Q-table avec valeurs et N visites.
    Filtre optionnel par mac_idx.
    """
    agent = load_qtable()
    if agent is None:
        raise HTTPException(404, "stats_agent.pkl introuvable.")

    q_table = agent.q_table  # shape (N_STATES, 3)

    states = []
    for state_id in range(q_table.shape[0]):
        dims = decode(state_id)
        if mac_idx is not None and dims["macro_ctx"] != mac_idx:
            continue

        q_vals = q_table[state_id]
        best   = int(np.argmax(q_vals))

        # N visites : si l'agent a un attribut visits/counts
        n = 0
        if hasattr(agent, "visits"):
            n = int(agent.visits[state_id]) if hasattr(agent.visits, "__getitem__") else 0
        elif hasattr(agent, "counts"):
            n = int(agent.counts[state_id].sum())

        states.append({
            "state":      state_id,
            "mc":         dims["month_ctx"],
            "dc":         dims["day_ctx"],
            "lc":         dims["london_ctx"],
            "mac":        dims["macro_ctx"],
            "mac_name":   MAC_NAMES.get(dims["macro_ctx"], "?"),
            "sc":         dims["sweep_ctx"],
            "pc":         dims["pool_ctx"],
            "lc_label":   LC_LABELS.get(dims["london_ctx"], "?"),
            "sc_label":   SC_LABELS.get(dims["sweep_ctx"], "?"),
            "pc_label":   PC_LABELS.get(dims["pool_ctx"], "?"),
            "q_flat":     round(float(q_vals[0]), 6),
            "q_long":     round(float(q_vals[1]), 6),
            "q_short":    round(float(q_vals[2]), 6),
            "best_action": AC_LABELS[best],
            "n":          n,
        })

    # Filtrer etats vides (tous q=0 et n=0)
    states = [s for s in states if any(abs(s[k]) > 1e-9 for k in ("q_long", "q_short"))]

    return {"states": states, "total": len(states)}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
def root():
    return {
        "service": "pi-api",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }
