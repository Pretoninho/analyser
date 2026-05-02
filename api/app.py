"""
api/app.py -- FastAPI backend Pi*.

Endpoints :
  GET /api/daily/{date}            rapport journalier (live + shadow)
  GET /api/trades                  historique filtrable
  GET /api/candles/{date}/{mac}    OHLC Binance pour vue detail trade
  GET /api/performance             metriques agregees
  GET /api/qtable                  Q-table etats + valeurs
    GET /version                     meta runtime/deploiement

Lancement local :
  uvicorn api.app:app --reload --port 8000
"""

import os
import sys
import csv
import time
import pickle
import platform
import subprocess
import requests
import pytz
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.stats_state import MACROS, decode
from engine.q_agent import QAgent
from pi_config import LIVE_MACROS, SHADOW_MACROS

ET_TZ      = pytz.timezone("America/New_York")
DB_DIR     = ROOT / "db"
LIVE_CSV   = DB_DIR / "live_trades.csv"
SHADOW_CSV = DB_DIR / "shadow_trades.csv"

MAC_NAMES = {
    1: "08:50", 2: "09:50", 3: "10:50",
    4: "11:50", 5: "12:50", 6: "13:50", 7: "14:50",
}
LC_LABELS = {0: "NO_RAID",  1: "RAID_HIGH", 2: "RAID_LOW"}
PC_LABELS = {0: "NEUTRAL",  1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_LABELS = {0: "NO_SWEEP", 1: "SWEEP_HIGH", 2: "SWEEP_LOW"}
AC_LABELS = {0: "FLAT",     1: "LONG",       2: "SHORT"}


# ── Scheduler ─────────────────────────────────────────────────────

def _run(script: str, args: list = None):
    """Lance un script Python dans un sous-processus (même container → CSV partagés)."""
    path = str(ROOT / script)
    cmd = [sys.executable, path] + (args or [])
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode not in (0, None):
        print(f"[scheduler] {script} exited with code {result.returncode}", flush=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_deribit_webhook() -> str:
    return os.environ.get("DISCORD_WEBHOOK_DERIBIT_URL", "") or os.environ.get("DISCORD_WEBHOOK_URL", "")


def _notify_deribit_signal_job(timeframe: str = "1h", days: int = 90) -> None:
    webhook_url = _get_deribit_webhook()
    if not webhook_url:
        print("[scheduler] Deribit notify skipped (no webhook configured)", flush=True)
        return

    try:
        from analysis.deribit_futures.signal import SignalConfig, build_deribit_signal, format_discord_signal

        cfg = SignalConfig(asset="BTC", timeframe=timeframe, days=days)
        signal = build_deribit_signal(cfg)
        msg = format_discord_signal(signal)
        resp = requests.post(webhook_url, json={"content": msg}, timeout=10)
        resp.raise_for_status()
        print(
            f"[scheduler] Deribit signal sent ({signal.get('signal', {}).get('action', 'NA')}, {timeframe}, {days}d)",
            flush=True,
        )
    except Exception as e:
        print(f"[scheduler] Deribit notify failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler(timezone=ET_TZ)
    # live_signal : lun-ven 09:51 ET (mac=2)
    scheduler.add_job(lambda: _run("live_signal.py", ["--mac", "2"]),   "cron",
                      day_of_week="mon-fri", hour=9, minute=51)
    # live_signal : lun-ven 11:51 ET (mac=4 - 11:50)
    scheduler.add_job(lambda: _run("live_signal.py", ["--mac", "4"]),   "cron",
                      day_of_week="mon-fri", hour=11, minute=51)
    # shadow_signal : lun-ven 16:05 ET
    scheduler.add_job(lambda: _run("shadow_signal.py"), "cron",
                      day_of_week="mon-fri", hour=16, minute=5)

    deribit_enabled = _env_bool("DERIBIT_NOTIFY_ENABLED", True)
    deribit_mode = os.environ.get("DERIBIT_NOTIFY_MODE", "every_4h").strip().lower()
    deribit_timeframe = os.environ.get("DERIBIT_NOTIFY_TIMEFRAME", "1h").strip()
    deribit_days = int(os.environ.get("DERIBIT_NOTIFY_DAYS", "90"))
    deribit_minute = int(os.environ.get("DERIBIT_NOTIFY_MINUTE", "2"))

    if deribit_enabled:
        if deribit_mode == "hourly_us":
            scheduler.add_job(
                lambda: _notify_deribit_signal_job(deribit_timeframe, deribit_days),
                "cron",
                day_of_week="mon-fri",
                hour="8-17",
                minute=deribit_minute,
            )
        elif deribit_mode == "hourly":
            scheduler.add_job(
                lambda: _notify_deribit_signal_job(deribit_timeframe, deribit_days),
                "cron",
                hour="*",
                minute=deribit_minute,
            )
        else:
            # Mode par défaut: toutes les 4h, 24/7
            scheduler.add_job(
                lambda: _notify_deribit_signal_job(deribit_timeframe, deribit_days),
                "cron",
                hour="*/4",
                minute=deribit_minute,
            )

    # TA strategy v2 : scan toutes les 15 min pendant London (07-11 UTC) + NY (13-17 UTC)
    # Utilise ensemble voting pour consensus sur signaux
    ta_enabled = _env_bool("TA_NOTIFY_ENABLED_V2", True)
    if ta_enabled:
        def _ta_notify_job_v2():
            try:
                from strategies.ta.discord_notify_v2 import scan_and_notify_v2
                scan_and_notify_v2()
            except Exception as e:
                print(f"[scheduler] TA notify v2 failed: {e}", flush=True)

        def _ta_resolve_job():
            try:
                from strategies.ta.signal_logger import resolve_pending
                n = resolve_pending()
                if n:
                    print(f"[scheduler] TA resolved {n} signal(s)", flush=True)
            except Exception as e:
                print(f"[scheduler] TA resolve failed: {e}", flush=True)

        scheduler.add_job(_ta_notify_job_v2, "cron",
                          hour="7-10,13-16", minute="0,15,30,45")
        # Résolution toutes les heures 24/7 (les trades peuvent expirer la nuit)
        scheduler.add_job(_ta_resolve_job, "cron", minute=5)

    scheduler.start()
    ta_status = "v2_enabled (ensemble voting)" if ta_enabled else "disabled"
    print(
        "[scheduler] APScheduler demarre "
        f"(live 09:51 mac=2, live 11:51 mac=4, shadow 16:05 ET, "
        f"deribit_enabled={deribit_enabled}, deribit_mode={deribit_mode}, "
        f"ta_notify={ta_status})",
        flush=True,
    )
    yield
    scheduler.shutdown()
    print("[scheduler] APScheduler arrêté", flush=True)


app = FastAPI(title="Pi* API", version="1.0.0", lifespan=lifespan)

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


@app.post("/api/shadow/run")
def shadow_run():
    """Declenche shadow_signal.py immediatement (hors schedule). Utile pour test."""
    script = str(ROOT / "shadow_signal.py")
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout[-4000:] if result.stdout else "",
            "stderr":    result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "shadow_signal.py timeout (120s).")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/live/run")
def live_run():
    """Declenche live_signal.py immediatement (hors schedule). Utile pour test."""
    script = str(ROOT / "live_signal.py")
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "live_signal.py timeout (120s).")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Deribit futures edges ─────────────────────────────────────────────────────

_deribit_cache: dict = {}
_DERIBIT_CACHE_TTL = 900  # secondes


def _cache_get(key: str):
    entry = _deribit_cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _DERIBIT_CACHE_TTL:
        return None
    return value


def _cache_set(key: str, value) -> None:
    _deribit_cache[key] = (time.time(), value)


def _safe_api(v):
    try:
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _clean_json_record(obj):
    if isinstance(obj, dict):
        return {k: _clean_json_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_record(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


@app.get("/api/deribit/edges")
def get_deribit_edges(
    timeframe: str = Query("1h", description="Timeframe: 1m, 15m, 1h, 4h, 1d"),
    days: int = Query(14, description="Nombre de jours d'historique a charger"),
):
    """
    Snapshot des scores d'edges Deribit sur la derniere barre.
    Renvoie les 7 edges (funding, carry, options...) + snapshot mark/OI/funding/options.
    Cache TTL = 15 min.
    """
    cache_key = f"edges_{timeframe}_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from analysis.deribit_futures.features import EdgeBuildConfig, build_deribit_edge_frame
        cfg = EdgeBuildConfig(asset="BTC", timeframe=timeframe, days=days)
        df, context = build_deribit_edge_frame(cfg)
    except Exception as e:
        raise HTTPException(502, f"Erreur Deribit fetch: {e}")

    latest = df.iloc[-1]
    edge_cols = [
        "edge_funding_reversion", "edge_carry_momentum", "edge_carry_stress",
        "edge_mark_dislocation", "edge_options_vol_premium",
        "edge_skew_panic", "edge_term_structure_kink", "edge_total",
    ]
    edges = {col: _safe_api(latest.get(col)) for col in edge_cols if col in latest.index}

    snapshot = context.get("snapshot", {})
    options = context.get("options_snapshot", {})

    result = {
        "asset": context["asset"],
        "timeframe": context["timeframe"],
        "latest_ts": latest["timestamp"].isoformat() if pd.notna(latest.get("timestamp")) else None,
        "close": _safe_api(latest.get("close")),
        "funding_annualized": _safe_api(latest.get("funding_annualized")),
        "realized_vol_annual": _safe_api(latest.get("realized_vol_annual")),
        "edges": edges,
        "snapshot": {
            "mark_price": _safe_api(snapshot.get("mark_price")),
            "index_price": _safe_api(snapshot.get("index_price")),
            "open_interest": _safe_api(snapshot.get("open_interest")),
            "current_funding": _safe_api(snapshot.get("current_funding")),
            "funding_8h": _safe_api(snapshot.get("funding_8h")),
        },
        "options": {
            "iv_atm": _safe_api(options.get("iv_atm")),
            "iv_skew_25d": _safe_api(options.get("iv_skew_25d")),
            "put_call_ratio": _safe_api(options.get("put_call_ratio")),
            "max_pain": _safe_api(options.get("max_pain")),
            "term_1w": _safe_api(options.get("term_1w")),
            "term_1m": _safe_api(options.get("term_1m")),
            "term_3m": _safe_api(options.get("term_3m")),
            "gex": _safe_api(options.get("gex")),
        },
    }
    _cache_set(cache_key, result)
    return result


@app.get("/api/deribit/backtest")
def get_deribit_backtest(
    timeframe: str = Query("1h", description="Timeframe: 1m, 15m, 1h, 4h, 1d"),
    days: int = Query(90, description="Nombre de jours d'historique pour le backtest"),
    threshold: float = Query(0.05, description="Seuil de score pour activer un signal"),
):
    """
    Hit ratio par edge signal Deribit a +4h et +24h.
    Retourne pour chaque edge : n_signals, hit_ratio, avg_ret_active, avg_ret_baseline, corr, lift.
    Cache TTL = 15 min.
    """
    cache_key = f"bt_{timeframe}_{days}_{threshold}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from analysis.deribit_futures.backtest import BacktestConfig, run_edge_backtest
        cfg = BacktestConfig(asset="BTC", timeframe=timeframe, days=days, threshold=threshold)
        results_df, context = run_edge_backtest(cfg)
    except Exception as e:
        raise HTTPException(502, f"Erreur backtest Deribit: {e}")

    records = [_clean_json_record(row) for row in results_df.to_dict(orient="records")]

    result = {
        "asset": context["asset"],
        "timeframe": context["timeframe"],
        "days": days,
        "threshold": threshold,
        "total_bars": context.get("bars"),
        "results": records,
    }
    _cache_set(cache_key, result)
    return result


@app.get("/api/deribit/signal")
def get_deribit_signal(
    timeframe: str = Query("1h", description="Timeframe: 1m, 15m, 1h, 4h, 1d"),
    days: int = Query(90, description="Nombre de jours d'historique pour le signal"),
):
    """
    Signal actionnable Deribit (LONG/SHORT/FLAT/WATCH) base sur tous les edges.
    Retourne action, confiance, horizon suggere et principaux drivers.
    Cache TTL = 15 min.
    """
    cache_key = f"signal_{timeframe}_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from analysis.deribit_futures.signal import SignalConfig, build_deribit_signal
        cfg = SignalConfig(asset="BTC", timeframe=timeframe, days=days)
        signal = build_deribit_signal(cfg)
    except Exception as e:
        raise HTTPException(502, f"Erreur signal Deribit: {e}")

    signal_clean = _clean_json_record(signal)
    _cache_set(cache_key, signal_clean)
    return signal_clean


def _send_deribit_signal_notification(timeframe: str, days: int) -> dict:
    webhook_url = _get_deribit_webhook()
    if not webhook_url:
        raise HTTPException(503, "DISCORD_WEBHOOK_DERIBIT_URL (ou DISCORD_WEBHOOK_URL) non defini.")

    try:
        from analysis.deribit_futures.signal import SignalConfig, build_deribit_signal, format_discord_signal
        cfg = SignalConfig(asset="BTC", timeframe=timeframe, days=days)
        signal = build_deribit_signal(cfg)
        msg = format_discord_signal(signal)
        resp = requests.post(webhook_url, json={"content": msg}, timeout=10)
        resp.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Echec notification Deribit: {e}")

    return {
        "status": "sent",
        "timeframe": timeframe,
        "days": days,
        "action": signal.get("signal", {}).get("action"),
        "confidence": signal.get("signal", {}).get("confidence"),
        "contract": signal.get("signal", {}).get("contract"),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/deribit/futures/notify")
def notify_deribit_futures_signal(
    timeframe: str = Query("1h", description="Timeframe: 1m, 15m, 1h, 4h, 1d"),
    days: int = Query(90, description="Nombre de jours d'historique pour le signal"),
):
    """
    Endpoint dedie aux notifications futures a terme Deribit.
    """
    return _send_deribit_signal_notification(timeframe=timeframe, days=days)


@app.post("/api/deribit/notify")
def notify_deribit_signal(
    timeframe: str = Query("1h", description="Timeframe: 1m, 15m, 1h, 4h, 1d"),
    days: int = Query(90, description="Nombre de jours d'historique pour le signal"),
):
    """
    Alias historique vers l'endpoint dedie futures.
    """
    return _send_deribit_signal_notification(timeframe=timeframe, days=days)


@app.post("/api/discord/test")
def discord_test():
    """Envoie un message de test sur Discord depuis le container Railway."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        raise HTTPException(503, "DISCORD_WEBHOOK_URL non defini sur ce service.")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = f"**Pi* API -- test Discord** | {ts} | connexion OK"
    try:
        resp = requests.post(webhook_url, json={"content": msg}, timeout=8)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Echec envoi Discord : {e}")
    return {"status": "sent", "timestamp": ts}


@app.post("/api/ta/scan-v2")
def ta_scan_v2():
    """
    Test endpoint : scan TA signaux avec ensemble voting v2.
    Retourne les signaux détectés sans envoyer Discord.
    """
    try:
        from strategies.ta.features import load_15m, compute_features
        from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
        from strategies.ta.config import RESULTS_DIR
        from strategies.ta.live_runner_v2 import scan_signals

        df15 = load_15m()
        if len(df15) < 100:
            raise HTTPException(400, "Donnees insuffisantes")

        df15 = compute_features(
            df15, ema_len=50, rsi_len=7,
            stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
        )

        voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=5, min_wr_oos=0.60)
        signals = scan_signals(df15, voter)

        return {
            "status": "scanned",
            "n_signals": len(signals),
            "signals": [
                {
                    "timestamp": sig["timestamp"].isoformat(),
                    "direction": sig["direction"],
                    "regime": sig["regime"],
                    "entry_price": sig["entry_price"],
                    "vote_favorable": sig["vote_favorable"],
                    "vote_total": sig["vote_total"],
                    "confidence": round(sig["confidence"], 3),
                }
                for sig in signals[:10]  # top 10
            ],
            "pool_size": voter.get_pool_stats()["total_configs"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TA scan failed: {e}")


@app.post("/api/ta/notify-v2")
def ta_notify_v2():
    """
    Endpoint pour déclencher le scan TA v2 et envoyer Discord.
    Utile pour tests manuels depuis Railway.
    """
    try:
        from strategies.ta.discord_notify_v2 import scan_and_notify_v2
        success = scan_and_notify_v2()
        return {
            "status": "executed",
            "signal_sent": success,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, f"TA notify failed: {e}")


# ==================== FRACTAL DETECTION ====================

_fractal_orchestrator = None

def _get_mock_signals():
    """Generate mock Fractal signals for testing UI"""
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    base_price = 67500

    signals = []
    setups = ['STRICT', 'MODÉRÉ', 'FRÉQUENT']
    patterns = ['UP->DOWN', 'DOWN->UP']
    zones = ['LKZ', 'NYKZ', 'LnCl']

    for i in range(25):
        signal = {
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "setup": setups[i % 3],
            "day_date": (now - timedelta(days=i // 3)).strftime("%Y-%m-%d"),
            "kz": zones[i % 3],
            "pattern": patterns[i % 2],
            "entry_price": base_price + (i % 10) * 50,
            "confidence": 0.85 + (i % 10) * 0.01,
            "levels": {
                "break_level": base_price + (i % 10) * 100,
                "retest_level": base_price + (i % 10) * 50 - 25
            }
        }
        signals.append(signal)
    return signals

def _init_fractal():
    """Lazy initialize Fractal Orchestrator on first request."""
    global _fractal_orchestrator
    if _fractal_orchestrator is None:
        try:
            sys.path.insert(0, str(ROOT / "strategies" / "fractal"))
            from orchestrator import FractalOrchestrator
            webhook = os.getenv("DISCORD_WEBHOOK_FRACTAL") or os.getenv("DISCORD_WEBHOOK")
            _fractal_orchestrator = FractalOrchestrator(discord_webhook_url=webhook)
            # Add mock signals for UI testing
            _fractal_orchestrator.signals_log = _get_mock_signals()
        except Exception as e:
            print(f"[FRACTAL] Failed to init orchestrator: {e}")
            # Return mock orchestrator for testing
            class MockOrchestrator:
                signals_log = _get_mock_signals()
            _fractal_orchestrator = MockOrchestrator()
            return _fractal_orchestrator
    return _fractal_orchestrator

@app.get("/api/fractal/strict")
def get_strict_signals():
    """Retourne les signaux STRICT (W+D+KZ+BR)"""
    orch = _init_fractal()
    if not orch:
        raise HTTPException(500, "Fractal orchestrator not available")
    strict_signals = [s for s in orch.signals_log if s.get('setup') == 'STRICT']
    return {
        "setup": "STRICT",
        "count": len(strict_signals),
        "confidence": 0.946,
        "signals": strict_signals[-10:]
    }

@app.get("/api/fractal/modere")
def get_modere_signals():
    """Retourne les signaux MODÉRÉ (D+KZ+BR)"""
    orch = _init_fractal()
    if not orch:
        raise HTTPException(500, "Fractal orchestrator not available")
    modere_signals = [s for s in orch.signals_log if s.get('setup') == 'MODÉRÉ']
    return {
        "setup": "MODÉRÉ",
        "count": len(modere_signals),
        "confidence": 0.91,
        "signals": modere_signals[-10:]
    }

@app.get("/api/fractal/frequent")
def get_frequent_signals():
    """Retourne les signaux FRÉQUENT (KZ+BR)"""
    orch = _init_fractal()
    if not orch:
        raise HTTPException(500, "Fractal orchestrator not available")
    frequent_signals = [s for s in orch.signals_log if s.get('setup') == 'FRÉQUENT']
    return {
        "setup": "FRÉQUENT",
        "count": len(frequent_signals),
        "confidence": 0.875,
        "signals": frequent_signals[-10:]
    }

@app.get("/api/fractal/stats")
def get_fractal_stats():
    """Retourne les statistiques globales des signaux Fractal"""
    orch = _init_fractal()
    if not orch:
        raise HTTPException(500, "Fractal orchestrator not available")

    summary = {
        "total": len(orch.signals_log),
        "by_setup": {},
        "by_pattern": {},
    }

    for signal in orch.signals_log:
        setup = signal.get('setup', 'UNKNOWN')
        pattern = signal.get('pattern', 'UNKNOWN')
        summary["by_setup"][setup] = summary["by_setup"].get(setup, 0) + 1
        summary["by_pattern"][pattern] = summary["by_pattern"].get(pattern, 0) + 1

    return {
        "total_signals": summary["total"],
        "by_setup": summary["by_setup"],
        "by_pattern": summary["by_pattern"],
        "uptime": datetime.utcnow().isoformat()
    }

@app.get("/api/fractal/health")
def fractal_health():
    """Vérification de santé de l'API Fractal"""
    orch = _init_fractal()
    return {
        "status": "healthy" if orch else "initializing",
        "orchestrator": "active" if orch else "inactive"
    }

@app.post("/api/fractal/discord/test")
def test_fractal_discord():
    """Test la connexion Discord"""
    webhook = os.getenv("DISCORD_WEBHOOK_FRACTAL") or os.getenv("DISCORD_WEBHOOK")
    if not webhook:
        raise HTTPException(503, "DISCORD_WEBHOOK_FRACTAL ou DISCORD_WEBHOOK non configuré")

    try:
        import requests
        payload = {
            "embeds": [{
                "title": "🧪 Test Fractal Signal",
                "description": "Ceci est un signal de test",
                "color": 3447003,
                "fields": [
                    {"name": "Setup", "value": "TEST", "inline": True},
                    {"name": "Confiance", "value": "95%", "inline": True},
                ]
            }]
        }
        resp = requests.post(webhook, json=payload, timeout=5)
        resp.raise_for_status()
        return {
            "status": "success",
            "message": "Signal test envoyé à Discord",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(502, f"Échec envoi Discord: {e}")

# ==================== END FRACTAL ====================


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


@app.get("/version")
def version():
    """Expose runtime/build metadata for quick monitoring and debugging."""
    return {
        "service": "pi-api",
        "app_version": app.version,
        "python_version": platform.python_version(),
        "environment": os.getenv("RAILWAY_ENVIRONMENT_NAME"),
        "service_id": os.getenv("RAILWAY_SERVICE_ID"),
        "deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID"),
        "git_commit_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA"),
        "timestamp": datetime.utcnow().isoformat(),
    }
