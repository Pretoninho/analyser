"""live_signal.py -- Signal live Pi* a l'ouverture d'une macro ICT.

Usage :
  python live_signal.py [--mac MAC_IDX]
  MAC_IDX defaut = 2 (09:50 ET)

Declenchement Railway APScheduler (dans api/app.py) :
  09:51 ET  --mac 2
  11:51 ET  --mac 4

Variables d'environnement requises :
  DISCORD_WEBHOOK_URL

Optionnelles :
  BINANCE_BASE_URL      (defaut: https://api.binance.com)
  USE_MICROSTRUCTURE    (defaut: 1)
  MICRO_OFI_THRESHOLD   (defaut: 0.10)
  MICRO_ALLOW_NEUTRAL   (defaut: 1)
"""

import argparse
import os
import sys
import requests
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Parse args (avant tout import config) ────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mac", type=int, default=2,
                   help="mac_idx ICT a trader (2=09:50, 4=11:50, ...)")
    return p.parse_args()

_ARGS = _parse_args()

# ── Config active Pi* (source : pi_config.py) ────────────────────
from pi_config import (
    SL_PCT, RR, EXIT_HM, Q_THRESHOLD, ALIGNED_ONLY, SKIP_DAYS, MACRO_RULES,
    LIVE_MACROS,
)

# ── Constantes macro derivees depuis mac_idx ─────────────────────
# Heures ET en minutes depuis minuit (source : engine/stats_state.py MACROS)
_MAC_STARTS = {
    1: 530,   # 08:50
    2: 590,   # 09:50
    3: 650,   # 10:50
    4: 710,   # 11:50
    5: 770,   # 12:50
    6: 830,   # 13:50
    7: 890,   # 14:50
}
_MAC_NAMES = {
    1: "08:50", 2: "09:50", 3: "10:50",
    4: "11:50", 5: "12:50", 6: "13:50", 7: "14:50",
}

MAC_IDX    = _ARGS.mac
MAC_NAME   = _MAC_NAMES[MAC_IDX]
MAC_START  = _MAC_STARTS[MAC_IDX]
PRE_START  = MAC_START - 20   # 20 min avant la macro
REF_WINDOW = 240              # 4h lookback pour BSL/SSL

# Fenetre de tolerance : ±5 min autour de MAC_START
WINDOW_LOW  = MAC_START - 5
WINDOW_HIGH = MAC_START + 5

ET_TZ = pytz.timezone("America/New_York")

# ── Imports engine ────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.stats_state import (
    encode, compute_pool_ctx, month_ctx, day_ctx,
    attach_microstructure_overlay, microstructure_trade_allowed,
)
from engine.q_agent import QAgent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# ── Binance API ───────────────────────────────────────────────────

def fetch_klines(interval: str, limit: int) -> pd.DataFrame:
    base = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
    url  = f"{base}/api/v3/klines"
    symbol = os.environ.get("TRADING_SYMBOL", "BTCUSDT").upper()
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    resp.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(resp.json(), columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "tbb"):
        df[c] = df[c].astype(float)
    out = df[["timestamp", "open", "high", "low", "close", "volume", "tbb"]].copy()
    out = out.rename(columns={"tbb": "taker_buy_vol"})
    return out


# ── Contextes ─────────────────────────────────────────────────────

def compute_pwh_pwl(daily_df: pd.DataFrame, today_et: date):
    df = daily_df.copy()
    df["ts_et"]   = df["timestamp"].dt.tz_convert(ET_TZ)
    df["iso_year"] = df["ts_et"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["ts_et"].dt.isocalendar().week.astype(int)

    ts_today  = pd.Timestamp(today_et).tz_localize(ET_TZ)
    cur_year  = int(ts_today.isocalendar().year)
    cur_week  = int(ts_today.isocalendar().week)
    prev_week = cur_week - 1
    prev_year = cur_year
    if prev_week == 0:
        prev_year -= 1
        prev_week  = date(prev_year, 12, 28).isocalendar().week

    mask = (df["iso_year"] == prev_year) & (df["iso_week"] == prev_week)
    sub  = df[mask]
    if sub.empty:
        return None, None
    return float(sub["high"].max()), float(sub["low"].min())


def compute_london_ctx(today_df: pd.DataFrame):
    hm_utc = today_df["timestamp"].dt.hour * 60 + today_df["timestamp"].dt.minute
    asia_m = (hm_utc >= 60)  & (hm_utc < 300)
    ldn_m  = (hm_utc >= 420) & (hm_utc < 600)

    asia_h = float(today_df["high"][asia_m].max()) if asia_m.any() else None
    asia_l = float(today_df["low"][asia_m].min())  if asia_m.any() else None
    ldn_h  = float(today_df["high"][ldn_m].max())  if ldn_m.any() else None
    ldn_l  = float(today_df["low"][ldn_m].min())   if ldn_m.any() else None

    lc = 0
    if asia_h is not None and ldn_h is not None:
        raid_h = ldn_h > asia_h
        raid_l = ldn_l < asia_l
        if raid_h and not raid_l:
            lc = 1
        elif raid_l and not raid_h:
            lc = 2
    return lc, ldn_h, ldn_l


# ── Discord ───────────────────────────────────────────────────────

LC_LABELS = {0: "NO_RAID", 1: "RAID_HIGH", 2: "RAID_LOW"}
PC_LABELS  = {0: "NEUTRAL", 1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_LABELS  = {0: "NO_SWEEP", 1: "SWEEP_HIGH", 2: "SWEEP_LOW"}


def build_message(action: int, direction: str, entry: float,
                  tp: float, sl: float,
                  lc: int, pc: int, sc: int,
                  q_val: float, now_et: datetime,
                  flat_reason: str = "") -> str:
    ts = now_et.strftime("%Y-%m-%d %H:%M ET")
    if action == 0:
        header = f"**Pi* -- FLAT [{MAC_NAME}]**"
        lines  = [header, ts,
                  f"London  : {LC_LABELS[lc]}",
                  f"Pool    : {PC_LABELS[pc]}",
                  f"Sweep   : {SC_LABELS[sc]}"]
        if flat_reason:
            lines.append(f"Raison  : {flat_reason}")
    else:
        arrow  = "[^]" if action == 1 else "[v]"
        header = f"**Pi* -- {direction} {arrow} [{MAC_NAME}]**"
        lines  = [header, ts,
                  f"London  : {LC_LABELS[lc]}",
                  f"Pool    : {PC_LABELS[pc]}",
                  f"Sweep   : {SC_LABELS[sc]}",
                  f"Q       : {q_val*100:+.3f}%",
                  f"Entry   : {entry:,.2f}",
                  f"TP      : {tp:,.2f}",
                  f"SL      : {sl:,.2f}",
                  f"EOD     : 16:00 ET"]
    return "\n".join(lines)


def send_discord(msg: str):
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        print("[live] DISCORD_WEBHOOK_URL non defini -- pas d'envoi.")
        return
    resp = requests.post(url, json={"content": msg}, timeout=10)
    resp.raise_for_status()


# ── CSV log ───────────────────────────────────────────────────────

CSV_PATH   = ROOT / "db" / "live_trades.csv"
CSV_FIELDS = [
    "date", "mac_idx", "mac_name",
    "mc", "dc", "lc", "sc", "pc", "state",
    "action", "q_val", "would_trade", "flat_reason",
    "entry_px", "tp_px", "sl_px",
    "pnl", "exit_reason", "n_candles",
]

import csv as _csv

def log_csv(date_str: str, row: dict):
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({"date": date_str, **{k: row.get(k, "") for k in CSV_FIELDS if k != "date"}})


# ── Main ──────────────────────────────────────────────────────────

def main():
    now_utc = datetime.now(pytz.utc)
    now_et  = now_utc.astimezone(ET_TZ)
    hm_et   = now_et.hour * 60 + now_et.minute
    today   = now_et.date()
    dow     = now_et.weekday()

    # Verification que la macro est autorisee en live
    if MAC_IDX not in LIVE_MACROS:
        print(f"[live] mac_idx={MAC_IDX} non autorise (LIVE_MACROS={LIVE_MACROS}) -- exit.")
        sys.exit(0)

    # Verification fenetre horaire
    if not (WINDOW_LOW <= hm_et <= WINDOW_HIGH):
        print(f"[live] Hors fenetre ({now_et.strftime('%H:%M')} ET) -- exit.")
        sys.exit(0)

    if dow in SKIP_DAYS:
        print(f"[live] Jour skippé (dow={dow}) -- exit.")
        mc = month_ctx(now_et.month)
        dc = day_ctx(dow)
        msg = (f"**Pi* -- SKIP [{MAC_NAME}]**\n{now_et.strftime('%Y-%m-%d %H:%M ET')}\n"
               f"Lundi -- pas de trade live.")
        send_discord(msg)
        log_csv(today.isoformat(), {
            "mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc,
            "lc": -1, "sc": -1, "pc": -1, "state": -1,
            "action": 0, "q_val": 0.0, "would_trade": False, "flat_reason": "skip_day",
        })
        sys.exit(0)

    print(f"[live] {now_et.strftime('%Y-%m-%d %H:%M')} ET -- calcul signal Pi* {MAC_NAME}...")

    use_microstructure = _env_bool("USE_MICROSTRUCTURE", True)
    micro_ofi_threshold = float(os.environ.get("MICRO_OFI_THRESHOLD", "0.10"))
    micro_allow_neutral = _env_bool("MICRO_ALLOW_NEUTRAL", True)

    # Fetch donnees
    df_1m = fetch_klines("1m", 1500)   # ~25h de bougies
    df_1d = fetch_klines("1d", 14)     # 14 jours pour PWH/PWL

    if use_microstructure:
        df_1m = attach_microstructure_overlay(
            df_1m,
            config={
                "ofi_threshold": micro_ofi_threshold,
                "allow_neutral": micro_allow_neutral,
            },
        )

    # Preparation du jour en ET
    df_1m["ts_et"]   = df_1m["timestamp"].dt.tz_convert(ET_TZ)
    df_1m["date_et"] = df_1m["ts_et"].dt.date
    df_1m["hm_et"]   = df_1m["ts_et"].dt.hour * 60 + df_1m["ts_et"].dt.minute
    today_df = df_1m[df_1m["date_et"] == today].copy()

    if len(today_df) < 60:
        print(f"[live] Donnees insuffisantes ({len(today_df)} bougies) -- exit.")
        sys.exit(1)

    # Contextes fixes du jour
    mc = month_ctx(now_et.month)
    dc = day_ctx(dow)
    lc, ldn_h, ldn_l = compute_london_ctx(today_df)
    pwh, pwl = compute_pwh_pwl(df_1d, today)

    # Session high/low (avant 08:30 ET = reference BSL/SSL fallback)
    sess_mask = today_df["hm_et"] < 510
    sess_h = float(today_df["high"][sess_mask].max()) if sess_mask.any() else ldn_h
    sess_l = float(today_df["low"][sess_mask].min())  if sess_mask.any() else ldn_l

    # Pre-macro (20 min avant la macro)
    pre_mask = (today_df["hm_et"] >= PRE_START) & (today_df["hm_et"] < MAC_START)
    pre_df   = today_df[pre_mask]

    # Premiere bougie macro
    first_df = today_df[today_df["hm_et"] == MAC_START]

    if len(pre_df) < 3 or first_df.empty:
        print("[live] Pre-macro ou premiere bougie manquante -- exit.")
        sys.exit(1)

    pre_high = float(pre_df["high"].max())
    pre_low  = float(pre_df["low"].min())
    first    = first_df.iloc[0]

    # Sweep context
    if float(first["high"]) > pre_high:
        sc = 1
    elif float(first["low"]) < pre_low:
        sc = 2
    else:
        sc = 0

    if ALIGNED_ONLY and sc == 0:
        print("[live] No sweep detecte -- FLAT (aligned_only).")
        msg = build_message(0, "FLAT", 0, 0, 0, lc, 0, sc, 0.0, now_et, "no sweep")
        send_discord(msg)
        log_csv(today.isoformat(), {"mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": 0, "state": encode(mc, dc, lc, MAC_IDX, sc, 0), "action": 0, "q_val": 0.0, "would_trade": False, "flat_reason": "no_sweep"})
        sys.exit(0)

    # Reference BSL/SSL : 4h lookback avant pre-macro (05:30-09:30 ET)
    ref_start = PRE_START - REF_WINDOW   # 570 - 240 = 330 (05:30 ET)
    ref_mask  = (today_df["hm_et"] >= max(0, ref_start)) & (today_df["hm_et"] < PRE_START)
    ref_df    = today_df[ref_mask]
    ref_h = float(ref_df["high"].max()) if len(ref_df) >= 5 else sess_h
    ref_l = float(ref_df["low"].min())  if len(ref_df) >= 5 else sess_l

    pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

    # Macro rules
    allowed_sc = MACRO_RULES.get((MAC_IDX, lc, pc))
    if allowed_sc is not None and sc not in allowed_sc:
        print(f"[live] Macro rule bloquee (mac={MAC_IDX}, lc={lc}, pc={pc}, sc={sc}) -- FLAT.")
        msg = build_message(0, "FLAT", 0, 0, 0, lc, pc, sc, 0.0, now_et, "macro rule")
        send_discord(msg)
        log_csv(today.isoformat(), {"mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": pc, "state": encode(mc, dc, lc, MAC_IDX, sc, pc), "action": 0, "q_val": 0.0, "would_trade": False, "flat_reason": "macro_rule"})
        sys.exit(0)

    state = encode(mc, dc, lc, MAC_IDX, sc, pc)

    # Q-table lookup
    model_path = ROOT / "db" / "stats_agent.pkl"
    if not model_path.exists():
        print(f"[live] stats_agent.pkl introuvable -- exit.")
        sys.exit(1)

    agent  = QAgent.load(str(model_path))
    action = agent.act(state, training=False)
    q_val  = float(agent.q_table[state, action]) if action > 0 else 0.0

    if action == 0 or q_val <= Q_THRESHOLD:
        print(f"[live] FLAT (Q={q_val*100:+.3f}%, action={action}) -- state={state}")
        msg = build_message(0, "FLAT", 0, 0, 0, lc, pc, sc, q_val, now_et, "Q <= threshold")
        send_discord(msg)
        log_csv(today.isoformat(), {"mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": pc, "state": state, "action": action, "q_val": q_val, "would_trade": False, "flat_reason": "q_flat"})
        sys.exit(0)

    if use_microstructure:
        micro_row = pre_df.iloc[-1]
        if not microstructure_trade_allowed(
            micro_row, action=action, allow_neutral=micro_allow_neutral,
        ):
            print(f"[live] FLAT (micro gate) -- state={state} action={action}")
            msg = build_message(0, "FLAT", 0, 0, 0, lc, pc, sc, q_val, now_et, "micro gate")
            send_discord(msg)
            log_csv(today.isoformat(), {
                "mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc,
                "lc": lc, "sc": sc, "pc": pc, "state": state,
                "action": action, "q_val": q_val, "would_trade": False,
                "flat_reason": "micro_gate",
            })
            sys.exit(0)

    direction = "LONG" if action == 1 else "SHORT"
    entry_px  = float(first["open"])
    print(f"[live] Signal {direction} (Q={q_val*100:+.3f}%) -- state={state} entry={entry_px:.2f}")

    # Calcul SL/TP (logique target_pool identique au backtest)
    sweep_high = max(float(first["high"]), pre_high)
    sweep_low  = min(float(first["low"]),  pre_low)
    tp_px = sl_px = 0.0

    if pc == 2 and action == 1 and ref_h is not None and ref_h > entry_px:
        # SSL swept -> LONG -> TP = ref_h (BSL)
        tp_pct = (ref_h - entry_px) / entry_px
        sl_pct = SL_PCT + max(0.0, (entry_px - sweep_low) / entry_px)
        if tp_pct > sl_pct > 0:
            tp_px = entry_px * (1 + tp_pct)
            sl_px = entry_px * (1 - sl_pct)
        else:
            action = 0
    elif pc == 1 and action == 2 and ref_l is not None and ref_l < entry_px:
        # BSL swept -> SHORT -> TP = ref_l (SSL)
        tp_pct = (entry_px - ref_l) / entry_px
        sl_pct = SL_PCT + max(0.0, (sweep_high - entry_px) / entry_px)
        if tp_pct > sl_pct > 0:
            tp_px = entry_px * (1 - tp_pct)
            sl_px = entry_px * (1 + sl_pct)
        else:
            action = 0
    else:
        # Fallback SL/TP fixe (pc==0)
        tp_pct = SL_PCT * RR
        if action == 1:
            tp_px = entry_px * (1 + tp_pct)
            sl_px = entry_px * (1 - SL_PCT)
        else:
            tp_px = entry_px * (1 - tp_pct)
            sl_px = entry_px * (1 + SL_PCT)

    if action == 0:
        print("[live] TP <= SL apres calcul -- FLAT.")
        msg = build_message(0, "FLAT", 0, 0, 0, lc, pc, sc, q_val, now_et, "TP <= SL")
        send_discord(msg)
        log_csv(today.isoformat(), {"mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": pc, "state": state, "action": 0, "q_val": q_val, "would_trade": False, "flat_reason": "tp_le_sl"})
        sys.exit(0)

    log_csv(today.isoformat(), {"mac_idx": MAC_IDX, "mac_name": MAC_NAME, "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": pc, "state": state, "action": action, "q_val": q_val, "would_trade": True, "flat_reason": "", "entry_px": round(entry_px, 2), "tp_px": round(tp_px, 2), "sl_px": round(sl_px, 2), "pnl": None, "exit_reason": "PENDING", "n_candles": 0})

    msg = build_message(action, direction, entry_px, tp_px, sl_px, lc, pc, sc, q_val, now_et)
    send_discord(msg)
    print(f"[live] Signal envoye : {direction} entry={entry_px:.2f} TP={tp_px:.2f} SL={sl_px:.2f}")


if __name__ == "__main__":
    main()
