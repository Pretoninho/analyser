"""
shadow_signal.py -- Paper trading EOD des macros sous silence de Pi*.

Pour chaque macro dans SHADOW_MACROS (pi_config.py) :
  - Calcule les contextes ICT au moment de la macro
  - Interroge la Q-table pour connaitre le signal hypothetique
  - Simule le trade sur les bougies reelles
  - Ecrit le resultat dans db/shadow_trades.csv
  - Envoie un resume journalier sur Discord avec label [SHADOW]

Declenchement Railway Cron (UTC) :
  EDT avril-octobre  : 5 20 * * 1-5   (16:05 ET = 20:05 UTC)
  EST novembre-mars  : 5 21 * * 1-5   (16:05 ET = 21:05 UTC)

Variables d'environnement requises :
  DISCORD_WEBHOOK_URL

Optionnelles :
  BINANCE_BASE_URL   (defaut: https://api.binance.com)
"""

import os
import sys
import csv
import requests
import pytz
import pandas as pd
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pi_config import (
    SHADOW_MACROS, LIVE_MACROS, SL_PCT, RR, EXIT_HM, Q_THRESHOLD,
    ALIGNED_ONLY, SKIP_DAYS, MACRO_RULES, FEE, SLIP,
)
from engine.stats_state import (
    MACROS, encode, compute_pool_ctx, compute_daily_context,
)
from engine.q_agent import QAgent

ET_TZ = pytz.timezone("America/New_York")

LC_LABELS = {0: "NO_RAID", 1: "RAID_H",  2: "RAID_L"}
PC_LABELS  = {0: "NEUTRAL", 1: "BSL_SWP", 2: "SSL_SWP"}
SC_LABELS  = {0: "NO_SWP",  1: "SWP_H",   2: "SWP_L"}
MAC_NAMES  = {
    1: "08:50", 2: "09:50", 3: "10:50",
    4: "11:50", 5: "12:50", 6: "13:50", 7: "14:50",
}


# ── Binance API ───────────────────────────────────────────────────

def fetch_klines(interval: str, limit: int) -> pd.DataFrame:
    base = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
    url  = f"{base}/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(resp.json(), columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]


# ── PWH/PWL depuis donnees daily ─────────────────────────────────

def compute_pwh_pwl(df_1d: pd.DataFrame, today_et: date):
    df = df_1d.copy()
    df["ts_et"]    = df["timestamp"].dt.tz_convert(ET_TZ)
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


# ── Simulation ────────────────────────────────────────────────────

def sim_trade(exit_df, entry_px, direction, sl_pct, tp_pct):
    """
    Simule un trade sur exit_df candle par candle.
    direction : +1 LONG, -1 SHORT
    Retourne (pnl, exit_reason, n_candles).
    """
    tp_px = entry_px * (1 + direction * tp_pct)
    sl_px = entry_px * (1 - direction * sl_pct)

    for i, (_, row) in enumerate(exit_df.iterrows(), 1):
        h, lo = float(row["high"]), float(row["low"])
        if direction == 1:
            if lo <= sl_px:
                return round(-sl_pct - FEE - SLIP, 6), "SL", i
            if h >= tp_px:
                return round(tp_pct - FEE - SLIP, 6), "TP", i
        else:
            if h >= sl_px:
                return round(-sl_pct - FEE - SLIP, 6), "SL", i
            if lo <= tp_px:
                return round(tp_pct - FEE - SLIP, 6), "TP", i

    if exit_df.empty:
        return 0.0, "NO_EXIT", 0
    last_px = float(exit_df["close"].iloc[-1])
    pnl = direction * (last_px - entry_px) / entry_px - FEE - SLIP
    return round(pnl, 6), "EOD", len(exit_df)


# ── Traitement d'une macro ────────────────────────────────────────

def process_macro(mac_idx, today_df, daily_ctx, pwh, pwl, agent):
    """
    Calcule le signal shadow pour mac_idx.
    Retourne un dict avec tous les champs de log.
    """
    mac_start, _ = MACROS[mac_idx]
    pre_start = mac_start - 20
    ref_start = pre_start - 240  # 4h lookback

    pre_df   = today_df[
        (today_df["hm_et"] >= pre_start) & (today_df["hm_et"] < mac_start)
    ]
    first_df = today_df[today_df["hm_et"] == mac_start]

    base = {
        "mac_idx": mac_idx,
        "mac_name": MAC_NAMES[mac_idx],
        "mc": daily_ctx["month_ctx"],
        "dc": daily_ctx["day_ctx"],
        "lc": daily_ctx["london_ctx"],
        "sc": -1, "pc": -1, "state": -1,
        "action": 0, "q_val": 0.0,
        "would_trade": False, "flat_reason": "no_data",
        "entry_px": None, "tp_px": None, "sl_px": None,
        "pnl": None, "exit_reason": "NO_DATA", "n_candles": 0,
    }

    if len(pre_df) < 3 or first_df.empty:
        return base

    mc = daily_ctx["month_ctx"]
    dc = daily_ctx["day_ctx"]
    lc = daily_ctx["london_ctx"]

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

    # Reference BSL/SSL (4h avant pre-macro)
    ref_df = today_df[
        (today_df["hm_et"] >= max(0, ref_start)) & (today_df["hm_et"] < pre_start)
    ]
    ref_h = float(ref_df["high"].max()) if len(ref_df) >= 5 else daily_ctx.get("session_high")
    ref_l = float(ref_df["low"].min())  if len(ref_df) >= 5 else daily_ctx.get("session_low")

    pc    = compute_pool_ctx(
        pre_high, pre_low,
        daily_ctx.get("london_high"), daily_ctx.get("london_low"),
        pwh, pwl,
    )
    state = encode(mc, dc, lc, mac_idx, sc, pc)

    action = agent.act(state, training=False)
    q_val  = float(agent.q_table[state, action]) if action > 0 else 0.0

    entry_px = float(first["open"])

    base.update({
        "sc": sc, "pc": pc, "state": state,
        "action": action, "q_val": q_val, "entry_px": entry_px,
        "flat_reason": "",
    })

    # Filtres (meme logique que live)
    if ALIGNED_ONLY and sc == 0:
        base["flat_reason"] = "no_sweep"
        return base
    allowed_sc = MACRO_RULES.get((mac_idx, lc, pc))
    if allowed_sc is not None and sc not in allowed_sc:
        base["flat_reason"] = "macro_rule"
        return base
    if action == 0 or q_val <= Q_THRESHOLD:
        base["flat_reason"] = "q_flat"
        return base

    # Calcul TP/SL (logique target_pool identique au backtest)
    direction  = 1 if action == 1 else -1
    sweep_high = max(float(first["high"]), pre_high)
    sweep_low  = min(float(first["low"]),  pre_low)

    if pc == 2 and action == 1 and ref_h is not None and ref_h > entry_px:
        tp_pct     = (ref_h - entry_px) / entry_px
        sl_pct_use = SL_PCT + max(0.0, (entry_px - sweep_low) / entry_px)
    elif pc == 1 and action == 2 and ref_l is not None and ref_l < entry_px:
        tp_pct     = (entry_px - ref_l) / entry_px
        sl_pct_use = SL_PCT + max(0.0, (sweep_high - entry_px) / entry_px)
    else:
        tp_pct     = SL_PCT * RR
        sl_pct_use = SL_PCT

    if tp_pct <= sl_pct_use or tp_pct <= 0:
        base["flat_reason"] = "tp_le_sl"
        return base

    tp_px = entry_px * (1 + direction * tp_pct)
    sl_px = entry_px * (1 - direction * sl_pct_use)

    base["tp_px"] = round(tp_px, 2)
    base["sl_px"] = round(sl_px, 2)

    # Simulation sur bougies reelles
    exit_mask = (today_df["hm_et"] > mac_start) & (today_df["hm_et"] <= EXIT_HM)
    exit_df   = today_df[exit_mask]

    pnl, exit_reason, n_candles = sim_trade(
        exit_df, entry_px, direction, sl_pct_use, tp_pct
    )

    base["would_trade"] = True
    base["pnl"]         = pnl
    base["exit_reason"] = exit_reason
    base["n_candles"]   = n_candles

    return base


# ── CSV ───────────────────────────────────────────────────────────

CSV_PATH      = ROOT / "db" / "shadow_trades.csv"
LIVE_CSV_PATH = ROOT / "db" / "live_trades.csv"
CSV_FIELDS = [
    "date", "mac_idx", "mac_name",
    "mc", "dc", "lc", "sc", "pc", "state",
    "action", "q_val", "would_trade", "flat_reason",
    "entry_px", "tp_px", "sl_px",
    "pnl", "exit_reason", "n_candles",
]


def append_csv(date_str, results, path=None):
    if path is None:
        path = CSV_PATH
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            row = {"date": date_str}
            row.update({k: r.get(k, "") for k in CSV_FIELDS if k != "date"})
            writer.writerow(row)


# ── Discord ───────────────────────────────────────────────────────

def _format_macro_line(r):
    mac = r["mac_name"]
    lc  = LC_LABELS.get(r["lc"], "?")
    sc  = SC_LABELS.get(r["sc"], "?") if r["sc"] >= 0 else "?"
    pc  = PC_LABELS.get(r["pc"], "?") if r["pc"] >= 0 else "?"
    ctx = f"{lc} | {sc} | {pc}"
    if r["exit_reason"] == "NO_DATA":
        return f"{mac} | {ctx} -> NO_DATA", False, 0.0, ""
    if r["flat_reason"]:
        return f"{mac} | {ctx} -> FLAT ({r['flat_reason']})", False, 0.0, ""
    direction = "LONG" if r["action"] == 1 else "SHORT"
    pnl_pct   = r["pnl"] * 100 if r["pnl"] is not None else 0.0
    sign      = "+" if pnl_pct >= 0 else ""
    line = (f"{mac} | {ctx} -> {direction} Q={r['q_val']*100:+.3f}%"
            f" | {r['exit_reason']} {sign}{pnl_pct:.2f}% (N={r['n_candles']})")
    return line, True, pnl_pct, r["exit_reason"]


def build_discord_message(date_str, shadow_results, live_results=None):
    lines = [f"**[SHADOW] Pi* | {date_str}**", ""]

    # Section LIVE observation (jours skippes)
    if live_results:
        lines.append("_[LIVE — observation skip_day]_")
        for r in live_results:
            line, _, _, _ = _format_macro_line(r)
            lines.append(line)
        lines.append("")

    n_trades = n_tp = n_sl = n_eod = 0
    pnl_total = 0.0

    for r in (shadow_results or []):
        line, traded, pnl_pct, exit_r = _format_macro_line(r)
        lines.append(line)
        if traded:
            n_trades += 1
            pnl_total += pnl_pct
            if exit_r == "TP":
                n_tp  += 1
            elif exit_r == "SL":
                n_sl  += 1
            elif exit_r == "EOD":
                n_eod += 1

    lines.append("")
    if n_trades > 0:
        lines.append(
            f"Trades: {n_trades} | TP: {n_tp} | SL: {n_sl} | EOD: {n_eod}"
            f" | P&L total: {pnl_total:+.2f}%"
        )
    else:
        lines.append("Aucun trade shadow aujourd'hui.")

    return "\n".join(lines)


def send_discord(msg: str):
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        print("[shadow] DISCORD_WEBHOOK_URL non defini -- pas d'envoi.")
        return
    resp = requests.post(url, json={"content": msg}, timeout=10)
    resp.raise_for_status()


# ── Main ──────────────────────────────────────────────────────────

def main():
    now_utc  = datetime.now(pytz.utc)
    now_et   = now_utc.astimezone(ET_TZ)
    today    = now_et.date()
    dow      = now_et.weekday()
    date_str = today.isoformat()

    skip_day = dow in SKIP_DAYS
    if skip_day:
        print(f"[shadow] Jour skippe (dow={dow}) -- observation LIVE_MACROS en mode shadow.")

    if not SHADOW_MACROS and not skip_day:
        print("[shadow] SHADOW_MACROS vide -- exit.")
        sys.exit(0)

    print(f"[shadow] {now_et.strftime('%Y-%m-%d %H:%M')} ET -- calcul shadow Pi*...")

    # Fetch donnees
    df_1m = fetch_klines("1m", 1500)  # ~25h de bougies
    df_1d = fetch_klines("1d", 14)   # 14 jours pour PWH/PWL

    # Preparation colonnes ET
    df_1m["ts_et"]   = df_1m["timestamp"].dt.tz_convert(ET_TZ)
    df_1m["date_et"] = df_1m["ts_et"].dt.date
    df_1m["hm_et"]   = df_1m["ts_et"].dt.hour * 60 + df_1m["ts_et"].dt.minute

    today_df = df_1m[df_1m["date_et"] == today].copy().reset_index(drop=True)

    if len(today_df) < 60:
        print(f"[shadow] Donnees insuffisantes ({len(today_df)} bougies) -- exit.")
        sys.exit(1)

    # Contexte fixe du jour (commun a toutes les macros)
    pwh, pwl    = compute_pwh_pwl(df_1d, today)
    daily_ctx   = compute_daily_context(today_df, pwh=pwh, pwl=pwl)

    # Q-table
    model_path = ROOT / "db" / "stats_agent.pkl"
    if not model_path.exists():
        print("[shadow] stats_agent.pkl introuvable -- exit.")
        sys.exit(1)
    agent = QAgent.load(str(model_path))

    # Macros a traiter : SHADOW toujours + LIVE si jour skippe
    macros_shadow = sorted(SHADOW_MACROS)
    macros_live   = sorted(LIVE_MACROS) if skip_day else []

    shadow_results = []
    live_results   = []

    for mac_idx in macros_shadow:
        r = process_macro(mac_idx, today_df, daily_ctx, pwh, pwl, agent)
        status = r["exit_reason"] if r["would_trade"] else f"FLAT ({r['flat_reason'] or 'no_data'})"
        print(f"[shadow] mac={mac_idx} ({MAC_NAMES[mac_idx]}) -> {status}")
        shadow_results.append(r)

    for mac_idx in macros_live:
        r = process_macro(mac_idx, today_df, daily_ctx, pwh, pwl, agent)
        # Jour skippe : le trade n'aurait pas ete pris en live
        if r["would_trade"]:
            r["would_trade"] = False
            r["flat_reason"] = "skip_day"
        status = f"FLAT ({r['flat_reason'] or 'no_data'})"
        print(f"[shadow/live-obs] mac={mac_idx} ({MAC_NAMES[mac_idx]}) -> {status}")
        live_results.append(r)

    # Log CSV
    if shadow_results:
        append_csv(date_str, shadow_results)
        print(f"[shadow] {len(shadow_results)} lignes shadow ecrites dans {CSV_PATH}")
    if live_results:
        append_csv(date_str, live_results, path=LIVE_CSV_PATH)
        print(f"[shadow] {len(live_results)} lignes live (skip_day obs) ecrites dans {LIVE_CSV_PATH}")

    # Discord
    msg = build_discord_message(date_str, shadow_results, live_results)
    send_discord(msg)
    print("[shadow] Message Discord envoye.")


if __name__ == "__main__":
    main()
