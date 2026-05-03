"""
stat_fractal.py — Backtest statistique Fractal ICT sur données historiques

Télécharge N mois de M15 via ccxt (paginé), Daily et Weekly, fait tourner
les 3 détecteurs, puis évalue chaque signal en simulant TP/SL sur les
bougies M15 suivantes.

Usage :
    python stat_fractal.py
    python stat_fractal.py --symbol ETH/USDT --months 3
    python stat_fractal.py --symbol BTC/USDT --months 6
    python stat_fractal.py --symbol ETH/USDT --months 3 --tp 1.5 --sl 0.75
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies" / "fractal"))


# ── Chargement paginé ──────────────────────────────────────────────────────────

def fetch_paginated_m15(exchange, symbol: str, months: int) -> pd.DataFrame:
    """Fetch M15 sur N mois via ccxt (multiple requests de 1000 bars)."""
    limit     = 1000
    end_dt    = datetime.utcnow()
    start_dt  = end_dt - timedelta(days=months * 30)
    since_ms  = int(start_dt.timestamp() * 1000)
    end_ms    = int(end_dt.timestamp() * 1000)

    print(f"  Pagination M15 : {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")
    all_rows = []
    since = since_ms
    req = 0

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "15m", since=since, limit=limit)
        except Exception as e:
            print(f"\n  [ERREUR] fetch_ohlcv: {e}")
            break
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        since = ohlcv[-1][0] + 15 * 60 * 1000
        req += 1
        print(f"  req={req}  bars={len(all_rows)}  last={datetime.utcfromtimestamp(ohlcv[-1][0]/1000).strftime('%Y-%m-%d')}", end="\r")
        time.sleep(exchange.rateLimit / 1000)

    print(f"\n  Total M15 : {len(all_rows)} bougies")
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_simple(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ── Simulation TP/SL ──────────────────────────────────────────────────────────

def simulate_outcome(pattern: str, entry: float, df_m15: pd.DataFrame,
                     after_ts: pd.Timestamp, tp_pct: float, sl_pct: float,
                     horizon_bars: int = 96) -> str:
    """
    Simule le trade sur les N bougies M15 qui suivent after_ts.
    Retourne 'WIN', 'LOSS' ou 'TIMEOUT'.
    """
    future = df_m15[df_m15["timestamp"] > after_ts].head(horizon_bars)
    if future.empty:
        return "NO_DATA"

    is_long = pattern == "DOWN->UP"
    tp_px = entry * (1 + tp_pct / 100) if is_long else entry * (1 - tp_pct / 100)
    sl_px = entry * (1 - sl_pct / 100) if is_long else entry * (1 + sl_pct / 100)

    for _, bar in future.iterrows():
        if is_long:
            if bar["low"] <= sl_px:
                return "LOSS"
            if bar["high"] >= tp_px:
                return "WIN"
        else:  # SHORT
            if bar["high"] >= sl_px:
                return "LOSS"
            if bar["low"] <= tp_px:
                return "WIN"

    return "TIMEOUT"


# ── Détection avec timestamps réels ──────────────────────────────────────────

KZ_MAP = {
    "LKZ":  (5, 7),
    "NYKZ": (16, 18),
    "LnCl": (20, 21),
    "AKZ":  (21, 23),
}


def detect_with_real_ts(df_m15: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame) -> list:
    """
    Variante des détecteurs qui stocke le timestamp réel de l'entrée
    (premier bar de kz_after qui déclenche la confirmation).
    Retourne une liste unifiée pour les 3 setups.
    """
    signals = []

    # ── Précalcul inside bars ────────────────────────────────────────
    daily = daily.copy()
    daily["is_inside"] = (daily["high"] < daily["high"].shift(1)) & (daily["low"] > daily["low"].shift(1))

    weekly = weekly.copy()
    weekly["is_inside"] = (weekly["high"] < weekly["high"].shift(1)) & (weekly["low"] > weekly["low"].shift(1))

    inside_days = daily[daily["is_inside"]].reset_index(drop=True)

    for _, day in inside_days.iterrows():
        day_start = day["timestamp"]
        day_end   = day_start + pd.Timedelta(days=1)

        # setup labels applicables
        setups = ["MODÉRÉ"]  # tout inside day → MODÉRÉ
        week_containing = weekly[
            (weekly["timestamp"] <= day_start) &
            (weekly["timestamp"] + pd.Timedelta(days=7) > day_start)
        ]
        if not week_containing.empty and week_containing.iloc[-1]["is_inside"]:
            setups.append("STRICT")  # inside week aussi → STRICT

        m15_day = df_m15[(df_m15["timestamp"] >= day_start) & (df_m15["timestamp"] < day_end)]
        if m15_day.empty:
            continue

        for kz_name, (kz_h0, kz_h1) in KZ_MAP.items():
            kz_m15 = m15_day[
                (m15_day["timestamp"].dt.hour >= kz_h0) &
                (m15_day["timestamp"].dt.hour < kz_h1)
            ]
            if kz_m15.empty:
                continue

            kz_high = kz_m15["high"].max()
            kz_low  = kz_m15["low"].min()

            prev_kz = df_m15[
                (df_m15["timestamp"] >= day_start - pd.Timedelta(days=1)) &
                (df_m15["timestamp"] < day_start) &
                (df_m15["timestamp"].dt.hour >= kz_h0) &
                (df_m15["timestamp"].dt.hour < kz_h1)
            ]
            if prev_kz.empty:
                continue
            if not (kz_high < prev_kz["high"].max() and kz_low > prev_kz["low"].min()):
                continue  # pas inside KZ

            # kz_after : prochaine occurrence de la même KZ (après day_end)
            kz_after = df_m15[
                (df_m15["timestamp"] >= day_end) &
                (df_m15["timestamp"].dt.hour >= kz_h0) &
                (df_m15["timestamp"].dt.hour < kz_h1)
            ]
            if kz_after.empty:
                continue

            has_up   = (kz_after["high"] > kz_high).any()
            has_down = (kz_after["low"]  < kz_low).any()
            if not (has_up and has_down):
                continue  # pas de break & retest

            # Timestamp réel = premier bar qui confirme la cassure
            if kz_after.iloc[0]["high"] > kz_high:
                direction = "UP->DOWN"
                # chercher premier bar down après le premier up
                first_up_idx = kz_after[kz_after["high"] > kz_high].index[0]
                bars_after_up = kz_after.loc[first_up_idx:]
                confirm_bars  = bars_after_up[bars_after_up["low"] < kz_low]
            else:
                direction = "DOWN->UP"
                first_dn_idx = kz_after[kz_after["low"] < kz_low].index[0]
                bars_after_dn = kz_after.loc[first_dn_idx:]
                confirm_bars  = bars_after_dn[bars_after_dn["high"] > kz_high]

            if confirm_bars.empty:
                confirm_ts = kz_after["timestamp"].iloc[0]
                entry_px   = kz_after["high"].iloc[0] if direction == "UP->DOWN" else kz_after["low"].iloc[0]
            else:
                confirm_ts = confirm_bars["timestamp"].iloc[0]
                entry_px   = confirm_bars["low"].iloc[0] if direction == "DOWN->UP" else confirm_bars["high"].iloc[0]

            for setup in setups:
                signals.append({
                    "setup":       setup,
                    "day_date":    day_start.date(),
                    "kz":          kz_name,
                    "pattern":     direction,
                    "entry_price": entry_px,
                    "confirm_ts":  confirm_ts,
                    "confidence":  0.946 if setup == "STRICT" else 0.91,
                })

        # ── FRÉQUENT : inside KZ seulement (pas besoin d'inside day) ──────────
        for kz_name, (kz_h0, kz_h1) in KZ_MAP.items():
            m15_all = df_m15  # on itère jour par jour séparément
            # On skippe car FRÉQUENT est traité dans sa propre boucle ci-dessous

    # ── FRÉQUENT (inside KZ uniquement) ─────────────────────────────────────
    df_m15_copy = df_m15.copy()
    df_m15_copy["date"] = df_m15_copy["timestamp"].dt.date
    days_all = sorted(df_m15_copy["date"].unique())

    for day in days_all:
        day_start = pd.Timestamp(day, tz=None)
        day_end   = day_start + pd.Timedelta(days=1)
        m15_day   = df_m15_copy[df_m15_copy["date"] == day]
        if m15_day.empty:
            continue

        for kz_name, (kz_h0, kz_h1) in KZ_MAP.items():
            kz_m15 = m15_day[
                (m15_day["timestamp"].dt.hour >= kz_h0) &
                (m15_day["timestamp"].dt.hour < kz_h1)
            ]
            if kz_m15.empty:
                continue

            kz_high = kz_m15["high"].max()
            kz_low  = kz_m15["low"].min()

            prev_kz = df_m15_copy[
                (df_m15_copy["timestamp"] >= day_start - pd.Timedelta(days=1)) &
                (df_m15_copy["timestamp"] < day_start) &
                (df_m15_copy["timestamp"].dt.hour >= kz_h0) &
                (df_m15_copy["timestamp"].dt.hour < kz_h1)
            ]
            if prev_kz.empty:
                continue
            if not (kz_high < prev_kz["high"].max() and kz_low > prev_kz["low"].min()):
                continue

            kz_after = df_m15_copy[
                (df_m15_copy["timestamp"] >= day_end) &
                (df_m15_copy["timestamp"].dt.hour >= kz_h0) &
                (df_m15_copy["timestamp"].dt.hour < kz_h1)
            ]
            if kz_after.empty:
                continue

            has_up   = (kz_after["high"] > kz_high).any()
            has_down = (kz_after["low"]  < kz_low).any()
            if not (has_up and has_down):
                continue

            if kz_after.iloc[0]["high"] > kz_high:
                direction = "UP->DOWN"
                first_up_idx  = kz_after[kz_after["high"] > kz_high].index[0]
                bars_after_up = kz_after.loc[first_up_idx:]
                confirm_bars  = bars_after_up[bars_after_up["low"] < kz_low]
            else:
                direction = "DOWN->UP"
                first_dn_idx  = kz_after[kz_after["low"] < kz_low].index[0]
                bars_after_dn = kz_after.loc[first_dn_idx:]
                confirm_bars  = bars_after_dn[bars_after_dn["high"] > kz_high]

            if confirm_bars.empty:
                confirm_ts = kz_after["timestamp"].iloc[0]
                entry_px   = kz_after["high"].iloc[0] if direction == "UP->DOWN" else kz_after["low"].iloc[0]
            else:
                confirm_ts = confirm_bars["timestamp"].iloc[0]
                entry_px   = confirm_bars["low"].iloc[0] if direction == "DOWN->UP" else confirm_bars["high"].iloc[0]

            signals.append({
                "setup":       "FRÉQUENT",
                "day_date":    day,
                "kz":          kz_name,
                "pattern":     direction,
                "entry_price": entry_px,
                "confirm_ts":  confirm_ts,
                "confidence":  0.875,
            })

    return signals


# ── Rapport ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_stats(df: pd.DataFrame, label: str, tp_pct: float, sl_pct: float):
    if df.empty:
        print(f"  {label}: aucun signal")
        return

    n      = len(df)
    wins   = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    to     = (df["outcome"] == "TIMEOUT").sum()
    nd     = (df["outcome"] == "NO_DATA").sum()
    resolved = wins + losses
    wr     = wins / resolved if resolved > 0 else 0.0
    expectancy = (wr * tp_pct - (1 - wr) * sl_pct) if resolved > 0 else 0.0

    print(f"\n  {label}  ({n} signaux, TP={tp_pct}% SL={sl_pct}%)")
    print(f"    WIN={wins}  LOSS={losses}  TIMEOUT={to}  NO_DATA={nd}")
    print(f"    Win Rate (résolu) : {wr:.1%}")
    print(f"    Expectancy R      : {expectancy:+.3f}R")

    # Par KZ
    print(f"    Par KZ :")
    for kz, grp in df.groupby("kz"):
        gw = (grp["outcome"] == "WIN").sum()
        gl = (grp["outcome"] == "LOSS").sum()
        gr = gw + gl
        print(f"      {kz:<8} {len(grp):3d} signaux  WR={gw/gr:.0%}" if gr > 0 else f"      {kz:<8} {len(grp):3d} signaux  WR=N/A")

    # Par pattern
    print(f"    Par pattern :")
    for pat, grp in df.groupby("pattern"):
        gw = (grp["outcome"] == "WIN").sum()
        gl = (grp["outcome"] == "LOSS").sum()
        gr = gw + gl
        print(f"      {pat:<12} {len(grp):3d} signaux  WR={gw/gr:.0%}" if gr > 0 else f"      {pat:<12} {len(grp):3d} signaux  WR=N/A")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--tp",     type=float, default=1.5, help="TP en %%")
    parser.add_argument("--sl",     type=float, default=0.75, help="SL en %%")
    parser.add_argument("--horizon", type=int, default=96, help="Barres M15 pour simuler (96=24h)")
    args = parser.parse_args()

    symbol  = args.symbol
    months  = args.months
    tp_pct  = args.tp
    sl_pct  = args.sl
    horizon = args.horizon

    print(f"\n{'='*60}")
    print(f"  FRACTAL STATS BACKTEST — {symbol}")
    print(f"  Période : {months} mois   TP={tp_pct}%   SL={sl_pct}%   horizon={horizon} bars M15")
    print(f"{'='*60}")

    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})

    # ── [1] Charger les données ──────────────────────────────────────
    print_section("1 / Chargement données Binance")
    df_m15   = fetch_paginated_m15(exchange, symbol, months)
    df_daily  = fetch_simple(exchange, symbol, "1d", 365)
    df_weekly = fetch_simple(exchange, symbol, "1w", 260)

    print(f"  M15    : {len(df_m15):,} bougies  "
          f"({df_m15['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df_m15['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  Daily  : {len(df_daily):,} bougies")
    print(f"  Weekly : {len(df_weekly):,} bougies")

    # ── [2] Détection ────────────────────────────────────────────────
    print_section("2 / Détection des patterns")
    signals = detect_with_real_ts(df_m15, df_daily, df_weekly)
    df_sig  = pd.DataFrame(signals)

    if df_sig.empty:
        print("  Aucun signal détecté sur la période.")
        return

    print(f"  {len(df_sig)} signal(s) total")
    for setup in ["STRICT", "MODÉRÉ", "FRÉQUENT"]:
        n = (df_sig["setup"] == setup).sum()
        print(f"    {setup:<10} : {n}")

    # ── [3] Simulation TP/SL ────────────────────────────────────────
    print_section("3 / Simulation TP/SL")
    outcomes = []
    for _, sig in df_sig.iterrows():
        out = simulate_outcome(
            pattern   = sig["pattern"],
            entry     = sig["entry_price"],
            df_m15    = df_m15,
            after_ts  = sig["confirm_ts"],
            tp_pct    = tp_pct,
            sl_pct    = sl_pct,
            horizon_bars = horizon,
        )
        outcomes.append(out)
    df_sig["outcome"] = outcomes

    # ── [4] Rapport par setup ────────────────────────────────────────
    print_section("4 / Résultats par setup")
    for setup in ["STRICT", "MODÉRÉ", "FRÉQUENT"]:
        sub = df_sig[df_sig["setup"] == setup]
        print_stats(sub, setup, tp_pct, sl_pct)

    # ── [5] Résumé global ────────────────────────────────────────────
    print_section("5 / Résumé global")
    print_stats(df_sig, "ALL SETUPS", tp_pct, sl_pct)

    # Distribution mensuelle
    df_sig["month"] = pd.to_datetime(df_sig["day_date"]).dt.to_period("M").astype(str)
    print(f"\n  Signaux par mois :")
    for month, grp in df_sig.groupby("month"):
        wins = (grp["outcome"] == "WIN").sum()
        total = len(grp)
        resolved = wins + (grp["outcome"] == "LOSS").sum()
        wr_str = f"WR={wins/resolved:.0%}" if resolved > 0 else "WR=N/A"
        print(f"    {month}  {total:3d} signaux  {wr_str}")

    print(f"\n{'='*60}")
    print(f"  FIN — {symbol}  {months} mois")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
