"""
analyse_silver_bullet.py — Silver Bullet 10:00 ET sur BTCUSDT Binance.

Setup : Opening Range (09:30-10:00 ET) forme OR_HIGH / OR_LOW.
        A 10:00, sweep de l'OR_HIGH (sc=SWEEP_H) ou OR_LOW (sc=SWEEP_L) -> reversal.
Pool  : OR_HIGH/LOW vs London session high/low + PWH/PWL.
SL    = 0.5% | TP = 1.0% | Sortie max = 16:00 ET.
"""

import sys
import pytz
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.binance import load_binance_1m
from engine.stats_state import (
    compute_daily_context, compute_pool_ctx, build_weekly_levels,
)

SL_PCT   = 0.005
TP_PCT   = 0.010
EXIT_HM  = 960    # 16:00 ET
FEE      = 0.0005
SLIP     = 0.0002

OR_START   = 570  # 09:30 ET en minutes
OR_END     = 600  # 10:00 ET
ENTRY_HM   = 600  # 10:00 ET
TEST_RATIO = 0.2

LC_NAMES = {0: "NO_RAID", 1: "RAID_H", 2: "RAID_L"}
SC_NAMES = {0: "NO_SWEEP", 1: "SWEEP_H", 2: "SWEEP_L"}
PC_NAMES = {0: "NEUTRAL",  1: "BSL_SWEPT", 2: "SSL_SWEPT"}


def sim_trade(exit_df, entry_px, direction):
    """direction: +1 LONG, -1 SHORT. Retourne pnl net."""
    if direction == 1:
        tp_px = entry_px * (1 + TP_PCT)
        sl_px = entry_px * (1 - SL_PCT)
    else:
        tp_px = entry_px * (1 - TP_PCT)
        sl_px = entry_px * (1 + SL_PCT)
    for _, row in exit_df.iterrows():
        h, lo = float(row["high"]), float(row["low"])
        if direction == 1:
            if lo <= sl_px: return -SL_PCT - FEE - SLIP * 2
            if h  >= tp_px: return  TP_PCT - FEE - SLIP * 2
        else:
            if h  >= sl_px: return -SL_PCT - FEE - SLIP * 2
            if lo <= tp_px: return  TP_PCT - FEE - SLIP * 2
    close = float(exit_df.iloc[-1]["close"])
    raw   = direction * (close - entry_px) / entry_px
    return raw - FEE - SLIP * 2


def _stats(subset):
    if len(subset) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, "-"
    n     = len(subset)
    wr_l  = (subset["pnl_l"] > 0).mean() * 100
    avg_l = subset["pnl_l"].mean() * 100
    wr_s  = (subset["pnl_s"] > 0).mean() * 100
    avg_s = subset["pnl_s"].mean() * 100
    best  = "LONG" if avg_l >= avg_s else "SHORT"
    return n, wr_l, avg_l, wr_s, avg_s, best


def main():
    et_tz = pytz.timezone("America/New_York")

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    print(f"[binance] {len(df):,} bougies chargees ({df['date_et'].min()} - {df['date_et'].max()})")

    weekly = build_weekly_levels(df)

    episodes, dates = [], []
    for date, grp in df.groupby("date_et"):
        hm_vals    = set(grp["hm_et"].values)
        has_asia   = any(60  <= h < 300 for h in hm_vals)
        has_london = any(420 <= h < 600 for h in hm_vals)
        has_or     = any(OR_START <= h < OR_END for h in hm_vals)
        has_entry  = ENTRY_HM in hm_vals
        if has_asia and has_london and has_or and has_entry and len(grp) >= 60:
            episodes.append(grp.sort_values("ts_et").reset_index(drop=True))
            dates.append(date)

    n_total = len(episodes)
    n_train = int(n_total * (1 - TEST_RATIO))
    print(f"[silver_bullet] {n_total} jours valides | train={n_train} | test={n_total - n_train}")

    rows = []
    for i, (day_df, date) in enumerate(zip(episodes, dates)):
        split    = "TRAIN" if i < n_train else "TEST"
        pwh, pwl = weekly.get(date, (None, None))
        ctx      = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        lc       = ctx["london_ctx"]

        or_mask = (day_df["hm_et"] >= OR_START) & (day_df["hm_et"] < OR_END)
        or_df   = day_df[or_mask]
        if len(or_df) < 5:
            continue

        or_high = float(or_df["high"].max())
        or_low  = float(or_df["low"].min())

        entry_mask = (day_df["hm_et"] >= ENTRY_HM) & (day_df["hm_et"] < EXIT_HM)
        entry_df   = day_df[entry_mask]
        if len(entry_df) < 5:
            continue

        first = entry_df.iloc[0]
        if float(first["high"]) > or_high:
            sc = 1   # SWEEP_H — OR_HIGH swept -> SHORT attendu
        elif float(first["low"]) < or_low:
            sc = 2   # SWEEP_L — OR_LOW swept  -> LONG attendu
        else:
            sc = 0   # NO_SWEEP

        # Pool context : OR_HIGH/LOW vs London session + PWH/PWL
        ref_h = ctx.get("session_high")
        ref_l = ctx.get("session_low")
        pc = compute_pool_ctx(or_high, or_low, ref_h, ref_l, pwh, pwl)

        entry_long  = float(first["open"]) * (1 + SLIP)
        entry_short = float(first["open"]) * (1 - SLIP)

        pnl_l = sim_trade(entry_df, entry_long,  +1)
        pnl_s = sim_trade(entry_df, entry_short, -1)

        rows.append({
            "date":  date, "year": date.year, "split": split,
            "lc": lc, "sc": sc, "pc": pc,
            "pnl_l": pnl_l, "pnl_s": pnl_s,
        })

    df_r    = pd.DataFrame(rows)
    train_r = df_r[df_r["split"] == "TRAIN"]
    test_r  = df_r[df_r["split"] == "TEST"]

    # ── Section 1 : Vue globale train (lc x sc x pc) ──────────────────────────
    print("\n" + "=" * 76)
    print("  SILVER BULLET 10:00 ET — Breakdown train (lc x sc x pc, N>=5)")
    print("=" * 76)
    print(f"  {'lc':<10} {'sc':<10} {'pc':<12} {'N':>5}  "
          f"{'WR_L%':>7} {'avg_L%':>8}  {'WR_S%':>7} {'avg_S%':>8}  Best")
    print("  " + "-" * 74)

    promising = []
    for lc in [0, 1, 2]:
        for sc in [1, 2, 0]:   # sweep first, then no_sweep
            for pc in [0, 1, 2]:
                sub = train_r[
                    (train_r["lc"] == lc) &
                    (train_r["sc"] == sc) &
                    (train_r["pc"] == pc)
                ]
                n, wr_l, avg_l, wr_s, avg_s, best = _stats(sub)
                if n < 5:
                    continue
                best_avg = avg_l if best == "LONG" else avg_s
                star = " *" if best_avg > 0.05 else ""
                print(f"  {LC_NAMES[lc]:<10} {SC_NAMES[sc]:<10} {PC_NAMES[pc]:<12} {n:>5}  "
                      f"{wr_l:>6.1f}% {avg_l:>+8.3f}%  {wr_s:>6.1f}% {avg_s:>+8.3f}%  {best}{star}")
                if best_avg > 0.05:
                    promising.append((lc, sc, pc, best, best_avg))

    # ── Section 2 : Contextes prometteurs annee par annee ─────────────────────
    promising.sort(key=lambda x: -x[4])

    if promising:
        print("\n" + "=" * 76)
        print("  CONTEXTES PROMETTEURS — annee par annee")
        print("=" * 76)

        for lc, sc, pc, best, _ in promising[:8]:
            sub = df_r[
                (df_r["lc"] == lc) &
                (df_r["sc"] == sc) &
                (df_r["pc"] == pc)
            ]
            sub_train = sub[sub["split"] == "TRAIN"]
            n, wr_l, avg_l, wr_s, avg_s, _ = _stats(sub_train)
            wr_best  = wr_l  if best == "LONG" else wr_s
            avg_best = avg_l if best == "LONG" else avg_s

            print(f"\n  [{LC_NAMES[lc]} x {SC_NAMES[sc]} x {PC_NAMES[pc]} -> {best}]")
            print(f"  TRAIN global : N={n}  WR={wr_best:.1f}%  avg={avg_best:+.3f}%")
            print(f"    An   Per    N     WR%    avg%   total%")
            print(f"  " + "-" * 40)

            for year in sorted(sub["year"].unique()):
                yr_sub = sub[sub["year"] == year]
                split  = yr_sub.iloc[0]["split"]
                n_yr, wr_l_yr, avg_l_yr, wr_s_yr, avg_s_yr, _ = _stats(yr_sub)
                wr_yr  = wr_l_yr  if best == "LONG" else wr_s_yr
                avg_yr = avg_l_yr if best == "LONG" else avg_s_yr
                total  = avg_yr * n_yr
                mark   = "*" if avg_yr > 0 else "x"
                print(f"  {year} {split:5s}  {n_yr:3d}  {wr_yr:6.1f}%  {avg_yr:+6.3f}%    {total:+5.2f}% {mark}")

    else:
        print("\n  Aucun contexte avec avg > 0.05% en train.")

    # ── Section 3 : Resume test set ───────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  RESUME TEST SET — contextes prometteurs")
    print("=" * 76)

    if promising:
        for lc, sc, pc, best, train_avg in promising[:6]:
            sub_test = test_r[
                (test_r["lc"] == lc) &
                (test_r["sc"] == sc) &
                (test_r["pc"] == pc)
            ]
            n, wr_l, avg_l, wr_s, avg_s, _ = _stats(sub_test)
            avg = avg_l if best == "LONG" else avg_s
            wr  = wr_l  if best == "LONG" else wr_s
            mark = "*" if avg > 0 else "x"
            print(f"  [{LC_NAMES[lc]} x {SC_NAMES[sc]} x {PC_NAMES[pc]} -> {best}] "
                  f"TEST N={n}  WR={wr:.1f}%  avg={avg:+.3f}% {mark}  "
                  f"(train avg={train_avg:+.3f}%)")

    # ── Section 4 : Distribution globale ──────────────────────────────────────
    print("\n" + "=" * 76)
    print("  DISTRIBUTION")
    print("=" * 76)
    sc_dist = df_r.groupby("sc").size()
    print(f"  Jours avec sweep (sc!=0) : {(df_r['sc'] != 0).sum()} / {len(df_r)}")
    print(f"  SWEEP_H : {sc_dist.get(1, 0)} | SWEEP_L : {sc_dist.get(2, 0)} | NO_SWEEP : {sc_dist.get(0, 0)}")
    pc_dist = df_r.groupby("pc").size()
    print(f"  BSL_SWEPT : {pc_dist.get(1, 0)} | SSL_SWEPT : {pc_dist.get(2, 0)} | NEUTRAL : {pc_dist.get(0, 0)}")
    print(f"  Total train : {len(train_r)} | test : {len(test_r)}")


if __name__ == "__main__":
    main()
