"""
analyse_1450.py — Analyse en isolation de la macro 14:50 (mac_idx=7).
Breakdown par contexte (lc x pc x sc) + annee par annee.
"""

import sys
import numpy as np
import pandas as pd
import pytz
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.stats_state import (
    MACROS, encode, decode,
    compute_daily_context, compute_pool_ctx, build_weekly_levels,
)
from main import _sim_trade_rr
from data.binance import load_binance_1m

MAC_IDX    = 7          # 14:50-15:10 ET
SL_PCT     = 0.005
RR         = 2.0
EXIT_HM    = 960        # 16:00 ET
TEST_RATIO = 0.2
SKIP_DAYS  = frozenset({0})   # Skip lundi

FEE_RATE   = 0.0005
SLIPPAGE   = 0.0002
REF_WINDOW = 240

LC_NAMES = {0: "NO_RAID", 1: "RAID_H ", 2: "RAID_L "}
PC_NAMES = {0: "NEUTRAL  ", 1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_NAMES = {0: "NO_SWEEP", 1: "SWEEP_H ", 2: "SWEEP_L "}


def sim_long(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    first = exit_df.iloc[0]
    ep    = entry_px * (1 + SLIPPAGE)
    if pc == 2 and ref_h is not None and ref_h > ep:
        tp_pct = (ref_h - ep) / ep
        sweep_low = min(float(first["low"]), pre_low)
        sl_pct_dyn = SL_PCT + max(0.0, (ep - sweep_low) / ep)
        if tp_pct > sl_pct_dyn > 0:
            return _sim_trade_rr(exit_df, ep, +1, sl_pct_dyn, tp_pct, FEE_RATE, SLIPPAGE)
    tp_pct_fixed = SL_PCT * RR
    return _sim_trade_rr(exit_df, ep, +1, SL_PCT, tp_pct_fixed, FEE_RATE, SLIPPAGE)


def sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    first = exit_df.iloc[0]
    ep    = entry_px * (1 - SLIPPAGE)
    if pc == 1 and ref_l is not None and ref_l < ep:
        tp_pct = (ep - ref_l) / ep
        sweep_high = max(float(first["high"]), pre_high)
        sl_pct_dyn = SL_PCT + max(0.0, (sweep_high - ep) / ep)
        if tp_pct > sl_pct_dyn > 0:
            return _sim_trade_rr(exit_df, ep, -1, sl_pct_dyn, tp_pct, FEE_RATE, SLIPPAGE)
    tp_pct_fixed = SL_PCT * RR
    return _sim_trade_rr(exit_df, ep, -1, SL_PCT, tp_pct_fixed, FEE_RATE, SLIPPAGE)


def run():
    et_tz = pytz.timezone("America/New_York")

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]     = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]     = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"]   = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    weekly = build_weekly_levels(df)

    episodes, dates = [], []
    for date, grp in df.groupby("date_et"):
        hm_vals    = set(grp["hm_et"].values)
        has_asia   = any(60  <= h < 300 for h in hm_vals)
        has_london = any(420 <= h < 600 for h in hm_vals)
        has_ny     = any(530 <= h < 910 for h in hm_vals)
        if has_asia and has_london and has_ny and len(grp) >= 60:
            episodes.append(grp.sort_values("ts_et").reset_index(drop=True))
            dates.append(date)

    n_total = len(episodes)
    n_train = int(n_total * (1 - TEST_RATIO))
    print(f"\n=== ANALYSE 14:50 (mac_idx={MAC_IDX}) ===")
    print(f"Params : SL={SL_PCT*100:.1f}%  RR={RR}  target_pool=True  exit={EXIT_HM//60}:{EXIT_HM%60:02d} ET")
    print(f"Donnees : {dates[0]} a {dates[-1]} | {n_total} jours | train={n_train} | test={n_total-n_train}\n")

    mac_start, mac_end = MACROS[MAC_IDX]
    pre_start = mac_start - 20

    rows = []

    for i, (day_df, date) in enumerate(zip(episodes, dates)):
        if date.weekday() in SKIP_DAYS:
            continue

        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc  = ctx["month_ctx"]
        dc  = ctx["day_ctx"]
        lc  = ctx["london_ctx"]

        pre_mask  = (day_df["hm_et"] >= pre_start) & (day_df["hm_et"] < mac_start)
        exit_mask = (day_df["hm_et"] >= mac_start) & (day_df["hm_et"] < EXIT_HM)

        pre_df  = day_df[pre_mask]
        exit_df = day_df[exit_mask]

        if len(pre_df) < 3 or len(exit_df) < 5:
            continue

        pre_high = float(pre_df["high"].max())
        pre_low  = float(pre_df["low"].min())
        first    = exit_df.iloc[0]

        if float(first["high"]) > pre_high:
            sc = 1
        elif float(first["low"]) < pre_low:
            sc = 2
        else:
            sc = 0

        ref_start = pre_start - REF_WINDOW
        ref_mask  = (day_df["hm_et"] >= max(0, ref_start)) & (day_df["hm_et"] < pre_start)
        ref_df    = day_df[ref_mask]
        if len(ref_df) >= 5:
            ref_h = float(ref_df["high"].max())
            ref_l = float(ref_df["low"].min())
        else:
            ref_h = ctx.get("session_high")
            ref_l = ctx.get("session_low")

        pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

        entry_px = float(first["open"])
        pnl_long  = sim_long(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low)
        pnl_short = sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low)

        period = "TRAIN" if i < n_train else "TEST"

        rows.append({
            "date":   date,
            "year":   date.year,
            "period": period,
            "mc": mc, "dc": dc, "lc": lc, "sc": sc, "pc": pc,
            "pnl_long":  pnl_long,
            "pnl_short": pnl_short,
        })

    data = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 1. BREAKDOWN PAR CONTEXTE (lc x sc x pc) — sur donnees TRAIN
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("BREAKDOWN PAR CONTEXTE (train seulement) — LONG / SHORT")
    print("=" * 72)
    print(f"{'lc':10} {'sc':10} {'pc':10} {'N':>5}  "
          f"{'WR_L%':>7} {'avg_L%':>8}  {'WR_S%':>7} {'avg_S%':>8}  Best")
    print("-" * 72)

    train = data[data["period"] == "TRAIN"]

    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < 2:
                    continue
                wr_l   = (sub["pnl_long"]  > 0).mean() * 100
                avg_l  = sub["pnl_long"].mean()  * 100
                wr_s   = (sub["pnl_short"] > 0).mean() * 100
                avg_s  = sub["pnl_short"].mean()  * 100
                best   = "LONG " if avg_l > avg_s else "SHORT"
                star   = "*" if max(avg_l, avg_s) > 0.05 else " "
                print(f"{LC_NAMES[lc]:10} {SC_NAMES[sc]:10} {PC_NAMES[pc]:10} {len(sub):>5}  "
                      f"{wr_l:>6.1f}% {avg_l:>+7.3f}%  {wr_s:>6.1f}% {avg_s:>+7.3f}%  {best} {star}")

    # ------------------------------------------------------------------ #
    # 2. ANNEE PAR ANNEE (best direction par contexte — comme en prod)
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print("ANNEE PAR ANNEE — ALL CONTEXTS (aligned_only=False)")
    print("=" * 72)
    print(f"{'Annee':>6} {'Period':>6} {'N':>5}  {'WR_L%':>7} {'tot_L%':>8}  {'WR_S%':>7} {'tot_S%':>8}")
    print("-" * 72)

    for yr in sorted(data["year"].unique()):
        sub = data[data["year"] == yr]
        period_label = "TRAIN" if sub.iloc[0]["period"] == "TRAIN" else "TEST "
        if "TEST" in sub["period"].values:
            period_label = "TEST "

        wr_l  = (sub["pnl_long"]  > 0).mean() * 100
        wr_s  = (sub["pnl_short"] > 0).mean() * 100
        tot_l = sub["pnl_long"].sum()  * 100
        tot_s = sub["pnl_short"].sum() * 100
        print(f"{yr:>6} {period_label:>6} {len(sub):>5}  "
              f"{wr_l:>6.1f}% {tot_l:>+8.2f}%  {wr_s:>6.1f}% {tot_s:>+8.2f}%")

    # ------------------------------------------------------------------ #
    # 3. ANNEE PAR ANNEE — aligned_only (sc != 0)
    # ------------------------------------------------------------------ #
    aligned = data[data["sc"] != 0]
    if not aligned.empty:
        print()
        print("=" * 72)
        print("ANNEE PAR ANNEE — aligned_only (sc != 0)")
        print("=" * 72)
        print(f"{'Annee':>6} {'Period':>6} {'N':>5}  {'WR_L%':>7} {'tot_L%':>8}  {'WR_S%':>7} {'tot_S%':>8}")
        print("-" * 72)
        for yr in sorted(aligned["year"].unique()):
            sub = aligned[aligned["year"] == yr]
            period_label = "TEST " if "TEST" in sub["period"].values else "TRAIN"
            wr_l  = (sub["pnl_long"]  > 0).mean() * 100
            wr_s  = (sub["pnl_short"] > 0).mean() * 100
            tot_l = sub["pnl_long"].sum()  * 100
            tot_s = sub["pnl_short"].sum() * 100
            print(f"{yr:>6} {period_label:>6} {len(sub):>5}  "
                  f"{wr_l:>6.1f}% {tot_l:>+8.2f}%  {wr_s:>6.1f}% {tot_s:>+8.2f}%")

    # ------------------------------------------------------------------ #
    # 4. RESUME — contextes prometteurs (train, N >= 5, avg > 0.03%)
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print("CONTEXTES PROMETTEURS (train, N>=5, avg_best > 0.05%)")
    print("=" * 72)
    print(f"{'lc':10} {'sc':10} {'pc':10} {'N':>5}  {'Dir':>6} {'WR%':>7} {'avg%':>8} {'total%':>8}")
    print("-" * 72)

    found = False
    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < 5:
                    continue
                best_dir = "LONG" if sub["pnl_long"].mean() >= sub["pnl_short"].mean() else "SHORT"
                pnls = sub["pnl_long"] if best_dir == "LONG" else sub["pnl_short"]
                wr   = (pnls > 0).mean() * 100
                avg  = pnls.mean() * 100
                tot  = pnls.sum()  * 100
                if avg > 0.05:
                    print(f"{LC_NAMES[lc]:10} {SC_NAMES[sc]:10} {PC_NAMES[pc]:10} {len(sub):>5}  "
                          f"{best_dir:>6} {wr:>6.1f}% {avg:>+7.3f}% {tot:>+8.2f}%  *")
                    found = True

    if not found:
        print("  Aucun contexte avec N>=5 et avg>0.05%")

    # ------------------------------------------------------------------ #
    # 5. TEST SET breakdown
    # ------------------------------------------------------------------ #
    test = data[data["period"] == "TEST"]
    print()
    print("=" * 72)
    print(f"TEST SET — {len(test)} jours 14:50 (aligned inclus : {(test['sc'] != 0).sum()})")
    print("=" * 72)
    if not test.empty:
        print(f"  LONG  : WR={( test['pnl_long']>0).mean()*100:.1f}%  avg={test['pnl_long'].mean()*100:+.3f}%  total={test['pnl_long'].sum()*100:+.2f}%")
        print(f"  SHORT : WR={(test['pnl_short']>0).mean()*100:.1f}%  avg={test['pnl_short'].mean()*100:+.3f}%  total={test['pnl_short'].sum()*100:+.2f}%")

        aligned_test = test[test["sc"] != 0]
        if not aligned_test.empty:
            print(f"\n  aligned_only (sc!=0) : N={len(aligned_test)}")
            print(f"  LONG  : WR={(aligned_test['pnl_long']>0).mean()*100:.1f}%  avg={aligned_test['pnl_long'].mean()*100:+.3f}%  total={aligned_test['pnl_long'].sum()*100:+.2f}%")
            print(f"  SHORT : WR={(aligned_test['pnl_short']>0).mean()*100:.1f}%  avg={aligned_test['pnl_short'].mean()*100:+.3f}%  total={aligned_test['pnl_short'].sum()*100:+.2f}%")


if __name__ == "__main__":
    run()
