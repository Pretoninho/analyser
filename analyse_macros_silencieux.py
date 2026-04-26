"""
analyse_macros_silencieux.py — Analyse en isolation des macros silencieuses.
08:50 (idx=1) | 10:50 (idx=3) | 12:50 (idx=5) | 13:50 (idx=6)

Pour chaque macro : breakdown (lc x sc x pc), annee par annee, contextes prometteurs.
"""

import sys
import numpy as np
import pandas as pd
import pytz
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.stats_state import (
    MACROS, encode, decode,
    compute_daily_context, compute_pool_ctx, build_weekly_levels,
)
from main import _sim_trade_rr
from data.binance import load_binance_1m

MACROS_CIBLES = [1, 3, 5, 6]   # 08:50, 10:50, 12:50, 13:50
MAC_NAMES     = {1: "08:50", 3: "10:50", 5: "12:50", 6: "13:50"}

SL_PCT     = 0.005
RR         = 2.0
EXIT_HM    = 960
TEST_RATIO = 0.2
SKIP_DAYS  = frozenset({0})

FEE_RATE   = 0.0005
SLIPPAGE   = 0.0002
REF_WINDOW = 240

LC_NAMES = {0: "NO_RAID", 1: "RAID_H ", 2: "RAID_L "}
PC_NAMES = {0: "NEUTRAL  ", 1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_NAMES = {0: "NO_SWEEP", 1: "SWEEP_H ", 2: "SWEEP_L "}


def sim_long(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    ep = entry_px * (1 + SLIPPAGE)
    if pc == 2 and ref_h is not None and ref_h > ep:
        tp_pct = (ref_h - ep) / ep
        sweep_low  = min(float(exit_df.iloc[0]["low"]), pre_low)
        sl_pct_dyn = SL_PCT + max(0.0, (ep - sweep_low) / ep)
        if tp_pct > sl_pct_dyn > 0:
            return _sim_trade_rr(exit_df, ep, +1, sl_pct_dyn, tp_pct, FEE_RATE, SLIPPAGE)
    return _sim_trade_rr(exit_df, ep, +1, SL_PCT, SL_PCT * RR, FEE_RATE, SLIPPAGE)


def sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    ep = entry_px * (1 - SLIPPAGE)
    if pc == 1 and ref_l is not None and ref_l < ep:
        tp_pct = (ep - ref_l) / ep
        sweep_high = max(float(exit_df.iloc[0]["high"]), pre_high)
        sl_pct_dyn = SL_PCT + max(0.0, (sweep_high - ep) / ep)
        if tp_pct > sl_pct_dyn > 0:
            return _sim_trade_rr(exit_df, ep, -1, sl_pct_dyn, tp_pct, FEE_RATE, SLIPPAGE)
    return _sim_trade_rr(exit_df, ep, -1, SL_PCT, SL_PCT * RR, FEE_RATE, SLIPPAGE)


def analyse_macro(mac_idx, episodes, dates, contexts, n_train):
    mac_start, mac_end = MACROS[mac_idx]
    pre_start = mac_start - 20
    rows = []

    for i, (day_df, date, ctx) in enumerate(zip(episodes, dates, contexts)):
        if date.weekday() in SKIP_DAYS:
            continue

        mc, dc, lc = ctx["month_ctx"], ctx["day_ctx"], ctx["london_ctx"]
        pwh, pwl   = ctx["pwh"], ctx["pwl"]

        pre_mask  = (day_df["hm_et"] >= pre_start) & (day_df["hm_et"] < mac_start)
        exit_mask = (day_df["hm_et"] >= mac_start) & (day_df["hm_et"] < EXIT_HM)

        pre_df  = day_df[pre_mask]
        exit_df = day_df[exit_mask]

        if len(pre_df) < 3 or len(exit_df) < 5:
            continue

        pre_high = float(pre_df["high"].max())
        pre_low  = float(pre_df["low"].min())
        first    = exit_df.iloc[0]

        sc = 1 if float(first["high"]) > pre_high else (2 if float(first["low"]) < pre_low else 0)

        ref_start = pre_start - REF_WINDOW
        ref_mask  = (day_df["hm_et"] >= max(0, ref_start)) & (day_df["hm_et"] < pre_start)
        ref_df    = day_df[ref_mask]
        if len(ref_df) >= 5:
            ref_h, ref_l = float(ref_df["high"].max()), float(ref_df["low"].min())
        else:
            ref_h = ctx.get("session_high")
            ref_l = ctx.get("session_low")

        pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

        entry_px  = float(first["open"])
        pnl_long  = sim_long(exit_df,  entry_px, pc, ref_h, ref_l, pre_high, pre_low)
        pnl_short = sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low)

        rows.append({
            "date": date, "year": date.year,
            "period": "TRAIN" if i < n_train else "TEST",
            "lc": lc, "sc": sc, "pc": pc,
            "pnl_long": pnl_long, "pnl_short": pnl_short,
        })

    return pd.DataFrame(rows)


def print_macro_report(mac_idx, data):
    name  = MAC_NAMES[mac_idx]
    train = data[data["period"] == "TRAIN"]
    test  = data[data["period"] == "TEST"]

    W = 70
    print()
    print("=" * W)
    print(f"  MACRO {name}  (mac_idx={mac_idx})  |  train N={len(train)}  test N={len(test)}")
    print("=" * W)

    # --- Annee par annee (aligned_only : sc != 0) ---
    aligned = data[data["sc"] != 0]
    print(f"\n  Annee par annee — aligned_only (sc!=0) :")
    print(f"  {'An':>4} {'Per':>5} {'N':>4}  {'WR_L%':>6} {'tot_L%':>8}  {'WR_S%':>6} {'tot_S%':>8}")
    print("  " + "-" * 48)
    for yr in sorted(data["year"].unique()):
        sub = aligned[aligned["year"] == yr]
        if sub.empty:
            continue
        per = "TEST " if "TEST" in sub["period"].values else "TRAIN"
        wr_l  = (sub["pnl_long"]  > 0).mean() * 100
        wr_s  = (sub["pnl_short"] > 0).mean() * 100
        tot_l = sub["pnl_long"].sum()  * 100
        tot_s = sub["pnl_short"].sum() * 100
        print(f"  {yr:>4} {per:>5} {len(sub):>4}  {wr_l:>5.1f}% {tot_l:>+8.2f}%  {wr_s:>5.1f}% {tot_s:>+8.2f}%")

    # --- Breakdown (lc x sc x pc) train ---
    print(f"\n  Breakdown (train, N>=3) — LONG / SHORT :")
    print(f"  {'lc':9} {'sc':9} {'pc':10} {'N':>4}  {'WR_L%':>6} {'avg_L%':>7}  {'WR_S%':>6} {'avg_S%':>7}  Best")
    print("  " + "-" * 68)

    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < 3:
                    continue
                wr_l  = (sub["pnl_long"]  > 0).mean() * 100
                wr_s  = (sub["pnl_short"] > 0).mean() * 100
                avg_l = sub["pnl_long"].mean()  * 100
                avg_s = sub["pnl_short"].mean() * 100
                best  = "LONG " if avg_l > avg_s else "SHORT"
                flag  = " *" if max(avg_l, avg_s) > 0.05 and len(sub) >= 5 else "  "
                print(f"  {LC_NAMES[lc]:9} {SC_NAMES[sc]:9} {PC_NAMES[pc]:10} {len(sub):>4}  "
                      f"{wr_l:>5.1f}% {avg_l:>+7.3f}%  {wr_s:>5.1f}% {avg_s:>+7.3f}%  {best}{flag}")

    # --- Contextes prometteurs (N >= 5, avg > 0.05%) ---
    prometteurs = []
    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < 5:
                    continue
                best_dir = "LONG" if sub["pnl_long"].mean() >= sub["pnl_short"].mean() else "SHORT"
                pnls = sub["pnl_long"] if best_dir == "LONG" else sub["pnl_short"]
                avg  = pnls.mean() * 100
                if avg > 0.05:
                    prometteurs.append((lc, sc, pc, len(sub), best_dir,
                                        (pnls > 0).mean() * 100, avg, pnls.sum() * 100))

    print(f"\n  Contextes prometteurs (train, N>=5, avg>0.05%) :")
    if prometteurs:
        print(f"  {'lc':9} {'sc':9} {'pc':10} {'N':>4}  {'Dir':>5} {'WR%':>6} {'avg%':>7} {'total%':>8}")
        print("  " + "-" * 60)
        for (lc, sc, pc, n, d, wr, avg, tot) in sorted(prometteurs, key=lambda x: -x[6]):
            print(f"  {LC_NAMES[lc]:9} {SC_NAMES[sc]:9} {PC_NAMES[pc]:10} {n:>4}  "
                  f"{d:>5} {wr:>5.1f}% {avg:>+7.3f}% {tot:>+8.2f}%  *")
    else:
        print("  Aucun contexte avec N>=5 et avg>0.05%")

    # --- Test set summary ---
    al_test = test[test["sc"] != 0]
    print(f"\n  Test set — aligned_only N={len(al_test)} :")
    if not al_test.empty:
        print(f"    LONG  WR={(al_test['pnl_long']>0).mean()*100:.1f}%  avg={al_test['pnl_long'].mean()*100:+.3f}%  total={al_test['pnl_long'].sum()*100:+.2f}%")
        print(f"    SHORT WR={(al_test['pnl_short']>0).mean()*100:.1f}%  avg={al_test['pnl_short'].mean()*100:+.3f}%  total={al_test['pnl_short'].sum()*100:+.2f}%")
    else:
        print("    Aucun trade aligned sur le test set.")


def run():
    et_tz = pytz.timezone("America/New_York")

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]     = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]     = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"]   = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    weekly = build_weekly_levels(df)

    episodes, dates, contexts = [], [], []
    for date, grp in df.groupby("date_et"):
        hm_vals = set(grp["hm_et"].values)
        if (any(60 <= h < 300 for h in hm_vals) and
                any(420 <= h < 600 for h in hm_vals) and
                any(530 <= h < 910 for h in hm_vals) and
                len(grp) >= 60):
            day_df = grp.sort_values("ts_et").reset_index(drop=True)
            pwh, pwl = weekly.get(date, (None, None))
            ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
            ctx["pwh"], ctx["pwl"] = pwh, pwl
            episodes.append(day_df)
            dates.append(date)
            contexts.append(ctx)

    n_total = len(episodes)
    n_train = int(n_total * (1 - TEST_RATIO))
    print(f"[data] {n_total} jours | train={n_train} | test={n_total - n_train}")

    for mac_idx in MACROS_CIBLES:
        data = analyse_macro(mac_idx, episodes, dates, contexts, n_train)
        print_macro_report(mac_idx, data)

    print()
    print("=" * 70)
    print("  RESUME — Verdict par macro")
    print("=" * 70)
    verdicts = {
        1:  "08:50 — A ANALYSER (premiere macro NY, SHORT historiquement mauvais)",
        3:  "10:50 — A ANALYSER (peu de donnees en train/test)",
        5:  "12:50 — A ANALYSER (drag PM constate)",
        6:  "13:50 — A ANALYSER (drag systematique constate)",
    }
    for k, v in verdicts.items():
        print(f"  mac_idx={k}  {v}")


if __name__ == "__main__":
    run()
