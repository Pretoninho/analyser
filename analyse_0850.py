"""
analyse_0850.py — Analyse approfondie de la macro 08:50 (mac_idx=1).

Questions :
  1. Quels contextes (lc x sc x pc) sont robustes sur train ET test ?
  2. Est-ce que le timing d'entree est le probleme ?
     -> sweep d'entree de 08:50 a 09:40 ET (toutes les 10 min)
  3. Est-ce que 09:30 (ouverture NYSE) est un meilleur point d'entree ?
  4. Stabilite annee par annee des contextes prometteurs.
"""

import sys
import numpy as np
import pandas as pd
import pytz
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from engine.stats_state import (
    MACROS, compute_daily_context, compute_pool_ctx, build_weekly_levels,
)
from main import _sim_trade_rr
from data.binance import load_binance_1m

# ── Parametres (coherents avec la prod) ──────────────────────────────────────
MAC_IDX    = 1          # 08:50-09:10 ET
MAC_START  = 530        # 08:50 en minutes ET
MAC_END    = 550        # 09:10 en minutes ET
PRE_START  = 510        # 08:30 (fenetre pre-macro)

NYSE_OPEN  = 570        # 09:30 ET
EXIT_HM    = 960        # 16:00 ET
TEST_RATIO = 0.20
SKIP_DAYS  = frozenset({0})   # lundi

SL_PCT     = 0.005
RR         = 2.0
FEE        = 0.0005
SLIP       = 0.0002
REF_WINDOW = 240

# Entrees a tester : 08:50 09:00 09:10 09:20 09:30 09:40
ENTRY_TIMES = [530, 540, 550, 560, 570, 580]
ENTRY_NAMES = {530:"08:50", 540:"09:00", 550:"09:10",
               560:"09:20", 570:"09:30 (NYSE)", 580:"09:40"}

LC_NAMES = {0:"NO_RAID", 1:"RAID_H ", 2:"RAID_L "}
PC_NAMES = {0:"NEUTRAL  ", 1:"BSL_SWEPT", 2:"SSL_SWEPT"}
SC_NAMES = {0:"NO_SWEEP", 1:"SWEEP_H ", 2:"SWEEP_L "}


# ─────────────────────────────────────────────────────────────────────────────
def sim(exit_df, entry_px, direction, sl_pct=SL_PCT, tp_pct=None):
    if tp_pct is None:
        tp_pct = sl_pct * RR
    ep = entry_px * (1 + SLIP * direction)
    return _sim_trade_rr(exit_df, ep, direction, sl_pct, tp_pct, FEE, SLIP)


def compute_ctx_at_time(day_df, entry_hm, ref_h, ref_l, pwh, pwl):
    """Recalcule sc et pc pour une entree a entry_hm."""
    pre_mask = (day_df["hm_et"] >= entry_hm - 20) & (day_df["hm_et"] < entry_hm)
    pre_df   = day_df[pre_mask]
    if len(pre_df) < 3:
        return None, None, None
    pre_high = float(pre_df["high"].max())
    pre_low  = float(pre_df["low"].min())

    exit_mask = (day_df["hm_et"] >= entry_hm) & (day_df["hm_et"] < EXIT_HM)
    exit_df   = day_df[exit_mask]
    if len(exit_df) < 5:
        return None, None, None

    first = exit_df.iloc[0]
    sc = 1 if float(first["high"]) > pre_high else (2 if float(first["low"]) < pre_low else 0)
    pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)
    return sc, pc, exit_df


# ─────────────────────────────────────────────────────────────────────────────
def run():
    et_tz = pytz.timezone("America/New_York")

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]     = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]     = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"]   = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    weekly = build_weekly_levels(df)

    episodes, dates, ctxs = [], [], []
    for date, grp in df.groupby("date_et"):
        hm_vals = set(grp["hm_et"].values)
        if (any(60  <= h < 300 for h in hm_vals) and
                any(420 <= h < 600 for h in hm_vals) and
                any(530 <= h < 910 for h in hm_vals) and
                len(grp) >= 60):
            day_df = grp.sort_values("ts_et").reset_index(drop=True)
            pwh, pwl = weekly.get(date, (None, None))
            ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
            ctx["pwh"], ctx["pwl"] = pwh, pwl
            episodes.append(day_df)
            dates.append(date)
            ctxs.append(ctx)

    n_total = len(episodes)
    n_train = int(n_total * (1 - TEST_RATIO))

    # ── Collecte des donnees brutes ───────────────────────────────────────────
    rows = []   # une ligne par (jour, entry_time)

    for i, (day_df, date, ctx) in enumerate(zip(episodes, dates, ctxs)):
        if date.weekday() in SKIP_DAYS:
            continue
        period = "TRAIN" if i < n_train else "TEST"

        mc = ctx["month_ctx"]
        dc = ctx["day_ctx"]
        lc = ctx["london_ctx"]
        pwh, pwl = ctx["pwh"], ctx["pwl"]

        # Contexte de reference 4h (BSL/SSL)
        ref_mask = (day_df["hm_et"] >= PRE_START - REF_WINDOW) & (day_df["hm_et"] < PRE_START)
        ref_df   = day_df[ref_mask]
        if len(ref_df) >= 5:
            ref_h = float(ref_df["high"].max())
            ref_l = float(ref_df["low"].min())
        else:
            ref_h = ctx.get("session_high")
            ref_l = ctx.get("session_low")

        # sc/pc de reference a 08:50
        pre_mask_850 = (day_df["hm_et"] >= PRE_START) & (day_df["hm_et"] < MAC_START)
        pre_df_850   = day_df[pre_mask_850]
        exit_mask_850= (day_df["hm_et"] >= MAC_START) & (day_df["hm_et"] < EXIT_HM)
        exit_df_850  = day_df[exit_mask_850]

        if len(pre_df_850) < 3 or len(exit_df_850) < 5:
            continue

        pre_high_850 = float(pre_df_850["high"].max())
        pre_low_850  = float(pre_df_850["low"].min())
        first_850    = exit_df_850.iloc[0]
        sc_850 = (1 if float(first_850["high"]) > pre_high_850
                  else (2 if float(first_850["low"]) < pre_low_850 else 0))
        pc_850 = compute_pool_ctx(pre_high_850, pre_low_850, ref_h, ref_l, pwh, pwl)

        # Simulation pour chaque temps d'entree
        entry_row = {
            "date": date, "year": date.year, "period": period,
            "mc": mc, "dc": dc, "lc": lc,
            "sc_850": sc_850, "pc_850": pc_850,
        }

        for entry_hm in ENTRY_TIMES:
            if entry_hm == MAC_START:
                sc, pc, exit_df = sc_850, pc_850, exit_df_850
                entry_px = float(first_850["open"])
            else:
                sc, pc, exit_df = compute_ctx_at_time(day_df, entry_hm, ref_h, ref_l, pwh, pwl)
                if sc is None:
                    entry_row[f"pnl_long_{entry_hm}"]  = None
                    entry_row[f"pnl_short_{entry_hm}"] = None
                    entry_row[f"sc_{entry_hm}"] = None
                    entry_row[f"pc_{entry_hm}"] = None
                    continue
                entry_px = float(exit_df.iloc[0]["open"])

            pnl_l = sim(exit_df, entry_px, +1)
            pnl_s = sim(exit_df, entry_px, -1)

            entry_row[f"pnl_long_{entry_hm}"]  = pnl_l
            entry_row[f"pnl_short_{entry_hm}"] = pnl_s
            entry_row[f"sc_{entry_hm}"]  = sc
            entry_row[f"pc_{entry_hm}"]  = pc

        rows.append(entry_row)

    data = pd.DataFrame(rows)
    train = data[data["period"] == "TRAIN"]
    test  = data[data["period"] == "TEST"]

    W = 72

    # ═════════════════════════════════════════════════════════════════════════
    # 1. SWEEP D'ENTREE : 08:50 -> 09:40 (aligned_only sur sc_850 != 0)
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  1. SWEEP D'ENTREE — aligned_only (sc_08:50 != 0)")
    print("     Meilleure direction par temps d'entree (train)")
    print("=" * W)
    print(f"  {'Entree':18} {'N':>5}  {'WR_L%':>6} {'avg_L%':>8}  {'WR_S%':>6} {'avg_S%':>8}  Best")
    print("  " + "-" * 66)

    aligned = data[data["sc_850"] != 0]
    al_train = aligned[aligned["period"] == "TRAIN"]
    al_test  = aligned[aligned["period"] == "TEST"]

    for entry_hm in ENTRY_TIMES:
        col_l = f"pnl_long_{entry_hm}"
        col_s = f"pnl_short_{entry_hm}"
        sub = al_train[[col_l, col_s]].dropna()
        if sub.empty:
            continue
        wr_l  = (sub[col_l] > 0).mean() * 100
        wr_s  = (sub[col_s] > 0).mean() * 100
        avg_l = sub[col_l].mean() * 100
        avg_s = sub[col_s].mean() * 100
        best  = "LONG " if avg_l > avg_s else "SHORT"
        flag  = " <<" if max(avg_l, avg_s) > 0.05 else ""
        print(f"  {ENTRY_NAMES[entry_hm]:18} {len(sub):>5}  "
              f"{wr_l:>5.1f}% {avg_l:>+8.3f}%  {wr_s:>5.1f}% {avg_s:>+8.3f}%  {best}{flag}")

    print()
    print("  TEST SET :")
    print(f"  {'Entree':18} {'N':>5}  {'WR_L%':>6} {'avg_L%':>8}  {'WR_S%':>6} {'avg_S%':>8}  Best")
    print("  " + "-" * 66)
    for entry_hm in ENTRY_TIMES:
        col_l = f"pnl_long_{entry_hm}"
        col_s = f"pnl_short_{entry_hm}"
        sub = al_test[[col_l, col_s]].dropna()
        if sub.empty:
            continue
        wr_l  = (sub[col_l] > 0).mean() * 100
        wr_s  = (sub[col_s] > 0).mean() * 100
        avg_l = sub[col_l].mean() * 100
        avg_s = sub[col_s].mean() * 100
        best  = "LONG " if avg_l > avg_s else "SHORT"
        flag  = " <<" if max(avg_l, avg_s) > 0.05 else ""
        print(f"  {ENTRY_NAMES[entry_hm]:18} {len(sub):>5}  "
              f"{wr_l:>5.1f}% {avg_l:>+8.3f}%  {wr_s:>5.1f}% {avg_s:>+8.3f}%  {best}{flag}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CONTEXTES FORTS @ 08:50 — ANNEE PAR ANNEE (train + test)
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  2. CONTEXTES LES PLUS FORTS @ 08:50 — annee par annee")
    print("=" * W)

    # Contextes selectionnees : WR >= 60% OU avg >= 0.3% en train, N>=5
    good_contexts = []
    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[
                    (train["lc"] == lc) & (train["sc_850"] == sc) & (train["pc_850"] == pc)
                ]
                if len(sub) < 5:
                    continue
                for col, direction in [(f"pnl_long_530", "LONG"), (f"pnl_short_530", "SHORT")]:
                    pnls = sub[col].dropna()
                    if len(pnls) < 5:
                        continue
                    wr  = (pnls > 0).mean() * 100
                    avg = pnls.mean() * 100
                    if wr >= 60 or avg >= 0.15:
                        good_contexts.append((lc, sc, pc, direction, wr, avg, len(pnls)))

    good_contexts.sort(key=lambda x: -x[5])

    for (lc, sc, pc, direction, wr_train, avg_train, n_train_ctx) in good_contexts[:6]:
        col = f"pnl_long_530" if direction == "LONG" else f"pnl_short_530"
        label = f"{LC_NAMES[lc].strip()} x {SC_NAMES[sc].strip()} x {PC_NAMES[pc].strip()} -> {direction}"
        print(f"\n  [{label}]")
        print(f"  TRAIN global : N={n_train_ctx}  WR={wr_train:.1f}%  avg={avg_train:+.3f}%")
        print(f"  {'An':>4} {'Per':>5} {'N':>4}  {'WR%':>6} {'avg%':>7} {'total%':>8}")
        print("  " + "-" * 40)

        for yr in sorted(data["year"].unique()):
            sub_yr = data[
                (data["year"] == yr) &
                (data["lc"] == lc) & (data["sc_850"] == sc) & (data["pc_850"] == pc)
            ]
            pnls = sub_yr[col].dropna()
            if len(pnls) < 1:
                continue
            per = "TEST " if "TEST" in sub_yr["period"].values else "TRAIN"
            wr  = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            tot = pnls.sum()  * 100
            flag = " *" if avg > 0.05 else ("  " if avg >= 0 else " x")
            print(f"  {yr:>4} {per:>5} {len(pnls):>4}  {wr:>5.1f}% {avg:>+7.3f}% {tot:>+8.2f}%{flag}")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ANALYSE 09:30 (NYSE OPEN) — sweep d'entree specifique
    #    Compare sc a 08:50 vs sc a 09:30 : coherence directionnelle ?
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  3. ANALYSE SPECIFIQUE 09:30 (ouverture NYSE)")
    print("     Question : entrer a 09:30 au lieu de 08:50 ?")
    print("=" * W)

    # sc_850 vs sc_930 : est-ce que le sweep de 09:30 confirme celui de 08:50 ?
    col_l_930 = "pnl_long_570"
    col_s_930 = "pnl_short_570"
    col_sc_930 = "sc_570"

    sub_both = data[
        (data["sc_850"] != 0) & (data[col_sc_930].notna()) & (data[col_sc_930] != 0)
    ].copy()

    sub_both["sc_align"] = sub_both["sc_850"] == sub_both[col_sc_930]

    print(f"\n  Jours ou sc_08:50 != 0 et sc_09:30 != 0 : N={len(sub_both)}")
    aligned_dir   = sub_both[sub_both["sc_align"] == True]
    unaligned_dir = sub_both[sub_both["sc_align"] == False]
    print(f"  Meme direction (08:50 confirme par 09:30) : N={len(aligned_dir)}")
    print(f"  Direction inverse                          : N={len(unaligned_dir)}")

    for label, sub_grp in [("CONFIRME  (sc_850 == sc_930)", aligned_dir),
                            ("INVERSE   (sc_850 != sc_930)", unaligned_dir)]:
        tr = sub_grp[sub_grp["period"] == "TRAIN"]
        te = sub_grp[sub_grp["period"] == "TEST"]
        if len(tr) == 0:
            continue
        print(f"\n  [{label}]")
        for (pset, pname) in [(tr, "TRAIN"), (te, "TEST ")]:
            if len(pset) == 0:
                continue
            wr_l  = (pset[col_l_930] > 0).mean() * 100
            wr_s  = (pset[col_s_930] > 0).mean() * 100
            avg_l = pset[col_l_930].mean() * 100
            avg_s = pset[col_s_930].mean() * 100
            best  = "LONG " if avg_l > avg_s else "SHORT"
            print(f"    {pname} N={len(pset):>3}  "
                  f"LONG WR={wr_l:.1f}% avg={avg_l:>+.3f}%  "
                  f"SHORT WR={wr_s:.1f}% avg={avg_s:>+.3f}%  best={best}")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. 09:30 SEUL — breakdown par sc/pc recalcule a 09:30
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  4. ENTREE 09:30 SEUL — breakdown (lc x sc_930 x pc_930, N>=5)")
    print("=" * W)
    print(f"  {'lc':9} {'sc_930':9} {'pc_930':10} {'N':>4}  "
          f"{'WR_L%':>6} {'avg_L%':>8}  {'WR_S%':>6} {'avg_S%':>8}  Best")
    print("  " + "-" * 68)

    col_sc = "sc_570"
    col_pc = "pc_570"
    col_l  = "pnl_long_570"
    col_s  = "pnl_short_570"

    prometteurs_930 = []
    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[
                    (train["lc"] == lc) &
                    (train[col_sc] == sc) & (train[col_pc] == pc)
                ][[col_l, col_s]].dropna()
                if len(sub) < 5:
                    continue
                wr_l  = (sub[col_l] > 0).mean() * 100
                wr_s  = (sub[col_s] > 0).mean() * 100
                avg_l = sub[col_l].mean() * 100
                avg_s = sub[col_s].mean() * 100
                best  = "LONG " if avg_l > avg_s else "SHORT"
                flag  = " *" if max(avg_l, avg_s) > 0.05 else "  "
                print(f"  {LC_NAMES[lc]:9} {SC_NAMES[sc]:9} {PC_NAMES[pc]:10} {len(sub):>4}  "
                      f"{wr_l:>5.1f}% {avg_l:>+8.3f}%  {wr_s:>5.1f}% {avg_s:>+8.3f}%  {best}{flag}")
                if max(avg_l, avg_s) > 0.05 and len(sub) >= 5:
                    dir_best = "LONG" if avg_l > avg_s else "SHORT"
                    avg_best = max(avg_l, avg_s)
                    wr_best  = wr_l if dir_best == "LONG" else wr_s
                    prometteurs_930.append((lc, sc, pc, dir_best, len(sub), wr_best, avg_best))

    # ═════════════════════════════════════════════════════════════════════════
    # 5. CONTEXTES PROMETTEURS 09:30 — annee par annee
    # ═════════════════════════════════════════════════════════════════════════
    if prometteurs_930:
        print()
        print("=" * W)
        print("  5. CONTEXTES PROMETTEURS @ 09:30 — annee par annee")
        print("=" * W)
        prometteurs_930.sort(key=lambda x: -x[6])

        for (lc, sc, pc, direction, n_ctx, wr_tr, avg_tr) in prometteurs_930[:5]:
            col = col_l if direction == "LONG" else col_s
            label = f"{LC_NAMES[lc].strip()} x {SC_NAMES[sc].strip()} x {PC_NAMES[pc].strip()} -> {direction}"
            print(f"\n  [{label}]")
            print(f"  TRAIN N={n_ctx}  WR={wr_tr:.1f}%  avg={avg_tr:+.3f}%")
            print(f"  {'An':>4} {'Per':>5} {'N':>4}  {'WR%':>6} {'avg%':>7} {'total%':>8}")
            print("  " + "-" * 40)
            for yr in sorted(data["year"].unique()):
                sub_yr = data[
                    (data["year"] == yr) &
                    (data["lc"] == lc) &
                    (data[col_sc] == sc) & (data[col_pc] == pc)
                ]
                pnls = sub_yr[col].dropna()
                if len(pnls) < 1:
                    continue
                per  = "TEST " if "TEST" in sub_yr["period"].values else "TRAIN"
                wr   = (pnls > 0).mean() * 100
                avg  = pnls.mean() * 100
                tot  = pnls.sum()  * 100
                flag = " *" if avg > 0.05 else ("  " if avg >= 0 else " x")
                print(f"  {yr:>4} {per:>5} {len(pnls):>4}  {wr:>5.1f}% {avg:>+7.3f}% {tot:>+8.2f}%{flag}")

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SYNTHESE
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  SYNTHESE")
    print("=" * W)
    print()
    print("  Meilleure entree (train, aligned_only, avg le + eleve) :")
    best_entry = None
    best_avg   = -999
    for entry_hm in ENTRY_TIMES:
        col_l = f"pnl_long_{entry_hm}"
        col_s = f"pnl_short_{entry_hm}"
        sub = al_train[[col_l, col_s]].dropna()
        if sub.empty:
            continue
        avg_l = sub[col_l].mean() * 100
        avg_s = sub[col_s].mean() * 100
        best_val = max(avg_l, avg_s)
        if best_val > best_avg:
            best_avg   = best_val
            best_entry = ENTRY_NAMES[entry_hm]
    print(f"  -> {best_entry}  (avg_best = {best_avg:+.3f}%)")
    print()
    print("  Rappel : trade actif si avg_best > 0.05% ET confirme sur test set")


if __name__ == "__main__":
    run()
