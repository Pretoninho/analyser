"""
backtest_3conditions.py — Backtest des 3 conditions cross-macro identifiées.

Conditions :
  C1 : NO_RAID x SWEEP_L x SSL_SWEPT -> SHORT   (macros 08:50, 10:50, 14:50)
  C2 : RAID_H  x SWEEP_H x NEUTRAL   -> LONG    (macros 10:50, 12:50)
  C3 : RAID_H  x SWEEP_H x BSL_SWEPT -> LONG    (macros 08:50, 10:50)

Protocole identique : walk-forward 80/20, sl=0.6%, rr=2.5, target_pool.
"""

import sys
import pandas as pd
import pytz
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.stats_state import (
    MACROS, compute_daily_context, compute_pool_ctx, build_weekly_levels,
)
from main import _sim_trade_rr
from data.binance import load_binance_1m

SL_PCT     = 0.006
RR         = 2.5
EXIT_HM    = 960
TEST_RATIO = 0.2
SKIP_DAYS  = frozenset({0})
FEE        = 0.0005
SLIP       = 0.0002
REF_WINDOW = 240

MAC_NAMES = {1: "08:50", 3: "10:50", 5: "12:50", 7: "14:50"}

# (label, lc, sc, pc, direction, macros_cibles)
CONDITIONS = [
    ("NO_RAID x SWEEP_L x SSL_SWEPT -> SHORT", 0, 2, 2, "SHORT", [1, 3, 7]),
    ("RAID_H  x SWEEP_H x NEUTRAL   -> LONG",  1, 1, 0, "LONG",  [3, 5]),
    ("RAID_H  x SWEEP_H x BSL_SWEPT -> LONG",  1, 1, 1, "LONG",  [1, 3]),
]


# ── Simulation ────────────────────────────────────────────────────

def sim_long(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    ep = entry_px * (1 + SLIP)
    if pc == 2 and ref_h is not None and ref_h > ep:
        tp_pct = (ref_h - ep) / ep
        sl_dyn = SL_PCT + max(0.0, (ep - min(float(exit_df.iloc[0]["low"]), pre_low)) / ep)
        if tp_pct > sl_dyn > 0:
            return _sim_trade_rr(exit_df, ep, +1, sl_dyn, tp_pct, FEE, SLIP)
    return _sim_trade_rr(exit_df, ep, +1, SL_PCT, SL_PCT * RR, FEE, SLIP)


def sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low):
    ep = entry_px * (1 - SLIP)
    if pc == 1 and ref_l is not None and ref_l < ep:
        tp_pct = (ep - ref_l) / ep
        sl_dyn = SL_PCT + max(0.0, (max(float(exit_df.iloc[0]["high"]), pre_high) - ep) / ep)
        if tp_pct > sl_dyn > 0:
            return _sim_trade_rr(exit_df, ep, -1, sl_dyn, tp_pct, FEE, SLIP)
    return _sim_trade_rr(exit_df, ep, -1, SL_PCT, SL_PCT * RR, FEE, SLIP)


# ── Collecte d'une macro pour une condition donnée ────────────────

def collect_condition_mac(mac_idx, lc_f, sc_f, pc_f, direction,
                           episodes, dates, contexts, n_train):
    mac_start, _ = MACROS[mac_idx]
    pre_start     = mac_start - 20
    col           = "pnl_long" if direction == "LONG" else "pnl_short"
    rows = []

    for i, (day_df, dt, ctx) in enumerate(zip(episodes, dates, contexts)):
        if dt.weekday() in SKIP_DAYS:
            continue

        lc = ctx["london_ctx"]
        if lc != lc_f:
            continue

        pre_df  = day_df[(day_df["hm_et"] >= pre_start) & (day_df["hm_et"] < mac_start)]
        exit_df = day_df[(day_df["hm_et"] >= mac_start) & (day_df["hm_et"] < EXIT_HM)]

        if len(pre_df) < 3 or len(exit_df) < 5:
            continue

        pre_high = float(pre_df["high"].max())
        pre_low  = float(pre_df["low"].min())
        first    = exit_df.iloc[0]

        sc = (1 if float(first["high"]) > pre_high
              else (2 if float(first["low"]) < pre_low else 0))
        if sc != sc_f:
            continue

        ref_df = day_df[(day_df["hm_et"] >= max(0, pre_start - REF_WINDOW))
                        & (day_df["hm_et"] < pre_start)]
        pwh, pwl = ctx["pwh"], ctx["pwl"]
        if len(ref_df) >= 5:
            ref_h, ref_l = float(ref_df["high"].max()), float(ref_df["low"].min())
        else:
            ref_h = ctx.get("session_high")
            ref_l = ctx.get("session_low")

        pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)
        if pc != pc_f:
            continue

        entry_px = float(first["open"])
        if direction == "LONG":
            pnl = sim_long(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low)
        else:
            pnl = sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low)

        rows.append({
            "date":   dt,
            "year":   dt.year,
            "mac":    mac_idx,
            "period": "TRAIN" if i < n_train else "TEST",
            "pnl":    pnl,
        })

    return pd.DataFrame(rows)


# ── Stats helpers ─────────────────────────────────────────────────

def stats(sub):
    pnls = sub["pnl"].dropna()
    if len(pnls) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(pnls), (pnls > 0).mean() * 100, pnls.mean() * 100, pnls.sum() * 100


def flag(avg):
    return " *" if avg > 0.05 else ("  " if avg >= 0 else " x")


# ── Rapport par condition ─────────────────────────────────────────

def print_condition_report(label, lc_f, sc_f, pc_f, direction, mac_list,
                            episodes, dates, contexts, n_train):
    W = 74
    print()
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    all_rows = []
    for mac_idx in mac_list:
        df = collect_condition_mac(mac_idx, lc_f, sc_f, pc_f, direction,
                                   episodes, dates, contexts, n_train)
        all_rows.append(df)

    if not all_rows or all(d.empty for d in all_rows):
        print("  Aucune donnee.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    train = combined[combined["period"] == "TRAIN"]
    test  = combined[combined["period"] == "TEST"]

    # ── 1. Stats par macro ───────────────────────────────────────
    print(f"\n  1. Par macro (TRAIN / TEST) :")
    print(f"  {'Macro':>6}  {'Per':>5}  {'N':>4}  {'WR%':>6}  {'avg%':>8}  {'total%':>9}")
    print("  " + "-" * 50)

    for mac_idx in mac_list:
        for per, subset in [("TRAIN", train), ("TEST", test)]:
            sub = subset[subset["mac"] == mac_idx]
            n, wr, avg, tot = stats(sub)
            if n == 0:
                print(f"  {MAC_NAMES[mac_idx]:>6}  {per:>5}  {'—':>4}")
                continue
            print(f"  {MAC_NAMES[mac_idx]:>6}  {per:>5}  {n:>4}  "
                  f"{wr:>5.1f}%  {avg:>+8.3f}%  {tot:>+9.2f}%{flag(avg)}")

    # ── 2. Agrégé TRAIN + TEST ───────────────────────────────────
    print(f"\n  2. Agrege toutes macros :")
    print(f"  {'Per':>5}  {'N':>4}  {'WR%':>6}  {'avg%':>8}  {'total%':>9}")
    print("  " + "-" * 46)
    for per, subset in [("TRAIN", train), ("TEST", test)]:
        n, wr, avg, tot = stats(subset)
        print(f"  {per:>5}  {n:>4}  {wr:>5.1f}%  {avg:>+8.3f}%  {tot:>+9.2f}%{flag(avg)}")

    # ── 3. Annee par annee (agrégé) ──────────────────────────────
    print(f"\n  3. Annee par annee (toutes macros) :")
    print(f"  {'An':>4}  {'Per':>5}  {'N':>4}  {'WR%':>6}  {'avg%':>8}  {'total%':>9}  equite")
    print("  " + "-" * 62)
    equity = 0.0
    for yr in sorted(combined["year"].unique()):
        sub  = combined[combined["year"] == yr]
        n, wr, avg, tot = stats(sub)
        equity += tot
        per  = "TEST " if "TEST" in sub["period"].values else "TRAIN"
        print(f"  {yr:>4}  {per:>5}  {n:>4}  {wr:>5.1f}%  {avg:>+8.3f}%  "
              f"{tot:>+9.2f}%{flag(avg)}  {equity:>+8.2f}%")

    # ── 4. Equity train / test séparées ─────────────────────────
    print(f"\n  4. Equity :")
    eq_tr = train.sort_values("date")["pnl"].sum() * 100
    eq_te = test.sort_values("date")["pnl"].sum()  * 100
    n_tr, wr_tr, avg_tr, _ = stats(train)
    n_te, wr_te, avg_te, _ = stats(test)
    print(f"     TRAIN  N={n_tr:>3}  WR={wr_tr:>4.1f}%  avg={avg_tr:>+.3f}%  "
          f"total={eq_tr:>+.2f}%")
    print(f"     TEST   N={n_te:>3}  WR={wr_te:>4.1f}%  avg={avg_te:>+.3f}%  "
          f"total={eq_te:>+.2f}%")

    # ── 5. Distribution des PnL en test ─────────────────────────
    if not test.empty:
        pnls_te = test["pnl"].dropna() * 100
        wins    = pnls_te[pnls_te > 0]
        losses  = pnls_te[pnls_te <= 0]
        print(f"\n  5. Distribution test :")
        print(f"     Wins   N={len(wins):>3}  avg={wins.mean():>+.3f}%  "
              f"max={wins.max():>+.3f}%") if len(wins) else None
        print(f"     Losses N={len(losses):>3}  avg={losses.mean():>+.3f}%  "
              f"min={losses.min():>+.3f}%") if len(losses) else None
        if len(wins) and len(losses):
            print(f"     Ratio  gain/loss = {abs(wins.mean()/losses.mean()):.2f}x")


# ── Main ──────────────────────────────────────────────────────────

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
    for dt, grp in df.groupby("date_et"):
        hm_vals = set(grp["hm_et"].values)
        if (any(60  <= h < 300 for h in hm_vals) and
                any(420 <= h < 600 for h in hm_vals) and
                any(530 <= h < 910 for h in hm_vals) and
                len(grp) >= 60):
            day_df = grp.sort_values("ts_et").reset_index(drop=True)
            pwh, pwl = weekly.get(dt, (None, None))
            ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
            ctx["pwh"], ctx["pwl"] = pwh, pwl
            episodes.append(day_df)
            dates.append(dt)
            contexts.append(ctx)

    n_total = len(episodes)
    n_train = int(n_total * (1 - TEST_RATIO))
    print(f"\n[data] {n_total} jours | train={n_train} | test={n_total - n_train}")

    for (label, lc_f, sc_f, pc_f, direction, mac_list) in CONDITIONS:
        print_condition_report(label, lc_f, sc_f, pc_f, direction, mac_list,
                               episodes, dates, contexts, n_train)

    print()
    print("=" * 74)
    print("  SEUIL : avg_test > 0.05%  ET  total_test > 0  => candidat LIVE")
    print("=" * 74)


if __name__ == "__main__":
    run()
