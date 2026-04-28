"""
analyse_macros_silencieux.py — Macros sous silence Pi*.
08:50 (idx=1) | 10:50 (idx=3) | 12:50 (idx=5) | 13:50 (idx=6) | 14:50 (idx=7)

Pour chaque macro :
  1. Breakdown global (lc x sc x pc), train
  2. Contextes prometteurs (N>=5, avg>0.05%) : condition vs -1condition (complement)
  3. Annee par annee sur les contextes prometteurs
  4. Test set summary

Protocole : walk-forward 80/20, sl=0.006, rr=2.5, target_pool, aligned_only.
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

MACROS_CIBLES = [1, 3, 5, 6, 7]
MAC_NAMES     = {1: "08:50", 3: "10:50", 5: "12:50", 6: "13:50", 7: "14:50"}

SL_PCT     = 0.006   # production actuelle
RR         = 2.5     # production actuelle
EXIT_HM    = 960
TEST_RATIO = 0.2
SKIP_DAYS  = frozenset({0})
FEE        = 0.0005
SLIP       = 0.0002
REF_WINDOW = 240
MIN_N      = 5       # seuil validation
AVG_THRESH = 0.05    # %

LC_N = {0: "NO_RAID", 1: "RAID_H ", 2: "RAID_L "}
PC_N = {0: "NEUTRAL  ", 1: "BSL_SWEPT", 2: "SSL_SWEPT"}
SC_N = {0: "NO_SWEEP", 1: "SWEEP_H ", 2: "SWEEP_L "}


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


# ── Collecte données ──────────────────────────────────────────────

def collect_macro(mac_idx, episodes, dates, contexts, n_train):
    mac_start, _ = MACROS[mac_idx]
    pre_start     = mac_start - 20
    rows = []

    for i, (day_df, dt, ctx) in enumerate(zip(episodes, dates, contexts)):
        if dt.weekday() in SKIP_DAYS:
            continue

        mc, dc, lc = ctx["month_ctx"], ctx["day_ctx"], ctx["london_ctx"]
        pwh, pwl   = ctx["pwh"], ctx["pwl"]

        pre_df  = day_df[(day_df["hm_et"] >= pre_start)  & (day_df["hm_et"] < mac_start)]
        exit_df = day_df[(day_df["hm_et"] >= mac_start)  & (day_df["hm_et"] < EXIT_HM)]

        if len(pre_df) < 3 or len(exit_df) < 5:
            continue

        pre_high = float(pre_df["high"].max())
        pre_low  = float(pre_df["low"].min())
        first    = exit_df.iloc[0]

        sc = (1 if float(first["high"]) > pre_high
              else (2 if float(first["low"]) < pre_low else 0))

        ref_df = day_df[(day_df["hm_et"] >= max(0, pre_start - REF_WINDOW))
                        & (day_df["hm_et"] < pre_start)]
        if len(ref_df) >= 5:
            ref_h, ref_l = float(ref_df["high"].max()), float(ref_df["low"].min())
        else:
            ref_h = ctx.get("session_high")
            ref_l = ctx.get("session_low")

        pc       = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)
        entry_px = float(first["open"])

        rows.append({
            "date": dt, "year": dt.year,
            "period": "TRAIN" if i < n_train else "TEST",
            "lc": lc, "sc": sc, "pc": pc,
            "pnl_long":  sim_long(exit_df,  entry_px, pc, ref_h, ref_l, pre_high, pre_low),
            "pnl_short": sim_short(exit_df, entry_px, pc, ref_h, ref_l, pre_high, pre_low),
        })

    return pd.DataFrame(rows)


# ── Affichage ─────────────────────────────────────────────────────

def _stats(sub, col):
    pnls = sub[col].dropna()
    if len(pnls) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(pnls), (pnls > 0).mean() * 100, pnls.mean() * 100, pnls.sum() * 100


def print_report(mac_idx, data):
    name  = MAC_NAMES[mac_idx]
    train = data[data["period"] == "TRAIN"]
    test  = data[data["period"] == "TEST"]

    W = 74
    print()
    print("=" * W)
    print(f"  MACRO {name}  (mac_idx={mac_idx})  |  "
          f"train N={len(train)}  test N={len(test)}  "
          f"sl={SL_PCT*100:.1f}%  rr={RR}")
    print("=" * W)

    # ── 1. Breakdown global (train, N>=3) ────────────────────────
    print(f"\n  1. Breakdown (train, N>=3) — LONG / SHORT :")
    print(f"  {'lc':9} {'sc':9} {'pc':10} {'N':>4}  "
          f"{'WR_L%':>6} {'avg_L%':>7}  {'WR_S%':>6} {'avg_S%':>7}  Best")
    print("  " + "-" * 70)

    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < 3:
                    continue
                n, wr_l, avg_l, _ = _stats(sub, "pnl_long")
                _, wr_s, avg_s, _ = _stats(sub, "pnl_short")
                best = "LONG " if avg_l > avg_s else "SHORT"
                flag = " *" if max(avg_l, avg_s) > AVG_THRESH and n >= MIN_N else "  "
                print(f"  {LC_N[lc]:9} {SC_N[sc]:9} {PC_N[pc]:10} {n:>4}  "
                      f"{wr_l:>5.1f}% {avg_l:>+7.3f}%  {wr_s:>5.1f}% {avg_s:>+7.3f}%  {best}{flag}")

    # ── 2. Condition vs -1Condition ──────────────────────────────
    prometteurs = []
    for lc in range(3):
        for sc in range(3):
            for pc in range(3):
                sub = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
                if len(sub) < MIN_N:
                    continue
                for col, direction in [("pnl_long", "LONG"), ("pnl_short", "SHORT")]:
                    n, wr, avg, tot = _stats(sub, col)
                    if avg > AVG_THRESH:
                        prometteurs.append((lc, sc, pc, direction, n, wr, avg, tot))

    prometteurs.sort(key=lambda x: -x[6])

    print(f"\n  2. Condition vs -1Condition (N>={MIN_N}, avg>{AVG_THRESH}%) :")

    if not prometteurs:
        print("  Aucun contexte valide.")
    else:
        for (lc, sc, pc, direction, n_tr, wr_tr, avg_tr, tot_tr) in prometteurs:
            col   = "pnl_long" if direction == "LONG" else "pnl_short"
            label = f"{LC_N[lc].strip()} x {SC_N[sc].strip()} x {PC_N[pc].strip()} -> {direction}"

            # Condition (train + test)
            cond_tr  = train[(train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc)]
            cond_te  = test[ (test["lc"]  == lc) & (test["sc"]  == sc) & (test["pc"]  == pc)]
            # -1Condition (complement)
            ncond_tr = train[~((train["lc"] == lc) & (train["sc"] == sc) & (train["pc"] == pc))]
            ncond_te = test[ ~((test["lc"]  == lc) & (test["sc"]  == sc) & (test["pc"]  == pc))]

            _, wr_te,   avg_te,   tot_te   = _stats(cond_te,  col)
            n_nc_tr, wr_nc_tr, avg_nc_tr, _ = _stats(ncond_tr, col)
            n_nc_te, wr_nc_te, avg_nc_te, _ = _stats(ncond_te, col)

            delta_tr = avg_tr  - avg_nc_tr
            delta_te = avg_te  - avg_nc_te
            ok_te    = " *" if avg_te > AVG_THRESH else (" ?" if avg_te >= 0 else " x")

            print(f"\n  [{label}]")
            print(f"  {'':20} {'N':>5}  {'WR%':>6} {'avg%':>8} {'total%':>9}  {'delta vs -1cond':>15}")
            print("  " + "-" * 65)
            print(f"  {'COND (train)':20} {n_tr:>5}  {wr_tr:>5.1f}% {avg_tr:>+8.3f}% {tot_tr:>+9.2f}%  "
                  f"(baseline)")
            print(f"  {'-1COND (train)':20} {n_nc_tr:>5}  {wr_nc_tr:>5.1f}% {avg_nc_tr:>+8.3f}%  "
                  f"{'':>9}  delta train: {delta_tr:>+.3f}%")
            print(f"  {'COND (test)':20} {len(cond_te):>5}  {wr_te:>5.1f}% {avg_te:>+8.3f}% {tot_te:>+9.2f}%  "
                  f"delta test : {delta_te:>+.3f}%{ok_te}")
            print(f"  {'-1COND (test)':20} {n_nc_te:>5}  {wr_nc_te:>5.1f}% {avg_nc_te:>+8.3f}%")

    # ── 3. Annee par annee (contexts prometteurs) ─────────────────
    if prometteurs:
        print(f"\n  3. Annee par annee — contextes prometteurs :")
        for (lc, sc, pc, direction, *_) in prometteurs:
            col   = "pnl_long" if direction == "LONG" else "pnl_short"
            label = f"{LC_N[lc].strip()} x {SC_N[sc].strip()} x {PC_N[pc].strip()} -> {direction}"
            cond  = data[(data["lc"] == lc) & (data["sc"] == sc) & (data["pc"] == pc)]
            print(f"\n  [{label}]")
            print(f"  {'An':>4} {'Per':>5} {'N':>4}  {'WR%':>6} {'avg%':>8} {'total%':>9}")
            print("  " + "-" * 42)
            for yr in sorted(cond["year"].unique()):
                sub = cond[cond["year"] == yr]
                n, wr, avg, tot = _stats(sub, col)
                if n == 0:
                    continue
                per  = "TEST " if "TEST" in sub["period"].values else "TRAIN"
                flag = " *" if avg > AVG_THRESH else ("  " if avg >= 0 else " x")
                print(f"  {yr:>4} {per:>5} {n:>4}  {wr:>5.1f}% {avg:>+8.3f}% {tot:>+9.2f}%{flag}")

    # ── 4. Test set summary ───────────────────────────────────────
    al_te = test[test["sc"] != 0]
    print(f"\n  4. Test set global — aligned_only (sc!=0)  N={len(al_te)} :")
    if not al_te.empty:
        _, wr_l, avg_l, tot_l = _stats(al_te, "pnl_long")
        _, wr_s, avg_s, tot_s = _stats(al_te, "pnl_short")
        print(f"     LONG  WR={wr_l:.1f}%  avg={avg_l:>+.3f}%  total={tot_l:>+.2f}%")
        print(f"     SHORT WR={wr_s:.1f}%  avg={avg_s:>+.3f}%  total={tot_s:>+.2f}%")
    else:
        print("     Aucun trade aligned sur le test set.")


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

    for mac_idx in MACROS_CIBLES:
        data = collect_macro(mac_idx, episodes, dates, contexts, n_train)
        print_report(mac_idx, data)

    print()
    print("=" * 74)
    print("  RESUME — Verdict par macro")
    print("=" * 74)
    verdicts = {
        1: "08:50 — pré-NYSE, signal historiquement négatif globalement",
        3: "10:50 — post-Silver Bullet, peu d'edge OOS détecté",
        5: "12:50 — drag PM systématique constaté",
        6: "13:50 — drag systématique constaté",
        7: "14:50 — Power Hour, phénomène equity, sans effet BTC 24/7",
    }
    for k, v in verdicts.items():
        print(f"  mac_idx={k}  {v}")
    print()
    print("  Seuil de passage en LIVE : avg_cond > 0.05% ET avg_test > 0.05%")
    print("  ET delta_test (cond - complement) > 0 (alpha reel confirme)")


if __name__ == "__main__":
    run()
