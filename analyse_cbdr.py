"""
analyse_cbdr.py — Test CBDR (Central Bank Dealers Range) comme filtre sur Pi*.

ICT April 2017 Lesson 5 : CBDR = range 14:00-20:00 ET de la veille.
Forex : CBDR < 40 pips -> setup protraction favorable.
Sur BTC : seuils percentiles determines sur TRAIN, valides OOS.

Config active replicquee exactement :
  sl=0.6%, rr=2.5, target_pool=True, aligned_only=True,
  skip_macros={1,3,5,6,7}, skip_days={0},
  macro_rules={(2,1,1):{1}, (2,0,1):empty}

Sections :
  1. Distribution CBDR
  2. Baseline Pi* (sans filtre) — sanity check
  3. Performance par tercile CBDR (seuils TRAIN)
  4. Sweep de seuils sur TRAIN seul
  5. Validation OOS du seuil retenu
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytz
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.binance import load_binance_1m
from engine.stats_state import (
    MACROS, encode, compute_daily_context,
    compute_pool_ctx, build_weekly_levels,
)
from engine.q_agent import QAgent
from config import DATA_DIR
from main import _sim_trade_rr

ET_TZ = pytz.timezone("America/New_York")

SL_PCT      = 0.006
RR          = 2.5
FEE         = 0.0005
SLIP        = 0.0002
EXIT_HM     = 960
TEST_RATIO  = 0.2
SKIP_MACROS = frozenset({1, 3, 5, 6, 7})
SKIP_DAYS   = frozenset({0})
MACRO_RULES = {(2, 1, 1): frozenset({1}), (2, 0, 1): frozenset()}
MAX_TRADES  = 2
REF_WINDOW  = 240

CBDR_START = 840   # 14:00 ET
CBDR_END   = 1200  # 20:00 ET


def compute_cbdr_map(df: pd.DataFrame) -> dict:
    """Retourne {date -> cbdr_pct} pour chaque jour avec >= 10 bougies 14:00-20:00 ET."""
    cbdr = {}
    sub  = df[(df["hm_et"] >= CBDR_START) & (df["hm_et"] < CBDR_END)]
    for d, grp in sub.groupby("date_et"):
        if len(grp) >= 10:
            h = float(grp["high"].max())
            l = float(grp["low"].min())
            cbdr[d] = (h - l) / l * 100
    return cbdr


def _stats(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "pf": 0.0,
                "total": 0.0, "sharpe": 0.0, "maxdd": 0.0}
    arr    = np.array([t["pnl"] for t in trades])
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    pf     = (abs(wins.sum() / losses.sum())
              if len(losses) > 0 and losses.sum() != 0 else float("inf"))
    eq    = np.cumprod(1.0 + arr)
    eq    = np.insert(eq, 0, 1.0)
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    maxdd = float(dd.min())
    daily = np.diff(eq) / eq[:-1]
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    return {
        "n":      int(len(arr)),
        "wr":     float((arr > 0).mean() * 100),
        "avg":    float(arr.mean() * 100),
        "pf":     float(pf),
        "total":  float(arr.sum() * 100),
        "sharpe": float(sharpe),
        "maxdd":  float(maxdd * 100),
    }


def _fmt(st, label=""):
    pf  = f"{st['pf']:.3f}" if st["pf"] != float("inf") else "inf"
    tag = f"  {label:<22}" if label else "  " + " " * 22
    return (f"{tag} N={st['n']:>4}  WR={st['wr']:>5.1f}%  avg={st['avg']:>+7.3f}%  "
            f"PF={pf:<7} Total={st['total']:>+8.2f}%  "
            f"Sharpe={st['sharpe']:>+6.3f}  MaxDD={st['maxdd']:>+6.2f}%")


def collect_all_trades(episodes, dates, n_train, cbdr_map, weekly, agent):
    """
    Replique run_backtest_stats sur train + test avec tag CBDR.
    Retourne une liste de dicts : pnl, split, cbdr_prev, mac_idx, reason, date.
    """
    rows = []

    for i, (day_df, date) in enumerate(zip(episodes, dates)):
        split = "TRAIN" if i < n_train else "TEST"

        if date.weekday() in SKIP_DAYS:
            continue

        prev_d    = date - timedelta(days=1)
        cbdr_prev = cbdr_map.get(prev_d)

        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc  = ctx["month_ctx"]
        dc  = ctx["day_ctx"]
        lc  = ctx["london_ctx"]

        ep_trades = 0

        for mac_idx, (mac_start, _) in MACROS.items():
            if ep_trades >= MAX_TRADES:
                break
            if mac_idx in SKIP_MACROS:
                continue

            pre_mask  = ((day_df["hm_et"] >= mac_start - 20) &
                         (day_df["hm_et"] < mac_start))
            exit_mask = ((day_df["hm_et"] >= mac_start) &
                         (day_df["hm_et"] < EXIT_HM))

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

            if sc == 0:   # aligned_only
                continue

            ref_start = mac_start - 20 - REF_WINDOW
            ref_mask  = ((day_df["hm_et"] >= max(0, ref_start)) &
                         (day_df["hm_et"] < mac_start - 20))
            ref_df    = day_df[ref_mask]
            if len(ref_df) >= 5:
                ref_h = float(ref_df["high"].max())
                ref_l = float(ref_df["low"].min())
            else:
                ref_h = ctx.get("session_high")
                ref_l = ctx.get("session_low")

            pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

            if MACRO_RULES:
                allowed_sc = MACRO_RULES.get((mac_idx, lc, pc))
                if allowed_sc is not None and sc not in allowed_sc:
                    continue

            state  = encode(mc, dc, lc, mac_idx, sc, pc)
            action = agent.act(state, training=False)

            if action == 0:
                continue
            if agent.q_table[state, action] <= 0.0:
                continue

            entry_px  = float(first["open"])
            direction = +1 if action == 1 else -1
            slipped   = entry_px * (1 + direction * SLIP)

            # target_pool=True : meme logique que run_backtest_stats
            if pc != 0:
                # TP dynamique vers pool oppose
                if direction == +1 and pc == 2 and ref_h is not None and ref_h > entry_px:
                    sweep_low = min(float(first["low"]), pre_low)
                    tp_pct_v  = (ref_h - entry_px) / entry_px
                    sl_pct_v  = SL_PCT + max(0.0, (entry_px - sweep_low) / entry_px)
                    if tp_pct_v <= sl_pct_v:
                        continue
                elif direction == -1 and pc == 1 and ref_l is not None and ref_l < entry_px:
                    sweep_high = max(float(first["high"]), pre_high)
                    tp_pct_v   = (entry_px - ref_l) / entry_px
                    sl_pct_v   = SL_PCT + max(0.0, (sweep_high - entry_px) / entry_px)
                    if tp_pct_v <= sl_pct_v:
                        continue
                else:
                    continue   # mismatch direction/pool -> skip
            else:
                # pc == 0 : fallback SL/TP fixe
                tp_pct_v = SL_PCT * RR
                sl_pct_v = SL_PCT

            pnl, reason, _, _, _, _ = _sim_trade_rr(
                exit_df, slipped, direction,
                sl_pct_v, tp_pct_v, FEE, SLIP, verbose=True
            )

            rows.append({
                "date":       date,
                "split":      split,
                "mac_idx":    mac_idx,
                "pnl":        float(pnl),
                "reason":     reason,
                "cbdr_prev":  cbdr_prev,
            })
            ep_trades += 1

    return rows


def main():
    # ── Chargement ─────────────────────────────────────────────────────────────
    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]     = df["timestamp"].dt.tz_convert(ET_TZ)
    df["hm_et"]     = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"]   = df["ts_et"].dt.date
    df = df.sort_values("ts_et").reset_index(drop=True)
    df = df[df["hm_et"] < CBDR_END]  # inclut 14:00-20:00 ET pour CBDR

    print(f"[binance] {len(df):,} bougies ({df['date_et'].min()} -> {df['date_et'].max()})")

    cbdr_map = compute_cbdr_map(df)
    weekly   = build_weekly_levels(df)

    model_path = str(DATA_DIR / "stats_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print("[cbdr] stats_agent.pkl introuvable. Lancez --build-qtable d'abord.")
        return

    # Episodes — meme filtre que run_backtest_stats
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
    n_test  = n_total - n_train
    print(f"[cbdr] {n_total} jours | train={n_train} | test={n_test}")

    all_trades = collect_all_trades(episodes, dates, n_train, cbdr_map, weekly, agent)
    if not all_trades:
        print("[cbdr] Aucun trade collecte.")
        return

    train_t = [t for t in all_trades if t["split"] == "TRAIN"]
    test_t  = [t for t in all_trades if t["split"] == "TEST"]

    # Seuils percentiles sur les dates TRAIN (toutes, pas seulement les jours trades)
    train_dates  = set(dates[:n_train])
    cbdr_train_all = sorted([cbdr_map[d] for d in cbdr_map if d in train_dates])

    # ── Section 1 : Distribution CBDR ─────────────────────────────────────────
    all_cbdr = np.array(list(cbdr_map.values()))
    print("\n" + "=" * 75)
    print("  SECTION 1 - DISTRIBUTION CBDR (14:00-20:00 ET)")
    print("=" * 75)
    print(f"  Jours avec CBDR calcule : {len(cbdr_map)}")
    print(f"  P10={np.percentile(all_cbdr,10):.3f}%  "
          f"P25={np.percentile(all_cbdr,25):.3f}%  "
          f"P33={np.percentile(all_cbdr,33):.3f}%  "
          f"P50={np.percentile(all_cbdr,50):.3f}%  "
          f"P67={np.percentile(all_cbdr,67):.3f}%  "
          f"P75={np.percentile(all_cbdr,75):.3f}%  "
          f"P90={np.percentile(all_cbdr,90):.3f}%")
    print(f"  Moyenne={all_cbdr.mean():.3f}%  Max={all_cbdr.max():.3f}%")

    n_with    = sum(1 for t in all_trades if t["cbdr_prev"] is not None)
    n_without = sum(1 for t in all_trades if t["cbdr_prev"] is None)
    print(f"\n  Trades avec CBDR veille    : {n_with}/{len(all_trades)}")
    if n_without:
        print(f"  Trades sans CBDR (week-end/manquant) : {n_without}")

    # ── Section 2 : Baseline ───────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SECTION 2 - BASELINE Pi* (sans filtre CBDR)")
    print("=" * 75)
    print(_fmt(_stats(train_t), "TRAIN"))
    print(_fmt(_stats(test_t),  "TEST"))

    # ── Section 3 : Terciles CBDR (seuils issus du TRAIN) ─────────────────────
    p33 = float(np.percentile(cbdr_train_all, 33))
    p67 = float(np.percentile(cbdr_train_all, 67))

    print("\n" + "=" * 75)
    print(f"  SECTION 3 - PAR TERCILE CBDR  "
          f"(seuils TRAIN : P33={p33:.3f}%  P67={p67:.3f}%)")
    print("=" * 75)

    for split_name, tlist in [("TRAIN", train_t), ("TEST", test_t)]:
        narrow = [t for t in tlist if t["cbdr_prev"] is not None
                  and t["cbdr_prev"] <= p33]
        medium = [t for t in tlist if t["cbdr_prev"] is not None
                  and p33 < t["cbdr_prev"] <= p67]
        wide   = [t for t in tlist if t["cbdr_prev"] is not None
                  and t["cbdr_prev"] > p67]
        na     = [t for t in tlist if t["cbdr_prev"] is None]
        print(f"\n  {split_name}")
        print(_fmt(_stats(narrow), f"  narrow (<=  {p33:.2f}%)"))
        print(_fmt(_stats(medium), f"  medium ({p33:.2f}-{p67:.2f}%)"))
        print(_fmt(_stats(wide),   f"  wide   (>   {p67:.2f}%)"))
        if na:
            print(_fmt(_stats(na), f"  N/A (pas de CBDR)"))

    # ── Section 4 : Sweep seuils CBDR sur TRAIN ───────────────────────────────
    pct_thresholds = [float(np.percentile(cbdr_train_all, p))
                      for p in [10, 20, 25, 33, 40, 50, 60, 67, 75, 80, 90]]
    pct_thresholds = sorted(set(round(v, 3) for v in pct_thresholds))

    print("\n" + "=" * 75)
    print("  SECTION 4 - SWEEP SEUILS CBDR (TRAIN seul, filtre : CBDR_veille <= seuil)")
    print("=" * 75)
    print(f"  {'Seuil%':>8}  {'N':>5}  {'WR%':>6}  {'avg%':>8}  "
          f"{'PF':>6}  {'Total%':>8}  {'Sharpe':>7}  {'MaxDD%':>7}")
    print("  " + "-" * 65)

    best_sharpe = -999
    best_thr    = None

    for thr in pct_thresholds:
        sub = [t for t in train_t
               if t["cbdr_prev"] is not None and t["cbdr_prev"] <= thr]
        st  = _stats(sub)
        mark = ""
        if st["n"] >= 10 and st["sharpe"] > best_sharpe:
            best_sharpe = st["sharpe"]
            best_thr    = thr
            mark = " <--"
        pf = f"{st['pf']:.3f}" if st["pf"] != float("inf") else "inf"
        print(f"  {thr:>8.3f}  {st['n']:>5}  {st['wr']:>5.1f}%  "
              f"{st['avg']:>+8.3f}%  {pf:>6}  {st['total']:>+8.2f}%  "
              f"{st['sharpe']:>+7.3f}  {st['maxdd']:>+7.2f}%{mark}")

    # ── Section 5 : Validation OOS ────────────────────────────────────────────
    print("\n" + "=" * 75)
    if best_thr is not None:
        print(f"  SECTION 5 - VALIDATION OOS  (seuil retenu TRAIN : CBDR <= {best_thr:.3f}%)")
    else:
        print("  SECTION 5 - VALIDATION OOS  (aucun seuil suffisant sur TRAIN)")
    print("=" * 75)

    st_base = _stats(test_t)
    print("\n  Baseline TEST (reference) :")
    print(_fmt(st_base, "  ALL TEST"))

    if best_thr is not None:
        narrow_oos = [t for t in test_t
                      if t["cbdr_prev"] is not None and t["cbdr_prev"] <= best_thr]
        wide_oos   = [t for t in test_t
                      if t["cbdr_prev"] is None or t["cbdr_prev"] > best_thr]
        print(f"\n  Filtre CBDR <= {best_thr:.3f}% :")
        print(_fmt(_stats(narrow_oos), "  NARROW OOS"))
        print(_fmt(_stats(wide_oos),   "  WIDE (exclus)"))

        # Distribution sorties
        for label, tlist in [("NARROW OOS", narrow_oos), ("ALL TEST", test_t)]:
            if not tlist:
                continue
            n   = len(tlist)
            tp  = sum(1 for t in tlist if t["reason"] == "TP")
            sl  = sum(1 for t in tlist if t["reason"] == "SL")
            eod = sum(1 for t in tlist if t["reason"] == "EOD")
            print(f"\n  Sorties {label:<12}: "
                  f"TP={tp} ({tp/n*100:.0f}%)  "
                  f"SL={sl} ({sl/n*100:.0f}%)  "
                  f"EOD={eod} ({eod/n*100:.0f}%)")

        # Verdict
        st_narrow = _stats(narrow_oos)
        print("\n  Verdict :")
        if st_narrow["n"] == 0:
            print("  -> Aucun trade OOS avec ce seuil.")
        elif st_narrow["avg"] > st_base["avg"] and st_narrow["sharpe"] > st_base["sharpe"]:
            print(f"  -> CBDR filtre AMELIORE : "
                  f"avg {st_base['avg']:+.3f}% -> {st_narrow['avg']:+.3f}%  "
                  f"Sharpe {st_base['sharpe']:+.3f} -> {st_narrow['sharpe']:+.3f}")
        elif st_narrow["avg"] > st_base["avg"]:
            print(f"  -> avg ameliore ({st_base['avg']:+.3f}% -> {st_narrow['avg']:+.3f}%) "
                  f"mais Sharpe degrade ({st_base['sharpe']:+.3f} -> {st_narrow['sharpe']:+.3f})")
        else:
            print(f"  -> CBDR filtre NEUTRE/NEGATIF OOS : "
                  f"avg {st_base['avg']:+.3f}% -> {st_narrow['avg']:+.3f}%  "
                  f"N {len(test_t)} -> {st_narrow['n']}")
    else:
        print("  -> Pas de seuil retenu (aucun bucket N>=10 sur TRAIN).")


if __name__ == "__main__":
    main()
