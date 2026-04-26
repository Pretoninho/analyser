"""
analyse_judas_swing.py — ICT Judas Swing sur session London (00:00-05:00 ET).

Setup :
  1. Asia range = previous day 20:00 ET → minuit (overnight consolidation)
  2. London scan 00:00-05:00 ET : detecte la sequence Judas
     - Bullish Judas : sweep Asia HIGH en premier (faux breakout) PUIS sweep Asia LOW -> LONG
     - Bearish Judas : sweep Asia LOW en premier (faux breakout) PUIS sweep Asia HIGH -> SHORT
  3. Entree : open de la bougie qui complete le Judas (2eme sweep)
  4. SL = 0.6% | RR = 2.5 | Exit max = 08:30 ET

Sections :
  1. Resultats globaux TRAIN / TEST
  2. Par direction (LONG vs SHORT)
  3. Par annee (walk-forward)
  4. Filtre Asia range etroit (ICT : tight consolidation)
  5. Distribution et stats
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytz
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.binance import load_binance_1m
from main import _sim_trade_rr

ET_TZ    = pytz.timezone("America/New_York")
SL_PCT   = 0.006
RR       = 2.5
TP_PCT   = SL_PCT * RR
FEE      = 0.0005
SLIP     = 0.0002
EXIT_HM  = 510    # 08:30 ET — fin session London per ICT
LONDON_END = 300  # 05:00 ET — fin fenetre Judas
SKIP_DAYS  = {0}  # lundi
TEST_RATIO = 0.2


def detect_judas(london_df, asia_high, asia_low):
    """
    Scan bougie par bougie dans la fenetre London (00:00-05:00 ET).
    Retourne (direction, idx, entry_px, sequence) ou None.
      direction : +1 LONG (bullish Judas), -1 SHORT (bearish Judas)
      sequence  : 'H_then_L' ou 'L_then_H'
    """
    high_swept = False
    low_swept  = False

    for idx, (_, row) in enumerate(london_df.iterrows()):
        h = float(row["high"])
        lo = float(row["low"])

        if not high_swept and not low_swept:
            both_at_once = h > asia_high and lo < asia_low
            if both_at_once:
                continue   # les deux en meme temps = pas de sequence claire
            if h > asia_high:
                high_swept = True   # Asia HIGH swept en premier -> anticipe Bullish Judas
            elif lo < asia_low:
                low_swept = True    # Asia LOW swept en premier -> anticipe Bearish Judas

        elif high_swept and not low_swept:
            if lo < asia_low:       # Bullish Judas complet
                return +1, idx, float(row["open"]), "H_then_L"

        elif low_swept and not high_swept:
            if h > asia_high:       # Bearish Judas complet
                return -1, idx, float(row["open"]), "L_then_H"

    return None


def _stats(sub):
    if len(sub) == 0:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "pf": 0.0, "total": 0.0}
    arr  = sub["pnl"].values
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    return {
        "n":     int(len(sub)),
        "wr":    float((arr > 0).mean() * 100),
        "avg":   float(arr.mean() * 100),
        "pf":    float(pf),
        "total": float(arr.sum() * 100),
    }


def main():
    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]   = df["timestamp"].dt.tz_convert(ET_TZ)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date
    df = df.sort_values("ts_et").reset_index(drop=True)

    print(f"[binance] {len(df):,} bougies ({df['date_et'].min()} -> {df['date_et'].max()})")

    by_date = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    days    = sorted(by_date.keys())
    n_train = int(len(days) * (1 - TEST_RATIO))

    rows         = []
    n_no_prev    = 0
    n_no_asia    = 0
    n_no_judas   = 0
    n_no_exit    = 0

    for i, d in enumerate(days):
        if d.weekday() in SKIP_DAYS:
            continue

        prev_d = d - timedelta(days=1)
        if prev_d not in by_date:
            n_no_prev += 1
            continue

        split   = "TRAIN" if i < n_train else "TEST"
        day_df  = by_date[d]
        prev_df = by_date[prev_d]

        # Asia range : previous day >= 20:00 ET (hm_et >= 1200)
        asia_df = prev_df[prev_df["hm_et"] >= 1200]
        if len(asia_df) < 10:
            n_no_asia += 1
            continue

        asia_high      = float(asia_df["high"].max())
        asia_low       = float(asia_df["low"].min())
        asia_range_pct = (asia_high - asia_low) / asia_low * 100

        # London window : 00:00-05:00 ET
        london_df = day_df[(day_df["hm_et"] >= 0) & (day_df["hm_et"] < LONDON_END)].copy()
        if len(london_df) < 10:
            continue

        result = detect_judas(london_df, asia_high, asia_low)
        if result is None:
            n_no_judas += 1
            continue

        direction, entry_idx, entry_px, sequence = result
        entry_hm = int(london_df.iloc[entry_idx]["hm_et"])

        exit_df = day_df[(day_df["hm_et"] >= entry_hm) & (day_df["hm_et"] < EXIT_HM)]
        if len(exit_df) < 3:
            n_no_exit += 1
            continue

        slipped = entry_px * (1 + direction * SLIP)
        pnl, reason, _, _, _, nc = _sim_trade_rr(
            exit_df, slipped, direction, SL_PCT, TP_PCT,
            fee=FEE, slip=SLIP, verbose=True
        )

        rows.append({
            "date":           d,
            "year":           d.year,
            "split":          split,
            "direction":      "LONG" if direction == 1 else "SHORT",
            "sequence":       sequence,
            "entry_hm":       entry_hm,
            "asia_range_pct": asia_range_pct,
            "pnl":            float(pnl),
            "reason":         reason,
            "n_candles":      int(nc),
        })

    df_r    = pd.DataFrame(rows)
    if df_r.empty:
        print("Aucun Judas Swing detecte.")
        return

    train_r = df_r[df_r["split"] == "TRAIN"]
    test_r  = df_r[df_r["split"] == "TEST"]

    # ── Section 1 : Resultats globaux ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  JUDAS SWING — RESULTATS GLOBAUX (SL=0.6% | RR=2.5 | exit 08:30 ET)")
    print("=" * 70)
    st = _stats(train_r)
    print(f"  TRAIN : N={st['n']:>4}  WR={st['wr']:>5.1f}%  avg={st['avg']:>+7.3f}%  PF={st['pf']:.3f}  Total={st['total']:>+7.2f}%")
    st = _stats(test_r)
    print(f"  TEST  : N={st['n']:>4}  WR={st['wr']:>5.1f}%  avg={st['avg']:>+7.3f}%  PF={st['pf']:.3f}  Total={st['total']:>+7.2f}%")

    # ── Section 2 : Par direction ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PAR DIRECTION")
    print("=" * 70)
    for direc in ["LONG", "SHORT"]:
        sub_tr = train_r[train_r["direction"] == direc]
        sub_te = test_r[test_r["direction"] == direc]
        st_tr  = _stats(sub_tr)
        st_te  = _stats(sub_te)
        print(f"\n  {direc}")
        print(f"    TRAIN : N={st_tr['n']:>4}  WR={st_tr['wr']:>5.1f}%  avg={st_tr['avg']:>+7.3f}%  PF={st_tr['pf']:.3f}  Total={st_tr['total']:>+7.2f}%")
        print(f"    TEST  : N={st_te['n']:>4}  WR={st_te['wr']:>5.1f}%  avg={st_te['avg']:>+7.3f}%  PF={st_te['pf']:.3f}  Total={st_te['total']:>+7.2f}%")

    # ── Section 3 : Par annee ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PAR ANNEE")
    print("=" * 70)
    print(f"  {'An':<6} {'Per':<6} {'N':>5} {'WR%':>7} {'avg%':>8} {'PF':>6} {'Total%':>8}")
    print("  " + "-" * 50)
    for year in sorted(df_r["year"].unique()):
        sub   = df_r[df_r["year"] == year]
        split = sub.iloc[0]["split"]
        st    = _stats(sub)
        mark  = "*" if st["avg"] > 0 else "x"
        print(f"  {year:<6} {split:<6} {st['n']:>5} {st['wr']:>6.1f}% {st['avg']:>+8.3f}% {st['pf']:>6.3f} {st['total']:>+8.2f}% {mark}")

    # ── Section 4 : Filtre Asia range etroit ──────────────────────────────────
    tight_thr = float(train_r["asia_range_pct"].quantile(0.33))
    print(f"\n" + "=" * 70)
    print(f"  FILTRE : ASIA RANGE ETROIT (bottom 33%, range < {tight_thr:.3f}%)")
    print("=" * 70)
    t_tr = train_r[train_r["asia_range_pct"] <= tight_thr]
    t_te = test_r[test_r["asia_range_pct"]   <= tight_thr]
    st_tr = _stats(t_tr)
    st_te = _stats(t_te)
    print(f"  TRAIN : N={st_tr['n']:>4}  WR={st_tr['wr']:>5.1f}%  avg={st_tr['avg']:>+7.3f}%  PF={st_tr['pf']:.3f}  Total={st_tr['total']:>+7.2f}%")
    print(f"  TEST  : N={st_te['n']:>4}  WR={st_te['wr']:>5.1f}%  avg={st_te['avg']:>+7.3f}%  PF={st_te['pf']:.3f}  Total={st_te['total']:>+7.2f}%")

    print(f"\n  Par annee (filtre etroit) :")
    print(f"  {'An':<6} {'Per':<6} {'N':>5} {'WR%':>7} {'avg%':>8} {'PF':>6} {'Total%':>8}")
    print("  " + "-" * 50)
    sub_tight = df_r[df_r["asia_range_pct"] <= tight_thr]
    for year in sorted(sub_tight["year"].unique()):
        sub   = sub_tight[sub_tight["year"] == year]
        split = sub.iloc[0]["split"]
        st    = _stats(sub)
        mark  = "*" if st["avg"] > 0 else "x"
        print(f"  {year:<6} {split:<6} {st['n']:>5} {st['wr']:>6.1f}% {st['avg']:>+8.3f}% {st['pf']:>6.3f} {st['total']:>+8.2f}% {mark}")

    # ── Section 5 : Distribution ───────────────────────────────────────────────
    n_days_eligible = sum(1 for d in days if d.weekday() not in SKIP_DAYS)
    print(f"\n" + "=" * 70)
    print(f"  DISTRIBUTION")
    print("=" * 70)
    print(f"  Jours eligibles  : {n_days_eligible}")
    print(f"  Judas detectes   : {len(df_r)} ({len(df_r)/n_days_eligible*100:.1f}%)")
    print(f"  LONG (Bullish)   : {(df_r['direction']=='LONG').sum()}")
    print(f"  SHORT (Bearish)  : {(df_r['direction']=='SHORT').sum()}")
    print(f"  Sorties TP       : {df_r['reason'].eq('TP').sum()} ({df_r['reason'].eq('TP').sum()/len(df_r)*100:.0f}%)")
    print(f"  Sorties SL       : {df_r['reason'].eq('SL').sum()} ({df_r['reason'].eq('SL').sum()/len(df_r)*100:.0f}%)")
    print(f"  Sorties EOD      : {df_r['reason'].eq('EOD').sum()} ({df_r['reason'].eq('EOD').sum()/len(df_r)*100:.0f}%)")
    print(f"  Asia range median: {df_r['asia_range_pct'].median():.3f}%")
    print(f"  Entry hm median  : {df_r['entry_hm'].median():.0f} min ET ({int(df_r['entry_hm'].median())//60:02d}:{int(df_r['entry_hm'].median())%60:02d})")
    print(f"\n  Echecs detection : no_prev={n_no_prev} | no_asia={n_no_asia} | no_judas={n_no_judas} | no_exit={n_no_exit}")


if __name__ == "__main__":
    main()
