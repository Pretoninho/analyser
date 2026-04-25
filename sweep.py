"""
sweep.py — Recherche automatique des meilleurs parametres SL/TP.
Lance plusieurs combinaisons de sl_pct x rr et sauvegarde les resultats.

Usage : python sweep.py
"""

import csv
import itertools
from pathlib import Path
from main import run_build_qtable, run_backtest_stats

SKIP_MACROS  = frozenset({1, 3, 5, 6})   # macros silencieuses
SKIP_DAYS    = frozenset({0})             # 0=Lundi retire
EXIT_HM      = 960                         # sortie 16:00 ET
TEST_RATIO   = 0.2
MIN_SAMPLES  = 5

# Regles directionnelles : (mac_idx, lc, pc) -> sc autorise (frozenset vide = toujours skip)
# 09:50 (mac=2) : RAID_H x BSL_swept -> SW_H uniquement (sc=1 -> SHORT)
# 09:50 (mac=2) : NO_RAID x BSL_swept -> signal non fiable, bloque
MACRO_RULES  = {
    (2, 1, 1): frozenset({1}),
    (2, 0, 1): frozenset(),
}

SL_VALUES = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
RR_VALUES = [1.5, 2.0, 2.5, 3.0]

OUT = Path("db/sweep_results.csv")


def _run_one(sl_pct, rr):
    import io, sys

    # Rebuild Q-table
    run_build_qtable(
        test_ratio   = TEST_RATIO,
        min_samples  = MIN_SAMPLES,
        exit_hm      = EXIT_HM,
        sl_pct       = sl_pct,
        rr           = rr,
        target_pool  = True,
        aligned_only = True,
        skip_macros  = SKIP_MACROS,
        skip_days    = SKIP_DAYS,
        macro_rules  = MACRO_RULES,
    )

    # Capture stdout du backtest
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()

    run_backtest_stats(
        test_ratio   = TEST_RATIO,
        q_threshold  = 0.0,
        exit_hm      = EXIT_HM,
        sl_pct       = sl_pct,
        rr           = rr,
        target_pool  = True,
        aligned_only = True,
        skip_macros  = SKIP_MACROS,
        skip_days    = SKIP_DAYS,
        entry_mode   = "baseline",
        macro_rules  = MACRO_RULES,
    )

    sys.stdout = old_stdout
    output = buf.getvalue()

    # Extraire les metriques
    def _extract(label, cast=float):
        for line in output.splitlines():
            if label in line:
                val = line.split(":")[-1].strip().rstrip("%")
                try:
                    return cast(val)
                except ValueError:
                    return None
        return None

    return {
        "sl_pct"       : sl_pct,
        "rr"           : rr,
        "n_trades"     : _extract("Trades totaux", int),
        "win_rate"     : _extract("Win rate"),
        "return_pct"   : _extract("Return total"),
        "profit_factor": _extract("Profit factor"),
        "max_dd"       : _extract("Max drawdown"),
        "sharpe"       : _extract("Sharpe annualise"),
        "avg_win"      : _extract("Avg win"),
        "avg_loss"     : _extract("Avg loss"),
        "n_tp"         : _extract("TP touche", int),
        "n_sl"         : _extract("SL touche", int),
        "n_eod"        : _extract("EOD", int),
    }


def main():
    combos = list(itertools.product(SL_VALUES, RR_VALUES))
    print(f"[sweep] {len(combos)} combinaisons a tester (sl x rr)\n")

    results = []
    for i, (sl, rr) in enumerate(combos, 1):
        print(f"[sweep] {i}/{len(combos)} — sl={sl*100:.1f}%  rr={rr}")
        try:
            row = _run_one(sl, rr)
            results.append(row)
            print(f"         -> return={row['return_pct']}%  wr={row['win_rate']}%  pf={row['profit_factor']}")
        except Exception as e:
            print(f"         -> ERREUR : {e}")

    if not results:
        print("[sweep] Aucun resultat.")
        return

    # Sauvegarder CSV
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\n[sweep] Resultats sauvegardes : {OUT}")

    # Top 5 par return
    results.sort(key=lambda r: r["return_pct"] or -999, reverse=True)
    print("\n[sweep] Top 5 par return total :")
    print(f"  {'sl%':>5} {'rr':>5} {'N':>5} {'WR%':>7} {'Ret%':>8} {'PF':>7} {'DD%':>7}")
    print("  " + "-" * 50)
    for r in results[:5]:
        print(f"  {r['sl_pct']*100:>5.1f} {r['rr']:>5.1f} {r['n_trades']:>5} "
              f"{r['win_rate']:>6.1f}% {r['return_pct']:>+7.2f}% "
              f"{r['profit_factor']:>7.3f} {r['max_dd']:>+6.2f}%")


if __name__ == "__main__":
    main()
