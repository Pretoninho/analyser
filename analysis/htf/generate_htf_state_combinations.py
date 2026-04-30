from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "htf" / "htf_state_combinations.db"

STATES = [
    {
        "code": "W1",
        "name": "W_OrderBias",
        "timeframe": "Weekly",
        "distribution": "[p_HL, p_LH]",
        "variables": "iv_skew_25d",
        "description": "Ordre probable High avant Low vs Low avant High sur la semaine",
    },
    {
        "code": "W2",
        "name": "W_MPReturnBias",
        "timeframe": "Weekly",
        "distribution": "[p_return, p_no_return]",
        "variables": "dist_mp_pct, gex, max_pain",
        "description": "Probabilite de retour vers Max Pain avant vendredi",
    },
    {
        "code": "W3",
        "name": "W_TSOrderBias",
        "timeframe": "Weekly",
        "distribution": "[p_HL, p_LH]",
        "variables": "term_1w, term_1m, term_3m",
        "description": "Effet de la term structure sur l ordre High/Low hebdo",
    },
    {
        "code": "W4",
        "name": "W_SkewPosBias",
        "timeframe": "Weekly",
        "distribution": "[p_LH, p_HL]",
        "variables": "skew extreme positif",
        "description": "En stress downside, biais Low avant High",
    },
    {
        "code": "W5",
        "name": "W_SkewNegBias",
        "timeframe": "Weekly",
        "distribution": "[p_HL, p_LH]",
        "variables": "skew extreme negatif",
        "description": "En appetit upside, biais High avant Low",
    },
    {
        "code": "D1",
        "name": "D_SweepRevBias",
        "timeframe": "Daily",
        "distribution": "[p_sweep_rev, p_no_sweep_rev]",
        "variables": "iv_slope_3d, structure daily",
        "description": "Probabilite de sweep puis reversal",
    },
    {
        "code": "D2",
        "name": "D_BreakBias",
        "timeframe": "Daily",
        "distribution": "[p_break, p_no_break]",
        "variables": "gex, structure daily",
        "description": "Probabilite de break valide du range daily",
    },
    {
        "code": "D3",
        "name": "D_TrendDayBias",
        "timeframe": "Daily",
        "distribution": "[p_trend, p_chop]",
        "variables": "iv_atm, gex, contexte daily",
        "description": "Typologie de journee trend vs chop",
    },
    {
        "code": "H4_1",
        "name": "H4_RegimeBias",
        "timeframe": "4H",
        "distribution": "[p_trend, p_range]",
        "variables": "gex, atr_4h, range precedent",
        "description": "Probabilite continuation trend vs compression range",
    },
    {
        "code": "H4_2",
        "name": "H4_PCRBias",
        "timeframe": "4H",
        "distribution": "[p_cont, p_revert]",
        "variables": "put_call_ratio, direction spot",
        "description": "Continuation ou mean reversion sur la fenetre 4H suivante",
    },
]


def build_rows(n: int):
    rows = []
    for idx, combo in enumerate(itertools.combinations(STATES, n), start=1):
        codes = [state["code"] for state in combo]
        names = [state["name"] for state in combo]
        timeframes = [state["timeframe"] for state in combo]
        distributions = [state["distribution"] for state in combo]
        variables = [state["variables"] for state in combo]
        descriptions = [state["description"] for state in combo]
        rows.append(
            {
                "combo_id": f"C{n}_{idx:03d}",
                "state_count": n,
                "state_codes": " + ".join(codes),
                "state_names": " + ".join(names),
                "timeframes": " + ".join(timeframes),
                "distributions": " | ".join(distributions),
                "variables": " | ".join(variables),
                "descriptions": " | ".join(descriptions),
            }
        )
    return rows


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows2 = build_rows(2)
    rows3 = build_rows(3)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS states")
        conn.execute("DROP TABLE IF EXISTS combinations_2")
        conn.execute("DROP TABLE IF EXISTS combinations_3")

        conn.execute(
            """
            CREATE TABLE states (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                distribution TEXT NOT NULL,
                variables TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE combinations_2 (
                combo_id TEXT PRIMARY KEY,
                state_count INTEGER NOT NULL,
                state_codes TEXT NOT NULL,
                state_names TEXT NOT NULL,
                timeframes TEXT NOT NULL,
                distributions TEXT NOT NULL,
                variables TEXT NOT NULL,
                descriptions TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE combinations_3 (
                combo_id TEXT PRIMARY KEY,
                state_count INTEGER NOT NULL,
                state_codes TEXT NOT NULL,
                state_names TEXT NOT NULL,
                timeframes TEXT NOT NULL,
                distributions TEXT NOT NULL,
                variables TEXT NOT NULL,
                descriptions TEXT NOT NULL
            )
            """
        )

        conn.executemany(
            "INSERT INTO states (code, name, timeframe, distribution, variables, description) VALUES (:code, :name, :timeframe, :distribution, :variables, :description)",
            STATES,
        )
        conn.executemany(
            "INSERT INTO combinations_2 (combo_id, state_count, state_codes, state_names, timeframes, distributions, variables, descriptions) VALUES (:combo_id, :state_count, :state_codes, :state_names, :timeframes, :distributions, :variables, :descriptions)",
            rows2,
        )
        conn.executemany(
            "INSERT INTO combinations_3 (combo_id, state_count, state_codes, state_names, timeframes, distributions, variables, descriptions) VALUES (:combo_id, :state_count, :state_codes, :state_names, :timeframes, :distributions, :variables, :descriptions)",
            rows3,
        )

    print(f"DB created: {DB_PATH}")
    print(f"states={len(STATES)}")
    print(f"combinations_2={len(rows2)}")
    print(f"combinations_3={len(rows3)}")


if __name__ == "__main__":
    main()
