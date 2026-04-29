from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "htf_state_combinations.db"
OUT_DIR = ROOT / "display" / "analysis"

CSV_2 = OUT_DIR / "htf_relevant_combo_stats_2.csv"
CSV_3 = OUT_DIR / "htf_relevant_combo_stats_3.csv"

Z_95 = 1.96


@dataclass(frozen=True)
class AdmissionProfile:
    name: str
    min_trades: int
    min_win_rate: float
    min_wilson_lb: float
    min_avg_trade: float
    min_total_return: float
    min_max_drawdown: float


STRICT = AdmissionProfile("strict", 40, 58.0, 50.0, 0.25, 5.0, -12.0)
BALANCED = AdmissionProfile("equilibre", 20, 54.0, 45.0, 0.15, 2.5, -15.0)
BALANCED_RELAXED = AdmissionProfile("equilibre_assoupli", 14, 53.0, 30.0, 0.15, 2.0, -16.0)
AGGRESSIVE = AdmissionProfile("agressif", 12, 50.0, 35.0, 0.10, 1.0, -18.0)
PROFILES = (STRICT, BALANCED, AGGRESSIVE)


def _wilson_lower_bound(wins: pd.Series, total: pd.Series, z: float = Z_95) -> pd.Series:
    total_safe = total.replace(0, np.nan)
    p = wins / total_safe
    z2 = z * z
    denom = 1.0 + z2 / total_safe
    center = p + z2 / (2.0 * total_safe)
    margin = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * total_safe)) / total_safe)
    lower = (center - margin) / denom
    return lower.fillna(0.0).clip(lower=0.0, upper=1.0)


def _passes_profile(df: pd.DataFrame, profile: AdmissionProfile) -> pd.Series:
    return (
        (df["n_trades"] >= profile.min_trades)
        & (df["win_rate_pct"] >= profile.min_win_rate)
        & (df["wilson_lb_pct"] >= profile.min_wilson_lb)
        & (df["avg_trade_pct"] >= profile.min_avg_trade)
        & (df["total_return_pct"] >= profile.min_total_return)
        & (df["max_drawdown_pct"] >= profile.min_max_drawdown)
    )


def _assign_qtable_status(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["passes_strict"] = _passes_profile(ranked, STRICT)
    ranked["passes_equilibre"] = _passes_profile(ranked, BALANCED)
    ranked["passes_equilibre_assoupli"] = _passes_profile(ranked, BALANCED_RELAXED)
    ranked["passes_agressif"] = _passes_profile(ranked, AGGRESSIVE)

    ranked["qtable_status"] = np.where(
        ranked["passes_strict"],
        "ROBUSTE_STRICT",
        np.where(
            ranked["passes_equilibre"],
            "ROBUSTE_EQUILIBRE",
            np.where(
                ranked["passes_equilibre_assoupli"],
                "VALIDE_EQUILIBRE_ASSOUPLI",
                np.where(ranked["passes_agressif"], "PROVISOIRE_AGRESSIF", "EXCLU"),
            ),
        ),
    )
    return ranked


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()

    wins = (ranked["win_rate_pct"] / 100.0 * ranked["n_trades"]).round()
    ranked["wins"] = wins.astype(int)
    ranked["wilson_lb_pct"] = _wilson_lower_bound(wins, ranked["n_trades"]) * 100.0
    ranked["avg_trade_score"] = ranked["avg_trade_pct"].clip(lower=0.0)
    ranked["return_score"] = ranked["total_return_pct"].clip(lower=0.0)
    ranked["drawdown_penalty"] = (1.0 - (ranked["max_drawdown_pct"].abs() / 25.0)).clip(lower=0.0, upper=1.0)
    ranked["sample_score"] = (ranked["n_trades"] / 30.0).clip(lower=0.0, upper=1.0)

    ranked["reliability_score"] = (
        ranked["wilson_lb_pct"] * 0.60
        + ranked["avg_trade_score"] * 18.0
        + ranked["return_score"] * 0.80
        + ranked["drawdown_penalty"] * 12.0
        + ranked["sample_score"] * 10.0
    )

    ranked = _assign_qtable_status(ranked)
    ranked["reliability_label"] = np.where(
        ranked["qtable_status"] == "ROBUSTE_STRICT",
        "Robuste",
        np.where(
            ranked["qtable_status"] == "ROBUSTE_EQUILIBRE",
            "Valide",
            np.where(
                ranked["qtable_status"] == "VALIDE_EQUILIBRE_ASSOUPLI",
                "ValideAssoupli",
                np.where(
                ranked["qtable_status"] == "PROVISOIRE_AGRESSIF",
                "Provisoire",
                np.where(ranked["n_trades"] < AGGRESSIVE.min_trades, "Speculatif", "Fragile"),
                ),
            ),
        ),
    )

    ranked = ranked.sort_values(
        by=["reliability_score", "wilson_lb_pct", "avg_trade_pct", "total_return_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def _profile_view(df: pd.DataFrame, profile: AdmissionProfile) -> pd.DataFrame:
    filtered = df[_passes_profile(df, profile)].copy()
    return filtered.reset_index(drop=True)


def _shortlist_view(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        df["qtable_status"].isin(
            [
                "ROBUSTE_STRICT",
                "ROBUSTE_EQUILIBRE",
                "VALIDE_EQUILIBRE_ASSOUPLI",
                "PROVISOIRE_AGRESSIF",
            ]
        )
    ].copy()
    return filtered.reset_index(drop=True)


def _profile_summary_lines(label: str, profile: AdmissionProfile, full_df: pd.DataFrame, profile_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    lines = [f"=== Classement {label} ===", ""]
    if profile_df.empty:
        lines.append(f"Aucune combinaison {label} n'a valide le profil {profile.name}.")
        lines.append("")
        return lines

    lines.append(
        f"Profil {profile.name}: trades>={profile.min_trades}, wr>={profile.min_win_rate:.0f}%, "
        f"wilson>={profile.min_wilson_lb:.0f}%, avg>={profile.min_avg_trade:.2f}%, "
        f"total>={profile.min_total_return:.1f}%, dd>={profile.min_max_drawdown:.0f}%"
    )
    lines.append("")
    lines.append("Top admissible:")
    for row in profile_df.head(top_n).itertuples(index=False):
        lines.append(
            " - "
            f"#{row.rank} {row.state_codes} | score={row.reliability_score:.2f} | "
            f"wr={row.win_rate_pct:.2f}% | wilson={row.wilson_lb_pct:.2f}% | "
            f"trades={row.n_trades} | avg={row.avg_trade_pct:.3f}% | total={row.total_return_pct:.2f}% | "
            f"dd={row.max_drawdown_pct:.2f}% | statut={row.qtable_status}"
        )

    speculative = full_df[full_df["n_trades"] < AGGRESSIVE.min_trades].head(5)
    if not speculative.empty:
        lines.append("")
        lines.append("Combinaisons a fort win rate mais echantillon faible:")
        for row in speculative.itertuples(index=False):
            lines.append(
                " - "
                f"#{row.rank} {row.state_codes} | wr={row.win_rate_pct:.2f}% | trades={row.n_trades} | "
                f"avg={row.avg_trade_pct:.3f}% | total={row.total_return_pct:.2f}%"
            )

    lines.append("")
    return lines


def _shortlist_summary_lines(label: str, shortlist_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    lines = [f"=== Shortlist Q-Table {label} ===", ""]
    if shortlist_df.empty:
        lines.append("Aucune combinaison admissible pour la Q-Table avec les profils definis.")
        lines.append("")
        return lines

    for row in shortlist_df.head(top_n).itertuples(index=False):
        lines.append(
            " - "
            f"#{row.rank} {row.state_codes} | statut={row.qtable_status} | wr={row.win_rate_pct:.2f}% | "
            f"wilson={row.wilson_lb_pct:.2f}% | trades={row.n_trades} | avg={row.avg_trade_pct:.3f}% | "
            f"total={row.total_return_pct:.2f}% | dd={row.max_drawdown_pct:.2f}%"
        )
    lines.append("")
    return lines


def _persist_db(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    df.to_sql(table_name, conn, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats2 = pd.read_csv(CSV_2)
    stats3 = pd.read_csv(CSV_3)

    ranked2 = _score_frame(stats2)
    ranked3 = _score_frame(stats3)

    strict2 = _profile_view(ranked2, STRICT)
    strict3 = _profile_view(ranked3, STRICT)
    balanced2 = _profile_view(ranked2, BALANCED)
    balanced3 = _profile_view(ranked3, BALANCED)
    balanced_relaxed2 = _profile_view(ranked2, BALANCED_RELAXED)
    balanced_relaxed3 = _profile_view(ranked3, BALANCED_RELAXED)
    aggressive2 = _profile_view(ranked2, AGGRESSIVE)
    aggressive3 = _profile_view(ranked3, AGGRESSIVE)
    shortlist2 = _shortlist_view(ranked2)
    shortlist3 = _shortlist_view(ranked3)

    ranked2.to_csv(OUT_DIR / "htf_combo_ranking_2_all.csv", index=False)
    ranked3.to_csv(OUT_DIR / "htf_combo_ranking_3_all.csv", index=False)
    strict2.to_csv(OUT_DIR / "htf_combo_ranking_2_strict.csv", index=False)
    strict3.to_csv(OUT_DIR / "htf_combo_ranking_3_strict.csv", index=False)
    balanced2.to_csv(OUT_DIR / "htf_combo_ranking_2_equilibre.csv", index=False)
    balanced3.to_csv(OUT_DIR / "htf_combo_ranking_3_equilibre.csv", index=False)
    balanced_relaxed2.to_csv(OUT_DIR / "htf_combo_ranking_2_equilibre_assoupli.csv", index=False)
    balanced_relaxed3.to_csv(OUT_DIR / "htf_combo_ranking_3_equilibre_assoupli.csv", index=False)
    aggressive2.to_csv(OUT_DIR / "htf_combo_ranking_2_agressif.csv", index=False)
    aggressive3.to_csv(OUT_DIR / "htf_combo_ranking_3_agressif.csv", index=False)
    shortlist2.to_csv(OUT_DIR / "htf_combo_qtable_shortlist_2.csv", index=False)
    shortlist3.to_csv(OUT_DIR / "htf_combo_qtable_shortlist_3.csv", index=False)

    with sqlite3.connect(DB_PATH) as conn:
        _persist_db(conn, "combo_ranking_2_all", ranked2)
        _persist_db(conn, "combo_ranking_3_all", ranked3)
        _persist_db(conn, "combo_ranking_2_strict", strict2)
        _persist_db(conn, "combo_ranking_3_strict", strict3)
        _persist_db(conn, "combo_ranking_2_equilibre", balanced2)
        _persist_db(conn, "combo_ranking_3_equilibre", balanced3)
        _persist_db(conn, "combo_ranking_2_equilibre_assoupli", balanced_relaxed2)
        _persist_db(conn, "combo_ranking_3_equilibre_assoupli", balanced_relaxed3)
        _persist_db(conn, "combo_ranking_2_agressif", aggressive2)
        _persist_db(conn, "combo_ranking_3_agressif", aggressive3)
        _persist_db(conn, "combo_qtable_shortlist_2", shortlist2)
        _persist_db(conn, "combo_qtable_shortlist_3", shortlist3)

    summary_lines = []
    summary_lines.extend(_profile_summary_lines("2 etats - strict", STRICT, ranked2, strict2))
    summary_lines.extend(_profile_summary_lines("3 etats - strict", STRICT, ranked3, strict3))
    summary_lines.extend(_profile_summary_lines("2 etats - equilibre", BALANCED, ranked2, balanced2))
    summary_lines.extend(_profile_summary_lines("3 etats - equilibre", BALANCED, ranked3, balanced3))
    summary_lines.extend(_profile_summary_lines("2 etats - equilibre assoupli", BALANCED_RELAXED, ranked2, balanced_relaxed2))
    summary_lines.extend(_profile_summary_lines("3 etats - equilibre assoupli", BALANCED_RELAXED, ranked3, balanced_relaxed3))
    summary_lines.extend(_profile_summary_lines("2 etats - agressif", AGGRESSIVE, ranked2, aggressive2))
    summary_lines.extend(_profile_summary_lines("3 etats - agressif", AGGRESSIVE, ranked3, aggressive3))
    summary_lines.extend(_shortlist_summary_lines("2 etats", shortlist2))
    summary_lines.extend(_shortlist_summary_lines("3 etats", shortlist3))
    summary_path = OUT_DIR / "htf_combo_ranking_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"db={DB_PATH}")
    print(f"ranking_2_all={OUT_DIR / 'htf_combo_ranking_2_all.csv'}")
    print(f"ranking_3_all={OUT_DIR / 'htf_combo_ranking_3_all.csv'}")
    print(f"ranking_2_equilibre={OUT_DIR / 'htf_combo_ranking_2_equilibre.csv'}")
    print(f"ranking_3_equilibre={OUT_DIR / 'htf_combo_ranking_3_equilibre.csv'}")
    print(f"ranking_2_equilibre_assoupli={OUT_DIR / 'htf_combo_ranking_2_equilibre_assoupli.csv'}")
    print(f"ranking_3_equilibre_assoupli={OUT_DIR / 'htf_combo_ranking_3_equilibre_assoupli.csv'}")
    print(f"shortlist_2={OUT_DIR / 'htf_combo_qtable_shortlist_2.csv'}")
    print(f"shortlist_3={OUT_DIR / 'htf_combo_qtable_shortlist_3.csv'}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()