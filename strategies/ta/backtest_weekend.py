# ── Backtest Weekend — Comparaison Mon-Fri vs 24/7 ──

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from strategies.ta.features import load_15m, compute_features
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
from strategies.ta.config import RESULTS_DIR


def attach_features_to_trades(trades_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Join trades avec features au moment de l'entry."""
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    trades = trades.set_index("entry_time")
    trades = trades.join(features_df, how="left")
    trades = trades.reset_index()
    return trades


def filter_by_day_of_week(trades_df: pd.DataFrame, days: list) -> pd.DataFrame:
    """Filtre trades par jours de semaine (0=Mon, 6=Sun)."""
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    dow = trades["entry_time"].dt.dayofweek
    return trades[dow.isin(days)]


def evaluate_trades(trades_df: pd.DataFrame, name: str = "") -> dict:
    """Compute stats on trades."""
    if trades_df.empty:
        return {
            "method": name,
            "n_trades": 0,
            "n_wins": 0,
            "wr": 0.0,
            "exp_R": 0.0,
        }

    n_total = len(trades_df)
    n_wins = (trades_df["outcome"] == "win").sum()
    wr = n_wins / n_total if n_total > 0 else 0.0

    # Expectancy en R (TP=2*ATR, SL=1*ATR)
    wins_avg = trades_df[trades_df["outcome"] == "win"]["atr_at_entry"].mean() if n_wins > 0 else 0.0
    loss_avg = trades_df[trades_df["outcome"] == "loss"]["atr_at_entry"].mean() if n_total - n_wins > 0 else 0.0

    if n_total > 0:
        exp_R = wr * (2.0 * wins_avg if wins_avg > 0 else 1.0) - (1 - wr) * (1.0 * loss_avg if loss_avg > 0 else 1.0)
    else:
        exp_R = 0.0

    return {
        "method": name,
        "n_trades": n_total,
        "n_wins": n_wins,
        "wr": wr,
        "exp_R": exp_R,
    }


def apply_voting_filter(trades_df: pd.DataFrame, voter: EnsembleVoterV2) -> pd.DataFrame:
    """Filtre les trades par ensemble voting."""
    filtered = []

    for idx, trade in trades_df.iterrows():
        regime = trade.get("regime", "bull")
        features = {
            "ema_state": int(trade.get("ema_state", 0)),
            "ema_slope": int(trade.get("ema_slope", 0)),
            "swing": int(trade.get("swing", 0)),
            "rsi_state": str(trade.get("rsi_state", "weak")),
            "stoch_state": str(trade.get("stoch_state", "weak")),
            "atr_state": str(trade.get("atr_state", "neutral")),
            "vwap_state": int(trade.get("vwap_state", 0)),
        }

        total_voters, favorable_votes, consensus_dir, confidence = voter.vote(
            regime, features, trade["direction"]
        )

        if consensus_dir is not None:
            trade_copy = trade.copy()
            trade_copy["vote_total"] = total_voters
            trade_copy["vote_favorable"] = favorable_votes
            trade_copy["vote_confidence"] = confidence
            filtered.append(trade_copy)

    return pd.DataFrame(filtered)


def main():
    print("[backtest_weekend] Chargement données...\n")

    df15 = load_15m()
    print(f"  Loaded {len(df15)} candles 15m")

    # Split IS (2020-2024) vs OOS (2025-2026)
    split_date = pd.Timestamp("2025-01-01", tz="UTC")
    df_is = df15[df15.index < split_date]
    df_oos = df15[df15.index >= split_date]

    print(f"  IS period: {len(df_is)} candles ({df_is.index[0].date()} - {df_is.index[-1].date()})")
    print(f"  OOS period: {len(df_oos)} candles ({df_oos.index[0].date()} - {df_oos.index[-1].date()})\n")

    # Features
    print("[backtest_weekend] Calcul features...\n")
    features_is = compute_features(
        df_is, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )
    features_oos = compute_features(
        df_oos, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )

    # Build trades (simple 2-bar reversal)
    print("[backtest_weekend] Building trades (2-bar reversal)...\n")
    from strategies.ta.backtest import build_trades

    trades_is = attach_features_to_trades(build_trades(df_is), features_is)
    trades_oos = attach_features_to_trades(build_trades(df_oos), features_oos)

    # Load voter
    print("[backtest_weekend] Loading voter (strict: n>=5, WR>=60%)...\n")
    voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=5, min_wr_oos=0.60)

    # Apply voting
    trades_is_voted = apply_voting_filter(trades_is, voter)
    trades_oos_voted = apply_voting_filter(trades_oos, voter)

    print("=== COMPARISON: Mon-Fri vs 24/7 ===\n")

    # Mon-Fri (0-4 = Mon-Fri)
    trades_is_mon_fri = filter_by_day_of_week(trades_is_voted, [0, 1, 2, 3, 4])
    trades_oos_mon_fri = filter_by_day_of_week(trades_oos_voted, [0, 1, 2, 3, 4])

    # Weekend only (5-6 = Sat-Sun)
    trades_is_weekend = filter_by_day_of_week(trades_is_voted, [5, 6])
    trades_oos_weekend = filter_by_day_of_week(trades_oos_voted, [5, 6])

    # 24/7 (all days)
    trades_is_all = trades_is_voted
    trades_oos_all = trades_oos_voted

    results = []

    # Mon-Fri
    print("MON-FRI ONLY:")
    stats_is = evaluate_trades(trades_is_mon_fri, "IS")
    stats_oos = evaluate_trades(trades_oos_mon_fri, "OOS")
    print(f"  IS:  n={stats_is['n_trades']:4d}  WR={stats_is['wr']:.3f}  Exp={stats_is['exp_R']:+8.2f}R")
    print(f"  OOS: n={stats_oos['n_trades']:4d}  WR={stats_oos['wr']:.3f}  Exp={stats_oos['exp_R']:+8.2f}R")
    print(f"  Drop: {stats_oos['wr'] - stats_is['wr']:+.3f}\n")
    results.append(("Mon-Fri", stats_is, stats_oos))

    # Weekend only
    print("WEEKEND ONLY:")
    stats_is = evaluate_trades(trades_is_weekend, "IS")
    stats_oos = evaluate_trades(trades_oos_weekend, "OOS")
    print(f"  IS:  n={stats_is['n_trades']:4d}  WR={stats_is['wr']:.3f}  Exp={stats_is['exp_R']:+8.2f}R")
    print(f"  OOS: n={stats_oos['n_trades']:4d}  WR={stats_oos['wr']:.3f}  Exp={stats_oos['exp_R']:+8.2f}R")
    print(f"  Drop: {stats_oos['wr'] - stats_is['wr']:+.3f}\n")
    results.append(("Weekend", stats_is, stats_oos))

    # 24/7
    print("24/7 (MON-FRI + WEEKEND):")
    stats_is = evaluate_trades(trades_is_all, "IS")
    stats_oos = evaluate_trades(trades_oos_all, "OOS")
    print(f"  IS:  n={stats_is['n_trades']:4d}  WR={stats_is['wr']:.3f}  Exp={stats_is['exp_R']:+8.2f}R")
    print(f"  OOS: n={stats_oos['n_trades']:4d}  WR={stats_oos['wr']:.3f}  Exp={stats_oos['exp_R']:+8.2f}R")
    print(f"  Drop: {stats_oos['wr'] - stats_is['wr']:+.3f}\n")
    results.append(("24/7", stats_is, stats_oos))

    # Summary
    print("\n=== SUMMARY ===\n")
    print("Mode       | IS_n  | IS_WR | OOS_n | OOS_WR | WR_drop | Recommendation")
    print("-----------|-------|-------|-------|--------|---------|----------------")
    for mode, stats_is, stats_oos in results:
        drop = stats_oos['wr'] - stats_is['wr']
        rec = "[OK] GOOD" if stats_oos['wr'] >= 0.48 and drop < 0.02 else "[!] CHECK"
        print(f"{mode:10s} | {stats_is['n_trades']:5d} | {stats_is['wr']:.1%} | "
              f"{stats_oos['n_trades']:5d} | {stats_oos['wr']:.1%}  | {drop:+.3f}  | {rec}")


if __name__ == "__main__":
    main()
