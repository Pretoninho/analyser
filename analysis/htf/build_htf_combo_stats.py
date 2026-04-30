from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.htf.backtest_htf_probabilistic import _build_daily_frame, _compute_probabilities
from analysis.htf.generate_htf_state_combinations import STATES, DB_PATH

OUT_DIR = ROOT / "display" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEE_BPS = 5.0
SLIPPAGE_BPS = 2.0
TRADE_COST = ((FEE_BPS + SLIPPAGE_BPS) * 2.0) / 10000.0


def _states_df() -> pd.DataFrame:
    return pd.DataFrame(STATES)


def _filter_combos() -> tuple[pd.DataFrame, pd.DataFrame]:
    states_df = _states_df()

    rows2 = []
    rows3 = []

    combos = states_df.to_dict("records")

    # 2-state: uniquement inter-timeframes pertinents
    idx2 = 1
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            c = [combos[i], combos[j]]
            tfs = {x["timeframe"] for x in c}
            if len(tfs) != 2:
                continue
            rows2.append(
                {
                    "combo_id": f"R2_{idx2:03d}",
                    "state_count": 2,
                    "state_codes": " + ".join(x["code"] for x in c),
                    "state_names": " + ".join(x["name"] for x in c),
                    "timeframes": " + ".join(x["timeframe"] for x in c),
                }
            )
            idx2 += 1

    # 3-state: exactement Weekly + Daily + 4H
    weekly = [x for x in combos if x["timeframe"] == "Weekly"]
    daily = [x for x in combos if x["timeframe"] == "Daily"]
    h4 = [x for x in combos if x["timeframe"] == "4H"]
    idx3 = 1
    for w in weekly:
        for d in daily:
            for h in h4:
                c = [w, d, h]
                rows3.append(
                    {
                        "combo_id": f"R3_{idx3:03d}",
                        "state_count": 3,
                        "state_codes": " + ".join(x["code"] for x in c),
                        "state_names": " + ".join(x["name"] for x in c),
                        "timeframes": " + ".join(x["timeframe"] for x in c),
                    }
                )
                idx3 += 1

    return pd.DataFrame(rows2), pd.DataFrame(rows3)


def _dominant_signal(long_score: pd.Series, short_score: pd.Series, conf_min: float = 0.55, edge_min: float = 0.08) -> pd.Series:
    confidence = np.maximum(long_score, short_score)
    edge = long_score - short_score
    signal = np.where((confidence >= conf_min) & (edge > edge_min), 1, 0)
    signal = np.where((confidence >= conf_min) & (edge < -edge_min), -1, signal)
    return pd.Series(signal, index=long_score.index, dtype="int64")


def _build_state_signals(model_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.copy()
    df["mom_sign"] = np.sign(df["close"].pct_change()).replace(0, 1).fillna(1)
    df["mp_dir"] = np.sign(df["max_pain"] - df["close"]).fillna(0)
    df["prev_ret_sign"] = np.sign(df["ret_d"]).replace(0, 1).fillna(1)
    df["fwd_ret_1d"] = df["close"].shift(-1) / df["close"] - 1.0

    out = pd.DataFrame(index=df.index)
    out["fwd_ret_1d"] = df["fwd_ret_1d"]

    # Weekly states
    out["W1"] = _dominant_signal(df["p_risk_on"], df["p_risk_off"])

    w2_long = np.where(df["mp_dir"] > 0, df["p_pinning"], 0.0)
    w2_short = np.where(df["mp_dir"] < 0, df["p_pinning"], 0.0)
    out["W2"] = _dominant_signal(pd.Series(w2_long, index=df.index), pd.Series(w2_short, index=df.index))

    out["W3"] = _dominant_signal(1.0 - df["p_backwardation"], df["p_backwardation"])

    w4_short = np.where(df["p_risk_off"] > 0.65, df["p_risk_off"], 0.0)
    out["W4"] = _dominant_signal(pd.Series(np.zeros(len(df)), index=df.index), pd.Series(w4_short, index=df.index), conf_min=0.60, edge_min=0.10)

    w5_long = np.where(df["p_risk_on"] > 0.65, df["p_risk_on"], 0.0)
    out["W5"] = _dominant_signal(pd.Series(w5_long, index=df.index), pd.Series(np.zeros(len(df)), index=df.index), conf_min=0.60, edge_min=0.10)

    # Daily states
    d1_long = np.where(df["prev_ret_sign"] < 0, df["p_mean_reversion"], 0.0)
    d1_short = np.where(df["prev_ret_sign"] > 0, df["p_mean_reversion"], 0.0)
    out["D1"] = _dominant_signal(pd.Series(d1_long, index=df.index), pd.Series(d1_short, index=df.index))

    d2_long = np.where(df["mom_sign"] > 0, df["p_expansion"], 0.0)
    d2_short = np.where(df["mom_sign"] < 0, df["p_expansion"], 0.0)
    out["D2"] = _dominant_signal(pd.Series(d2_long, index=df.index), pd.Series(d2_short, index=df.index))

    d3_long = np.where(df["mom_sign"] > 0, df["p_continuation"], 0.0)
    d3_short = np.where(df["mom_sign"] < 0, df["p_continuation"], 0.0)
    out["D3"] = _dominant_signal(pd.Series(d3_long, index=df.index), pd.Series(d3_short, index=df.index))

    # 4H states (mapped on daily bar for immediate stats)
    h41_long = np.where(df["mom_sign"] > 0, df["p_expansion"], 0.0)
    h41_short = np.where(df["mom_sign"] < 0, df["p_expansion"], 0.0)
    out["H4_1"] = _dominant_signal(pd.Series(h41_long, index=df.index), pd.Series(h41_short, index=df.index), conf_min=0.53, edge_min=0.05)

    h42_long = np.where(df["p_continuation"] >= df["p_mean_reversion"], np.where(df["mom_sign"] > 0, df["p_continuation"], 0.0), np.where(df["mom_sign"] < 0, df["p_mean_reversion"], 0.0))
    h42_short = np.where(df["p_continuation"] >= df["p_mean_reversion"], np.where(df["mom_sign"] < 0, df["p_continuation"], 0.0), np.where(df["mom_sign"] > 0, df["p_mean_reversion"], 0.0))
    out["H4_2"] = _dominant_signal(pd.Series(h42_long, index=df.index), pd.Series(h42_short, index=df.index), conf_min=0.53, edge_min=0.05)

    return out.dropna(subset=["fwd_ret_1d"])


def _aggregate_combo_signal(state_signal_df: pd.DataFrame, codes: list[str]) -> pd.Series:
    sub = state_signal_df[codes].copy()
    score = sub.sum(axis=1)
    signal = np.where(score > 0, 1, np.where(score < 0, -1, 0))
    return pd.Series(signal, index=sub.index, dtype="int64")


def _combo_stats(combo_id: str, codes: list[str], signal: pd.Series, fwd_ret: pd.Series) -> dict:
    long_mask = signal == 1
    short_mask = signal == -1
    flat_mask = signal == 0

    strategy_ret = np.where(long_mask, fwd_ret, np.where(short_mask, -fwd_ret, 0.0))
    strategy_ret = np.where(signal != 0, strategy_ret - TRADE_COST, 0.0)
    strategy_ret = pd.Series(strategy_ret, index=signal.index)

    traded = strategy_ret[signal != 0]
    eq = (1.0 + strategy_ret).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak

    return {
        "combo_id": combo_id,
        "state_codes": " + ".join(codes),
        "n_obs": int(len(signal)),
        "long_n": int(long_mask.sum()),
        "short_n": int(short_mask.sum()),
        "flat_n": int(flat_mask.sum()),
        "long_pct": float(long_mask.mean() * 100.0),
        "short_pct": float(short_mask.mean() * 100.0),
        "flat_pct": float(flat_mask.mean() * 100.0),
        "long_avg_fwd_ret_pct": float(fwd_ret[long_mask].mean() * 100.0) if long_mask.any() else 0.0,
        "short_avg_fwd_ret_pct": float((-fwd_ret[short_mask]).mean() * 100.0) if short_mask.any() else 0.0,
        "flat_avg_abs_move_pct": float(fwd_ret[flat_mask].abs().mean() * 100.0) if flat_mask.any() else 0.0,
        "n_trades": int((signal != 0).sum()),
        "win_rate_pct": float((traded > 0).mean() * 100.0) if len(traded) else 0.0,
        "avg_trade_pct": float(traded.mean() * 100.0) if len(traded) else 0.0,
        "total_return_pct": float((eq.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(dd.min() * 100.0),
    }


def _persist_tables(conn: sqlite3.Connection, df2: pd.DataFrame, df3: pd.DataFrame, st2: pd.DataFrame, st3: pd.DataFrame) -> None:
    conn.execute("DROP TABLE IF EXISTS relevant_combinations_2")
    conn.execute("DROP TABLE IF EXISTS relevant_combinations_3")
    conn.execute("DROP TABLE IF EXISTS relevant_combo_stats_2")
    conn.execute("DROP TABLE IF EXISTS relevant_combo_stats_3")

    df2.to_sql("relevant_combinations_2", conn, index=False)
    df3.to_sql("relevant_combinations_3", conn, index=False)
    st2.to_sql("relevant_combo_stats_2", conn, index=False)
    st3.to_sql("relevant_combo_stats_3", conn, index=False)


def main():
    combos2, combos3 = _filter_combos()

    base_df = _build_daily_frame("BTC", derivatives_mode="proxy", market_source="binance")
    model_df = _compute_probabilities(base_df)
    state_signal_df = _build_state_signals(model_df)
    fwd_ret = state_signal_df["fwd_ret_1d"]

    stats2 = []
    for row in combos2.itertuples(index=False):
        codes = row.state_codes.split(" + ")
        signal = _aggregate_combo_signal(state_signal_df, codes)
        stats2.append(_combo_stats(row.combo_id, codes, signal, fwd_ret))

    stats3 = []
    for row in combos3.itertuples(index=False):
        codes = row.state_codes.split(" + ")
        signal = _aggregate_combo_signal(state_signal_df, codes)
        stats3.append(_combo_stats(row.combo_id, codes, signal, fwd_ret))

    stats2_df = pd.DataFrame(stats2).merge(combos2[["combo_id", "state_names", "timeframes"]], on="combo_id", how="left")
    stats3_df = pd.DataFrame(stats3).merge(combos3[["combo_id", "state_names", "timeframes"]], on="combo_id", how="left")

    with sqlite3.connect(DB_PATH) as conn:
        _persist_tables(conn, combos2, combos3, stats2_df, stats3_df)

    stats2_df.to_csv(OUT_DIR / "htf_relevant_combo_stats_2.csv", index=False)
    stats3_df.to_csv(OUT_DIR / "htf_relevant_combo_stats_3.csv", index=False)

    print(f"DB updated: {DB_PATH}")
    print(f"relevant_combinations_2={len(combos2)}")
    print(f"relevant_combinations_3={len(combos3)}")
    print(f"stats_2={len(stats2_df)}")
    print(f"stats_3={len(stats3_df)}")
    print(f"csv_2={OUT_DIR / 'htf_relevant_combo_stats_2.csv'}")
    print(f"csv_3={OUT_DIR / 'htf_relevant_combo_stats_3.csv'}")


if __name__ == "__main__":
    main()
