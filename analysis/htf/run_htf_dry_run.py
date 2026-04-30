from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.htf.backtest_htf_probabilistic import _build_daily_frame, _compute_probabilities
from analysis.htf.build_htf_combo_stats import _aggregate_combo_signal, _build_state_signals

OUT_DIR = ROOT / "display" / "analysis"
DB_DIR = ROOT / "db"


def _load_seed(profile: str) -> dict:
    path = DB_DIR / "htf" / f"stats_agent_htf_seed_{profile}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Seed introuvable: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _baseline_signal(model_df: pd.DataFrame, edge_threshold: float = 0.08) -> pd.Series:
    mom_sign = np.sign(model_df["close"].pct_change()).replace(0, 1).fillna(1)
    edge = model_df["p_continuation"] - model_df["p_mean_reversion"]
    sig = np.where(edge > edge_threshold, mom_sign, np.where(edge < -edge_threshold, -mom_sign, 0))
    sig = np.where(model_df["p_no_trade"] > 0.45, 0, sig)
    return pd.Series(sig, index=model_df.index, dtype="int64")


def _run(seed: dict, asset: str, market_source: str, derivatives_mode: str) -> pd.DataFrame:
    base_df = _build_daily_frame(asset, derivatives_mode=derivatives_mode, market_source=market_source)
    model_df = _compute_probabilities(base_df)
    state_signal_df = _build_state_signals(model_df)

    out = pd.DataFrame(index=state_signal_df.index)
    out["close"] = model_df.loc[state_signal_df.index, "close"]

    long_score = pd.Series(0.0, index=out.index)
    short_score = pd.Series(0.0, index=out.index)

    for state in seed.get("states", []):
        codes = [c.strip() for c in str(state.get("state_codes", "")).split("+") if c.strip()]
        if not codes:
            continue

        combo_signal = _aggregate_combo_signal(state_signal_df, codes)
        q_long = float(state.get("q_long", 0.0))
        q_short = float(state.get("q_short", 0.0))

        if q_long > 0:
            long_score = long_score + np.where(combo_signal == 1, q_long, 0.0)
        if q_short > 0:
            short_score = short_score + np.where(combo_signal == -1, q_short, 0.0)

    out["long_score"] = long_score
    out["short_score"] = short_score

    out["htf_signal"] = np.where(
        out["long_score"] > out["short_score"],
        1,
        np.where(out["short_score"] > out["long_score"], -1, 0),
    ).astype(int)

    baseline = _baseline_signal(model_df.loc[out.index])
    out["baseline_signal"] = baseline
    out["agreement"] = (out["htf_signal"] == out["baseline_signal"]).astype(int)

    # Forward return diagnostic only (no execution)
    out["fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0
    out = out.dropna(subset=["fwd_ret_1d"]).copy()

    out["htf_ret_proxy"] = np.where(out["htf_signal"] == 1, out["fwd_ret_1d"], np.where(out["htf_signal"] == -1, -out["fwd_ret_1d"], 0.0))
    out["baseline_ret_proxy"] = np.where(out["baseline_signal"] == 1, out["fwd_ret_1d"], np.where(out["baseline_signal"] == -1, -out["fwd_ret_1d"], 0.0))

    return out


def _summary(df: pd.DataFrame, profile: str, seed_states: int) -> str:
    htf_trades = int((df["htf_signal"] != 0).sum())
    base_trades = int((df["baseline_signal"] != 0).sum())

    htf_avg = float(df.loc[df["htf_signal"] != 0, "htf_ret_proxy"].mean() * 100.0) if htf_trades else 0.0
    base_avg = float(df.loc[df["baseline_signal"] != 0, "baseline_ret_proxy"].mean() * 100.0) if base_trades else 0.0

    agreement_pct = float(df["agreement"].mean() * 100.0) if len(df) else 0.0
    htf_total = float(((1.0 + df["htf_ret_proxy"]).cumprod().iloc[-1] - 1.0) * 100.0) if len(df) else 0.0
    base_total = float(((1.0 + df["baseline_ret_proxy"]).cumprod().iloc[-1] - 1.0) * 100.0) if len(df) else 0.0

    lines = [
        "HTF DRY-RUN REPORT",
        f"profile={profile}",
        f"seed_states={seed_states}",
        f"period={df.index.min().date()} -> {df.index.max().date()}",
        f"n_days={len(df)}",
        f"htf_trades={htf_trades}",
        f"baseline_trades={base_trades}",
        f"agreement_pct={agreement_pct:.2f}%",
        f"htf_avg_trade_proxy={htf_avg:+.4f}%",
        f"baseline_avg_trade_proxy={base_avg:+.4f}%",
        f"htf_total_proxy={htf_total:+.2f}%",
        f"baseline_total_proxy={base_total:+.2f}%",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTF dry-run (no live execution)")
    parser.add_argument("--profile", default="equilibre_assoupli", choices=["agressif", "equilibre_assoupli"])
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--market-source", default="binance", choices=["auto", "db", "binance"])
    parser.add_argument("--derivatives-mode", default="proxy", choices=["strict", "hybrid", "proxy"])
    args = parser.parse_args()

    seed = _load_seed(args.profile)
    df = _run(seed, asset=args.asset.upper(), market_source=args.market_source, derivatives_mode=args.derivatives_mode)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"htf_dry_run_{args.profile}.csv"
    report_path = OUT_DIR / f"htf_dry_run_{args.profile}_report.txt"

    df.to_csv(csv_path, index=True)
    report = _summary(df, args.profile, int(seed.get("n_states", 0)))
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"csv={csv_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
