"""
Backtest HTF probabiliste (MVP) base sur donnees options Deribit.

Objectif:
- Permettre un backtesting immediat avec la base existante (prices.db)
- Utiliser un schema regime-first probabiliste (sans etats figes)
- Produire un rapport simple + journal des trades

Usage:
    python analysis/backtest_htf_probabilistic.py --asset BTC --min-days 30 --edge-threshold 0.08
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.binance import load_binance_1m

PRICES_DB = ROOT / "db" / "prices.db"
OUT_DIR = ROOT / "display" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    asset: str = "BTC"
    min_days: int = 30
    edge_threshold: float = 0.08
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    derivatives_mode: str = "hybrid"  # strict | hybrid | proxy
    market_source: str = "auto"  # auto | db | binance


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def _zscore(s: pd.Series, window: int = 20) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    z = (s - m) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def _load_market_1m(asset: str, market_source: str = "auto") -> pd.DataFrame:
    if market_source == "binance":
        b = load_binance_1m()
        out = b[["timestamp", "open", "high", "low", "close", "volume", "open_interest", "funding_rate"]].copy()
        out = out.rename(columns={"timestamp": "ts"})
        return out

    with sqlite3.connect(PRICES_DB) as conn:
        q = """
            SELECT ts, open, high, low, close, volume, open_interest, funding_rate
            FROM market_1m
            WHERE asset = ?
            ORDER BY ts ASC
        """
        df = pd.read_sql_query(q, conn, params=(asset,))

    if df.empty:
        if market_source == "auto":
            b = load_binance_1m()
            out = b[["timestamp", "open", "high", "low", "close", "volume", "open_interest", "funding_rate"]].copy()
            out = out.rename(columns={"timestamp": "ts"})
            return out
        return df

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def _load_derivatives(asset: str) -> pd.DataFrame:
    with sqlite3.connect(PRICES_DB) as conn:
        q = """
            SELECT timestamp, iv_atm, iv_skew_25d, iv_skew_10d,
                   term_1w, term_1m, term_3m, term_6m,
                   put_call_ratio, index_price, max_pain, gex
            FROM derivatives
            WHERE asset = ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(q, conn, params=(asset,))

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def _build_proxy_derivatives(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un proxy options journalier pour backtest immediate quand l'historique
    options Deribit est trop court.
    """
    p = pd.DataFrame(index=daily.index)
    ret = daily["close"].pct_change()

    # Proxy IV ATM via vol realisee rolling
    rv = ret.rolling(20).std(ddof=0) * np.sqrt(365) * 100.0
    p["iv_atm"] = rv

    # Proxy skew: asymetrie downside/upside recente
    dn = ret.clip(upper=0).abs().rolling(20).mean()
    up = ret.clip(lower=0).rolling(20).mean()
    p["iv_skew_25d"] = ((dn - up) * 10000.0).replace([np.inf, -np.inf], np.nan)
    p["iv_skew_10d"] = p["iv_skew_25d"] * 1.2

    # Proxy term structure
    iv_short = ret.rolling(7).std(ddof=0) * np.sqrt(365) * 100.0
    iv_mid = ret.rolling(30).std(ddof=0) * np.sqrt(365) * 100.0
    iv_long = ret.rolling(90).std(ddof=0) * np.sqrt(365) * 100.0
    p["term_1w"] = iv_short
    p["term_1m"] = iv_mid
    p["term_3m"] = iv_long
    p["term_6m"] = iv_long.rolling(2).mean()

    # Proxy put/call ratio: funding + stress downside
    fr = daily["funding_rate"].fillna(0.0)
    pcr = 1.0 + (dn.fillna(0.0) * 2000.0) - (fr * 50000.0)
    p["put_call_ratio"] = pcr.clip(0.3, 2.5)

    # Proxy max pain: centre de gravite 5 jours
    p["max_pain"] = daily["close"].rolling(5).mean()
    p["index_price"] = daily["close"]

    # Proxy GEX: OI z-score inverse de la vol
    oi = daily["open_interest"].ffill()
    oi_z = (oi - oi.rolling(20).mean()) / oi.rolling(20).std(ddof=0).replace(0, np.nan)
    rv_z = (rv - rv.rolling(20).mean()) / rv.rolling(20).std(ddof=0).replace(0, np.nan)
    # Si OI est plat (cas Binance fallback), garder un proxy gex fonde sur la vol.
    gex_proxy = (oi_z - rv_z) * 1000.0
    if gex_proxy.notna().sum() == 0:
        gex_proxy = (-rv_z).fillna(0.0) * 1000.0
    p["gex"] = gex_proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return p


def _build_daily_frame(asset: str, derivatives_mode: str, market_source: str) -> pd.DataFrame:
    m1 = _load_market_1m(asset, market_source=market_source)
    der = _load_derivatives(asset)

    if m1.empty:
        raise RuntimeError("Aucune donnee market_1m disponible. Lancez d'abord la collecte/backfill.")

    if der.empty:
        raise RuntimeError("Aucune donnee derivatives disponible. Lancez la collecte options d'abord.")

    m1 = m1.set_index("ts")
    daily = pd.DataFrame(
        {
            "open": m1["open"].resample("1D").first(),
            "high": m1["high"].resample("1D").max(),
            "low": m1["low"].resample("1D").min(),
            "close": m1["close"].resample("1D").last(),
            "volume": m1["volume"].resample("1D").sum(),
            "open_interest": m1["open_interest"].resample("1D").last(),
            "funding_rate": m1["funding_rate"].resample("1D").mean(),
        }
    ).dropna()

    der = der.set_index("timestamp")
    dder = (
        der.resample("1D")
        .last()[
            [
                "iv_atm",
                "iv_skew_25d",
                "iv_skew_10d",
                "term_1w",
                "term_1m",
                "term_3m",
                "term_6m",
                "put_call_ratio",
                "index_price",
                "max_pain",
                "gex",
            ]
        ]
        .ffill()
    )

    proxy = _build_proxy_derivatives(daily)
    if derivatives_mode == "proxy":
        feat = proxy
    elif derivatives_mode == "hybrid":
        feat = dder.combine_first(proxy)
    else:
        feat = dder

    df = daily.join(feat, how="left").ffill().dropna(subset=["iv_atm", "iv_skew_25d", "put_call_ratio", "gex", "max_pain"])
    return df


def _compute_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_d"] = out["close"].pct_change()
    out["atr_proxy"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    # Features centraux des fiches validees
    out["iv_slope_3d"] = out["iv_atm"] - out["iv_atm"].shift(3)
    out["dist_mp_pct"] = (out["close"] - out["max_pain"]).abs() / out["close"].replace(0, np.nan)
    out["term_score"] = out["term_1w"] - out["term_1m"]

    # Z-scores pour transformer en croyances probabilistes
    z_skew = _zscore(out["iv_skew_25d"], 20)
    z_pcr = _zscore(out["put_call_ratio"], 20)
    z_gex = _zscore(out["gex"], 20)
    z_ivs = _zscore(out["iv_slope_3d"], 20)
    z_dist = _zscore(out["dist_mp_pct"], 20)
    z_term = _zscore(out["term_score"], 20)

    # Croyances primaires (0..1)
    out["p_risk_off"] = _sigmoid(0.9 * z_skew + 0.6 * z_pcr + 0.3 * z_ivs)
    out["p_risk_on"] = 1.0 - out["p_risk_off"]

    out["p_pinning"] = _sigmoid(0.9 * z_gex + 0.5 * z_dist)
    out["p_expansion"] = 1.0 - out["p_pinning"]

    out["p_backwardation"] = _sigmoid(0.8 * z_term)

    # Probabilites de classes decisionnelles
    # continuation favorisee par risk_on + expansion
    out["p_continuation"] = (
        0.50 * out["p_risk_on"]
        + 0.40 * out["p_expansion"]
        + 0.10 * (1.0 - out["p_backwardation"])
    )

    # mean reversion favorisee par pinning + risk_off
    out["p_mean_reversion"] = (
        0.45 * out["p_pinning"]
        + 0.45 * out["p_risk_off"]
        + 0.10 * out["p_backwardation"]
    )

    # no-trade quand aucune classe ne domine
    score_gap = (out["p_continuation"] - out["p_mean_reversion"]).abs()
    out["p_no_trade"] = (1.0 - score_gap).clip(0.0, 1.0)

    # normalisation douce
    s = out[["p_continuation", "p_mean_reversion", "p_no_trade"]].sum(axis=1).replace(0, np.nan)
    out[["p_continuation", "p_mean_reversion", "p_no_trade"]] = out[
        ["p_continuation", "p_mean_reversion", "p_no_trade"]
    ].div(s, axis=0)

    out["confidence"] = out[["p_continuation", "p_mean_reversion", "p_no_trade"]].max(axis=1)
    return out


def _run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    # Base directionnelle simple: momentum daily (peut etre remplacee ensuite)
    out["mom_sign"] = np.sign(out["close"].pct_change()).replace(0, 1)

    edge = out["p_continuation"] - out["p_mean_reversion"]

    signal = np.where(
        edge > cfg.edge_threshold,
        out["mom_sign"],
        np.where(edge < -cfg.edge_threshold, -out["mom_sign"], 0),
    )

    # No-trade si p_no_trade domine franchement
    signal = np.where(out["p_no_trade"] > 0.45, 0, signal)

    out["signal"] = signal.astype(int)

    # Execution au close D -> close D+1
    out["fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0
    gross = out["signal"] * out["fwd_ret_1d"]

    # Cout fixe par trade (aller-retour approx)
    cost = ((cfg.fee_bps + cfg.slippage_bps) * 2.0) / 10000.0
    traded = out["signal"].abs() > 0
    out["net_ret"] = np.where(traded, gross - cost, 0.0)

    out = out.dropna(subset=["fwd_ret_1d", "net_ret"])
    return out


def _summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {
            "n_days": 0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    ret = bt["net_ret"].astype(float)
    traded = bt[bt["signal"] != 0]
    trade_ret = traded["net_ret"].astype(float)

    eq = (1.0 + ret).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak

    sharpe = 0.0
    if ret.std(ddof=0) > 0:
        sharpe = float((ret.mean() / ret.std(ddof=0)) * math.sqrt(252))

    return {
        "n_days": int(len(bt)),
        "n_trades": int((bt["signal"] != 0).sum()),
        "win_rate": float((trade_ret > 0).mean() * 100.0) if len(trade_ret) else 0.0,
        "avg_trade": float(trade_ret.mean() * 100.0) if len(trade_ret) else 0.0,
        "total_return": float((eq.iloc[-1] - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest HTF probabiliste (MVP)")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--edge-threshold", type=float, default=0.08)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--derivatives-mode", choices=["strict", "hybrid", "proxy"], default="hybrid")
    parser.add_argument("--market-source", choices=["auto", "db", "binance"], default="auto")
    args = parser.parse_args()

    cfg = BacktestConfig(
        asset=args.asset.upper(),
        min_days=args.min_days,
        edge_threshold=args.edge_threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        derivatives_mode=args.derivatives_mode,
        market_source=args.market_source,
    )

    df = _build_daily_frame(cfg.asset, cfg.derivatives_mode, cfg.market_source)

    if cfg.market_source == "auto" and len(df) < cfg.min_days:
        print(
            f"[info] Historique DB trop court ({len(df)} jours). "
            "Fallback automatique sur historique Binance."
        )
        df = _build_daily_frame(cfg.asset, cfg.derivatives_mode, "binance")

    if len(df) < cfg.min_days:
        raise RuntimeError(
            f"Historique insuffisant: {len(df)} jours exploitables, min requis={cfg.min_days}. "
            "Laissez tourner la collecte options plus longtemps."
        )

    model_df = _compute_probabilities(df)
    bt = _run_backtest(model_df, cfg)
    st = _summary(bt)

    trades_path = OUT_DIR / "htf_probabilistic_backtest_trades.csv"
    report_path = OUT_DIR / "htf_probabilistic_backtest_report.txt"

    bt.to_csv(trades_path, index=True)

    lines = [
        "HTF Probabilistic Backtest (MVP)",
        f"asset={cfg.asset}",
        f"period={bt.index.min().date()} -> {bt.index.max().date()}",
        f"n_days={st['n_days']}",
        f"n_trades={st['n_trades']}",
        f"win_rate={st['win_rate']:.2f}%",
        f"avg_trade={st['avg_trade']:+.4f}%",
        f"total_return={st['total_return']:+.2f}%",
        f"sharpe={st['sharpe']:+.3f}",
        f"max_drawdown={st['max_drawdown']:+.2f}%",
        f"edge_threshold={cfg.edge_threshold:.3f}",
        f"fee_bps={cfg.fee_bps:.2f}",
        f"slippage_bps={cfg.slippage_bps:.2f}",
        f"derivatives_mode={cfg.derivatives_mode}",
        f"market_source={cfg.market_source}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[ok] Trades CSV : {trades_path}")
    print(f"[ok] Report TXT : {report_path}")


if __name__ == "__main__":
    main()
