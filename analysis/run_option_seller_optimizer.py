from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.deribit import fetch_option_chain_snapshot


@dataclass
class SellOptionConfig:
    asset: str = "BTC"
    option_type: str = "put"  # put, call, both
    dte_min: int = 20
    dte_max: int = 45
    dte_target: int = 30
    strike_otm_min_pct: float = 0.03
    min_open_interest: float = 50.0
    min_volume_usd: float = 10000.0
    top_k: int = 10


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_itm(spot: float, strike: float, iv_pct: float, t_years: float, option_type: str) -> float:
    """Approximation Black-Scholes risque d'ITM a maturite (r=0)."""
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return np.nan

    sigma = max(1e-6, iv_pct / 100.0)
    den = sigma * math.sqrt(t_years)
    if den <= 0:
        return np.nan

    d2 = (math.log(spot / strike) - 0.5 * sigma * sigma * t_years) / den

    if option_type == "call":
        return float(_norm_cdf(d2))
    return float(_norm_cdf(-d2))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending).fillna(0.0)


def build_sell_candidates(cfg: SellOptionConfig) -> pd.DataFrame:
    chain = fetch_option_chain_snapshot(cfg.asset)
    if chain.empty:
        raise RuntimeError("Option chain vide.")

    df = chain.copy()

    if cfg.option_type in {"put", "call"}:
        df = df[df["option_type"] == cfg.option_type].copy()

    # Filtre DTE
    df = df[(df["dte_days"] >= cfg.dte_min) & (df["dte_days"] <= cfg.dte_max)].copy()
    if df.empty:
        raise RuntimeError("Aucune option dans la fenetre DTE demandee.")

    # Premium vendeur conservateur: bid si dispo sinon mark
    df["premium"] = df["bid_price"].fillna(df["mark_price"])

    # Moneyness OTM (positif = OTM)
    df["otm_pct"] = np.where(
        df["option_type"] == "call",
        (df["strike"] / df["index_price"]) - 1.0,
        1.0 - (df["strike"] / df["index_price"]),
    )

    # Filtres liquidite + moneyness
    df = df[df["otm_pct"] >= cfg.strike_otm_min_pct].copy()
    df = df[df["open_interest"].fillna(0.0) >= cfg.min_open_interest].copy()
    df = df[df["volume_usd"].fillna(0.0) >= cfg.min_volume_usd].copy()
    df = df[df["premium"].fillna(0.0) > 0].copy()
    df = df[df["mark_iv"].fillna(0.0) > 0].copy()

    if df.empty:
        raise RuntimeError("Aucun candidat apres filtres (DTE/OTM/liquidite/premium).")

    # Yield annualise en unite sous-jacent (Deribit options quotees en coin)
    df["annualized_yield"] = df["premium"] * (365.0 / df["dte_days"])

    # Probabilite ITM approchee et POP = 1 - P(ITM)
    t_years = df["dte_days"] / 365.0
    df["prob_itm"] = [
        _prob_itm(s, k, iv, t, tpe)
        for s, k, iv, t, tpe in zip(
            df["index_price"],
            df["strike"],
            df["mark_iv"],
            t_years,
            df["option_type"],
        )
    ]
    df["prob_otm"] = (1.0 - df["prob_itm"]).clip(lower=0.0, upper=1.0)

    # Distance au DTE cible (plus proche = mieux)
    df["dte_distance"] = (df["dte_days"] - float(cfg.dte_target)).abs()

    # Score multi-critere
    df["score_yield"] = _pct_rank(df["annualized_yield"], ascending=True)
    df["score_pop"] = _pct_rank(df["prob_otm"], ascending=True)
    df["score_otm"] = _pct_rank(df["otm_pct"], ascending=True)
    liq_proxy = (df["open_interest"].fillna(0.0) * df["volume_usd"].fillna(0.0)).replace(0.0, np.nan)
    df["score_liq"] = _pct_rank(np.log1p(liq_proxy.fillna(0.0)), ascending=True)
    df["score_dte_fit"] = 1.0 - _pct_rank(df["dte_distance"], ascending=True)

    df["sell_score"] = (
        0.35 * df["score_yield"]
        + 0.30 * df["score_pop"]
        + 0.15 * df["score_otm"]
        + 0.10 * df["score_liq"]
        + 0.10 * df["score_dte_fit"]
    )

    return df.sort_values("sell_score", ascending=False).reset_index(drop=True)


def _format_output(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    cols = [
        "instrument_name", "option_type", "dte_days", "index_price", "strike",
        "premium", "mark_iv", "otm_pct", "prob_otm", "annualized_yield",
        "open_interest", "volume_usd", "sell_score",
    ]
    out = df.head(top_k)[cols].copy()
    out["dte_days"] = out["dte_days"].round(1)
    out["index_price"] = out["index_price"].round(2)
    out["strike"] = out["strike"].round(2)
    out["premium"] = out["premium"].round(4)
    out["mark_iv"] = out["mark_iv"].round(2)
    out["otm_pct"] = (100 * out["otm_pct"]).round(2)
    out["prob_otm"] = (100 * out["prob_otm"]).round(1)
    out["annualized_yield"] = (100 * out["annualized_yield"]).round(2)
    out["sell_score"] = out["sell_score"].round(3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trouve les meilleurs strikes/options a vendre selon DTE + risque (DVOL/IV proxy)."
    )
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--type", type=str, default="put", choices=["put", "call", "both"])
    parser.add_argument("--dte-min", type=int, default=20)
    parser.add_argument("--dte-max", type=int, default=45)
    parser.add_argument("--dte-target", type=int, default=30)
    parser.add_argument("--otm-min", type=float, default=0.03, help="Min OTM percent en decimal (0.03 = 3%%)")
    parser.add_argument("--min-oi", type=float, default=50.0)
    parser.add_argument("--min-vol-usd", type=float, default=10000.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=str, default="db/options_sell_candidates.csv")
    args = parser.parse_args()

    cfg = SellOptionConfig(
        asset=args.asset.upper(),
        option_type=args.type,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        dte_target=args.dte_target,
        strike_otm_min_pct=args.otm_min,
        min_open_interest=args.min_oi,
        min_volume_usd=args.min_vol_usd,
        top_k=args.top,
    )

    candidates = build_sell_candidates(cfg)
    formatted = _format_output(candidates, cfg.top_k)

    print("\n=== Option Selling Optimizer ===")
    print(
        f"asset={cfg.asset} type={cfg.option_type} DTE=[{cfg.dte_min},{cfg.dte_max}] "
        f"target={cfg.dte_target} OTM>={cfg.strike_otm_min_pct:.1%}"
    )
    print("\nTop candidates:")
    print(formatted.to_string(index=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)
    print(f"\nSaved full candidates: {out_path}")


if __name__ == "__main__":
    main()
