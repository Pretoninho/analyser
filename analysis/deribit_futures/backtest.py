"""
analysis/deribit_futures/backtest.py — Hit ratio par edge signal Deribit.

Pour chaque edge (funding_reversion, carry_momentum, etc.) :
  - Seuil d'activation (score > threshold)
  - Forward returns à +4h et +24h (en barres selon le timeframe)
  - Stats : n_signals, hit_ratio, avg_ret_active, avg_ret_baseline, corr, lift

Edges directionnels (funding implicite une direction) :
  - carry_momentum   → direction = signe du funding_annualized        (+1)
  - carry_stress     → direction = -signe du funding_annualized       (-1 vs funding)
  - skew_panic       → direction = +1 (expect bounce from put panic)

Edges non-directionnels (anticipent un déplacement, pas une direction) :
  - funding_reversion, mark_dislocation, options_vol_premium, term_structure_kink
  → hit = |ret_fwd| > médiane de l'ensemble

Usage CLI:
  python analysis/run_deribit_backtest.py --days 90 --timeframe 1h
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import EdgeBuildConfig, build_deribit_edge_frame

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 90
    threshold: float = 0.05          # score minimum pour qu'un signal soit "actif"
    min_signals: int = 5             # nombre minimum de barres actives pour reporter
    horizons_hours: list[int] = field(default_factory=lambda: [4, 24])


# ── Méta par edge ─────────────────────────────────────────────────────────────

_EDGE_META: dict[str, dict] = {
    "edge_funding_reversion":   {"directional": False},
    "edge_carry_momentum":      {"directional": True,  "direction_col": "funding_annualized", "direction_sign": +1},
    "edge_carry_stress":        {"directional": True,  "direction_col": "funding_annualized", "direction_sign": -1},
    "edge_mark_dislocation":    {"directional": False},
    "edge_options_vol_premium": {"directional": False},
    "edge_skew_panic":          {"directional": True,  "direction_col": None, "direction_sign": +1},
    "edge_term_structure_kink": {"directional": False},
}

_ALL_EDGES = list(_EDGE_META.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 6:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = math.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom < 1e-12:
        return np.nan
    return float((xm * ym).sum() / denom)


def _forward_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    return (close.shift(-horizon_bars) / close - 1.0)


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_edge_backtest(
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Construit le frame de features Deribit puis évalue chaque edge signal.

    Returns
    -------
    results_df : DataFrame (edge, horizon_h, n_signals, hit_ratio, avg_ret_active,
                            avg_ret_baseline, corr, lift, note)
    context    : dict avec metadata (asset, timeframe, bars, backtest params, options snapshot)
    """
    cfg = config or BacktestConfig()

    feat_cfg = EdgeBuildConfig(
        asset=cfg.asset,
        timeframe=cfg.timeframe,
        days=cfg.days,
    )
    df, context = build_deribit_edge_frame(feat_cfg)
    bar_seconds = context["bar_seconds"]

    rows = []

    for horizon_h in cfg.horizons_hours:
        horizon_bars = max(1, int(round(horizon_h * 3600 / bar_seconds)))
        ret_fwd = _forward_return(df["close"], horizon_bars)

        for edge in _ALL_EDGES:
            if edge not in df.columns:
                continue

            score = df[edge].fillna(0.0)
            meta = _EDGE_META[edge]
            active_mask = (score > cfg.threshold) & score.notna() & ret_fwd.notna()
            n_signals = int(active_mask.sum())

            corr = _pearson_corr(score.values, ret_fwd.values)

            if n_signals < cfg.min_signals:
                rows.append({
                    "edge":             edge,
                    "horizon_h":        horizon_h,
                    "n_signals":        n_signals,
                    "hit_ratio":        None,
                    "avg_ret_active":   None,
                    "avg_ret_baseline": None,
                    "corr":             round(float(corr), 4) if np.isfinite(corr) else None,
                    "lift":             None,
                    "note":             "insufficient_signals",
                })
                continue

            active_ret = ret_fwd[active_mask].values
            all_ret = ret_fwd.dropna().values

            if meta["directional"]:
                direction_col = meta.get("direction_col")
                direction_sign = meta.get("direction_sign", 1)

                if direction_col and direction_col in df.columns:
                    dir_series = np.sign(df[direction_col].fillna(0.0)).values
                    expected_dir = (dir_series * direction_sign)[active_mask.values]
                else:
                    # skew_panic: always expect +1
                    expected_dir = np.full(n_signals, float(direction_sign))

                valid_mask = expected_dir != 0
                n_valid = valid_mask.sum()
                if n_valid == 0:
                    hit_ratio = None
                else:
                    hits = ((expected_dir[valid_mask] * active_ret[valid_mask]) > 0).sum()
                    hit_ratio = round(float(hits / n_valid), 4)

                # avg return in expected direction (signed P&L proxy)
                signed_ret = expected_dir * active_ret
                avg_ret_active = float(np.nanmean(signed_ret))

                # baseline: avg signed return across all bars
                if direction_col and direction_col in df.columns:
                    all_dir = np.sign(df[direction_col].fillna(0.0)).values * direction_sign
                else:
                    all_dir = np.ones(len(all_ret))
                all_valid = ret_fwd.dropna()
                bl_idx = all_valid.index
                if direction_col and direction_col in df.columns:
                    bl_dir = np.sign(df.loc[bl_idx, direction_col].fillna(0.0)).values * direction_sign
                else:
                    bl_dir = np.ones(len(all_valid))
                avg_ret_baseline = float(np.nanmean(bl_dir * all_valid.values))

            else:
                # Non-directional: edge predicts larger-than-median move
                abs_active = np.abs(active_ret)
                abs_all = np.abs(all_ret)
                median_vol = float(np.median(abs_all)) if len(abs_all) > 0 else 0.0

                hits = int((abs_active > median_vol).sum())
                hit_ratio = round(float(hits / n_signals), 4)

                avg_ret_active = float(np.nanmean(abs_active))
                avg_ret_baseline = float(np.nanmean(abs_all)) if len(abs_all) > 0 else None

            lift = None
            if avg_ret_baseline is not None and abs(avg_ret_baseline) > 1e-9:
                lift = round(float(avg_ret_active / avg_ret_baseline), 3)

            rows.append({
                "edge":             edge,
                "horizon_h":        horizon_h,
                "n_signals":        n_signals,
                "hit_ratio":        hit_ratio,
                "avg_ret_active":   round(float(avg_ret_active), 6),
                "avg_ret_baseline": round(float(avg_ret_baseline), 6) if avg_ret_baseline is not None else None,
                "corr":             round(float(corr), 4) if np.isfinite(corr) else None,
                "lift":             lift,
                "note":             "ok",
            })

    results_df = pd.DataFrame(rows)
    context["backtest"] = {
        "threshold":  cfg.threshold,
        "horizons":   cfg.horizons_hours,
        "total_bars": int(len(df)),
    }
    return results_df, context


# ── Affichage console ─────────────────────────────────────────────────────────

def _print_results(results_df: pd.DataFrame, context: dict) -> None:
    print(f"\n=== Deribit Edge Backtest | {context['asset']} {context['timeframe']} | {context['days']}j ===")
    print(f"bars={context['bars']}  threshold={context['backtest']['threshold']}")

    for horizon_h in sorted(results_df["horizon_h"].unique()):
        sub = results_df[results_df["horizon_h"] == horizon_h].copy()
        sub = sub.sort_values("hit_ratio", ascending=False)
        print(f"\n--- Horizon +{horizon_h}h ---")
        print(f"{'edge':<28} {'n':>5} {'hit_ratio':>10} {'avg_ret':>10} {'baseline':>10} {'corr':>8} {'lift':>6} {'note'}")
        for _, r in sub.iterrows():
            hr = f"{r['hit_ratio']:.4f}" if r["hit_ratio"] is not None else "  N/A  "
            ar = f"{r['avg_ret_active']:+.5f}" if r["avg_ret_active"] is not None else "   N/A   "
            bl = f"{r['avg_ret_baseline']:+.5f}" if r["avg_ret_baseline"] is not None else "   N/A   "
            co = f"{r['corr']:+.4f}" if r["corr"] is not None else "  N/A "
            li = f"{r['lift']:.3f}" if r["lift"] is not None else " N/A"
            print(f"  {r['edge']:<26} {r['n_signals']:>5} {hr:>10} {ar:>10} {bl:>10} {co:>8} {li:>6}  {r['note']}")
