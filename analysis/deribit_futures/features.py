from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from config import SOURCES
from data.deribit import (
    DERIBIT_INSTRUMENTS,
    fetch_funding_rate_historical,
    fetch_ohlcv_historical,
    fetch_options_analytics,
)


@dataclass
class EdgeBuildConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 60
    funding_days: int = 90
    zscore_lookback_days: int = 14


def _safe_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(3, window // 4)).mean()
    std = series.rolling(window=window, min_periods=max(3, window // 4)).std().replace(0.0, np.nan)
    return (series - mean) / std


def _fetch_book_summary_snapshot(asset: str) -> dict:
    instrument = DERIBIT_INSTRUMENTS.get(asset.upper())
    if instrument is None:
        return {}

    endpoint = f"{SOURCES['deribit_base_url']}/get_book_summary_by_instrument"
    params = {"instrument_name": instrument}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return {}

    result = payload.get("result", [])
    if not result:
        return {}

    item = result[0]
    return {
        "snapshot_ts": int(time.time()),
        "instrument_name": instrument,
        "mark_price": _safe_float(item.get("mark_price")),
        "index_price": _safe_float(item.get("index_price")),
        "open_interest": _safe_float(item.get("open_interest")),
        "current_funding": _safe_float(item.get("current_funding")),
        "funding_8h": _safe_float(item.get("funding_8h")),
        "best_bid_price": _safe_float(item.get("bid_price")),
        "best_ask_price": _safe_float(item.get("ask_price")),
    }


def _infer_bar_seconds(ts: pd.Series) -> int:
    if len(ts) < 4:
        return 3600
    diffs = ts.diff().dropna()
    if diffs.empty:
        return 3600
    median_dt = int(diffs.median())
    return max(60, median_dt)


def _compute_edge_scores(df: pd.DataFrame, snapshot: dict, options: dict) -> pd.DataFrame:
    out = df.copy()

    out["edge_funding_reversion"] = (
        out["funding_extreme_score"].fillna(0.0)
        * out["price_stretch_score"].fillna(0.0)
        * out["funding_price_conflict"].fillna(0.0)
    )

    out["edge_carry_momentum"] = (
        out["funding_directional_score"].fillna(0.0)
        * out["trend_alignment_score"].fillna(0.0)
    )

    out["edge_carry_stress"] = out["funding_extreme_score"].fillna(0.0) * (1.0 - out["trend_alignment_score"].fillna(0.0))

    mark_price = _safe_float(snapshot.get("mark_price"))
    index_price = _safe_float(snapshot.get("index_price"))
    if np.isfinite(mark_price) and np.isfinite(index_price) and index_price > 0:
        dislocation = abs(mark_price - index_price) / index_price
        out["edge_mark_dislocation"] = min(1.0, dislocation / 0.003)
    else:
        out["edge_mark_dislocation"] = 0.0

    iv_atm = _safe_float(options.get("iv_atm"))
    if np.isfinite(iv_atm):
        vol_premium = (iv_atm / 100.0) - out["realized_vol_annual"].fillna(0.0)
        out["edge_options_vol_premium"] = vol_premium.clip(lower=0.0, upper=0.35) / 0.35
    else:
        out["edge_options_vol_premium"] = 0.0

    skew_25d = _safe_float(options.get("iv_skew_25d"))
    pcr = _safe_float(options.get("put_call_ratio"))
    skew_component = 0.0 if not np.isfinite(skew_25d) else min(1.0, max(0.0, skew_25d / 6.0))
    pcr_component = 0.0 if not np.isfinite(pcr) else min(1.0, max(0.0, (pcr - 0.9) / 0.8))
    out["edge_skew_panic"] = 0.5 * skew_component + 0.5 * pcr_component

    term_1w = _safe_float(options.get("term_1w"))
    term_3m = _safe_float(options.get("term_3m"))
    if np.isfinite(term_1w) and np.isfinite(term_3m):
        term_gap = abs(term_1w - term_3m)
        out["edge_term_structure_kink"] = min(1.0, term_gap / 12.0)
    else:
        out["edge_term_structure_kink"] = 0.0

    out["edge_total"] = (
        0.24 * out["edge_funding_reversion"]
        + 0.14 * out["edge_carry_momentum"]
        + 0.14 * out["edge_carry_stress"]
        + 0.12 * out["edge_mark_dislocation"]
        + 0.12 * out["edge_options_vol_premium"]
        + 0.12 * out["edge_skew_panic"]
        + 0.12 * out["edge_term_structure_kink"]
    ).fillna(0.0)

    return out


def build_deribit_edge_frame(config: EdgeBuildConfig | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = config or EdgeBuildConfig()

    ohlcv = fetch_ohlcv_historical(cfg.asset, cfg.timeframe, days=cfg.days)
    if ohlcv.empty:
        raise RuntimeError("No Deribit OHLCV data returned.")

    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)

    funding = fetch_funding_rate_historical(cfg.asset, days=max(cfg.days, cfg.funding_days))
    if funding.empty:
        ohlcv["funding_rate"] = np.nan
    else:
        funding = funding[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)
        ohlcv = pd.merge_asof(
            ohlcv,
            funding,
            on="ts",
            direction="backward",
            allow_exact_matches=True,
        )

    ohlcv["timestamp"] = pd.to_datetime(ohlcv["ts"], unit="s", utc=True)
    bar_seconds = _infer_bar_seconds(ohlcv["ts"])
    bars_per_day = max(1, int(round(86400 / bar_seconds)))

    ohlcv["ret_1"] = ohlcv["close"].pct_change().fillna(0.0)
    ohlcv["ret_log"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1)).fillna(0.0)

    rv_window = max(5, bars_per_day)
    z_window = max(10, cfg.zscore_lookback_days * bars_per_day)

    annual_factor = math.sqrt(365 * bars_per_day)
    ohlcv["realized_vol_annual"] = ohlcv["ret_log"].rolling(rv_window).std() * annual_factor
    ohlcv["price_stretch_z"] = _rolling_zscore(ohlcv["ret_1"], z_window)

    if "funding_rate" in ohlcv.columns:
        ohlcv["funding_rate"] = ohlcv["funding_rate"].astype(float)
        ohlcv["funding_annualized"] = ohlcv["funding_rate"] * 3.0 * 365.0
        ohlcv["funding_z"] = _rolling_zscore(ohlcv["funding_annualized"], z_window)
    else:
        ohlcv["funding_rate"] = np.nan
        ohlcv["funding_annualized"] = np.nan
        ohlcv["funding_z"] = np.nan

    ohlcv["trend_ema_fast"] = ohlcv["close"].ewm(span=max(5, bars_per_day // 2), adjust=False).mean()
    ohlcv["trend_ema_slow"] = ohlcv["close"].ewm(span=max(12, bars_per_day * 3), adjust=False).mean()

    trend_dir = np.sign(ohlcv["trend_ema_fast"] - ohlcv["trend_ema_slow"]).replace(0.0, np.nan)
    funding_dir = np.sign(ohlcv["funding_annualized"]).replace(0.0, np.nan)
    return_dir = np.sign(ohlcv["ret_1"]).replace(0.0, np.nan)

    valid_price_funding = funding_dir.notna() & return_dir.notna()
    valid_trend_funding = funding_dir.notna() & trend_dir.notna()

    ohlcv["funding_extreme_score"] = (ohlcv["funding_z"].abs().clip(upper=3.0) / 3.0).fillna(0.0)
    ohlcv["price_stretch_score"] = (ohlcv["price_stretch_z"].abs().clip(upper=3.0) / 3.0).fillna(0.0)
    ohlcv["funding_price_conflict"] = ((funding_dir != return_dir) & valid_price_funding).astype(float)
    ohlcv["funding_directional_score"] = (ohlcv["funding_annualized"].abs().clip(upper=0.45) / 0.45).fillna(0.0)
    ohlcv["trend_alignment_score"] = ((funding_dir == trend_dir) & valid_trend_funding).astype(float)

    snapshot = _fetch_book_summary_snapshot(cfg.asset)
    options = fetch_options_analytics(cfg.asset) or {}
    enriched = _compute_edge_scores(ohlcv, snapshot=snapshot, options=options)

    context = {
        "asset": cfg.asset,
        "timeframe": cfg.timeframe,
        "days": cfg.days,
        "bars": int(len(enriched)),
        "bar_seconds": bar_seconds,
        "bars_per_day": bars_per_day,
        "snapshot": snapshot,
        "options_snapshot": options,
    }

    return enriched, context
