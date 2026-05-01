from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.deribit import fetch_dvol_history


@dataclass
class DvolDetectorConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 60
    zscore_window_days: int = 14
    shock_threshold_z: float = 2.0
    crush_threshold_z: float = -2.0
    roc_up_threshold: float = 0.08
    roc_down_threshold: float = -0.08


def _infer_bars_per_day(ts: pd.Series) -> int:
    if len(ts) < 4:
        return 24
    diffs = ts.diff().dropna()
    if diffs.empty:
        return 24
    med = int(diffs.median())
    if med <= 0:
        return 24
    return max(1, int(round(86400 / med)))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_p = max(5, window // 4)
    mean = series.rolling(window=window, min_periods=min_p).mean()
    std = series.rolling(window=window, min_periods=min_p).std().replace(0.0, np.nan)
    return (series - mean) / std


def build_dvol_frame(config: DvolDetectorConfig | None = None) -> pd.DataFrame:
    cfg = config or DvolDetectorConfig()

    df = fetch_dvol_history(asset=cfg.asset, timeframe=cfg.timeframe, days=cfg.days)
    if df.empty:
        raise RuntimeError("No DVOL data returned from Deribit.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["ts"], unit="s", utc=True)

    close = out["dvol_close"].astype(float)
    bars_per_day = _infer_bars_per_day(out["ts"])
    lookback_24h = max(1, bars_per_day)
    z_window = max(10, cfg.zscore_window_days * bars_per_day)

    out["dvol_ret_1"] = close.pct_change()
    out["dvol_roc_24h"] = close.pct_change(lookback_24h)
    out["dvol_z"] = _rolling_zscore(close, z_window)
    out["dvol_ret_std_24h"] = out["dvol_ret_1"].rolling(lookback_24h).std()

    # Détection de régime de variation DVOL
    shock_cond = (out["dvol_z"] >= cfg.shock_threshold_z) | (out["dvol_roc_24h"] >= cfg.roc_up_threshold)
    crush_cond = (out["dvol_z"] <= cfg.crush_threshold_z) | (out["dvol_roc_24h"] <= cfg.roc_down_threshold)

    out["dvol_state"] = "NEUTRAL"
    out.loc[shock_cond, "dvol_state"] = "VOL_SHOCK_UP"
    out.loc[crush_cond, "dvol_state"] = "VOL_CRUSH_DOWN"

    # Score d'intensité normalisé [0,1]
    z_component = (out["dvol_z"].abs().clip(upper=4.0) / 4.0).fillna(0.0)
    roc_component = (out["dvol_roc_24h"].abs().clip(upper=0.20) / 0.20).fillna(0.0)
    out["dvol_intensity"] = (0.6 * z_component + 0.4 * roc_component).clip(0.0, 1.0)

    return out


def detect_dvol_variation(config: DvolDetectorConfig | None = None) -> dict:
    cfg = config or DvolDetectorConfig()
    frame = build_dvol_frame(cfg)
    latest = frame.iloc[-1]

    state = str(latest.get("dvol_state", "NEUTRAL"))
    z = float(latest.get("dvol_z") or 0.0)
    roc_24h = float(latest.get("dvol_roc_24h") or 0.0)
    intensity = float(latest.get("dvol_intensity") or 0.0)

    # Direction simplifiée pour exploitation downstream
    if state == "VOL_SHOCK_UP":
        regime = "RISK_OFF"
    elif state == "VOL_CRUSH_DOWN":
        regime = "RISK_ON"
    else:
        regime = "BALANCED"

    return {
        "asset": cfg.asset,
        "timeframe": cfg.timeframe,
        "days": cfg.days,
        "latest_ts": latest["timestamp"].isoformat() if latest.get("timestamp") is not None else None,
        "dvol_close": float(latest.get("dvol_close") or 0.0),
        "dvol_z": round(z, 4),
        "dvol_roc_24h": round(roc_24h, 4),
        "dvol_ret_std_24h": round(float(latest.get("dvol_ret_std_24h") or 0.0), 6),
        "dvol_state": state,
        "risk_regime": regime,
        "intensity": round(intensity, 4),
        "bars": int(len(frame)),
        "frame": frame,
    }


def format_dvol_signal(payload: dict) -> str:
    state = payload.get("dvol_state", "NEUTRAL")
    reg = payload.get("risk_regime", "BALANCED")
    dvol = float(payload.get("dvol_close", 0.0))
    z = float(payload.get("dvol_z", 0.0))
    roc = float(payload.get("dvol_roc_24h", 0.0))
    intensity = float(payload.get("intensity", 0.0))
    ts = payload.get("latest_ts", "")

    icon = {
        "VOL_SHOCK_UP": "🔺",
        "VOL_CRUSH_DOWN": "🔻",
        "NEUTRAL": "⚪",
    }.get(state, "⚪")

    lines = [
        "**Deribit DVOL Detector**",
        f"State     : {icon} {state}",
        f"Regime    : {reg}",
        f"DVOL      : {dvol:.2f}",
        f"Z-Score   : {z:+.2f}",
        f"ROC 24h   : {roc:+.2%}",
        f"Intensity : {intensity:.2f}",
        f"Timestamp : {ts}",
    ]
    return "\n".join(lines)
