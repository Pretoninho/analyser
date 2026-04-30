from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import EdgeBuildConfig, build_deribit_edge_frame


@dataclass
class SignalConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 90
    min_net_score: float = 0.03


def _edge_value(latest, key: str) -> float:
    try:
        value = float(latest.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _suggest_contract(action: str, horizon: str, funding_ann: float, top_driver: str) -> dict:
    if action in {"FLAT", "WATCH"}:
        return {
            "instrument": "NONE",
            "tenor": "NONE",
            "why": "Pas de signal directionnel assez fort pour proposer une echeance.",
        }

    if top_driver in {"carry_momentum", "carry_stress"} or abs(funding_ann) >= 0.08:
        return {
            "instrument": "PERPETUAL",
            "tenor": "PERP",
            "why": "Signal domine par le carry/funding, le perp est le plus direct.",
        }

    if horizon == "24h":
        return {
            "instrument": "DATED_FUTURE",
            "tenor": "1M",
            "why": "Horizon swing court, preferer un future mensuel pour lisser le bruit intraday.",
        }

    if horizon == "4h-24h":
        return {
            "instrument": "DATED_FUTURE",
            "tenor": "1W",
            "why": "Horizon intermediaire, echeance hebdo adaptee a une conviction tactique.",
        }

    return {
        "instrument": "PERPETUAL",
        "tenor": "PERP",
        "why": "Signal court terme orientee execution rapide.",
    }


def build_deribit_signal(config: SignalConfig | None = None) -> dict:
    cfg = config or SignalConfig()

    edge_cfg = EdgeBuildConfig(asset=cfg.asset, timeframe=cfg.timeframe, days=cfg.days)
    df, context = build_deribit_edge_frame(edge_cfg)
    latest = df.iloc[-1]

    funding_ann = float(latest.get("funding_annualized") or 0.0)
    funding_sign = 1 if funding_ann > 0 else -1 if funding_ann < 0 else 0

    mark_price = context.get("snapshot", {}).get("mark_price")
    index_price = context.get("snapshot", {}).get("index_price")
    mark_bias = 0
    if mark_price is not None and index_price is not None:
        try:
            mark_bias = -1 if float(mark_price) > float(index_price) else 1
        except (TypeError, ValueError):
            mark_bias = 0

    e_funding_reversion = _edge_value(latest, "edge_funding_reversion")
    e_carry_momentum = _edge_value(latest, "edge_carry_momentum")
    e_carry_stress = _edge_value(latest, "edge_carry_stress")
    e_mark_dislocation = _edge_value(latest, "edge_mark_dislocation")
    e_options_vol_premium = _edge_value(latest, "edge_options_vol_premium")
    e_skew_panic = _edge_value(latest, "edge_skew_panic")
    e_term_kink = _edge_value(latest, "edge_term_structure_kink")
    e_total = _edge_value(latest, "edge_total")

    directional_contrib = {
        "funding_reversion": (-funding_sign) * e_funding_reversion,
        "carry_momentum": funding_sign * e_carry_momentum,
        "carry_stress": (-funding_sign) * e_carry_stress,
        "mark_dislocation": mark_bias * e_mark_dislocation,
        "skew_panic": 1.0 * e_skew_panic,
    }

    long_score = sum(max(0.0, v) for v in directional_contrib.values())
    short_score = sum(max(0.0, -v) for v in directional_contrib.values())
    net_score = long_score - short_score

    non_directional = {
        "options_vol_premium": e_options_vol_premium,
        "term_structure_kink": e_term_kink,
    }
    regime_score = float(np.mean(list(non_directional.values()))) if non_directional else 0.0

    if net_score >= cfg.min_net_score:
        action = "LONG"
    elif net_score <= -cfg.min_net_score:
        action = "SHORT"
    else:
        action = "FLAT"

    if action == "FLAT" and e_total >= 0.08 and regime_score >= 0.15:
        action = "WATCH"

    confidence = min(1.0, abs(net_score) * 2.2 + e_total * 0.9 + regime_score * 0.4)

    ordered_drivers = sorted(
        [
            ("funding_reversion", abs(directional_contrib["funding_reversion"])),
            ("carry_momentum", abs(directional_contrib["carry_momentum"])),
            ("carry_stress", abs(directional_contrib["carry_stress"])),
            ("mark_dislocation", abs(directional_contrib["mark_dislocation"])),
            ("skew_panic", abs(directional_contrib["skew_panic"])),
            ("options_vol_premium", non_directional["options_vol_premium"]),
            ("term_structure_kink", non_directional["term_structure_kink"]),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    top_drivers = [{"name": name, "score": round(score, 4)} for name, score in ordered_drivers[:4]]

    horizon = "4h"
    if top_drivers:
        top_name = top_drivers[0]["name"]
        if top_name in {"carry_momentum", "carry_stress", "options_vol_premium"}:
            horizon = "24h"
        elif top_name in {"term_structure_kink", "skew_panic"}:
            horizon = "4h-24h"

    contract_suggestion = _suggest_contract(action, horizon, funding_ann, top_drivers[0]["name"] if top_drivers else "")

    return {
        "asset": context["asset"],
        "timeframe": context["timeframe"],
        "days": context["days"],
        "latest_ts": latest["timestamp"].isoformat() if latest.get("timestamp") is not None else None,
        "close": float(latest.get("close") or 0.0),
        "funding_annualized": funding_ann,
        "realized_vol_annual": float(latest.get("realized_vol_annual") or 0.0),
        "signal": {
            "action": action,
            "horizon": horizon,
            "confidence": round(confidence, 4),
            "net_score": round(net_score, 4),
            "long_score": round(long_score, 4),
            "short_score": round(short_score, 4),
            "edge_total": round(e_total, 4),
            "contract": contract_suggestion,
        },
        "edges": {
            "edge_funding_reversion": round(e_funding_reversion, 4),
            "edge_carry_momentum": round(e_carry_momentum, 4),
            "edge_carry_stress": round(e_carry_stress, 4),
            "edge_mark_dislocation": round(e_mark_dislocation, 4),
            "edge_options_vol_premium": round(e_options_vol_premium, 4),
            "edge_skew_panic": round(e_skew_panic, 4),
            "edge_term_structure_kink": round(e_term_kink, 4),
            "edge_total": round(e_total, 4),
        },
        "drivers": top_drivers,
        "snapshot": context.get("snapshot", {}),
        "options": context.get("options_snapshot", {}),
    }


def format_discord_signal(signal: dict) -> str:
    s = signal.get("signal", {})
    action = str(s.get("action", "FLAT"))
    horizon = str(s.get("horizon", "4h"))
    contract = s.get("contract", {}) if isinstance(s.get("contract"), dict) else {}
    confidence = float(s.get("confidence", 0.0))
    edge_total = float(s.get("edge_total", 0.0))
    close = float(signal.get("close", 0.0))
    ts = str(signal.get("latest_ts", ""))
    funding_ann = float(signal.get("funding_annualized", 0.0))

    lines = [
        "**Deribit Futures Signal**",
        f"Action   : {action}",
        f"Horizon  : {horizon}",
        f"Tenor    : {contract.get('tenor', 'N/A')}",
        f"Contract : {contract.get('instrument', 'N/A')}",
        f"Confidence: {confidence:.2f}",
        f"EdgeTotal: {edge_total:.4f}",
        f"Price    : {close:,.2f}",
        f"FundingA : {funding_ann:+.4f}",
        f"Timestamp: {ts}",
        "Top drivers:",
    ]

    drivers = signal.get("drivers", [])
    for d in drivers[:4]:
        name = d.get("name", "unknown")
        score = float(d.get("score", 0.0))
        lines.append(f"- {name}: {score:.4f}")

    why = str(contract.get("why", "")).strip()
    if why:
        lines.append(f"Rationale: {why}")

    return "\n".join(lines)