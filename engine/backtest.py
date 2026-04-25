"""
engine/backtest.py — Moteur de simulation de trading réaliste + backtest RL Pi*.

Hypothèses :
    - Exécution au prix d'ouverture de la bougie suivant le signal (no look-ahead)
    - Frais Kraken : 0.26% taker sur chaque trade (entrée + sortie)
    - Slippage : 0.05% défavorable sur chaque exécution
    - Sizing : fraction fixe du capital (défaut 100%, pas de levier)
    - Stop loss : ATR × atr_sl_mult (optionnel)
    - Take profit : ATR × atr_tp_mult (optionnel)
    - Une seule position ouverte à la fois
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ──────────────────────────────────────────────

@dataclass
class BacktestConfig:
    capital:        float = 10_000.0  # capital initial en USD
    fee_rate:       float = 0.0026    # 0.26% taker Kraken
    slippage:       float = 0.0005    # 0.05% par exécution
    position_pct:   float = 1.0       # fraction du capital engagée (1.0 = 100%)
    atr_sl_mult:    Optional[float] = 2.0   # stop loss = ATR × mult (None = désactivé)
    atr_tp_mult:    Optional[float] = 4.0   # take profit = ATR × mult (None = désactivé)


# ── Résultats ──────────────────────────────────────────────────

@dataclass
class Trade:
    direction:        int     # +1 long / -1 short
    entry_ts:         object
    entry_price:      float
    exit_ts:          object  = None
    exit_price:       float   = None
    qty:              float   = 0.0
    capital_before:   float   = 0.0
    capital_after:    float   = 0.0
    pnl:              float   = 0.0
    pnl_pct:          float   = 0.0
    fees:             float   = 0.0
    slippage_cost:    float   = 0.0
    exit_reason:      str     = ""
    signal_label_entry: str   = ""
    signal_label_exit:  str   = ""


@dataclass
class BacktestResult:
    signal_name:    str
    asset:          str
    timeframe:      str
    params:         dict
    config:         BacktestConfig
    trades:         list = field(default_factory=list)
    equity_curve:   list = field(default_factory=list)  # [(timestamp, capital)]

    # Métriques calculées après run
    capital_start:   float = 0.0
    capital_end:     float = 0.0
    total_return_pct: float = 0.0
    sharpe:          float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    total_trades:    int   = 0
    winning_trades:  int   = 0
    losing_trades:   int   = 0
    avg_win_pct:     float = 0.0
    avg_loss_pct:    float = 0.0


# ── Moteur ─────────────────────────────────────────────────────

def run_backtest(
    signal,
    df: pd.DataFrame,
    asset: str,
    timeframe: str,
    config: BacktestConfig = None,
) -> BacktestResult:
    """
    Exécute un backtest complet sur un DataFrame enrichi de signaux.

    df doit contenir les colonnes produites par signal.generate() :
        signal, signal_raw, signal_label
    ainsi que : timestamp, open, close, atr (optionnel pour SL/TP)
    """
    if config is None:
        config = BacktestConfig()

    result = BacktestResult(
        signal_name = signal.name,
        asset       = asset,
        timeframe   = timeframe,
        params      = signal.params.copy(),
        config      = config,
        capital_start = config.capital,
    )

    capital   = config.capital
    position  = None   # Trade en cours
    trades    = []
    equity    = []

    rows = df.reset_index(drop=True)

    for i, row in rows.iterrows():
        ts    = row["timestamp"]
        sig   = int(row.get("signal", 0))
        label = str(row.get("signal_label", ""))
        atr   = row.get("atr", None)

        # Prix d'exécution = open de la bougie courante
        # (le signal est généré sur la bougie précédente → pas de look-ahead)
        exec_price = row["open"]

        # ── Vérification SL / TP si position ouverte ──────────
        if position is not None and atr and not pd.isna(atr):
            sl_mult = config.atr_sl_mult
            tp_mult = config.atr_tp_mult

            if position.direction == 1:
                sl_price = position.entry_price - atr * sl_mult if sl_mult else None
                tp_price = position.entry_price + atr * tp_mult if tp_mult else None
                low_price  = row.get("low",  exec_price)
                high_price = row.get("high", exec_price)

                if sl_price and low_price <= sl_price:
                    capital = _close_position(position, sl_price, capital, config, "STOP_LOSS", label, trades, ts)
                    position = None
                    equity.append((ts, capital))
                    continue
                if tp_price and high_price >= tp_price:
                    capital = _close_position(position, tp_price, capital, config, "TAKE_PROFIT", label, trades, ts)
                    position = None
                    equity.append((ts, capital))
                    continue

            elif position.direction == -1:
                sl_price = position.entry_price + atr * sl_mult if sl_mult else None
                tp_price = position.entry_price - atr * tp_mult if tp_mult else None
                high_price = row.get("high", exec_price)
                low_price  = row.get("low",  exec_price)

                if sl_price and high_price >= sl_price:
                    capital = _close_position(position, sl_price, capital, config, "STOP_LOSS", label, trades, ts)
                    position = None
                    equity.append((ts, capital))
                    continue
                if tp_price and low_price <= tp_price:
                    capital = _close_position(position, tp_price, capital, config, "TAKE_PROFIT", label, trades, ts)
                    position = None
                    equity.append((ts, capital))
                    continue

        # ── Gestion des signaux ────────────────────────────────

        # Fermeture de position existante sur signal contraire ou neutre
        if position is not None:
            should_close = (
                sig == 0 or
                (position.direction == 1  and sig == -1) or
                (position.direction == -1 and sig ==  1)
            )
            if should_close:
                capital = _close_position(position, exec_price, capital, config, "SIGNAL", label, trades, ts)
                position = None

        # Ouverture d'une nouvelle position
        if position is None and sig != 0:
            position = _open_position(sig, exec_price, capital, config, label, ts)

        equity.append((ts, capital))

    # Clôture forcée en fin de période
    if position is not None and not rows.empty:
        last = rows.iloc[-1]
        capital = _close_position(
            position, last["close"], capital, config, "END_OF_PERIOD",
            "FIN DE BACKTEST", trades, last["timestamp"]
        )

    result.trades       = trades
    result.equity_curve = equity
    result.capital_end  = capital
    result = _compute_metrics(result)
    return result


# ── Helpers ────────────────────────────────────────────────────

def _apply_slippage(price: float, direction: int, config: BacktestConfig, opening: bool) -> float:
    """Applique un slippage défavorable."""
    slip = config.slippage
    if opening:
        return price * (1 + slip * direction)
    else:
        return price * (1 - slip * direction)


def _open_position(direction: int, price: float, capital: float,
                   config: BacktestConfig, label: str, ts) -> Trade:
    exec_price = _apply_slippage(price, direction, config, opening=True)
    notional   = capital * config.position_pct
    fees       = notional * config.fee_rate
    qty        = (notional - fees) / exec_price

    t = Trade(
        direction    = direction,
        entry_ts     = ts,
        entry_price  = exec_price,
        qty          = qty,
        capital_before = capital,
        fees         = fees,
        slippage_cost = abs(exec_price - price) * qty,
        signal_label_entry = label,
    )
    return t


def _close_position(position: Trade, price: float, capital: float,
                    config: BacktestConfig, reason: str, label: str,
                    trades: list, ts=None) -> float:
    exec_price    = _apply_slippage(price, position.direction, config, opening=False)
    exit_notional = position.qty * exec_price
    exit_fees     = exit_notional * config.fee_rate

    if position.direction == 1:
        gross_pnl = (exec_price - position.entry_price) * position.qty
    else:
        gross_pnl = (position.entry_price - exec_price) * position.qty

    net_pnl     = gross_pnl - exit_fees
    new_capital = capital + net_pnl

    position.exit_ts          = ts
    position.exit_price       = exec_price
    position.capital_after    = new_capital
    position.pnl              = round(net_pnl, 4)
    position.pnl_pct          = round(net_pnl / position.capital_before * 100, 4)
    position.fees             += exit_fees
    position.exit_reason      = reason
    position.signal_label_exit = label

    trades.append(position)
    return new_capital


def _compute_metrics(result: BacktestResult) -> BacktestResult:
    trades = result.trades

    if not trades:
        return result

    result.total_trades   = len(trades)
    winning = [t for t in trades if t.pnl > 0]
    losing  = [t for t in trades if t.pnl <= 0]

    result.winning_trades = len(winning)
    result.losing_trades  = len(losing)
    result.win_rate       = round(len(winning) / len(trades) * 100, 2)

    gross_win  = sum(t.pnl for t in winning)
    gross_loss = abs(sum(t.pnl for t in losing))
    result.profit_factor  = round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf")

    result.avg_win_pct  = round(np.mean([t.pnl_pct for t in winning]), 4) if winning else 0.0
    result.avg_loss_pct = round(np.mean([t.pnl_pct for t in losing]),  4) if losing  else 0.0

    result.total_return_pct = round(
        (result.capital_end - result.capital_start) / result.capital_start * 100, 4
    )

    # Sharpe annualisé depuis les rendements des trades
    rets = [t.pnl_pct / 100 for t in trades]
    if len(rets) > 1 and np.std(rets) > 0:
        result.sharpe = round(np.mean(rets) / np.std(rets) * np.sqrt(252), 3)

    # Max drawdown depuis la courbe equity
    if result.equity_curve:
        caps   = [c for _, c in result.equity_curve]
        peak   = caps[0]
        max_dd = 0.0
        for c in caps:
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = round(max_dd, 4)

    return result


# ═══════════════════════════════════════════════════════════════
# Backtest RL (Pi* walk-forward)
# ═══════════════════════════════════════════════════════════════

_RL_SESSION_LABELS = {0: "ASIA", 1: "LONDON", 2: "OVERLAP", 3: "NY"}
_ACTION_TO_DIR     = {0: 0, 1: 1, 2: -1}


@dataclass
class RLTrade:
    session:     int
    state_id:    int
    direction:   int        # +1=LONG, -1=SHORT
    entry_ts:    object
    entry_price: float
    exit_ts:     object
    exit_price:  float
    pnl_pct:     float
    exit_reason: str = ""


@dataclass
class RLBacktestResult:
    trades:           list
    equity_curve:     object    # pd.Series — cumulative PnL par trade
    n_episodes:       int
    n_trades:         int
    total_return_pct: float
    sharpe:           float
    max_drawdown_pct: float
    win_rate:         float
    profit_factor:    float
    avg_win_pct:      float
    avg_loss_pct:     float
    session_stats:    object    # pd.DataFrame


from engine.state import HTFBias, apply_htf_mask
from engine.rl_env import _full_state

_RL_FEE_RATE         = 0.0005
_RL_SLIPPAGE         = 0.0002
_CONVICTION_THRESHOLD = 0.0

# SL/TP adaptatifs par niveau de volatilite — R:R = 2:1 dans tous les cas
# Encodage : vol = (state_id % 35) // 7
_RL_VOL_SL_TP = {
    0: (0.003, 0.006),  # LOW
    1: (0.003, 0.006),  # RISING
    2: (0.004, 0.008),  # HIGH
    3: (0.006, 0.012),  # EXTREME
    4: (0.003, 0.006),  # FALLING
}

def _rl_sl_tp(state_id: int):
    vol = (state_id % 35) // 7
    return _RL_VOL_SL_TP.get(vol, (0.003, 0.006))


def run_rl_backtest(
    agent,
    sessions: list,
    test_ratio: float = 0.2,
) -> RLBacktestResult:
    """
    Rejoue la politique greedy de l'agent sur les derniers test_ratio des sessions.

    Args:
        agent      : QAgent entraine
        sessions   : TradingEnv.sessions (liste de DataFrames episodes)
        test_ratio : fraction des sessions utilisee comme test out-of-sample

    Returns:
        RLBacktestResult avec metriques, trades, equity curve
    """
    n_test = max(1, int(len(sessions) * test_ratio))
    test_sessions = sessions[-n_test:]

    trades = []
    cumulative_pnl = 0.0
    equity = [0.0]

    for ep_df in test_sessions:
        ep_df = ep_df.reset_index(drop=True)
        position    = 0
        entry_price = None
        entry_ts    = None
        entry_state = None
        entry_sess  = None
        entry_sl    = 0.003
        entry_tp    = 0.006

        for i in range(len(ep_df)):
            row      = ep_df.iloc[i]
            state_id = int(row["state_id"])
            price    = float(row["close"])
            low      = float(row["low"])  if "low"  in row.index else price
            high     = float(row["high"]) if "high" in row.index else price
            ts       = row["timestamp"]
            session  = int(row["session"])

            htf_bias = int(row.get("htf_bias", int(HTFBias.NEUTRAL))) if "htf_bias" in row.index else int(HTFBias.NEUTRAL)
            q_vals   = agent.q_table[_full_state(state_id, position)].copy().astype(float)
            allowed  = apply_htf_mask(np.ones(3, dtype=bool), htf_bias)
            q_vals[~allowed] = -np.inf
            best_a   = int(np.argmax(q_vals))
            conviction = q_vals[best_a] - q_vals[0] if best_a != 0 else 0.0
            if conviction < _CONVICTION_THRESHOLD:
                best_a = 0
            desired  = _ACTION_TO_DIR[best_a]

            # Calcul SL/TP sur les extremes intracandle pour eviter le slippage de bougie
            sl_hit = False
            tp_hit = False
            exit_price_sl_tp = price
            if position != 0 and entry_price is not None:
                if position == 1:   # LONG
                    worst = (low  - entry_price) / entry_price - _RL_FEE_RATE - _RL_SLIPPAGE
                    best  = (high - entry_price) / entry_price - _RL_FEE_RATE - _RL_SLIPPAGE
                    if worst <= -entry_sl:
                        sl_hit = True
                        exit_price_sl_tp = entry_price * (1 - entry_sl)
                    elif best >= entry_tp:
                        tp_hit = True
                        exit_price_sl_tp = entry_price * (1 + entry_tp)
                else:               # SHORT
                    worst = (entry_price - high) / entry_price - _RL_FEE_RATE - _RL_SLIPPAGE
                    best  = (entry_price - low)  / entry_price - _RL_FEE_RATE - _RL_SLIPPAGE
                    if worst <= -entry_sl:
                        sl_hit = True
                        exit_price_sl_tp = entry_price * (1 + entry_sl)
                    elif best >= entry_tp:
                        tp_hit = True
                        exit_price_sl_tp = entry_price * (1 - entry_tp)

            if sl_hit or tp_hit or (position != 0 and desired != position):
                ep = exit_price_sl_tp if (sl_hit or tp_hit) else price
                raw = (ep - entry_price) / entry_price
                pnl = float(position * raw - _RL_FEE_RATE - _RL_SLIPPAGE)
                exit_reason = "SL" if sl_hit else ("TP" if tp_hit else "signal")
                trades.append(RLTrade(
                    session=entry_sess, state_id=entry_state,
                    direction=position, entry_ts=entry_ts,
                    entry_price=entry_price, exit_ts=ts,
                    exit_price=ep, pnl_pct=pnl,
                    exit_reason=exit_reason,
                ))
                cumulative_pnl += pnl
                equity.append(cumulative_pnl)
                position = 0
                entry_price = None

            if desired != 0 and position == 0:
                position    = desired
                entry_price = price * (1 + desired * _RL_SLIPPAGE)
                entry_ts    = ts
                entry_state = state_id
                entry_sess  = session
                entry_sl, entry_tp = _rl_sl_tp(state_id)

        # Fermeture forcee fin d'episode
        if position != 0:
            last  = ep_df.iloc[-1]
            price = float(last["close"])
            raw   = (price - entry_price) / entry_price
            pnl   = float(position * raw - _RL_FEE_RATE - _RL_SLIPPAGE)
            trades.append(RLTrade(
                session=entry_sess, state_id=entry_state,
                direction=position, entry_ts=entry_ts,
                entry_price=entry_price, exit_ts=last["timestamp"],
                exit_price=price, pnl_pct=pnl,
            ))
            cumulative_pnl += pnl
            equity.append(cumulative_pnl)

    equity_series = pd.Series(equity, name="cumulative_pnl")

    if not trades:
        return RLBacktestResult(
            trades=[], equity_curve=equity_series, n_episodes=n_test,
            n_trades=0, total_return_pct=0.0, sharpe=0.0,
            max_drawdown_pct=0.0, win_rate=0.0, profit_factor=0.0,
            avg_win_pct=0.0, avg_loss_pct=0.0, session_stats=pd.DataFrame(),
        )

    pnl_arr = np.array([t.pnl_pct for t in trades])
    wins    = pnl_arr[pnl_arr > 0]
    losses  = pnl_arr[pnl_arr <= 0]

    total_return  = float(pnl_arr.sum())
    sharpe        = float(pnl_arr.mean() / (pnl_arr.std() + 1e-9)) * np.sqrt(max(len(pnl_arr), 1))
    peak          = equity_series.cummax()
    max_dd        = float((equity_series - peak).min())
    win_rate      = float((pnl_arr > 0).mean())
    pf_denom      = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = float(wins.sum()) / pf_denom if pf_denom > 1e-9 else float("inf")
    avg_win       = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss      = float(losses.mean()) if len(losses) > 0 else 0.0

    session_rows = []
    for s in range(4):
        s_pnls = np.array([t.pnl_pct for t in trades if t.session == s])
        if len(s_pnls):
            session_rows.append({
                "session":       _RL_SESSION_LABELS[s],
                "n_trades":      len(s_pnls),
                "win_rate":      float((s_pnls > 0).mean()),
                "avg_pnl_pct":   float(s_pnls.mean()),
                "total_pnl_pct": float(s_pnls.sum()),
            })

    return RLBacktestResult(
        trades=trades,
        equity_curve=equity_series,
        n_episodes=n_test,
        n_trades=len(trades),
        total_return_pct=total_return,
        sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        session_stats=pd.DataFrame(session_rows),
    )


def rl_trades_to_df(trades: list) -> pd.DataFrame:
    """Convertit la liste de RLTrade en DataFrame exportable."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "session":     _RL_SESSION_LABELS.get(t.session, t.session),
        "direction":   "LONG" if t.direction == 1 else "SHORT",
        "state_id":    t.state_id,
        "entry_ts":    t.entry_ts,
        "entry_price": round(t.entry_price, 2),
        "exit_ts":     t.exit_ts,
        "exit_price":  round(t.exit_price, 2),
        "pnl_pct":     round(t.pnl_pct * 100, 4),
        "exit_reason": t.exit_reason,
    } for t in trades])
