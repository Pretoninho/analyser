"""
engine/rl_env.py — Environnement MDP pour l'agent RL Pi*.

Episode   : une session de trading (ASIA_DEAD / LONDON / OVERLAP / NY)
Step      : une bougie 5min (~36-72 steps par episode)
State     : state_id entier 0-419 (140 états marché × 3 positions)
Action    : 0=FLAT, 1=LONG, 2=SHORT
Reward    : P&L complet à la clôture du trade uniquement (0 pendant le holding)
Sortie    : fin de session ou signal contraire
"""

import numpy as np
import pandas as pd
from engine.state import compute_states, HTFBias
from engine.patterns import detect_pattern, apply_pattern_to_state_mask
from engine.sltp_config import SLTPConfig, DEFAULT_CONFIG

ACTIONS   = {0: "FLAT", 1: "LONG", 2: "SHORT"}
_MARKET_STATES = 140  # 4 sessions × 5 vol × 7 price_structure
N_STATES  = _MARKET_STATES * 3   # × 3 positions (FLAT/LONG/SHORT) = 420
N_ACTIONS = 3

# Encodage position dans le state_id :
#   full_state = base_state_id + _POS_OFFSET[position] * _MARKET_STATES
#   FLAT=0→ états 0-139, LONG=1→ états 140-279, SHORT=-1→ états 280-419
_POS_OFFSET = {0: 0, 1: 1, -1: 2}


def _full_state(base_state_id: int, position: int) -> int:
    return base_state_id + _POS_OFFSET[position] * _MARKET_STATES

FEE_RATE     = 0.0005   # 0.05% taker Deribit
SLIPPAGE     = 0.0002   # 0.02% slippage estime
LOSS_PENALTY = 1.5      # multiplicateur sur les P&L négatifs à la clôture


class TradingEnv:
    """
    Environnement de trading episodique base sur les sessions.

    Usage :
        env   = TradingEnv(df_1m)
        state = env.reset()
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
    """

    def __init__(self, df_1m: pd.DataFrame, sltp_config: SLTPConfig = None):
        self.df5       = compute_states(df_1m)
        self.sltp_config = sltp_config or DEFAULT_CONFIG
        self._sessions = self._split_sessions()
        self._ep_idx   = 0

        self._ep_df           = None
        self._step            = 0
        self._position        = 0       # -1 short / 0 flat / 1 long
        self._entry_price     = None
        self._entry_sl        = None
        self._entry_tp        = None
        self._prev_structure  = None
        self._current_pattern = None
        self._current_htf     = int(HTFBias.NEUTRAL)
        self._current_vol     = 0

    # ── API publique ───────────────────────────────────────────

    def reset(self, ep_idx: int = None) -> int:
        if ep_idx is not None:
            self._ep_idx = ep_idx % len(self._sessions)

        self._ep_df           = self._sessions[self._ep_idx].reset_index(drop=True)
        self._step            = 0
        self._position        = 0
        self._entry_price     = None
        self._entry_sl        = None
        self._entry_tp        = None
        self._prev_structure  = None
        self._current_pattern = None
        self._current_htf     = int(HTFBias.NEUTRAL)
        self._current_vol     = 0
        self._ep_idx          = (self._ep_idx + 1) % len(self._sessions)

        return _full_state(int(self._ep_df.iloc[0]["state_id"]), 0)

    def step(self, action: int):
        row   = self._ep_df.iloc[self._step]
        price = float(row["close"])
        curr_structure = int(row["price_structure"])

        if self._prev_structure is not None:
            self._current_pattern = detect_pattern(self._prev_structure, curr_structure)
        self._prev_structure = curr_structure
        self._current_htf = int(row.get("htf_bias", int(HTFBias.NEUTRAL)))

        pnl_pct      = self._compute_step_pnl(price)
        prev_position = self._position

        # SL/TP : force FLAT si les seuils sont atteints
        sl_tp_hit = False
        if self._position != 0:
            if pnl_pct <= -self._entry_sl or pnl_pct >= self._entry_tp:
                action    = 0
                sl_tp_hit = True

        # Reward uniquement à la clôture du trade
        desired = {0: 0, 1: 1, 2: -1}[action]
        closing = (prev_position != 0) and (desired == 0 or desired != prev_position)
        reward  = 0.0

        if closing or sl_tp_hit:
            reward = pnl_pct * LOSS_PENALTY if pnl_pct < 0 else pnl_pct

        self._apply_action(action, price)

        self._step += 1
        done = self._step >= len(self._ep_df)

        # Clôture forcée en fin d'episode
        if done and self._position != 0:
            last_price = float(self._ep_df.iloc[-1]["close"])
            close_pnl  = self._compute_step_pnl(last_price)
            reward    += close_pnl * LOSS_PENALTY if close_pnl < 0 else close_pnl
            self._position    = 0
            self._entry_price = None

        next_base  = int(self._ep_df.iloc[min(self._step, len(self._ep_df) - 1)]["state_id"])
        next_state = _full_state(next_base, self._position)

        info = {
            "step":     self._step,
            "price":    price,
            "position": self._position,
            "pnl_pct":  pnl_pct,
            "session":  int(row["session"]),
            "state_id": int(row["state_id"]),
            "pattern":  self._current_pattern,
        }
        return next_state, reward, done, info

    @property
    def current_pattern(self) -> str | None:
        return self._current_pattern

    @property
    def current_htf_bias(self) -> int:
        return self._current_htf

    @property
    def n_episodes(self) -> int:
        return len(self._sessions)

    @property
    def sessions(self) -> list:
        return self._sessions

    # ── Helpers ────────────────────────────────────────────────

    def _split_sessions(self) -> list:
        df = self.df5.copy()
        df["_key"] = (
            df["timestamp"].dt.date.astype(str) + "_" +
            df["session"].astype(str)
        )
        episodes = []
        for _, grp in df.groupby("_key", sort=True):
            if len(grp) >= 6:
                episodes.append(grp)
        return episodes

    def _compute_step_pnl(self, current_price: float) -> float:
        if self._position == 0 or self._entry_price is None:
            return 0.0
        raw = (current_price - self._entry_price) / self._entry_price
        return float(self._position * raw - FEE_RATE - SLIPPAGE)

    def _apply_action(self, action: int, price: float):
        desired = {0: 0, 1: 1, 2: -1}[action]
        if desired == self._position:
            return

        if self._position != 0:
            self._position    = 0
            self._entry_price = None

        if desired != 0:
            self._position    = desired
            self._entry_price = price * (1 + desired * SLIPPAGE)
            # Récupère le niveau de volatilité courant pour adapter SL/TP
            vol_level = int(self._ep_df.iloc[self._step]["volatility"]) if self._ep_df is not None else 0
            self._entry_sl, self._entry_tp = self.sltp_config.get_sl_tp(vol_level)
