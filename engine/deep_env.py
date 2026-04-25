"""
engine/deep_env.py — Environnement Gymnasium pour l'agent Deep RL (RecurrentPPO).

Episode  : une session de trading (ASIA_DEAD / LONDON / OVERLAP / NY)
Step     : une bougie 5min
Obs      : vecteur (N_FEATURES,) = 8 features marché + 1 position courante
Action   : 0=FLAT, 1=LONG, 2=SHORT
Reward   : P&L à la clôture du trade (× LOSS_PENALTY si négatif)

Le LSTM interne à RecurrentPPO construit la mémoire temporelle —
l'environnement n'a pas besoin de gérer de fenêtre glissante.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from engine.features import compute_features, FEATURE_COLS, N_FEATURES, obs_from_row
from engine.sltp_config import SLTPConfig, DEFAULT_CONFIG

# ── Paramètres de simulation ───────────────────────────────────

FEE_RATE     = 0.0005   # 0.05% taker Deribit
SLIPPAGE     = 0.0002   # 0.02% slippage estimé
LOSS_PENALTY = 1.5      # multiplicateur sur les P&L négatifs


def _sl_tp_from_vol_ratio(vol_ratio: float, sltp_config: SLTPConfig) -> tuple:
    """
    Détermine le niveau de volatilité basé sur le volume relatif, 
    puis retourne SL/TP via la config.
    
    vol_ratio normalisé [0,1]:
        [0.00-0.33] → LOW
        [0.33-0.66] → RISING  
        [0.66-0.80] → HIGH
        [0.80-1.00] → EXTREME
    """
    if vol_ratio > 0.8:
        vol_level = 3      # EXTREME
    elif vol_ratio > 0.66:
        vol_level = 2      # HIGH
    elif vol_ratio > 0.33:
        vol_level = 1      # RISING
    else:
        vol_level = 0      # LOW
    return sltp_config.get_sl_tp(vol_level)


# ── Environnement ──────────────────────────────────────────────

class DeepTradingEnv(gym.Env):
    """
    Environnement de trading épisodique compatible RecurrentPPO (sb3-contrib).

    Chaque épisode est une session de trading. L'agent reçoit à chaque step
    les features normalisées de la bougie courante. Le LSTM de la politique
    accumule le contexte temporel (la "narration") en interne.
    """

    metadata = {"render_modes": []}

    def __init__(self, df_1m: pd.DataFrame, sltp_config: SLTPConfig = None):
        super().__init__()

        self._df_feat  = compute_features(df_1m)
        self.sltp_config = sltp_config or DEFAULT_CONFIG
        self._sessions = self._split_sessions()
        self._ep_idx   = 0

        # Observation : 8 features marché + 1 position = 9 float32
        self.observation_space = spaces.Box(
            low  = np.full(N_FEATURES, -1.0, dtype=np.float32),
            high = np.full(N_FEATURES,  2.0, dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=FLAT 1=LONG 2=SHORT

        # État interne de l'épisode courant
        self._ep_df       = None
        self._step_idx    = 0
        self._position    = 0       # -1 short / 0 flat / 1 long
        self._entry_price = None
        self._entry_sl    = None
        self._entry_tp    = None

    # ── Gymnasium API ──────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._ep_df       = self._sessions[self._ep_idx].reset_index(drop=True)
        self._ep_idx      = (self._ep_idx + 1) % len(self._sessions)
        self._step_idx    = 0
        self._position    = 0
        self._entry_price = None
        # Initialise SL/TP avec le niveau courant
        sl, tp = _sl_tp_from_vol_ratio(0.5, self.sltp_config)  # 0.5 = médiane
        self._entry_sl, self._entry_tp = sl, tp

        return self._get_obs(), {}

    def step(self, action: int):
        row   = self._ep_df.iloc[self._step_idx]
        price = float(row["close"])

        pnl_pct   = self._compute_pnl(price)
        sl_tp_hit = False

        # SL/TP — force FLAT si seuil atteint
        if self._position != 0:
            if pnl_pct <= -self._entry_sl or pnl_pct >= self._entry_tp:
                action    = 0
                sl_tp_hit = True

        desired = {0: 0, 1: 1, 2: -1}[int(action)]
        closing = (self._position != 0) and (
            desired == 0 or desired != self._position
        )
        reward = 0.0

        if closing or sl_tp_hit:
            reward = pnl_pct * LOSS_PENALTY if pnl_pct < 0 else pnl_pct

        self._apply_action(int(action), price, row)
        self._step_idx += 1

        terminated = self._step_idx >= len(self._ep_df)
        truncated  = False

        # Clôture forcée en fin d'épisode
        if terminated and self._position != 0:
            last_price = float(self._ep_df.iloc[-1]["close"])
            close_pnl  = self._compute_pnl(last_price)
            reward    += close_pnl * LOSS_PENALTY if close_pnl < 0 else close_pnl
            self._position    = 0
            self._entry_price = None

        obs  = self._get_obs() if not terminated else np.zeros(N_FEATURES, dtype=np.float32)
        info = {
            "pnl_pct":  pnl_pct,
            "position": self._position,
            "session":  int(row["session"]),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    # ── Propriétés publiques ───────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self._sessions)

    # ── Helpers privés ─────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_idx, len(self._ep_df) - 1)
        return obs_from_row(self._ep_df.iloc[idx], self._position)

    def _compute_pnl(self, price: float) -> float:
        if self._position == 0 or self._entry_price is None:
            return 0.0
        raw = (price - self._entry_price) / self._entry_price
        return float(self._position * raw - FEE_RATE - SLIPPAGE)

    def _apply_action(self, action: int, price: float, row: pd.Series):
        desired = {0: 0, 1: 1, 2: -1}[action]
        if desired == self._position:
            return

        if self._position != 0:
            self._position    = 0
            self._entry_price = None

        if desired != 0:
            self._position    = desired
            self._entry_price = price * (1 + desired * SLIPPAGE)
            # SL/TP adaptatifs selon le volume relatif de la bougie d'entrée
            vol_ratio = float(row.get("f_vol_ratio", 0.5))
            self._entry_sl, self._entry_tp = _sl_tp_from_vol_ratio(vol_ratio, self.sltp_config)

    def _split_sessions(self) -> list:
        df = self._df_feat.copy()
        df["_key"] = (
            df["timestamp"].dt.date.astype(str) + "_" +
            df["session"].astype(str)
        )
        sessions = [
            grp.drop(columns=["_key"])
            for _, grp in df.groupby("_key", sort=True)
            if len(grp) >= 6
        ]
        return sessions
