"""
engine/stats_env.py — Environnement RL base sur le contexte statistique ICT.

Episode  : une journee de trading (session NY, 08:30-16:00 ET)
Step     : une bougie 1min
State    : encodage ICT a 1944 etats (month x day x london x macro x sweep x pool)
Action   : 0=FLAT  1=LONG  2=SHORT
Reward   : P&L realise a la cloture du trade

pool_ctx :
    1=BSL_SWEPT (pre_high > London High ou PWH -> reversal baissier)
    2=SSL_SWEPT (pre_low  < London Low  ou PWL -> reversal haussier)
"""

import numpy as np
import pandas as pd
import pytz

from engine.stats_state import (
    N_STATES, N_ACTIONS, ET_TZ, MACROS, PRE_MACRO_WINDOWS,
    encode, macro_ctx, compute_daily_context, compute_pool_ctx,
    build_weekly_levels,
)

FEE_RATE   = 0.0005   # 0.05% taker
SLIPPAGE   = 0.0002   # 0.02% estime
MAX_TRADES = 2        # trades max par jour


class StatsEnv:

    def __init__(self, df_1m: pd.DataFrame):
        self._weekly = build_weekly_levels(df_1m)
        self._episodes, self._dates = self._build_episodes(df_1m)
        self._ep_idx  = 0

        self._ep          = None
        self._ctx         = None
        self._step_idx    = 0
        self._position    = 0
        self._entry_px    = None
        self._pre_high    = None
        self._pre_low     = None
        self._cur_macro   = 0
        self._trades_done = 0

    # ── API publique ───────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def reset(self, ep_idx: int = None) -> int:
        if ep_idx is not None:
            self._ep_idx = ep_idx % len(self._episodes)

        self._ep      = self._episodes[self._ep_idx].reset_index(drop=True)
        date          = self._dates[self._ep_idx]
        pwh, pwl      = self._weekly.get(date, (None, None))
        self._ctx     = compute_daily_context(self._ep, pwh=pwh, pwl=pwl)
        self._ep_idx  = (self._ep_idx + 1) % len(self._episodes)

        self._step_idx    = 0
        self._position    = 0
        self._entry_px    = None
        self._pre_high    = None
        self._pre_low     = None
        self._cur_macro   = 0
        self._trades_done = 0

        return self._get_state()

    def step(self, action: int):
        row   = self._ep.iloc[self._step_idx]
        price = float(row["close"])
        hm_et = self._hm_et(row["timestamp"])

        self._update_pre_window(hm_et, row)

        mac = macro_ctx(hm_et)

        # Fin de macro : cloture forcee
        if self._cur_macro != 0 and mac != self._cur_macro:
            reward = self._close_position(price)
            self._cur_macro = mac
        else:
            reward = 0.0
            self._cur_macro = mac

        # Hors macro ou quota atteint : seul FLAT autorise
        if mac == 0 or self._trades_done >= MAX_TRADES:
            action = 0

        trade_opened = False
        desired = {0: 0, 1: 1, 2: -1}[int(action)]
        if desired != self._position:
            if self._position != 0:
                reward += self._close_position(price)
            if desired != 0 and mac != 0 and self._trades_done < MAX_TRADES:
                self._position    = desired
                self._entry_px    = price * (1 + desired * SLIPPAGE)
                self._trades_done += 1
                trade_opened      = True

        self._step_idx += 1
        done = self._step_idx >= len(self._ep)

        if done and self._position != 0:
            last_px = float(self._ep.iloc[-1]["close"])
            reward += self._close_position(last_px)

        state = self._get_state() if not done else 0
        return state, float(reward), done, {"trade_opened": trade_opened}

    # ── Helpers prives ─────────────────────────────────────────

    def _get_state(self) -> int:
        if self._step_idx >= len(self._ep):
            return 0
        row   = self._ep.iloc[self._step_idx]
        hm_et = self._hm_et(row["timestamp"])
        mac   = macro_ctx(hm_et)

        # Sweep : comparaison avec le pre-macro high/low
        sc = 0
        if mac != 0 and self._pre_high is not None:
            if float(row["high"]) > self._pre_high:
                sc = 1
            elif float(row["low"]) < self._pre_low:
                sc = 2

        # Pool : sweep de session_high/low (00:00-08:30 ET) ou PWH/PWL
        pc = 0
        if mac != 0 and self._pre_high is not None:
            pc = compute_pool_ctx(
                self._pre_high, self._pre_low,
                self._ctx.get("session_high"), self._ctx.get("session_low"),
                self._ctx.get("pwh"),          self._ctx.get("pwl"),
            )

        return encode(
            self._ctx["month_ctx"],
            self._ctx["day_ctx"],
            self._ctx["london_ctx"],
            mac, sc, pc,
        )

    def _close_position(self, price: float) -> float:
        if self._position == 0 or self._entry_px is None:
            return 0.0
        raw = (price - self._entry_px) / self._entry_px
        pnl = float(self._position * raw - FEE_RATE - SLIPPAGE)
        self._position = 0
        self._entry_px = None
        return pnl

    def _update_pre_window(self, hm_et: int, row: pd.Series):
        for mac_idx, (pre_start, pre_end) in PRE_MACRO_WINDOWS.items():
            if pre_start <= hm_et < pre_end:
                h = float(row["high"])
                l = float(row["low"])
                if self._pre_high is None:
                    self._pre_high = h
                    self._pre_low  = l
                else:
                    self._pre_high = max(self._pre_high, h)
                    self._pre_low  = min(self._pre_low,  l)
                return
        if macro_ctx(hm_et) == 0:
            cur_pre = any(
                pre_s <= hm_et < pre_e
                for pre_s, pre_e in PRE_MACRO_WINDOWS.values()
            )
            if not cur_pre:
                next_pre = min(
                    (pre_s for pre_s, _ in PRE_MACRO_WINDOWS.values() if pre_s > hm_et),
                    default=None
                )
                if next_pre is not None and hm_et < next_pre - 5:
                    self._pre_high = None
                    self._pre_low  = None

    @staticmethod
    def _hm_et(ts) -> int:
        ts_et = ts.tz_convert(ET_TZ) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(ET_TZ)
        return ts_et.hour * 60 + ts_et.minute

    def _build_episodes(self, df_1m: pd.DataFrame):
        df = df_1m.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        et_tz = pytz.timezone("America/New_York")
        df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
        df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
        df["date_et"] = df["ts_et"].dt.date

        df = df[df["hm_et"] < 20 * 60]

        episodes, dates = [], []
        for date, grp in df.groupby("date_et"):
            hm_vals    = set(grp["hm_et"].values)
            has_asia   = any(60  <= h < 300 for h in hm_vals)
            has_london = any(420 <= h < 600 for h in hm_vals)
            has_ny     = any(530 <= h < 910 for h in hm_vals)
            if has_asia and has_london and has_ny and len(grp) >= 60:
                episodes.append(grp.drop(columns=["ts_et", "hm_et", "date_et"]))
                dates.append(date)

        print(f"[stats_env] {len(episodes)} episodes (jours complets avec Asia+London+NY)")
        return episodes, dates
