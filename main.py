"""
main.py — Point d'entree : collecte, stockage et entrainement de Pi*.
"""

import sys
import time
import argparse
from pathlib import Path

from config import FETCH, DATA_DIR
from data   import init_db, collect_btc_1m, load_latest_btc_1m, backfill_btc_1m, db_summary


def run_once(limit: int = 720):
    df = collect_btc_1m(limit=limit)
    if df.empty:
        print("[main] Echec de la collecte.")
        sys.exit(1)


def run_watch(limit: int = 720):
    interval = FETCH["refresh_seconds"]
    print(f"[main] Mode surveillance — rafraichissement toutes les {interval // 60} min (Ctrl+C pour arreter)")
    while True:
        try:
            collect_btc_1m(limit=limit)
            print(f"[main] Prochain rafraichissement dans {interval // 60} min...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[main] Arret.")
            break


def run_gen_masks(ev_threshold: float = -0.01, rr: float = 2.0):
    from engine import TradingEnv, QAgent, compute_transition_stats, build_action_mask, mask_summary, SLTPConfig
    import numpy as np

    model_path = str(DATA_DIR / "q_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print(f"[masks] Modele introuvable. Lancez --train d'abord.")
        sys.exit(1)

    df = load_latest_btc_1m(limit=50_000)
    if df.empty:
        print("[masks] Aucune donnee.")
        sys.exit(1)

    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    env = TradingEnv(df, sltp_config=sltp_config)

    print("\n[masks] === Analyse des transitions ===")
    trans = compute_transition_stats(env.sessions)
    print(f"  Transitions uniques observees : {len(trans)}")
    print(f"  Transitions impossibles       : {100*100 - len(trans)} (jamais vues en donnees)")
    print(f"\n  Top 10 transitions les plus fréquentes :")
    for _, row in trans.head(10).iterrows():
        print(f"    etat {int(row['from_state']):3d} -> {int(row['to_state']):3d}  "
              f"({int(row['count'])} fois, {row['freq_pct']:.2f}%)")

    print(f"\n[masks] === Construction du masque (seuil EV={ev_threshold}) ===")
    mask = build_action_mask(agent.q_table, ev_threshold=ev_threshold)
    stats = mask_summary(mask)
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")

    mask_path = str(DATA_DIR / "action_mask.npy")
    np.save(mask_path, mask)
    print(f"\n[masks] Masque sauvegarde : {mask_path}")


def run_build_markov(rr: float = 2.0):
    from engine import TradingEnv, MarkovChain, SLTPConfig
    import numpy as np

    df = load_latest_btc_1m(limit=50_000)
    if df.empty:
        print("[markov] Aucune donnee.")
        sys.exit(1)

    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    env = TradingEnv(df, sltp_config=sltp_config)
    mc  = MarkovChain()
    mc.fit(env.sessions)

    stats = mc.summary()
    acc   = mc.backtest_accuracy(env.sessions[-max(1, int(len(env.sessions)*0.2)):])

    print(f"\n[markov] === Chaine de Markov ===")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    print(f"\n[markov] === Precision de prediction (test set 20%) ===")
    print(f"  Top-1 accuracy : {acc['top1_accuracy']*100:.1f}%")
    print(f"  Top-3 accuracy : {acc['top3_accuracy']*100:.1f}%")
    print(f"  Transitions evaluees : {acc['n_evaluated']}")

    markov_path = str(DATA_DIR / "markov.pkl")
    mc.save(markov_path)
    print(f"\n[markov] Sauvegarde : {markov_path}")


def run_train(episodes: int = 500, alpha: float = None, epsilon_decay: float = None,
              reset: bool = False, use_masks: bool = False, rr: float = 2.0):
    from engine import TradingEnv, QAgent, MarkovChain, SLTPConfig
    import numpy as np

    df = load_latest_btc_1m(limit=50_000)
    if df.empty:
        print("[main] Aucune donnee. Lancez une collecte d'abord.")
        sys.exit(1)

    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    env        = TradingEnv(df, sltp_config=sltp_config)
    model_path = str(DATA_DIR / "q_agent.pkl")

    if not reset and Path(model_path).exists():
        agent = QAgent.load(model_path)
        print(f"[main] Reprise de l'entrainement ({agent._episode_count} episodes deja effectues)")
        if alpha is not None:
            agent.alpha = alpha
            print(f"[main] alpha force : {alpha}")
        if epsilon_decay is not None:
            agent.epsilon_decay = epsilon_decay
            print(f"[main] epsilon_decay force : {epsilon_decay}")
    else:
        kw = {}
        if alpha         is not None: kw["alpha"]         = alpha
        if epsilon_decay is not None: kw["epsilon_decay"] = epsilon_decay
        agent = QAgent(**kw)
        print(f"[main] Nouvel agent — alpha={agent.alpha} epsilon_decay={agent.epsilon_decay}")

    mask = None
    mask_path = DATA_DIR / "action_mask.npy"
    if use_masks and mask_path.exists():
        mask = np.load(str(mask_path))
        from engine import mask_summary
        stats = mask_summary(mask)
        print(f"[main] Masque charge : {stats['actions_masquées']} actions masquees "
              f"({stats['pct_masquées']}%)")
    elif use_masks:
        print("[main] --use-masks specifie mais action_mask.npy introuvable. Lancez --gen-masks.")

    # Chargement ou creation de la Markov Chain
    markov_path = DATA_DIR / "markov.pkl"
    if markov_path.exists():
        mc = MarkovChain.load(str(markov_path))
    else:
        mc = MarkovChain()
        mc.fit(env.sessions)
        print(f"[main] Markov initialisee depuis les sessions existantes")

    n_sessions = env.n_episodes
    n_passes   = episodes / n_sessions
    print(f"[main] {episodes} episodes | {n_sessions} sessions | ~{n_passes:.1f} passes par session\n")

    log_every  = max(100, episodes // 20)
    rewards_window = []

    from engine.patterns import apply_pattern_to_state_mask
    from engine.state    import apply_htf_mask

    try:
        for ep in range(episodes):
            state = env.reset()
            done  = False
            total_reward = 0.0
            prev_state = None

            while not done:
                # Mise a jour Markov sur chaque transition observee
                if prev_state is not None:
                    mc.update(prev_state % 100, state % 100)

                pattern   = env.current_pattern
                step_mask = apply_pattern_to_state_mask(mask, pattern, state)
                step_mask = apply_htf_mask(step_mask, env.current_htf_bias)
                action                         = agent.act(state, mask=step_mask)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                prev_state = state
                state      = next_state
                total_reward += reward

            agent.end_episode()
            rewards_window.append(total_reward)

            if (ep + 1) % log_every == 0:
                policy   = agent.policy_summary()
                avg_r    = np.mean(rewards_window[-log_every:])
                pct_done = (ep + 1) / episodes * 100
                print(
                    f"  [{pct_done:5.1f}%] ep {agent._episode_count:6d} | "
                    f"reward moy={avg_r:+.4f} | eps={agent.epsilon:.4f} | "
                    f"visites={policy['visited_states']:3d} | "
                    f"FLAT={policy['FLAT']:3d} LONG={policy['LONG']:3d} SHORT={policy['SHORT']:3d}"
                )

    except KeyboardInterrupt:
        print(f"\n[main] Interruption — sauvegarde en cours...")

    agent.save(model_path)
    mc.save(str(markov_path))
    print(f"\n[main] Entrainement termine ({agent._episode_count} episodes total). Modele : {model_path}")
    stats = mc.summary()
    print(f"[main] Markov : {stats['transitions_tot']} transitions | "
          f"{stats['etats_fiables']} etats fiables | entropie moy={stats['entropie_moy']:.2f}")


def run_backtest(test_ratio: float = 0.2, export_csv: bool = False, rr: float = 2.0):
    from engine import TradingEnv, QAgent, run_rl_backtest, rl_trades_to_df, SLTPConfig
    import numpy as np

    model_path = str(DATA_DIR / "q_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print(f"[main] Modele introuvable : {model_path}. Lancez --train d'abord.")
        sys.exit(1)

    df = load_latest_btc_1m(limit=50_000)
    if df.empty:
        print("[main] Aucune donnee.")
        sys.exit(1)

    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    env = TradingEnv(df, sltp_config=sltp_config)
    n_total = env.n_episodes
    n_test  = max(1, int(n_total * test_ratio))
    print(f"\n[backtest] {n_total} sessions | test set = derniers {n_test} ({test_ratio*100:.0f}%)\n")

    result = run_rl_backtest(agent, env.sessions, test_ratio=test_ratio)

    if result.n_trades == 0:
        print("[backtest] Aucun trade execute. L'agent recommande peut-etre uniquement FLAT.")
        return

    pf_str = f"{result.profit_factor:.3f}" if result.profit_factor != float("inf") else "∞"
    print(
        f"  Episodes test    : {result.n_episodes}\n"
        f"  Trades           : {result.n_trades}\n"
        f"  Return total     : {result.total_return_pct*100:+.2f}%\n"
        f"  Sharpe           : {result.sharpe:+.3f}\n"
        f"  Max drawdown     : {result.max_drawdown_pct*100:.2f}%\n"
        f"  Win rate         : {result.win_rate*100:.1f}%\n"
        f"  Profit factor    : {pf_str}\n"
        f"  Avg win          : {result.avg_win_pct*100:+.3f}%\n"
        f"  Avg loss         : {result.avg_loss_pct*100:+.3f}%\n"
    )

    if not result.session_stats.empty:
        print("  Par session :")
        for _, row in result.session_stats.iterrows():
            print(
                f"    {row['session']:8s} | {int(row['n_trades']):3d} trades | "
                f"win={row['win_rate']*100:.0f}% | avg={row['avg_pnl_pct']*100:+.3f}% | "
                f"total={row['total_pnl_pct']*100:+.2f}%"
            )

    if export_csv:
        csv_path = str(DATA_DIR / "backtest_trades.csv")
        rl_trades_to_df(result.trades).to_csv(csv_path, index=False)
        print(f"\n[backtest] Trades exportes : {csv_path}")


def run_eval(rr: float = 2.0):
    from engine import TradingEnv, QAgent, SLTPConfig
    import numpy as np

    model_path = str(DATA_DIR / "q_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print(f"[main] Modele introuvable : {model_path}. Lancez --train d'abord.")
        sys.exit(1)

    df = load_latest_btc_1m(limit=50_000)
    if df.empty:
        print("[main] Aucune donnee.")
        sys.exit(1)

    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    env    = TradingEnv(df, sltp_config=sltp_config)
    rewards = []

    for ep_idx in range(env.n_episodes):
        state = env.reset(ep_idx=ep_idx)
        done  = False
        ep_r  = 0.0
        while not done:
            action                      = agent.act(state, training=False)
            state, reward, done, _      = env.step(action)
            ep_r += reward
        rewards.append(ep_r)

    print(
        f"[eval] {len(rewards)} episodes | "
        f"reward moyen={np.mean(rewards):+.4f} | "
        f"std={np.std(rewards):.4f} | "
        f"positifs={sum(r > 0 for r in rewards)}/{len(rewards)}"
    )
    print(agent.policy_summary())


def run_download_binance(start_year: int, start_month: int,
                         end_year: int = None, end_month: int = None):
    from data.binance import download_binance_1m
    start = (start_year, start_month)
    end   = (end_year, end_month) if end_year else None
    download_binance_1m(start=start, end=end)


def _sim_trade_rr(exit_df: "pd.DataFrame", entry_px: float, direction: int,
                  sl_pct: float, tp_pct: float,
                  fee: float = 0.0005, slip: float = 0.0002,
                  verbose: bool = False,
                  trailing_delta: float = 0.0):
    """
    Simule un trade avec SL et TP sur les bougies 1min de exit_df.
    direction : +1 LONG, -1 SHORT
    trailing_delta > 0 : sortie TRAIL si drawdown du PnL non realise >= trailing_delta
                         depuis le meilleur PnL non realise observe.
    verbose=False : retourne float (pnl)
    verbose=True  : retourne (pnl, exit_reason, exit_px, tp_px, sl_px, n_candles)
    """
    if direction == 1:
        tp_px = entry_px * (1 + tp_pct)
        sl_px = entry_px * (1 - sl_pct)
    else:
        tp_px = entry_px * (1 - tp_pct)
        sl_px = entry_px * (1 + sl_pct)

    best_unrealized = 0.0

    for i, (_, row) in enumerate(exit_df.iterrows()):
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        if direction == 1:
            if l <= sl_px:
                pnl = -sl_pct - fee - slip * 2
                return (pnl, "SL", sl_px, tp_px, sl_px, i + 1) if verbose else pnl
            if h >= tp_px:
                pnl = tp_pct - fee - slip * 2
                return (pnl, "TP", tp_px, tp_px, sl_px, i + 1) if verbose else pnl
        else:
            if h >= sl_px:
                pnl = -sl_pct - fee - slip * 2
                return (pnl, "SL", sl_px, tp_px, sl_px, i + 1) if verbose else pnl
            if l <= tp_px:
                pnl = tp_pct - fee - slip * 2
                return (pnl, "TP", tp_px, tp_px, sl_px, i + 1) if verbose else pnl

        if trailing_delta > 0:
            unrealized = direction * (c - entry_px) / entry_px - fee - slip * 2
            if unrealized > best_unrealized:
                best_unrealized = unrealized
            if (best_unrealized - unrealized) >= trailing_delta:
                return (unrealized, "TRAIL", c, tp_px, sl_px, i + 1) if verbose else unrealized

    close = float(exit_df.iloc[-1]["close"])
    raw = direction * (close - entry_px) / entry_px
    pnl = raw - fee - slip * 2
    return (pnl, "EOD", close, tp_px, sl_px, len(exit_df)) if verbose else pnl


def _find_ote_entry(macro_df: "pd.DataFrame", direction: int, n_disp: int = 5):
    """
    Cherche une entree OTE (62-79% retracement) dans les bougies de la macro.
    direction : +1 LONG, -1 SHORT
    n_disp    : nombre de bougies pour definir le displacement initial
    Retourne  : (entry_idx, entry_px, sl_px) ou None si pas de setup.
    """
    if len(macro_df) < n_disp + 3:
        return None

    disp    = macro_df.iloc[:n_disp]
    swing_h = float(disp["high"].max())
    swing_l = float(disp["low"].min())
    rng     = swing_h - swing_l

    if rng / max(swing_l, 1) < 0.0003:   # displacement < 0.03% — trop plat
        return None

    if direction == +1:
        fib_lo = swing_h - 0.786 * rng   # 78.6% retrace
        fib_hi = swing_h - 0.618 * rng   # 61.8% retrace
        sl_px  = swing_l * 0.9995
        for i, (_, row) in enumerate(macro_df.iloc[n_disp:].iterrows()):
            if float(row["low"]) <= fib_hi:
                return (n_disp + i, (fib_lo + fib_hi) / 2, sl_px)
    else:
        fib_lo = swing_l + 0.618 * rng
        fib_hi = swing_l + 0.786 * rng
        sl_px  = swing_h * 1.0005
        for i, (_, row) in enumerate(macro_df.iloc[n_disp:].iterrows()):
            if float(row["high"]) >= fib_lo:
                return (n_disp + i, (fib_lo + fib_hi) / 2, sl_px)
    return None


def _find_fvg_entry(macro_df: "pd.DataFrame", direction: int, n_search: int = 12):
    """
    Cherche un FVG dans les bougies de la macro puis attend que le prix y revienne.
    direction : +1 LONG (FVG bullish), -1 SHORT (FVG bearish)
    Retourne  : (entry_idx, entry_px, sl_px) ou None si pas de setup.
    """
    if len(macro_df) < 6:
        return None

    scan   = macro_df.iloc[:n_search]
    highs  = scan["high"].values.astype(float)
    lows   = scan["low"].values.astype(float)

    best_fvg = None
    for i in range(2, len(scan)):
        if direction == +1 and lows[i] > highs[i - 2]:       # bullish FVG
            best_fvg = (i, highs[i - 2], lows[i])            # (idx, bot, top)
        elif direction == -1 and highs[i] < lows[i - 2]:     # bearish FVG
            best_fvg = (i, highs[i], lows[i - 2])            # (idx, bot, top)

    if best_fvg is None:
        return None

    fvg_i, fvg_bot, fvg_top = best_fvg
    mid = (fvg_bot + fvg_top) / 2

    for i, (_, row) in enumerate(macro_df.iloc[fvg_i + 1:].iterrows()):
        lo, hi = float(row["low"]), float(row["high"])
        if direction == +1 and lo <= fvg_top:                 # retour dans le FVG bullish
            return (fvg_i + 1 + i, mid, fvg_bot * 0.9995)
        if direction == -1 and hi >= fvg_bot:                 # retour dans le FVG bearish
            return (fvg_i + 1 + i, mid, fvg_top * 1.0005)
    return None


def run_build_qtable(test_ratio: float = 0.2, min_samples: int = 10,
                     exit_hm: int = 960, sl_pct: float = 0.0, rr: float = 2.0,
                     trailing_delta: float = 0.0,
                     target_pool: bool = False, aligned_only: bool = False,
                     skip_macros: frozenset = frozenset(),
                     skip_days: frozenset = frozenset(),
                     macro_rules: dict = None):
    """
    Construit la Q-table empiriquement depuis les donnees historiques.
    Entre a l'ouverture de la macro, sort a exit_hm (16:00 ET par defaut).
    sl_pct > 0  : simulation SL/TP fixes (TP = sl_pct * rr).
    trailing_delta > 0 : sortie TRAIL si drawdown du PnL non realise >= trailing_delta.
    target_pool : TP dynamique = pool oppose (SSL pour SHORT apres BSL swept,
                  BSL pour LONG apres SSL swept). SL = sl_pct au-dessus du sweep.
    Etat 1944 : month x day x london x macro x sweep x pool(BSL/SSL)
    """
    from engine.stats_state import (
        N_STATES, N_ACTIONS, MACROS,
        encode, decode, compute_daily_context,
        compute_pool_ctx, build_weekly_levels,
    )
    from engine.q_agent import QAgent
    from data.binance   import load_binance_1m
    import numpy as np
    import pandas as pd
    import pytz

    FEE_RATE = 0.0005
    SLIPPAGE = 0.0002

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    et_tz = pytz.timezone("America/New_York")
    df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    # Precompute PWH/PWL pour toutes les dates
    weekly = build_weekly_levels(df)

    # Construire les episodes (meme logique que StatsEnv)
    episodes, dates = [], []
    for date, grp in df.groupby("date_et"):
        hm_vals    = set(grp["hm_et"].values)
        has_asia   = any(60  <= h < 300 for h in hm_vals)
        has_london = any(420 <= h < 600 for h in hm_vals)
        has_ny     = any(530 <= h < 910 for h in hm_vals)
        if has_asia and has_london and has_ny and len(grp) >= 60:
            episodes.append(grp.sort_values("ts_et").reset_index(drop=True))
            dates.append(date)

    n_total = len(episodes)
    n_train = int(n_total * (1 - test_ratio))
    print(f"[qtable] {n_total} jours | train={n_train} | test={n_total - n_train} ({test_ratio*100:.0f}%)")

    # Accumulateurs : somme des P&L et compte par (state, action)
    pnl_sum   = np.zeros((N_STATES, N_ACTIONS))
    pnl_count = np.zeros((N_STATES, N_ACTIONS), dtype=int)

    REF_WINDOW = 240   # minutes de lookback pour BSL/SSL par macro

    for day_df, date in zip(episodes[:n_train], dates[:n_train]):
        if date.weekday() in skip_days:
            continue
        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc  = ctx["month_ctx"]
        dc  = ctx["day_ctx"]
        lc  = ctx["london_ctx"]

        for mac_idx, (mac_start, mac_end) in MACROS.items():
            if mac_idx in skip_macros:
                continue

            pre_start = mac_start - 20
            pre_mask  = (day_df["hm_et"] >= pre_start) & (day_df["hm_et"] < mac_start)
            exit_mask = (day_df["hm_et"] >= mac_start) & (day_df["hm_et"] < exit_hm)

            pre_df  = day_df[pre_mask]
            exit_df = day_df[exit_mask]

            if len(pre_df) < 3 or len(exit_df) < 5:
                continue

            # Sweep a l'entree (premiere bougie de la macro vs pre-macro high/low)
            pre_high = float(pre_df["high"].max())
            pre_low  = float(pre_df["low"].min())
            first    = exit_df.iloc[0]
            if float(first["high"]) > pre_high:
                sc = 1
            elif float(first["low"]) < pre_low:
                sc = 2
            else:
                sc = 0

            if aligned_only and sc == 0:
                continue

            # Reference BSL/SSL : lookback 4h avant la fenetrep re-macro
            # Capture le high/low de la session precedente (London pour NY, Asia pour London)
            ref_start = pre_start - REF_WINDOW
            ref_mask  = (day_df["hm_et"] >= max(0, ref_start)) & (day_df["hm_et"] < pre_start)
            ref_df    = day_df[ref_mask]
            if len(ref_df) >= 5:
                ref_h = float(ref_df["high"].max())
                ref_l = float(ref_df["low"].min())
            else:
                ref_h = ctx.get("session_high")
                ref_l = ctx.get("session_low")

            pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

            # Regles directionnelles : (mac_idx, lc, pc) -> sc autorise
            if macro_rules:
                allowed_sc = macro_rules.get((mac_idx, lc, pc))
                if allowed_sc is not None and sc not in allowed_sc:
                    continue

            state    = encode(mc, dc, lc, mac_idx, sc, pc)
            entry_px = float(first["open"])

            if target_pool and sl_pct > 0 and pc != 0:
                # TP dynamique : pool oppose (ref_h pour LONG apres SSL, ref_l pour SHORT apres BSL)
                sweep_high = max(float(first["high"]), pre_high)
                sweep_low  = min(float(first["low"]),  pre_low)

                tp_long = tp_short = None
                if pc == 2 and ref_h is not None and ref_h > entry_px:  # SSL swept -> LONG -> TP = ref_h (BSL)
                    tp_long_pct = (ref_h - entry_px) / entry_px
                    sl_long_pct = sl_pct + max(0.0, (entry_px - sweep_low) / entry_px)
                    if tp_long_pct > sl_long_pct > 0:
                        pnl_long = _sim_trade_rr(exit_df, entry_px * (1 + SLIPPAGE), +1,
                                                 sl_long_pct, tp_long_pct, FEE_RATE, SLIPPAGE,
                                                 trailing_delta=trailing_delta)
                        tp_long = True
                if pc == 1 and ref_l is not None and ref_l < entry_px:  # BSL swept -> SHORT -> TP = ref_l (SSL)
                    tp_short_pct = (entry_px - ref_l) / entry_px
                    sl_short_pct = sl_pct + max(0.0, (sweep_high - entry_px) / entry_px)
                    if tp_short_pct > sl_short_pct > 0:
                        pnl_short = _sim_trade_rr(exit_df, entry_px * (1 - SLIPPAGE), -1,
                                                  sl_short_pct, tp_short_pct, FEE_RATE, SLIPPAGE,
                                                  trailing_delta=trailing_delta)
                        tp_short = True

                if tp_long is None:
                    exit_px    = float(exit_df.iloc[-1]["close"])
                    entry_long = entry_px * (1 + SLIPPAGE)
                    pnl_long   = (exit_px - entry_long) / entry_long - FEE_RATE - SLIPPAGE
                if tp_short is None:
                    exit_px     = float(exit_df.iloc[-1]["close"])
                    entry_short = entry_px * (1 - SLIPPAGE)
                    pnl_short   = -(exit_px - entry_short) / entry_short - FEE_RATE - SLIPPAGE
            elif sl_pct > 0:
                tp_pct_val = sl_pct * rr
                pnl_long  = _sim_trade_rr(exit_df, entry_px * (1 + SLIPPAGE), +1,
                                          sl_pct, tp_pct_val, FEE_RATE, SLIPPAGE,
                                          trailing_delta=trailing_delta)
                pnl_short = _sim_trade_rr(exit_df, entry_px * (1 - SLIPPAGE), -1,
                                          sl_pct, tp_pct_val, FEE_RATE, SLIPPAGE,
                                          trailing_delta=trailing_delta)
            else:
                exit_px     = float(exit_df.iloc[-1]["close"])
                entry_long  = entry_px * (1 + SLIPPAGE)
                pnl_long    = (exit_px - entry_long) / entry_long - FEE_RATE - SLIPPAGE
                entry_short = entry_px * (1 - SLIPPAGE)
                pnl_short   = -(exit_px - entry_short) / entry_short - FEE_RATE - SLIPPAGE

            pnl_sum[state, 1]   += pnl_long
            pnl_sum[state, 2]   += pnl_short
            pnl_count[state, 1] += 1
            pnl_count[state, 2] += 1

    # Q-table : moyenne empirique (NaN -> 0, etats insuffisants -> 0)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a in (1, 2):
            n = pnl_count[s, a]
            if n >= min_samples:
                q_table[s, a] = pnl_sum[s, a] / n
            # Q[FLAT] = 0 toujours

    # Sauvegarder dans le meme format que QAgent
    agent = QAgent()
    agent.q_table        = q_table
    agent.epsilon        = 0.0
    agent._episode_count = n_train

    model_path = str(DATA_DIR / "stats_agent.pkl")
    agent.save(model_path)

    # Resume
    visited    = int((pnl_count[:, 1] > 0).sum())
    sufficient = int((pnl_count[:, 1] >= min_samples).sum())
    n_pos_long = int((q_table[:, 1] > 0).sum())
    n_pos_short= int((q_table[:, 2] > 0).sum())
    avg_n      = float(pnl_count[pnl_count[:, 1] > 0, 1].mean()) if visited > 0 else 0

    print(f"\n[qtable] Etats visites       : {visited} / {N_STATES}")
    print(f"[qtable] Etats >= {min_samples} samples  : {sufficient} / {N_STATES}")
    print(f"[qtable] Moyenne N / etat    : {avg_n:.1f}")
    print(f"[qtable] Q(LONG)  > 0        : {n_pos_long} etats")
    print(f"[qtable] Q(SHORT) > 0        : {n_pos_short} etats")

    print("\n[qtable] Top 10 etats LONG :")
    for s in np.argsort(q_table[:, 1])[::-1][:10]:
        if q_table[s, 1] > 0:
            n = pnl_count[s, 1]
            print(f"  state {s:3d}  Q={q_table[s,1]*100:+.4f}%  N={n:3d}  {decode(s)}")

    print("\n[qtable] Top 10 etats SHORT :")
    for s in np.argsort(q_table[:, 2])[::-1][:10]:
        if q_table[s, 2] > 0:
            n = pnl_count[s, 2]
            print(f"  state {s:3d}  Q={q_table[s,2]*100:+.4f}%  N={n:3d}  {decode(s)}")


def run_train_stats(episodes: int = 2000, reset: bool = False, rr: float = 2.0):
    from engine.stats_env   import StatsEnv
    from engine.stats_state import N_STATES, N_ACTIONS, decode
    from engine.q_agent     import QAgent
    from data.binance       import load_binance_1m
    import numpy as np

    df = load_binance_1m()
    env = StatsEnv(df)

    model_path = str(DATA_DIR / "stats_agent.pkl")

    # MC hyperparams : alpha eleve + epsilon_min haut pour explorer suffisamment (2 trades/jour)
    mc_kwargs = dict(alpha=0.2, epsilon_min=0.20, epsilon_decay=0.00008)

    if not reset and Path(model_path).exists():
        agent = QAgent.load(model_path)
        if agent.q_table.shape[0] != N_STATES:
            print(f"[stats] Q-table incompatible ({agent.q_table.shape[0]} vs {N_STATES}) — nouvel agent")
            agent = QAgent(**mc_kwargs)
    else:
        agent = QAgent(**mc_kwargs)
        agent.q_table = np.zeros((N_STATES, N_ACTIONS))

    print(f"[stats] {episodes} episodes | {env.n_episodes} jours disponibles\n")
    log_every = max(100, episodes // 20)
    rewards_window = []

    try:
        for ep in range(episodes):
            state         = env.reset()
            done          = False
            total_r       = 0.0
            pending_trade = None  # (entry_state, entry_action)

            while not done:
                action                        = agent.act(state)
                next_state, reward, done, info = env.step(action)

                # MC : mettre a jour l'etat d'entree avec le P&L realise
                if reward != 0.0 and pending_trade is not None:
                    agent.update_mc(pending_trade[0], pending_trade[1], reward)
                    pending_trade = None

                # Enregistrer l'ouverture de position pour le prochain update MC
                if info.get("trade_opened"):
                    pending_trade = (state, action)

                state    = next_state
                total_r += reward

            agent.end_episode()
            rewards_window.append(total_r)

            if (ep + 1) % log_every == 0:
                avg_r    = np.mean(rewards_window[-log_every:])
                pct_done = (ep + 1) / episodes * 100
                policy   = agent.policy_summary()
                print(
                    f"  [{pct_done:5.1f}%] ep {agent._episode_count:5d} | "
                    f"reward moy={avg_r:+.4f} | eps={agent.epsilon:.3f} | "
                    f"FLAT={policy['FLAT']} LONG={policy['LONG']} SHORT={policy['SHORT']} "
                    f"(etats visites={policy['visited_states']})"
                )

    except KeyboardInterrupt:
        print("\n[stats] Interruption — sauvegarde...")

    agent.save(model_path)

    # Resume des etats les plus actionnes
    print("\n[stats] Top etats LONG :")
    long_states = np.argsort(agent.q_table[:, 1])[::-1][:10]
    for s in long_states:
        if agent.q_table[s, 1] > 0:
            ctx = decode(s)
            print(f"  state {s:3d} Q={agent.q_table[s,1]:+.4f} | {ctx}")

    print("\n[stats] Top etats SHORT :")
    short_states = np.argsort(agent.q_table[:, 2])[::-1][:10]
    for s in short_states:
        if agent.q_table[s, 2] > 0:
            ctx = decode(s)
            print(f"  state {s:3d} Q={agent.q_table[s,2]:+.4f} | {ctx}")


def run_backtest_stats(test_ratio: float = 0.2, q_threshold: float = 0.0,
                       exit_hm: int = 960, sl_pct: float = 0.0, rr: float = 2.0,
                       trailing_delta: float = 0.0,
                       target_pool: bool = False, aligned_only: bool = False,
                       skip_macros: frozenset = frozenset(),
                       skip_days: frozenset = frozenset(),
                       entry_mode: str = "baseline",
                       macro_rules: dict = None):
    """
    Backtest coherent avec run_build_qtable : meme boucle directe, meme exit_hm.
    exit_hm     : sortie en minutes ET (defaut 960 = 16:00 ET)
    sl_pct      : si > 0, SL en % du prix d'entree
    trailing_delta : si > 0, sortie TRAIL sur drawdown du PnL non realise.
    target_pool : TP dynamique = pool oppose (SSL pour SHORT apres BSL, BSL pour LONG apres SSL)
    """
    from engine.stats_state import (
        MACROS, encode, decode,
        compute_daily_context, compute_pool_ctx, build_weekly_levels,
    )
    from engine.q_agent   import QAgent
    from data.binance     import load_binance_1m
    import numpy as np
    import pandas as pd
    import pytz

    FEE_RATE   = 0.0005
    SLIPPAGE   = 0.0002
    MAX_TRADES = 2

    model_path = str(DATA_DIR / "stats_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print(f"[backtest_stats] Modele introuvable. Lancez --build-qtable d'abord.")
        sys.exit(1)

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    et_tz = pytz.timezone("America/New_York")
    df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date
    df = df[df["hm_et"] < 20 * 60]

    weekly = build_weekly_levels(df)

    episodes, dates = [], []
    for date, grp in df.groupby("date_et"):
        hm_vals    = set(grp["hm_et"].values)
        has_asia   = any(60  <= h < 300 for h in hm_vals)
        has_london = any(420 <= h < 600 for h in hm_vals)
        has_ny     = any(530 <= h < 910 for h in hm_vals)
        if has_asia and has_london and has_ny and len(grp) >= 60:
            episodes.append(grp.sort_values("ts_et").reset_index(drop=True))
            dates.append(date)

    n_total = len(episodes)
    n_test  = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test
    exit_label = f"{exit_hm // 60:02d}:{exit_hm % 60:02d} ET"

    print(f"\n[backtest_stats] {n_total} jours | train={n_train} | test={n_test} ({test_ratio*100:.0f}%)")
    filter_label  = " | aligned-only" if aligned_only else ""
    skip_label    = f" | muettes={sorted(skip_macros)}" if skip_macros else ""
    days_label    = f" | skip-days={sorted(skip_days)}" if skip_days else ""
    rules_label   = f" | regles={macro_rules}" if macro_rules else ""
    trail_label = f" | trail={trailing_delta*100:.2f}%" if trailing_delta > 0 else ""
    print(f"[backtest_stats] Agent : {agent._episode_count} episodes | seuil Q > {q_threshold} | sortie {exit_label}{filter_label}{skip_label}{days_label}{rules_label}{trail_label} | entree={entry_mode}\n")

    trades_list = []   # (mac_idx, pnl, exit_reason, entry_px, exit_px, tp_px, sl_px, n_candles, date, direction, state)
    eq_curve    = [1.0]
    equity      = 1.0
    days_traded = 0
    days_flat   = 0

    REF_WINDOW = 240   # minutes de lookback pour BSL/SSL par macro

    for day_df, date in zip(episodes[n_train:], dates[n_train:]):
        if date.weekday() in skip_days:
            continue
        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc  = ctx["month_ctx"]
        dc  = ctx["day_ctx"]
        lc  = ctx["london_ctx"]

        ep_pnl    = 0.0
        ep_trades = 0

        for mac_idx, (mac_start, _) in MACROS.items():
            if ep_trades >= MAX_TRADES:
                break
            if mac_idx in skip_macros:
                continue

            pre_mask  = (day_df["hm_et"] >= mac_start - 20) & (day_df["hm_et"] < mac_start)
            exit_mask = (day_df["hm_et"] >= mac_start)       & (day_df["hm_et"] < exit_hm)

            pre_df  = day_df[pre_mask]
            exit_df = day_df[exit_mask]

            if len(pre_df) < 3 or len(exit_df) < 5:
                continue

            pre_high = float(pre_df["high"].max())
            pre_low  = float(pre_df["low"].min())
            first    = exit_df.iloc[0]

            if float(first["high"]) > pre_high:
                sc = 1
            elif float(first["low"]) < pre_low:
                sc = 2
            else:
                sc = 0

            if aligned_only and sc == 0:
                continue

            # Reference BSL/SSL per-macro (4h lookback)
            ref_start = mac_start - 20 - REF_WINDOW
            ref_mask  = (day_df["hm_et"] >= max(0, ref_start)) & (day_df["hm_et"] < mac_start - 20)
            ref_df    = day_df[ref_mask]
            if len(ref_df) >= 5:
                ref_h = float(ref_df["high"].max())
                ref_l = float(ref_df["low"].min())
            else:
                ref_h = ctx.get("session_high")
                ref_l = ctx.get("session_low")

            pc    = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)

            # Regles directionnelles : (mac_idx, lc, pc) -> sc autorise
            if macro_rules:
                allowed_sc = macro_rules.get((mac_idx, lc, pc))
                if allowed_sc is not None and sc not in allowed_sc:
                    continue

            state = encode(mc, dc, lc, mac_idx, sc, pc)

            action = agent.act(state, training=False)
            if action == 0:
                continue
            if agent.q_table[state, action] <= q_threshold:
                continue

            entry_px  = float(first["open"])
            direction = +1 if action == 1 else -1

            # Technique d'entree
            struct_sl_px = None
            if entry_mode in ("ote", "fvg"):
                finder = _find_ote_entry if entry_mode == "ote" else _find_fvg_entry
                setup  = finder(exit_df, direction)
                if setup is None:
                    continue   # pas de setup trouve, on skippe
                entry_candle_idx, entry_px, struct_sl_px = setup
                exit_df = exit_df.iloc[entry_candle_idx:].reset_index(drop=True)
                if len(exit_df) < 2:
                    continue

            macro_names_map = {1:'08:50',2:'09:50',3:'10:50',4:'11:50',5:'12:50',6:'13:50',7:'14:50'}
            slipped = entry_px * (1 + direction * SLIPPAGE)
            exit_reason = "EOD"
            tp_px_log = sl_px_log = None
            n_candles  = len(exit_df)

            # SL structurel OTE/FVG remplace sl_pct si disponible
            effective_sl = sl_pct
            if struct_sl_px is not None and entry_px > 0:
                effective_sl = max(abs(entry_px - struct_sl_px) / entry_px, 0.001)

            if target_pool and effective_sl > 0 and pc != 0:
                sweep_high = max(float(first["high"]), pre_high)
                sweep_low  = min(float(first["low"]),  pre_low)

                if direction == +1 and pc == 2 and ref_h is not None and ref_h > entry_px:
                    tp_pct_v = (ref_h - entry_px) / entry_px
                    sl_pct_v = effective_sl + max(0.0, (entry_px - sweep_low) / entry_px)
                    if tp_pct_v > sl_pct_v > 0:
                        pnl, exit_reason, exit_px_log, tp_px_log, sl_px_log, n_candles = \
                            _sim_trade_rr(exit_df, slipped, +1, sl_pct_v, tp_pct_v,
                                          FEE_RATE, SLIPPAGE, verbose=True,
                                          trailing_delta=trailing_delta)
                    else:
                        continue
                elif direction == -1 and pc == 1 and ref_l is not None and ref_l < entry_px:
                    tp_pct_v = (entry_px - ref_l) / entry_px
                    sl_pct_v = effective_sl + max(0.0, (sweep_high - entry_px) / entry_px)
                    if tp_pct_v > sl_pct_v > 0:
                        pnl, exit_reason, exit_px_log, tp_px_log, sl_px_log, n_candles = \
                            _sim_trade_rr(exit_df, slipped, -1, sl_pct_v, tp_pct_v,
                                          FEE_RATE, SLIPPAGE, verbose=True,
                                          trailing_delta=trailing_delta)
                    else:
                        continue
                else:
                    continue
            elif effective_sl > 0:
                pnl, exit_reason, exit_px_log, tp_px_log, sl_px_log, n_candles = \
                    _sim_trade_rr(exit_df, slipped, direction,
                                  effective_sl, effective_sl * rr, FEE_RATE, SLIPPAGE, verbose=True,
                                  trailing_delta=trailing_delta)
            else:
                exit_px_v = float(exit_df.iloc[-1]["close"])
                entry_v   = entry_px * (1 + direction * SLIPPAGE)
                pnl       = direction * (exit_px_v - entry_v) / entry_v - FEE_RATE - SLIPPAGE
                exit_px_log = exit_px_v

            ctx_dec = decode(state)
            trades_list.append((
                mac_idx, pnl, exit_reason,
                entry_px, exit_px_log, tp_px_log, sl_px_log, n_candles,
                str(date), "LONG" if direction == 1 else "SHORT",
                ctx_dec["london_ctx"], ctx_dec["pool_ctx"], ctx_dec["sweep_ctx"],
                macro_names_map.get(mac_idx, str(mac_idx)),
            ))
            ep_pnl    += pnl
            ep_trades += 1

        equity *= (1 + ep_pnl)
        eq_curve.append(equity)
        if ep_trades > 0:
            days_traded += 1
        else:
            days_flat += 1

    if not trades_list:
        print("[backtest_stats] Aucun trade — seuil Q trop eleve ou pas de setup.")
        return

    pnls_arr    = np.array([t[1] for t in trades_list])
    reasons     = [t[2] for t in trades_list]
    wins        = pnls_arr[pnls_arr > 0]
    losses      = pnls_arr[pnls_arr < 0]

    n_tp  = reasons.count("TP")
    n_sl  = reasons.count("SL")
    n_tr  = reasons.count("TRAIL")
    n_eod = reasons.count("EOD")

    total_return  = equity - 1.0
    win_rate      = len(wins) / len(pnls_arr)
    avg_win       = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss      = losses.mean() if len(losses) > 0 else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    eq_arr    = np.array(eq_curve)
    peak      = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peak) / peak
    max_dd    = drawdowns.min()

    daily_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    pf_str = f"{profit_factor:.3f}" if profit_factor != float("inf") else "inf"

    n_total_trades = len(pnls_arr)
    print("=" * 58)
    print(f"  Jours test          : {n_test}")
    print(f"  Jours trades        : {days_traded}  ({days_traded/n_test*100:.0f}%)")
    print(f"  Jours flat          : {days_flat}  ({days_flat/n_test*100:.0f}%)")
    print(f"  Trades totaux       : {n_total_trades}")
    if days_traded:
        print(f"  Trades / jour trade : {n_total_trades/days_traded:.1f}")
    print(f"  --- Sorties -------------------------------------------")
    print(f"  TP touche           : {n_tp:3d}  ({n_tp/n_total_trades*100:4.0f}%)")
    print(f"  SL touche           : {n_sl:3d}  ({n_sl/n_total_trades*100:4.0f}%)")
    print(f"  TRAIL touche        : {n_tr:3d}  ({n_tr/n_total_trades*100:4.0f}%)")
    print(f"  EOD (sans SL/TP/TRAIL): {n_eod:3d}  ({n_eod/n_total_trades*100:4.0f}%)")
    if n_tp + n_sl + n_tr > 0:
        if n_tp > 0:
            avg_tp_dur = np.mean([t[7] for t in trades_list if t[2] == "TP"])
            print(f"  Duree moy TP        : {avg_tp_dur:.0f} min")
        if n_sl > 0:
            avg_sl_dur = np.mean([t[7] for t in trades_list if t[2] == "SL"])
            print(f"  Duree moy SL        : {avg_sl_dur:.0f} min")
        if n_tr > 0:
            avg_tr_dur = np.mean([t[7] for t in trades_list if t[2] == "TRAIL"])
            print(f"  Duree moy TRAIL     : {avg_tr_dur:.0f} min")
    print(f"  --- Performance ---------------------------------------")
    print(f"  Return total        : {total_return*100:+.2f}%")
    print(f"  Sharpe annualise    : {sharpe:+.3f}")
    print(f"  Max drawdown        : {max_dd*100:.2f}%")
    print(f"  Win rate            : {win_rate*100:.1f}%")
    print(f"  Profit factor       : {pf_str}")
    print(f"  Avg win             : {avg_win*100:+.3f}%")
    print(f"  Avg loss            : {avg_loss*100:+.3f}%")
    print(f"  Expectancy/trade    : {pnls_arr.mean()*100:+.4f}%")
    print("=" * 58)

    macro_names  = {0:'AUCUNE',1:'08:50',2:'09:50*',3:'10:50',4:'11:50',5:'12:50',6:'13:50',7:'14:50'}
    macro_trades = {i: [] for i in range(8)}
    for t in trades_list:
        macro_trades[t[0]].append(t[1])

    print(f"\n[backtest_stats] Performance par macro :")
    print(f"  {'Macro':<8} | {'N':>5} | {'Win%':>6} | {'Avg%':>8} | {'Total%':>8}")
    print("  " + "-" * 45)
    for m, tlist in macro_trades.items():
        if not tlist: continue
        t = np.array(tlist)
        print(f"  {macro_names[m]:<8} | {len(t):>5} | {(t>0).mean()*100:>5.1f}% | {t.mean()*100:>+7.3f}% | {t.sum()*100:>+7.2f}%")

    # Export CSV des trades
    import csv
    csv_path = str(DATA_DIR / "trades_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","macro","direction","entry_px","exit_px","tp_px","sl_px",
                    "n_candles","exit_reason","pnl_pct","london_ctx","pool_ctx","sweep_ctx"])
        for t in trades_list:
            (mac_idx, pnl, reason, entry, exit_p, tp_p, sl_p, nc, d, dirstr,
             lc_v, pc_v, sc_v, mac_name) = t
            w.writerow([d, mac_name, dirstr,
                        round(entry, 2), round(exit_p, 2) if exit_p else "",
                        round(tp_p, 2) if tp_p else "", round(sl_p, 2) if sl_p else "",
                        nc, reason, round(pnl * 100, 4),
                        ["NO_RAID","RAID_H","RAID_L"][lc_v],
                        ["NEUTRAL","BSL_SWEPT","SSL_SWEPT"][pc_v],
                        ["NO_SW","SW_H","SW_L"][sc_v]])
    print(f"\n[backtest_stats] Log detaille : {csv_path}")


def run_plot_trades(n: int = 20, filter_reason: str = None):
    """
    Charge le CSV de trades et genere des graphiques matplotlib.
    Pour chaque trade : prix 1min, entree, TP, SL, sortie effective.
    filter_reason : 'TP', 'SL', 'EOD' ou None pour tout afficher.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")   # pas besoin d'ecran
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from data.binance   import load_binance_1m
    import pandas as pd
    import pytz

    csv_path = str(DATA_DIR / "trades_log.csv")
    if not Path(csv_path).exists():
        print("[plot] Aucun log trouve. Lancez --backtest-stats d'abord.")
        return

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if filter_reason:
        rows = [r for r in rows if r["exit_reason"] == filter_reason]

    if not rows:
        print(f"[plot] Aucun trade {filter_reason or ''} dans le log.")
        return

    rows = rows[:n]
    print(f"[plot] Generation de {len(rows)} graphiques...")

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    et_tz = pytz.timezone("America/New_York")
    df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
    df["date_et"] = df["ts_et"].dt.date.astype(str)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute

    # Precompute NWOG map (CME times : ven 17:00 ET -> dim 19:00 ET)
    from engine.entry_stats import _build_nwog_map
    nwog_map = _build_nwog_map(df)
    nwog_by_date_str = {str(k): v for k, v in nwog_map.items()}

    plots_dir = DATA_DIR.parent / "display" / "trades"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        date_str  = row["date"]
        macro_str = row["macro"]
        direction = row["direction"]
        entry_px  = float(row["entry_px"])
        tp_px     = float(row["tp_px"]) if row["tp_px"] else None
        sl_px     = float(row["sl_px"]) if row["sl_px"] else None
        exit_px   = float(row["exit_px"]) if row["exit_px"] else None
        n_can     = int(row["n_candles"])
        reason    = row["exit_reason"]
        pnl       = float(row["pnl_pct"])

        day_df = df[df["date_et"] == date_str].sort_values("hm_et")
        if day_df.empty:
            continue

        # Fenetrep : 1h avant le debut de la macro jusqu'a 16h ET
        mac_hm = {
            "08:50": 530, "09:50": 590, "10:50": 650,
            "11:50": 710, "12:50": 770, "13:50": 830, "14:50": 890,
        }.get(macro_str, 530)
        plot_df = day_df[(day_df["hm_et"] >= mac_hm - 60) & (day_df["hm_et"] < 960)]
        if len(plot_df) < 5:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))

        xs     = range(len(plot_df))
        closes = plot_df["close"].values.astype(float)
        highs  = plot_df["high"].values.astype(float)
        lows   = plot_df["low"].values.astype(float)

        ax.plot(xs, closes, color="#aaaaaa", linewidth=0.8, label="Close")
        ax.fill_between(xs, lows, highs, alpha=0.12, color="#666666")

        # Marquer l'entree (premier candle de la macro)
        entry_offset = mac_hm - (mac_hm - 60)  # position relative dans plot_df
        entry_idx = plot_df["hm_et"].searchsorted(mac_hm)
        if entry_idx < len(plot_df):
            ax.axvline(entry_idx, color="#2196F3", linestyle="--", linewidth=1, alpha=0.7)
            ax.scatter([entry_idx], [entry_px], color="#2196F3", zorder=5, s=80,
                       marker="^" if direction == "LONG" else "v", label="Entree")

        # TP et SL horizontaux
        if tp_px:
            ax.axhline(tp_px, color="#4CAF50", linestyle="--", linewidth=1.2, label=f"TP {tp_px:.0f}")
        if sl_px:
            ax.axhline(sl_px, color="#F44336", linestyle="--", linewidth=1.2, label=f"SL {sl_px:.0f}")

        # NWOG (CME gap : ven 17:00 ET -> dim 19:00 ET)
        nwog_info = nwog_by_date_str.get(date_str)
        if nwog_info is not None:
            nwog_top = max(nwog_info[0], nwog_info[1])
            nwog_bot = min(nwog_info[0], nwog_info[1])
            price_range = closes.max() - closes.min()
            if price_range > 0 and (nwog_top - nwog_bot) / (closes.mean()) >= 0.0001:
                ax.axhspan(nwog_bot, nwog_top, alpha=0.15, color="#FF9800", zorder=0)
                ax.axhline(nwog_top, color="#FF9800", linewidth=0.8, linestyle=":")
                ax.axhline(nwog_bot, color="#FF9800", linewidth=0.8, linestyle=":")
                nwog_mid = (nwog_top + nwog_bot) / 2
                ax.axhline(nwog_mid, color="#FF9800", linewidth=1.0, linestyle="-",
                           alpha=0.6, label=f"NWOG mid {nwog_mid:.0f}")

        # Marquer la sortie
        exit_idx = entry_idx + n_can
        if exit_px and exit_idx < len(plot_df):
            color_exit = "#4CAF50" if pnl > 0 else "#F44336"
            ax.scatter([exit_idx], [exit_px], color=color_exit, zorder=5, s=100,
                       marker="x", linewidths=2, label=f"Sortie ({reason})")

        color_bg = "#e8f5e9" if pnl > 0 else "#ffebee"
        ax.set_facecolor(color_bg)

        pnl_sign = "+" if pnl >= 0 else ""
        ax.set_title(
            f"{date_str} | {macro_str} | {direction} | {reason} | PnL: {pnl_sign}{pnl:.3f}%"
            f"\n{row['london_ctx']} | pool={row['pool_ctx']} | sweep={row['sweep_ctx']}",
            fontsize=10
        )
        ax.set_xlabel("Bougies 1min depuis 1h avant la macro")
        ax.set_ylabel("Prix (USD)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

        fname = plots_dir / f"{i+1:03d}_{date_str}_{macro_str.replace(':','')}_{reason}_{direction}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()

    print(f"[plot] {len(rows)} graphiques sauvegardes dans : {plots_dir}")


def run_train_deep(timesteps: int = 1_000_000, reset: bool = False, rr: float = 2.0):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from data.binance import load_binance_1m
    from engine.deep_env import DeepTradingEnv
    from engine.sltp_config import SLTPConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[deep] Device : {device}")

    df = load_binance_1m()
    sltp_config = SLTPConfig(risk_reward_ratio=rr)

    model_path = str(DATA_DIR / "deep_agent")
    norm_path  = str(DATA_DIR / "deep_vecnorm.pkl")

    env = make_vec_env(lambda: DeepTradingEnv(df, sltp_config=sltp_config), n_envs=1)
    if not reset and Path(norm_path).exists():
        env = VecNormalize.load(norm_path, env)
        print("[deep] VecNormalize chargé.")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if not reset and Path(model_path + ".zip").exists():
        model = RecurrentPPO.load(model_path, env=env, device=device)
        print(f"[deep] Modèle chargé : {model_path}.zip")
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose         = 1,
            device          = device,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            learning_rate   = 3e-4,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            policy_kwargs   = {"lstm_hidden_size": 64, "n_lstm_layers": 1,
                               "net_arch": [64, 64]},
        )
        print("[deep] Nouvel agent RecurrentPPO (LSTM 64)")

    n_sessions = DeepTradingEnv(df, sltp_config=sltp_config).n_episodes
    print(f"[deep] {timesteps:,} timesteps | ~{n_sessions} sessions disponibles\n")

    try:
        resuming = not reset and Path(model_path + ".zip").exists()
        model.learn(total_timesteps=timesteps, progress_bar=True,
                    reset_num_timesteps=not resuming)
    except KeyboardInterrupt:
        print("\n[deep] Interruption — sauvegarde...")

    model.save(model_path)
    env.save(norm_path)
    print(f"[deep] Modèle sauvegardé : {model_path}.zip")
    print(f"[deep] VecNormalize    : {norm_path}")


def run_backtest_deep(test_ratio: float = 0.2, rr: float = 2.0):
    import torch
    import numpy as np
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from data.binance import load_binance_1m
    from engine.deep_env import DeepTradingEnv
    from engine.sltp_config import SLTPConfig

    model_path = str(DATA_DIR / "deep_agent")
    norm_path  = str(DATA_DIR / "deep_vecnorm.pkl")

    for p in (model_path + ".zip", norm_path):
        if not Path(p).exists():
            print(f"[backtest_deep] Fichier manquant : {p}. Lancez --train-deep d'abord.")
            return

    df = load_binance_1m()
    sltp_config = SLTPConfig(risk_reward_ratio=rr)
    base_env    = DeepTradingEnv(df, sltp_config=sltp_config)
    all_sessions = base_env._sessions

    n_total = len(all_sessions)
    n_train = int(n_total * (1 - test_ratio))
    n_test  = n_total - n_train
    test_sessions = all_sessions[n_train:]

    print(f"[backtest_deep] {n_total} sessions | train={n_train} | test={n_test} ({test_ratio*100:.0f}%)")

    class _TestEnv(DeepTradingEnv):
        def __init__(self):
            super().__init__(df, sltp_config=sltp_config)
            self._sessions = test_sessions
            self._ep_idx   = 0

    vec_env = DummyVecEnv([_TestEnv])
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=vec_env)

    obs         = vec_env.reset()
    lstm_state  = None
    ep_start    = np.ones((1,), dtype=bool)
    ep_return   = 0.0
    ep_count    = 0

    episode_returns = []
    trade_pnls      = []

    while ep_count < n_test:
        action, lstm_state = model.predict(
            obs, state=lstm_state, episode_start=ep_start, deterministic=True
        )
        obs, rewards, dones, _ = vec_env.step(action)
        ep_start  = dones
        raw_r     = float(rewards[0])
        ep_return += raw_r

        if raw_r != 0.0:
            trade_pnls.append(raw_r)

        if dones[0]:
            episode_returns.append(ep_return)
            ep_return  = 0.0
            ep_count  += 1
            lstm_state = None

    if not trade_pnls:
        print("[backtest_deep] Aucun trade execute.")
        return

    ret_arr  = np.array(episode_returns)
    pnl_arr  = np.array(trade_pnls)
    wins     = pnl_arr[pnl_arr > 0]
    losses   = pnl_arr[pnl_arr < 0]

    total_ret = float(ret_arr.sum())
    win_rate  = len(wins) / len(pnl_arr) * 100
    avg_win   = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0
    pf        = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float("inf")
    exp_trade = float(pnl_arr.mean())

    sharpe = 0.0
    if ret_arr.std() > 0:
        sharpe = (ret_arr.mean() / ret_arr.std()) * np.sqrt(252)

    cum  = np.cumsum(ret_arr)
    dd   = cum - np.maximum.accumulate(cum)
    max_dd = float(dd.min())

    print("\n" + "=" * 52)
    print(f"  Sessions test       : {n_test}")
    print(f"  Trades totaux       : {len(pnl_arr)}")
    print(f"  Trades / session    : {len(pnl_arr)/n_test:.1f}")
    print(f"  Return total        : {total_ret*100:+.2f}%")
    print(f"  Sharpe annualise    : {sharpe:.3f}")
    print(f"  Max drawdown        : {max_dd*100:.2f}%")
    print(f"  Win rate            : {win_rate:.1f}%")
    print(f"  Profit factor       : {pf:.3f}")
    print(f"  Avg win             : {avg_win*100:+.3f}%")
    print(f"  Avg loss            : {avg_loss*100:+.3f}%")
    print(f"  Expectancy/trade    : {exp_trade*100:+.4f}%")
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(description="Pi* — agent RL sur BTC-PERPETUAL Deribit")
    parser.add_argument("--watch",    action="store_true", help="Boucle de collecte continue")
    parser.add_argument("--load",     action="store_true", help="Afficher les dernieres bougies en base")
    parser.add_argument("--db",       action="store_true", help="Resume de la base")
    parser.add_argument("--train",    action="store_true", help="Entrainer l'agent Q-learning")
    parser.add_argument("--eval",     action="store_true", help="Evaluer la Q-table courante")
    parser.add_argument("--backtest", action="store_true", help="Backtest walk-forward de la politique Pi*")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction des sessions en test set (defaut: 0.2)")
    parser.add_argument("--export-csv", action="store_true", help="Exporter les trades backtest en CSV")
    parser.add_argument("--backfill", action="store_true", help="Remplir la DB avec l'historique Deribit")
    parser.add_argument("--limit",    type=int, default=720, help="Bougies a fetcher (defaut: 720)")
    parser.add_argument("--days",     type=int, default=30,  help="Jours d'historique pour --backfill (defaut: 30)")
    parser.add_argument("--episodes",      type=int,   default=500,   help="Episodes d'entrainement Q-learning (defaut: 500)")
    parser.add_argument("--alpha",         type=float, default=None,  help="Taux d'apprentissage (defaut: 0.1)")
    parser.add_argument("--epsilon-decay", type=float, default=None,  help="Decay epsilon par episode (defaut: 0.001)")
    parser.add_argument("--reset",         action="store_true",       help="Repart d'un agent vierge (ignore le modele existant)")
    parser.add_argument("--gen-masks",     action="store_true",       help="Generer les masques d'actions depuis la Q-table et les transitions")
    parser.add_argument("--use-masks",     action="store_true",       help="Appliquer le masque action_mask.npy pendant l'entrainement")
    parser.add_argument("--ev-threshold",  type=float, default=-0.01, help="Seuil Q-valeur pour masquer une action (defaut: -0.01)")
    parser.add_argument("--build-markov",  action="store_true",       help="Construire la chaine de Markov depuis les sessions et evaluer sa precision")
    # ── Deep RL ──────────────────────────────────────────────────
    parser.add_argument("--download-binance", action="store_true",
                        help="Telecharger l'historique Binance BTCUSDT futures 1m")
    parser.add_argument("--start-year",  type=int, default=2020, help="Annee de debut (defaut: 2020)")
    parser.add_argument("--start-month", type=int, default=1,    help="Mois de debut (defaut: 1)")
    parser.add_argument("--end-year",    type=int, default=None,  help="Annee de fin (defaut: mois precedent)")
    parser.add_argument("--end-month",   type=int, default=None,  help="Mois de fin")
    parser.add_argument("--train-deep",  action="store_true",
                        help="Entrainer l'agent RecurrentPPO (LSTM) sur donnees Binance")
    parser.add_argument("--timesteps",   type=int, default=1_000_000,
                        help="Timesteps d'entrainement deep RL (defaut: 1 000 000)")
    parser.add_argument("--train-stats",    action="store_true",
                        help="Entrainer la Q-table sur le contexte statistique ICT")
    parser.add_argument("--backtest-stats", action="store_true",
                        help="Backtester la Q-table statistique ICT sur le test set")
    parser.add_argument("--backtest-deep",  action="store_true",
                        help="Backtester l'agent RecurrentPPO sur le test set")
    parser.add_argument("--entry-stats",    action="store_true",
                        help="Statistiques comparatives FVG / OTE / NWOG / Breaker sur les macros ICT")
    parser.add_argument("--build-qtable",   action="store_true",
                        help="Construire la Q-table empiriquement depuis les donnees historiques")
    parser.add_argument("--min-samples",    type=int, default=10,
                        help="Echantillons minimum par etat pour inclure dans la Q-table (defaut: 10)")
    parser.add_argument("--exit-hm",        type=int, default=960,
                        help="Heure de sortie en minutes ET pour --build-qtable et --backtest-stats (defaut: 960 = 16:00 ET)")
    parser.add_argument("--q-threshold",    type=float, default=0.0,
                        help="Seuil Q minimum pour trader (defaut: 0.0, ex: 0.001 = 0.1%%)")
    parser.add_argument("--sl-pct",         type=float, default=0.0,
                        help="Stop-loss en %% du prix d'entree (defaut: 0 = pas de SL/TP, ex: 0.005 = 0.5%%)")
    parser.add_argument("--rr",             type=float, default=2.0,
                        help="Ratio Risk:Reward (defaut: 2.0 => TP = SL * 2)")
    parser.add_argument("--target-pool",    action="store_true",
                        help="TP dynamique ICT : SSL pour SHORT apres BSL swept, BSL pour LONG apres SSL swept. Necessite --sl-pct.")
    parser.add_argument("--trailing-delta", type=float, default=0.0,
                        help="Trailing stop en %% decimal (ex: 0.002 = 0.2%%). Sortie si drawdown du PnL non realise >= trailing_delta.")
    parser.add_argument("--plot-trades",    type=int, default=0, metavar="N",
                        help="Generer N graphiques de trades depuis le dernier log CSV (ex: --plot-trades 20)")
    parser.add_argument("--plot-filter",    type=str, default=None, choices=["TP","SL","TRAIL","EOD"],
                        help="Filtrer les graphiques par type de sortie : TP, SL, TRAIL ou EOD")
    parser.add_argument("--aligned-only",   action="store_true",
                        help="Ne trader que si sweep_ctx != 0 (sweep detecte a l'entree macro). Filtre les setups sans signal ICT.")
    parser.add_argument("--skip-macros",    type=str, default="",
                        help="Indices de macros a mettre au silence, separes par virgule (ex: 3,5,6 pour 10:50/12:50/13:50)")
    parser.add_argument("--skip-days",      type=str, default="",
                        help="Jours de la semaine a ignorer (0=Lun, 1=Mar, ..., 6=Dim), ex: 0 pour ignorer lundi")
    parser.add_argument("--macro-rules",    type=str, default="",
                        help="Regles directionnelles par contexte. Format: 'mac,lc,pc:sc[+sc]' separes par '|'. "
                             "Ex: '2,1,1:1' = macro 09:50 en contexte RAID_H+BSL_swept -> sc=1 (SW_H) uniquement")
    parser.add_argument("--entry-mode",     type=str, default="baseline",
                        choices=["baseline", "ote", "fvg"],
                        help="Technique d'entree : baseline=open macro, ote=retracement 62-79%%, fvg=retour dans le gap")
    args = parser.parse_args()

    skip_macros = frozenset(int(x) for x in args.skip_macros.split(",") if x.strip().isdigit())
    skip_days   = frozenset(int(x) for x in args.skip_days.split(",")   if x.strip().isdigit())

    macro_rules = None
    if args.macro_rules.strip():
        macro_rules = {}
        for part in args.macro_rules.split("|"):
            part = part.strip()
            if ":" not in part:
                continue
            key_str, val_str = part.split(":", 1)
            key_parts = [x.strip() for x in key_str.split(",")]
            if len(key_parts) != 3:
                continue
            try:
                key = (int(key_parts[0]), int(key_parts[1]), int(key_parts[2]))
                allowed = frozenset(int(x) for x in val_str.split("+") if x.strip().isdigit())
                macro_rules[key] = allowed
            except ValueError:
                continue

    init_db()

    if args.db:
        print(db_summary())
        return

    if args.load:
        df = load_latest_btc_1m(limit=args.limit)
        if df.empty:
            print("[main] Aucune donnee en base. Lancez sans --load pour collecter.")
        else:
            print(df[["timestamp", "open", "close", "volume", "open_interest", "funding_rate"]].tail(20).to_string())
        return

    if args.backfill:
        backfill_btc_1m(days=args.days)
        return

    if args.gen_masks:
        run_gen_masks(ev_threshold=args.ev_threshold, rr=args.rr)
        return

    if args.build_markov:
        run_build_markov(rr=args.rr)
        return

    if args.train:
        run_train(
            episodes      = args.episodes,
            alpha         = args.alpha,
            epsilon_decay = args.epsilon_decay,
            reset         = args.reset,
            use_masks     = args.use_masks,
            rr            = args.rr,
        )
        return

    if args.eval:
        run_eval(rr=args.rr)
        return

    if args.backtest:
        run_backtest(test_ratio=args.test_ratio, export_csv=args.export_csv, rr=args.rr)
        return

    if args.download_binance:
        run_download_binance(
            start_year  = args.start_year,
            start_month = args.start_month,
            end_year    = args.end_year,
            end_month   = args.end_month,
        )
        return

    if args.train_deep:
        run_train_deep(timesteps=args.timesteps, reset=args.reset, rr=args.rr)
        return

    if args.train_stats:
        run_train_stats(episodes=args.episodes, reset=args.reset, rr=args.rr)
        return

    if args.backtest_stats:
        run_backtest_stats(test_ratio=args.test_ratio, q_threshold=args.q_threshold,
                           exit_hm=args.exit_hm, sl_pct=args.sl_pct, rr=args.rr,
                           trailing_delta=args.trailing_delta,
                           target_pool=args.target_pool, aligned_only=args.aligned_only,
                           skip_macros=skip_macros, skip_days=skip_days,
                           entry_mode=args.entry_mode, macro_rules=macro_rules)
        return

    if args.backtest_deep:
        run_backtest_deep(test_ratio=args.test_ratio, rr=args.rr)
        return

    if args.build_qtable:
        run_build_qtable(test_ratio=args.test_ratio, min_samples=args.min_samples,
                         exit_hm=args.exit_hm, sl_pct=args.sl_pct, rr=args.rr,
                         trailing_delta=args.trailing_delta,
                         target_pool=args.target_pool, aligned_only=args.aligned_only,
                         skip_macros=skip_macros, skip_days=skip_days,
                         macro_rules=macro_rules)
        return

    if args.plot_trades:
        run_plot_trades(n=args.plot_trades, filter_reason=args.plot_filter)
        return

    if args.entry_stats:
        from engine.entry_stats import compute_entry_stats, print_entry_stats
        from data.binance import load_binance_1m
        print("[entry_stats] Analyse sur donnees Binance 1m...")
        df = load_binance_1m()
        results = compute_entry_stats(df)
        print_entry_stats(results)
        return

    if args.watch:
        run_watch(limit=args.limit)
    else:
        run_once(limit=args.limit)


if __name__ == "__main__":
    main()
