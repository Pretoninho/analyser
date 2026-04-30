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
                  fee: float = 0.0005, slip: float = 0.0002) -> float:
    """
    Simule un trade avec SL et TP fixes sur les bougies 1min de exit_df.
    direction : +1 LONG, -1 SHORT
    Retourne le P&L net (fees + slippage).
    SL conservateur : si SL et TP dans le meme candle, on assume SL touche en premier.
    """
    if direction == 1:
        tp_px = entry_px * (1 + tp_pct)
        sl_px = entry_px * (1 - sl_pct)
    else:
        tp_px = entry_px * (1 - tp_pct)
        sl_px = entry_px * (1 + sl_pct)

    for _, row in exit_df.iterrows():
        h = float(row["high"])
        l = float(row["low"])
        if direction == 1:
            if l <= sl_px:
                return -sl_pct - fee - slip * 2
            if h >= tp_px:
                return  tp_pct - fee - slip * 2
        else:
            if h >= sl_px:
                return -sl_pct - fee - slip * 2
            if l <= tp_px:
                return  tp_pct - fee - slip * 2

    close = float(exit_df.iloc[-1]["close"])
    raw = direction * (close - entry_px) / entry_px
    return raw - fee - slip * 2


def run_build_qtable(test_ratio: float = 0.2, min_samples: int = 10,
                     exit_hm: int = 960, sl_pct: float = 0.0, rr: float = 2.0,
                     use_microstructure: bool = False,
                     micro_ofi_threshold: float = 0.10,
                     micro_allow_neutral: bool = True,
                     output: str = "",
                     exclude_macros: set = None):
    """
    Construit la Q-table empiriquement depuis les donnees historiques.
    Entre a l'ouverture de la macro, sort a exit_hm (16:00 ET par defaut).
    Si sl_pct > 0 : simulation candle-par-candle avec SL=sl_pct et TP=sl_pct*rr.
    Q[etat, action] = moyenne des P&L nets sur tous les exemples.
    Etat 1944 : month x day x london x macro x sweep x pool(BSL/SSL)
    """
    from engine.stats_state import (
        N_STATES, N_ACTIONS, MACROS,
        encode, decode, compute_daily_context,
        compute_pool_ctx, build_weekly_levels,
        attach_microstructure_overlay, microstructure_trade_allowed,
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

    if use_microstructure:
        df = attach_microstructure_overlay(
            df,
            config={
                "ofi_threshold": micro_ofi_threshold,
                "allow_neutral": micro_allow_neutral,
            },
        )

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
    micro_blocked = 0

    for day_df, date in zip(episodes[:n_train], dates[:n_train]):
        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc  = ctx["month_ctx"]
        dc  = ctx["day_ctx"]
        lc  = ctx["london_ctx"]
        ldn_h = ctx.get("london_high")
        ldn_l = ctx.get("london_low")

        for mac_idx, (mac_start, mac_end) in MACROS.items():
            if exclude_macros and mac_idx in exclude_macros:
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
            micro_row = pre_df.iloc[-1]
            if float(first["high"]) > pre_high:
                sc = 1
            elif float(first["low"]) < pre_low:
                sc = 2
            else:
                sc = 0

            # Pool ctx : sweep de London High/Low ou PWH/PWL
            pc = compute_pool_ctx(pre_high, pre_low, ldn_h, ldn_l, pwh, pwl)

            state    = encode(mc, dc, lc, mac_idx, sc, pc)
            entry_px = float(first["open"])

            if sl_pct > 0:
                tp_pct_val = sl_pct * rr
                pnl_long  = _sim_trade_rr(exit_df, entry_px * (1 + SLIPPAGE), +1,
                                          sl_pct, tp_pct_val, FEE_RATE, SLIPPAGE)
                pnl_short = _sim_trade_rr(exit_df, entry_px * (1 - SLIPPAGE), -1,
                                          sl_pct, tp_pct_val, FEE_RATE, SLIPPAGE)
            else:
                exit_px     = float(exit_df.iloc[-1]["close"])
                entry_long  = entry_px * (1 + SLIPPAGE)
                pnl_long    = (exit_px - entry_long) / entry_long - FEE_RATE - SLIPPAGE
                entry_short = entry_px * (1 - SLIPPAGE)
                pnl_short   = -(exit_px - entry_short) / entry_short - FEE_RATE - SLIPPAGE

            allow_long = True
            allow_short = True
            if use_microstructure:
                allow_long = microstructure_trade_allowed(
                    micro_row, action=1, allow_neutral=micro_allow_neutral,
                )
                allow_short = microstructure_trade_allowed(
                    micro_row, action=2, allow_neutral=micro_allow_neutral,
                )

            if allow_long:
                pnl_sum[state, 1]   += pnl_long
                pnl_count[state, 1] += 1
            else:
                micro_blocked += 1

            if allow_short:
                pnl_sum[state, 2]   += pnl_short
                pnl_count[state, 2] += 1
            else:
                micro_blocked += 1

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

    model_path = output if output else str(DATA_DIR / "stats_agent.pkl")
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
    if use_microstructure:
        print(f"[qtable] Actions filtrees micro : {micro_blocked}")

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


def run_train_stats(episodes: int = 2000, reset: bool = False, rr: float = 2.0,
                    use_microstructure: bool = False,
                    micro_ofi_threshold: float = 0.10,
                    micro_allow_neutral: bool = True):
    from engine.stats_env   import StatsEnv
    from engine.stats_state import N_STATES, N_ACTIONS, decode
    from engine.q_agent     import QAgent
    from data.binance       import load_binance_1m
    import numpy as np

    df = load_binance_1m()
    env = StatsEnv(
        df,
        microstructure_enabled=use_microstructure,
        microstructure_config={
            "ofi_threshold": micro_ofi_threshold,
            "allow_neutral": micro_allow_neutral,
        },
    )

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

    print(f"[stats] {episodes} episodes | {env.n_episodes} jours disponibles")
    if use_microstructure:
        print(f"[stats] Filtre microstructure actif | ofi_threshold={micro_ofi_threshold:.3f} | allow_neutral={micro_allow_neutral}")
    print()
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
                       use_microstructure: bool = False,
                       micro_ofi_threshold: float = 0.10,
                       micro_allow_neutral: bool = True,
                       model: str = "",
                       exclude_macros: set = None):
    """
    Backtest coherent avec run_build_qtable : meme boucle directe, meme exit_hm.
    exit_hm : sortie en minutes ET (defaut 960 = 16:00 ET)
    sl_pct  : si > 0, simulation SL/TP (SL=sl_pct, TP=sl_pct*rr)
    """
    from engine.stats_state import (
        MACROS, encode, decode,
        compute_daily_context, compute_pool_ctx, build_weekly_levels,
        attach_microstructure_overlay, microstructure_trade_allowed,
    )
    from engine.q_agent   import QAgent
    from data.binance     import load_binance_1m
    import numpy as np
    import pandas as pd
    import pytz

    FEE_RATE   = 0.0005
    SLIPPAGE   = 0.0002
    MAX_TRADES = 2

    model_path = model if model else str(DATA_DIR / "stats_agent.pkl")
    try:
        agent = QAgent.load(model_path)
    except FileNotFoundError:
        print(f"[backtest_stats] Modele introuvable. Lancez --build-qtable d'abord.")
        sys.exit(1)

    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if use_microstructure:
        df = attach_microstructure_overlay(
            df,
            config={
                "ofi_threshold": micro_ofi_threshold,
                "allow_neutral": micro_allow_neutral,
            },
        )
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
    print(f"[backtest_stats] Agent : {agent._episode_count} episodes | seuil Q > {q_threshold} | sortie {exit_label}\n")

    trades_list = []   # (mac_idx, pnl)
    eq_curve    = [1.0]
    equity      = 1.0
    days_traded = 0
    days_flat   = 0
    micro_blocked = 0

    for day_df, date in zip(episodes[n_train:], dates[n_train:]):
        pwh, pwl = weekly.get(date, (None, None))
        ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
        mc    = ctx["month_ctx"]
        dc    = ctx["day_ctx"]
        lc    = ctx["london_ctx"]
        ldn_h = ctx.get("london_high")
        ldn_l = ctx.get("london_low")

        ep_pnl    = 0.0
        ep_trades = 0

        for mac_idx, (mac_start, _) in MACROS.items():
            if ep_trades >= MAX_TRADES:
                break
            if exclude_macros and mac_idx in exclude_macros:
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
            micro_row = pre_df.iloc[-1]

            if float(first["high"]) > pre_high:
                sc = 1
            elif float(first["low"]) < pre_low:
                sc = 2
            else:
                sc = 0

            pc    = compute_pool_ctx(pre_high, pre_low, ldn_h, ldn_l, pwh, pwl)
            state = encode(mc, dc, lc, mac_idx, sc, pc)

            action = agent.act(state, training=False)
            if action == 0:
                continue
            if agent.q_table[state, action] <= q_threshold:
                continue
            if use_microstructure and not microstructure_trade_allowed(
                micro_row, action=action, allow_neutral=micro_allow_neutral,
            ):
                micro_blocked += 1
                continue

            entry_px = float(first["open"])
            direction = +1 if action == 1 else -1

            if sl_pct > 0:
                slipped = entry_px * (1 + direction * SLIPPAGE)
                pnl = _sim_trade_rr(exit_df, slipped, direction,
                                    sl_pct, sl_pct * rr, FEE_RATE, SLIPPAGE)
            else:
                exit_px = float(exit_df.iloc[-1]["close"])
                entry   = entry_px * (1 + direction * SLIPPAGE)
                pnl     = direction * (exit_px - entry) / entry - FEE_RATE - SLIPPAGE

            trades_list.append((mac_idx, pnl))
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

    pnls_arr = np.array([p for _, p in trades_list])
    wins      = pnls_arr[pnls_arr > 0]
    losses    = pnls_arr[pnls_arr < 0]

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

    print("=" * 52)
    print(f"  Jours test          : {n_test}")
    print(f"  Jours trades        : {days_traded}  ({days_traded/n_test*100:.0f}%)")
    print(f"  Jours flat          : {days_flat}  ({days_flat/n_test*100:.0f}%)")
    print(f"  Trades totaux       : {len(pnls_arr)}")
    if days_traded:
        print(f"  Trades / jour trade : {len(pnls_arr)/days_traded:.1f}")
    print(f"  Return total        : {total_return*100:+.2f}%")
    print(f"  Sharpe annualise    : {sharpe:+.3f}")
    print(f"  Max drawdown        : {max_dd*100:.2f}%")
    print(f"  Win rate            : {win_rate*100:.1f}%")
    print(f"  Profit factor       : {pf_str}")
    print(f"  Avg win             : {avg_win*100:+.3f}%")
    print(f"  Avg loss            : {avg_loss*100:+.3f}%")
    print(f"  Expectancy/trade    : {pnls_arr.mean()*100:+.4f}%")
    if use_microstructure:
        print(f"  Trades bloques micro: {micro_blocked}")
    print("=" * 52)

    macro_names  = {0:'AUCUNE',1:'08:50',2:'09:50*',3:'10:50',4:'11:50',5:'12:50',6:'13:50',7:'14:50'}
    macro_trades = {i: [] for i in range(8)}
    for mac, r in trades_list:
        macro_trades[mac].append(r)

    print(f"\n[backtest_stats] Performance par macro :")
    print(f"  {'Macro':<8} | {'N':>5} | {'Win%':>6} | {'Avg%':>8} | {'Total%':>8}")
    print("  " + "-" * 45)
    for m, tlist in macro_trades.items():
        if not tlist: continue
        t = np.array(tlist)
        print(f"  {macro_names[m]:<8} | {len(t):>5} | {(t>0).mean()*100:>5.1f}% | {t.mean()*100:>+7.3f}% | {t.sum()*100:>+7.2f}%")


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
    parser.add_argument("--use-microstructure", action="store_true",
                        help="Activer le filtre microstructure 1m dans le flux stats (OFF par defaut)")
    parser.add_argument("--micro-ofi-threshold", type=float, default=0.10,
                        help="Seuil absolu OFI pour le biais directionnel microstructure (defaut: 0.10)")
    parser.add_argument("--micro-no-neutral", action="store_true",
                        help="Refuser les entrees quand le biais microstructure est neutre")
    parser.add_argument("--qtable-output",     type=str, default="",
                        help="Chemin de sauvegarde de la Q-table (defaut: data/stats_agent.pkl)")
    parser.add_argument("--qtable-model",      type=str, default="",
                        help="Chemin du modele Q-table a charger pour --backtest-stats")
    parser.add_argument("--exclude-macros",    type=int, nargs="+", default=[],
                        help="Indices de macros a exclure (ex: 4 = 11:50, 5 = 12:50)")
    args = parser.parse_args()

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
        run_train_stats(
            episodes=args.episodes,
            reset=args.reset,
            rr=args.rr,
            use_microstructure=args.use_microstructure,
            micro_ofi_threshold=args.micro_ofi_threshold,
            micro_allow_neutral=not args.micro_no_neutral,
        )
        return

    if args.backtest_stats:
        run_backtest_stats(
            test_ratio=args.test_ratio,
            q_threshold=args.q_threshold,
            exit_hm=args.exit_hm,
            sl_pct=args.sl_pct,
            rr=args.rr,
            use_microstructure=args.use_microstructure,
            micro_ofi_threshold=args.micro_ofi_threshold,
            micro_allow_neutral=not args.micro_no_neutral,
            model=args.qtable_model,
            exclude_macros=set(args.exclude_macros) if args.exclude_macros else None,
        )
        return

    if args.backtest_deep:
        run_backtest_deep(test_ratio=args.test_ratio, rr=args.rr)
        return

    if args.build_qtable:
        run_build_qtable(
            test_ratio=args.test_ratio,
            min_samples=args.min_samples,
            exit_hm=args.exit_hm,
            sl_pct=args.sl_pct,
            rr=args.rr,
            use_microstructure=args.use_microstructure,
            micro_ofi_threshold=args.micro_ofi_threshold,
            micro_allow_neutral=not args.micro_no_neutral,
            output=args.qtable_output,
            exclude_macros=set(args.exclude_macros) if args.exclude_macros else None,
        )
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
