"""
dashboard/app.py — Interface de consultation Pi*.

Lancement : streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR
from data   import init_db, load_latest_btc_1m, db_summary
from engine import (
    compute_states, aggregate_5m,
    Session, Volatility, PriceStructure, HTFBias,
    TradingEnv, QAgent, N_STATES, N_ACTIONS,
    run_rl_backtest, rl_trades_to_df,
    compute_transition_stats, build_action_mask, mask_summary,
    detect_pattern, pattern_description, PATTERN_NAMES,
    MarkovChain,
)

# ── Config page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pi* — BTC Dashboard",
    page_icon="₿",
    layout="wide",
)

# ── Labels & couleurs ──────────────────────────────────────────
SESSION_LABELS      = {0: "ASIA", 1: "LONDON", 2: "NY AM", 3: "NY PM"}
POSITION_LABELS     = {0: "FLAT", 1: "LONG", 2: "SHORT"}
VOLATILITY_LABELS   = {0: "LOW", 1: "RISING", 2: "HIGH", 3: "EXTREME", 4: "FALLING"}
PRICE_STRUCT_LABELS = {0: "RANGE", 1: "UPTREND", 2: "DOWNTREND", 3: "BREAKOUT", 4: "REJECTION"}
ACTION_LABELS       = {0: "FLAT", 1: "LONG", 2: "SHORT"}
ACTION_COLORS       = {0: "#888888", 1: "#00C853", 2: "#D50000"}
SESSION_COLORS      = {0: "#1565C0", 1: "#6A1B9A", 2: "#E65100", 3: "#1B5E20"}
VOL_COLORS          = {0: "#4CAF50", 1: "#FFC107", 2: "#FF9800", 3: "#F44336", 4: "#2196F3"}
PS_COLORS           = {0: "#9E9E9E", 1: "#4CAF50", 2: "#F44336", 3: "#FF9800", 4: "#9C27B0"}
HTF_LABELS          = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}
HTF_COLORS          = {0: "#00C853", 1: "#888888", 2: "#D50000"}

_DARK = dict(
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white",
    xaxis=dict(gridcolor="#2a2a2a"),
    yaxis=dict(gridcolor="#2a2a2a"),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── Chargement ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _init():
    init_db()

@st.cache_data(ttl=60, show_spinner=False)
def load_data(limit: int = 10000):
    return load_latest_btc_1m(limit=limit)

@st.cache_data(ttl=60, show_spinner=False)
def load_states(limit: int = 10000):
    df = load_latest_btc_1m(limit=limit)
    if df.empty:
        return pd.DataFrame()
    return compute_states(df)

def load_agent():
    path = DATA_DIR / "q_agent.pkl"
    if path.exists():
        return QAgent.load(str(path))
    return None

def load_markov():
    path = DATA_DIR / "markov.pkl"
    if path.exists():
        return MarkovChain.load(str(path))
    return None


# ── Utilitaires UI ─────────────────────────────────────────────

def badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:8px;font-weight:bold;font-size:14px">{label}</span>'
    )


def _dark_fig(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(height=height, **_DARK)
    return fig


# ── Graphiques live ────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, n: int = 120) -> go.Figure:
    df5 = aggregate_5m(df).tail(n)
    fig = go.Figure(go.Candlestick(
        x=df5["timestamp"],
        open=df5["open"], high=df5["high"],
        low=df5["low"],   close=df5["close"],
        increasing_line_color="#00C853",
        decreasing_line_color="#D50000",
        name="BTC 5m",
    ))
    fig.update_layout(title="BTC-PERPETUAL — bougies 5min", xaxis_rangeslider_visible=False)
    return _dark_fig(fig, 400)


def oi_chart(df: pd.DataFrame, n: int = 120) -> go.Figure:
    df5 = aggregate_5m(df).tail(n)
    fig = go.Figure(go.Scatter(
        x=df5["timestamp"], y=df5["open_interest"],
        mode="lines", line=dict(color="#FF9800", width=1.5), name="OI",
    ))
    fig.update_layout(title="Open Interest")
    return _dark_fig(fig, 200)


def qtable_heatmap(agent: QAgent) -> go.Figure:
    actions = np.argmax(agent.q_table, axis=1)
    visited = agent.q_table.sum(axis=1) != 0
    n_market = 4 * 25  # 100 market states (session × intra-session)
    grid = np.full((4, 25), -1.0)
    for s in range(N_STATES):
        ms = s % n_market  # collapse position dimension onto market states
        r, c = divmod(ms, 25)
        if visited[s]:
            grid[r, c] = float(actions[s])

    fig = go.Figure(go.Heatmap(
        z=grid,
        y=[SESSION_LABELS[i] for i in range(4)],
        colorscale=[
            [0.0, "#333333"], [0.25, "#888888"],
            [0.5, "#00C853"], [0.75, "#00C853"], [1.0, "#D50000"],
        ],
        zmin=-1, zmax=2, showscale=False,
        hovertemplate="Session=%{y}<br>Etat=%{x}<br>Action=%{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Q-table — action dominante (gris=non visite, vert=LONG, rouge=SHORT)",
        xaxis=dict(title="Etat intra-session (0-24)", gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
    )
    return _dark_fig(fig, 250)


# ── Tab Live ───────────────────────────────────────────────────

def tab_live(df_1m, df5, agent, n_candles):
    last      = df_1m.iloc[-1]
    prev      = df_1m.iloc[-2]
    price_chg = (last["close"] - prev["close"]) / prev["close"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Prix",          f"${last['close']:,.2f}",  f"{price_chg:+.3f}%")
    c2.metric("Volume (1min)",     f"{last['volume']:.3f} BTC")
    oi_val = last.get("open_interest")
    c3.metric("Open Interest",     f"{oi_val:,.0f} BTC" if pd.notna(oi_val) else "N/A")
    fr_val = last.get("funding_rate")
    c4.metric("Funding Rate (8h)", f"{fr_val:.6f}" if pd.notna(fr_val) else "N/A")
    st.caption(f"Derniere bougie : {last['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
    st.divider()

    # Etat courant
    st.subheader("Etat courant")
    if not df5.empty:
        last5 = df5.iloc[-1]
        sess  = int(last5["session"])
        vol   = int(last5["volatility"])
        ps    = int(last5["price_structure"])
        sid   = int(last5["state_id"])

        htf = int(last5.get("htf_bias", int(HTFBias.NEUTRAL)))

        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        with cc1:
            st.markdown("**Session**")
            st.markdown(badge(SESSION_LABELS[sess], SESSION_COLORS[sess]), unsafe_allow_html=True)
        with cc2:
            st.markdown("**Volatilite**")
            st.markdown(badge(VOLATILITY_LABELS[vol], VOL_COLORS[vol]), unsafe_allow_html=True)
        with cc3:
            st.markdown("**Structure**")
            st.markdown(badge(PRICE_STRUCT_LABELS[ps], PS_COLORS[ps]), unsafe_allow_html=True)
        with cc4:
            st.markdown("**Biais HTF (1h)**")
            st.markdown(badge(HTF_LABELS[htf], HTF_COLORS[htf]), unsafe_allow_html=True)
        with cc5:
            st.markdown("**State ID**")
            st.markdown(f"<span style='font-size:24px;font-weight:bold'>{sid}</span>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Recommandation Pi*")
        # Pattern Price Action courant
        if len(df5) >= 2:
            prev_ps = int(df5.iloc[-2]["price_structure"])
            curr_ps = int(df5.iloc[-1]["price_structure"])
            pattern = detect_pattern(prev_ps, curr_ps)
        else:
            pattern = None

        PATTERN_COLORS = {
            "FAUX_BREAKOUT":    "#D50000",
            "CONTINUATION_H":  "#00C853",
            "CONTINUATION_B":  "#D50000",
            "REVERSAL_H":      "#00C853",
            "REVERSAL_B":      "#D50000",
            "DOUBLE_REJECTION":"#FF6F00",
            "RANGE_BREAKOUT":  "#FF9800",
            "TOP":             "#D50000",
            "BOTTOM":          "#00C853",
            "UPTREND_CONT":    "#00C853",
            "DOWNTREND_CONT":  "#D50000",
        }
        p_color = PATTERN_COLORS.get(pattern, "#555555")
        p_label = pattern if pattern else "Aucun pattern"
        p_desc  = pattern_description(pattern)
        st.markdown(
            f'<div style="border:1px solid {p_color};padding:10px 16px;border-radius:8px;margin-bottom:12px">'
            f'<span style="color:{p_color};font-weight:bold;font-size:16px">⬡ {p_label}</span>'
            f'<span style="color:#aaa;font-size:13px;margin-left:12px">{p_desc}</span>'
            f'</div>', unsafe_allow_html=True,
        )

        if agent is not None:
            from engine.patterns import apply_pattern_to_state_mask
            q_vals    = agent.q_table[sid].copy()
            pmask     = apply_pattern_to_state_mask(None, pattern, sid)
            q_filtered = q_vals.copy()
            q_filtered[~pmask] = float("nan")
            action = int(np.argmax(q_vals))
            action_filtered = int(np.nanargmax(q_filtered))

            col_rec, col_qvals = st.columns([1, 2])
            with col_rec:
                color = ACTION_COLORS[action_filtered]
                st.markdown(
                    f'<div style="background:{color};padding:20px;border-radius:12px;text-align:center">'
                    f'<span style="color:white;font-size:32px;font-weight:bold">{ACTION_LABELS[action_filtered]}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
                if action != action_filtered:
                    st.caption(f"Q-table: {ACTION_LABELS[action]} → filtré par pattern: **{ACTION_LABELS[action_filtered]}**")
                st.caption(f"Epsilon : {agent.epsilon:.3f} | Episodes : {agent._episode_count}")
            with col_qvals:
                bar_colors = [
                    ACTION_COLORS[i] if pmask[i] else "#333333"
                    for i in range(N_ACTIONS)
                ]
                fig_q = go.Figure(go.Bar(
                    x=[ACTION_LABELS[i] for i in range(N_ACTIONS)],
                    y=q_vals,
                    marker_color=bar_colors,
                    text=["✓" if pmask[i] else "✗" for i in range(N_ACTIONS)],
                    textposition="outside",
                ))
                fig_q.update_layout(title=f"Q-valeurs — etat {sid} (gris=masqué par pattern)")
                st.plotly_chart(_dark_fig(fig_q, 220), use_container_width=True)
        else:
            st.info("Aucun modele entraine. Lancez : `python main.py --train`")

    st.divider()
    st.subheader("Prix")
    st.plotly_chart(candlestick_chart(df_1m, n=n_candles), use_container_width=True)
    if df_1m["open_interest"].notna().any():
        st.plotly_chart(oi_chart(df_1m, n=n_candles), use_container_width=True)

    if agent is not None:
        st.divider()
        st.subheader("Q-table — politique apprise")
        policy = agent.policy_summary()
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Etats visites",  policy["visited_states"])
        pc2.metric("FLAT dominant",  policy["FLAT"])
        pc3.metric("LONG dominant",  policy["LONG"])
        pc4.metric("SHORT dominant", policy["SHORT"])
        st.plotly_chart(qtable_heatmap(agent), use_container_width=True)

    if not df5.empty:
        st.divider()
        st.subheader("Historique des etats (5min)")
        cols_show = ["timestamp", "close", "session", "volatility", "price_structure", "state_id"]
        dfd = df5[cols_show].tail(50).copy()
        dfd["session"]         = dfd["session"].map(SESSION_LABELS)
        dfd["volatility"]      = dfd["volatility"].map(VOLATILITY_LABELS)
        dfd["price_structure"] = dfd["price_structure"].map(PRICE_STRUCT_LABELS)
        dfd["close"]           = dfd["close"].map("${:,.2f}".format)
        dfd["timestamp"]       = dfd["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(dfd, use_container_width=True, hide_index=True)


# ── Tab Backtest ───────────────────────────────────────────────

def tab_backtest(df_1m, agent, test_ratio, run_btn, export_btn):
    st.subheader("Backtest walk-forward — Pi*")
    st.caption(
        "Le test set correspond aux dernieres sessions (out-of-sample). "
        "L'agent agit en mode greedy (epsilon=0)."
    )

    if agent is None:
        st.warning("Aucun modele charge. Lancez `python main.py --train` d'abord.")
        return

    # Gestion du resultat en session_state
    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None
    if "bt_df" not in st.session_state:
        st.session_state.bt_df = None

    if run_btn:
        with st.spinner("Calcul du backtest..."):
            env = TradingEnv(df_1m)
            result = run_rl_backtest(agent, env.sessions, test_ratio=test_ratio)
            st.session_state.bt_result = result
            st.session_state.bt_df     = rl_trades_to_df(result.trades)

    result = st.session_state.bt_result
    df_trades = st.session_state.bt_df

    if result is None:
        st.info("Cliquez sur **Lancer le backtest** dans la barre laterale.")
        return

    if result.n_trades == 0:
        st.warning("Aucun trade execute. L'agent recommande probablement uniquement FLAT sur ce test set.")
        return

    # ── Metriques globales ─────────────────────────────────────
    pf_str = f"{result.profit_factor:.3f}" if result.profit_factor != float("inf") else "∞"
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Episodes test",   result.n_episodes)
    m2.metric("Trades",          result.n_trades)
    m3.metric("Return total",    f"{result.total_return_pct*100:+.2f}%")
    m4.metric("Sharpe",          f"{result.sharpe:+.3f}")
    m5.metric("Max drawdown",    f"{result.max_drawdown_pct*100:.2f}%")
    m6.metric("Win rate",        f"{result.win_rate*100:.1f}%")

    m7, m8, _, _, _, _ = st.columns(6)
    m7.metric("Profit factor",   pf_str)
    m8.metric("Avg win / loss",  f"{result.avg_win_pct*100:+.3f}% / {result.avg_loss_pct*100:+.3f}%")

    st.divider()

    col_eq, col_sess = st.columns([3, 2])

    # ── Courbe equity ─────────────────────────────────────────
    with col_eq:
        eq = result.equity_curve
        is_pos   = float(eq.iloc[-1]) >= 0
        color_eq = "#00C853" if is_pos else "#D50000"
        fill_eq  = "rgba(0,200,83,0.15)" if is_pos else "rgba(213,0,0,0.15)"
        fig_eq = go.Figure(go.Scatter(
            y=eq, mode="lines",
            line=dict(color=color_eq, width=2),
            fill="tozeroy",
            fillcolor=fill_eq,
            name="PnL cumulatif",
        ))
        fig_eq.add_hline(y=0, line_dash="dot", line_color="#555")
        fig_eq.update_layout(
            title="Courbe equity (PnL cumulatif en %)",
            xaxis_title="Trade #",
            yaxis_title="PnL cumulatif",
        )
        st.plotly_chart(_dark_fig(fig_eq, 320), use_container_width=True)

    # ── Stats par session ─────────────────────────────────────
    with col_sess:
        if not result.session_stats.empty:
            ss = result.session_stats.copy()
            ss["avg_pnl_pct"]   = (ss["avg_pnl_pct"]   * 100).round(3)
            ss["total_pnl_pct"] = (ss["total_pnl_pct"]  * 100).round(2)
            ss["win_rate"]      = (ss["win_rate"]        * 100).round(1)
            colors_sess = [
                "#00C853" if v >= 0 else "#D50000"
                for v in ss["avg_pnl_pct"]
            ]
            fig_sess = go.Figure(go.Bar(
                x=ss["session"], y=ss["avg_pnl_pct"],
                marker_color=colors_sess,
                text=[f"{v:+.3f}%" for v in ss["avg_pnl_pct"]],
                textposition="outside",
            ))
            fig_sess.update_layout(title="PnL moyen par session (%)")
            st.plotly_chart(_dark_fig(fig_sess, 320), use_container_width=True)

    # ── Graphique prix + trades ───────────────────────────────
    st.divider()
    st.subheader("Trades sur le prix")

    # Periode couverte par les trades
    t_min = df_trades["entry_ts"].min()
    t_max = df_trades["exit_ts"].max()
    df_price = aggregate_5m(df_1m)
    df_price = df_price[(df_price["timestamp"] >= t_min) & (df_price["timestamp"] <= t_max)]

    if not df_price.empty:
        fig_trades = go.Figure()

        # Chandeliers
        fig_trades.add_trace(go.Candlestick(
            x=df_price["timestamp"],
            open=df_price["open"], high=df_price["high"],
            low=df_price["low"],   close=df_price["close"],
            increasing_line_color="#00C853", decreasing_line_color="#D50000",
            name="BTC 5m", showlegend=False,
        ))

        # Entrees LONG ▲
        long_entries = df_trades[df_trades["direction"] == "LONG"]
        if not long_entries.empty:
            fig_trades.add_trace(go.Scatter(
                x=long_entries["entry_ts"], y=long_entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00C853",
                            line=dict(color="white", width=1)),
                name="Entree LONG",
                hovertemplate="LONG<br>%{x}<br>Prix: %{y:,.2f}<extra></extra>",
            ))

        # Entrees SHORT ▼
        short_entries = df_trades[df_trades["direction"] == "SHORT"]
        if not short_entries.empty:
            fig_trades.add_trace(go.Scatter(
                x=short_entries["entry_ts"], y=short_entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#D50000",
                            line=dict(color="white", width=1)),
                name="Entree SHORT",
                hovertemplate="SHORT<br>%{x}<br>Prix: %{y:,.2f}<extra></extra>",
            ))

        # Sorties — couleur selon PnL
        exit_colors = ["#00C853" if p > 0 else "#D50000" for p in df_trades["pnl_pct"]]
        fig_trades.add_trace(go.Scatter(
            x=df_trades["exit_ts"], y=df_trades["exit_price"],
            mode="markers",
            marker=dict(symbol="circle", size=7, color=exit_colors,
                        line=dict(color="white", width=0.5)),
            name="Sortie",
            hovertemplate="Sortie<br>%{x}<br>Prix: %{y:,.2f}<extra></extra>",
        ))

        fig_trades.update_layout(
            title="Trades executes (▲ LONG  ▼ SHORT  ● Sortie)",
            xaxis_rangeslider_visible=False,
            xaxis_title="Timestamp",
            yaxis_title="Prix (USD)",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(_dark_fig(fig_trades, 500), use_container_width=True)

    # ── Distribution PnL ──────────────────────────────────────
    col_dist, col_dir = st.columns(2)

    with col_dist:
        pnls = df_trades["pnl_pct"].values
        fig_hist = go.Figure(go.Histogram(
            x=pnls, nbinsx=30,
            marker_color="#1565C0", opacity=0.8, name="PnL %",
        ))
        fig_hist.add_vline(x=0, line_dash="dot", line_color="#aaa")
        fig_hist.update_layout(title="Distribution des PnL (%)", xaxis_title="PnL (%)", yaxis_title="Trades")
        st.plotly_chart(_dark_fig(fig_hist, 280), use_container_width=True)

    with col_dir:
        dir_counts = df_trades["direction"].value_counts()
        fig_dir = go.Figure(go.Bar(
            x=dir_counts.index, y=dir_counts.values,
            marker_color=["#00C853" if d == "LONG" else "#D50000" for d in dir_counts.index],
            text=dir_counts.values, textposition="outside",
        ))
        fig_dir.update_layout(title="Repartition LONG / SHORT", yaxis_title="Nombre de trades")
        st.plotly_chart(_dark_fig(fig_dir, 280), use_container_width=True)

    # ── Table des trades ──────────────────────────────────────
    st.divider()
    st.subheader("Trades detailles")

    filter_sess = st.multiselect(
        "Filtrer par session",
        options=["ASIA", "LONDON", "NY_AM", "NY_PM"],
        default=[],
    )
    filter_dir = st.multiselect(
        "Filtrer par direction",
        options=["LONG", "SHORT"],
        default=[],
    )
    dft = df_trades.copy()
    if filter_sess:
        dft = dft[dft["session"].isin(filter_sess)]
    if filter_dir:
        dft = dft[dft["direction"].isin(filter_dir)]

    def _color_pnl(val):
        color = "#00C853" if val > 0 else ("#D50000" if val < 0 else "#888")
        return f"color: {color}"

    st.dataframe(
        dft.style.map(_color_pnl, subset=["pnl_pct"]),
        use_container_width=True,
        hide_index=True,
    )

    if export_btn and df_trades is not None:
        csv_path = str(DATA_DIR / "backtest_trades.csv")
        df_trades.to_csv(csv_path, index=False)
        st.success(f"CSV exporte : {csv_path}")


# ── Tab États ─────────────────────────────────────────────────

def tab_states(agent):
    st.subheader("Tableau des 100 états — politique Pi*")
    st.caption(
        "state_id = SESSION×25 + VOLATILITY×5 + PRICE_STRUCTURE  |  "
        "Grisé = état non visité pendant l'entraînement"
    )

    if agent is None:
        st.info("Aucun modèle chargé. Lancez `python main.py --train` d'abord.")
        return

    rows = []
    for sid in range(N_STATES):
        base = sid % 100
        pos  = sid // 100
        sess = base // 25
        rest = base % 25
        vol  = rest // 5
        ps   = rest % 5

        q       = agent.q_table[sid]
        visited = q.sum() != 0
        action  = int(np.argmax(q)) if visited else -1

        rows.append({
            "state_id":       sid,
            "Position":       POSITION_LABELS.get(pos, pos),
            "Session":        SESSION_LABELS.get(sess, sess),
            "Volatilité":     VOLATILITY_LABELS.get(vol, vol),
            "Structure prix": PRICE_STRUCT_LABELS.get(ps, ps),
            "Action":         ACTION_LABELS[action] if visited else "—",
            "Q(FLAT)":        round(float(q[0]), 5),
            "Q(LONG)":        round(float(q[1]), 5),
            "Q(SHORT)":       round(float(q[2]), 5),
            "Visité":         visited,
        })

    df_all = pd.DataFrame(rows)

    # ── Filtres ────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        f_sess = st.multiselect("Session", options=list(SESSION_LABELS.values()), default=[])
    with fc2:
        f_vol  = st.multiselect("Volatilité", options=list(VOLATILITY_LABELS.values()), default=[])
    with fc3:
        f_ps   = st.multiselect("Structure", options=list(PRICE_STRUCT_LABELS.values()), default=[])
    with fc4:
        f_act  = st.multiselect("Action", options=["FLAT", "LONG", "SHORT", "—"], default=[])

    dft = df_all.copy()
    if f_sess: dft = dft[dft["Session"].isin(f_sess)]
    if f_vol:  dft = dft[dft["Volatilité"].isin(f_vol)]
    if f_ps:   dft = dft[dft["Structure prix"].isin(f_ps)]
    if f_act:  dft = dft[dft["Action"].isin(f_act)]

    # ── Métriques résumées ──────────────────────────────────────
    visited_df = dft[dft["Visité"]]
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("États affichés",  len(dft))
    s2.metric("Visités",         len(visited_df))
    s3.metric("FLAT",  len(visited_df[visited_df["Action"] == "FLAT"]))
    s4.metric("LONG",  len(visited_df[visited_df["Action"] == "LONG"]))
    s5.metric("SHORT", len(visited_df[visited_df["Action"] == "SHORT"]))

    # ── Heatmap Q-valeur dominante filtrée ────────────────────
    if not dft.empty:
        action_num = dft["Action"].map({"FLAT": 0, "LONG": 1, "SHORT": 2, "—": -1})
        color_map  = {-1: "#333", 0: "#888888", 1: "#00C853", 2: "#D50000"}
        bar_colors = [color_map[a] for a in action_num]
        fig_bar = go.Figure(go.Bar(
            x=dft["state_id"], y=[1] * len(dft),
            marker_color=bar_colors,
            hovertemplate=(
                "State %{x}<br>"
                + dft["Session"].reset_index(drop=True).astype(str)
                + " | "
                + dft["Volatilité"].reset_index(drop=True).astype(str)
                + " | "
                + dft["Structure prix"].reset_index(drop=True).astype(str)
                + "<br>→ "
                + dft["Action"].reset_index(drop=True).astype(str)
                + "<extra></extra>"
            ).tolist(),
            showlegend=False,
        ))
        fig_bar.update_layout(
            title="Couverture des états (gris=non visité  vert=LONG  rouge=SHORT  gris clair=FLAT)",
            xaxis_title="state_id", yaxis=dict(visible=False),
            bargap=0.05,
        )
        st.plotly_chart(_dark_fig(fig_bar, 180), use_container_width=True)

    # ── Tableau ────────────────────────────────────────────────
    def _color_action(val):
        return {
            "LONG":  "color: #00C853; font-weight: bold",
            "SHORT": "color: #D50000; font-weight: bold",
            "FLAT":  "color: #888888",
            "—":     "color: #444444",
        }.get(val, "")

    def _color_q(val):
        if val > 0:   return "color: #00C853"
        if val < 0:   return "color: #D50000"
        return "color: #555"

    display_df = dft.drop(columns=["Visité"])
    styled = (
        display_df.style
        .map(_color_action, subset=["Action"])
        .map(_color_q, subset=["Q(FLAT)", "Q(LONG)", "Q(SHORT)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # ── Successeurs les plus fréquents ────────────────────────
    st.divider()
    st.subheader("Successeurs les plus fréquents (hors boucle)")
    st.caption(
        "Pour chaque état, les 2 états suivants les plus rencontrés "
        "dans les données historiques, en excluant les auto-transitions."
    )

    with st.spinner("Calcul des transitions..."):
        df_1m_tmp = load_data()
        if not df_1m_tmp.empty:
            env_tmp = TradingEnv(df_1m_tmp)
            trans   = compute_transition_stats(env_tmp.sessions)

    if df_1m_tmp.empty or trans.empty:
        st.info("Pas assez de données pour calculer les transitions.")
        return

    # Filtrer les auto-boucles
    trans_ext = trans[trans["from_state"] != trans["to_state"]].copy()

    # Top 2 successeurs par état source
    top2 = (
        trans_ext
        .sort_values(["from_state", "count"], ascending=[True, False])
        .groupby("from_state")
        .head(2)
        .reset_index(drop=True)
    )
    top2["rank"] = top2.groupby("from_state").cumcount() + 1

    def _decode(sid):
        sid = int(sid)
        sess = sid // 25
        rest = sid % 25
        vol  = rest // 5
        ps   = rest % 5
        return (
            f"{SESSION_LABELS.get(sess, sid)} | "
            f"{VOLATILITY_LABELS.get(vol, '')} | "
            f"{PRICE_STRUCT_LABELS.get(ps, '')}"
        )

    top2["État source"]     = top2["from_state"].apply(_decode)
    top2["État suivant"]    = top2["to_state"].apply(_decode)
    top2["from_state_id"]   = top2["from_state"].astype(int)
    top2["to_state_id"]     = top2["to_state"].astype(int)

    # Pivoter : une ligne par état source, colonnes Succ#1 et Succ#2
    rows_succ = []
    for from_sid, grp in top2.groupby("from_state_id"):
        row = {
            "state_id":  from_sid,
            "État":      _decode(from_sid),
            "Action":    ACTION_LABELS.get(
                int(np.argmax(agent.q_table[from_sid]))
                if agent.q_table[from_sid].sum() != 0 else 0, "—"
            ),
        }
        for _, r in grp.iterrows():
            rank = int(r["rank"])
            row[f"Succ #{rank} (id)"]    = int(r["to_state_id"])
            row[f"Succ #{rank}"]         = r["État suivant"]
            row[f"Succ #{rank} (count)"] = int(r["count"])
            row[f"Succ #{rank} (%)"]     = round(float(r["freq_pct"]), 2)
        rows_succ.append(row)

    df_succ = pd.DataFrame(rows_succ).fillna("—")

    # Filtre optionnel par session
    sessions_filter = st.multiselect(
        "Filtrer par session (état source)",
        options=list(SESSION_LABELS.values()), default=[],
        key="succ_sess_filter",
    )
    if sessions_filter:
        df_succ = df_succ[df_succ["État"].str.startswith(tuple(sessions_filter))]

    cols_show = [c for c in [
        "state_id", "État", "Action",
        "Succ #1 (id)", "Succ #1", "Succ #1 (count)", "Succ #1 (%)",
        "Succ #2 (id)", "Succ #2", "Succ #2 (count)", "Succ #2 (%)",
    ] if c in df_succ.columns]

    st.dataframe(df_succ[cols_show], use_container_width=True, hide_index=True, height=500)
    st.caption(f"{len(df_succ)} états avec au moins un successeur externe | {len(trans_ext)} transitions hors auto-boucles")


# ── Tab Masques ────────────────────────────────────────────────

def tab_masks(agent, df_1m):
    st.subheader("Masques d'actions — dérivés automatiquement")
    st.caption(
        "Les masques interdisent à l'agent certaines actions dans certains états, "
        "basé sur les Q-valeurs apprises et les transitions réelles."
    )

    if agent is None:
        st.info("Aucun modèle chargé. Lancez `python main.py --train` d'abord.")
        return

    mask_path = DATA_DIR / "action_mask.npy"

    col_ev, col_btn = st.columns([2, 1])
    with col_ev:
        ev_threshold = st.slider(
            "Seuil Q-valeur (actions en dessous = masquées)",
            min_value=-0.10, max_value=0.00, value=-0.01, step=0.005,
            format="%.3f",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("Générer / Mettre à jour les masques", use_container_width=True)

    if gen_btn:
        with st.spinner("Calcul des masques..."):
            env   = TradingEnv(df_1m)
            mask  = build_action_mask(agent.q_table, ev_threshold=ev_threshold)
            trans = compute_transition_stats(env.sessions)
            import numpy as np
            np.save(str(mask_path), mask)
            st.session_state.mask       = mask
            st.session_state.mask_trans = trans
            st.success(f"Masque sauvegardé : {mask_path}")

    # Chargement depuis fichier si non en session_state
    if "mask" not in st.session_state:
        if mask_path.exists():
            import numpy as np
            _m = np.load(str(mask_path))
            if _m.shape[0] == N_STATES:
                st.session_state.mask = _m
            else:
                st.warning(
                    f"Masque obsolète ({_m.shape[0]} états) — N_STATES={N_STATES}. "
                    "Cliquez sur **Générer** pour le reconstruire."
                )
                return
        else:
            st.info("Aucun masque généré. Cliquez sur **Générer / Mettre à jour les masques**.")
            return

    mask = st.session_state.mask

    # ── Métriques ──────────────────────────────────────────────
    stats = mask_summary(mask)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("États avec ≥1 masque", stats["états_avec_masque"])
    s2.metric("Actions masquées",     f"{stats['actions_masquées']} ({stats['pct_masquées']}%)")
    s3.metric("FLAT masqué",          stats["FLAT_masqué"])
    s4.metric("LONG masqué",          stats["LONG_masqué"])
    s5.metric("SHORT masqué",         stats["SHORT_masqué"])

    st.divider()

    # ── Visualisation masque par état ─────────────────────────
    col_viz, col_trans = st.columns([3, 2])

    with col_viz:
        # Heatmap : 0=masqué, 1=autorisé — 3 lignes (FLAT/LONG/SHORT), 400 colonnes
        import numpy as np
        fig_mask = go.Figure()
        for a_idx, a_label in enumerate(["FLAT", "LONG", "SHORT"]):
            row_vals = mask[:, a_idx].astype(float)
            colors   = ["#00C853" if v else "#D50000" for v in row_vals]
            fig_mask.add_trace(go.Bar(
                name=a_label,
                x=list(range(N_STATES)),
                y=[a_idx + 1] * N_STATES,
                marker_color=colors,
                showlegend=False,
                hovertemplate=f"état %{{x}}<br>{a_label}: %{{customdata}}<extra></extra>",
                customdata=["autorisé" if v else "masqué" for v in row_vals],
            ))
        fig_mask.update_layout(
            title="Masque par état (vert=autorisé  rouge=masqué)",
            barmode="overlay",
            xaxis_title="state_id",
            yaxis=dict(
                tickvals=[1, 2, 3],
                ticktext=["FLAT", "LONG", "SHORT"],
            ),
            bargap=0.05,
        )
        st.plotly_chart(_dark_fig(fig_mask, 260), use_container_width=True)

    # ── Transitions réelles ───────────────────────────────────
    with col_trans:
        if "mask_trans" in st.session_state and not st.session_state.mask_trans.empty:
            trans = st.session_state.mask_trans
            st.markdown("**Top 15 transitions les plus fréquentes**")
            st.dataframe(
                trans.head(15)[["from_state", "to_state", "count", "freq_pct"]],
                use_container_width=True, hide_index=True, height=280,
            )
            st.caption(f"{len(trans)} transitions uniques observées sur {100*100} possibles")
        else:
            st.info("Cliquez sur **Générer** pour voir les transitions.")

    # ── Tableau détaillé des masques ─────────────────────────
    st.divider()
    st.subheader("Détail par état")

    show_masked_only = st.checkbox("Afficher uniquement les états avec au moins un masque", value=True)

    rows = []
    for sid in range(N_STATES):
        flat_ok  = bool(mask[sid, 0])
        long_ok  = bool(mask[sid, 1])
        short_ok = bool(mask[sid, 2])
        if show_masked_only and (flat_ok and long_ok and short_ok):
            continue

        base = sid % 100
        pos  = sid // 100
        sess = base // 25
        rest = base % 25
        vol  = rest // 5
        ps   = rest % 5

        rows.append({
            "state_id":       sid,
            "Position":       POSITION_LABELS.get(pos, pos),
            "Session":        SESSION_LABELS.get(sess, sess),
            "Volatilité":     VOLATILITY_LABELS.get(vol, vol),
            "Structure":      PRICE_STRUCT_LABELS.get(ps, ps),
            "FLAT":           "✓" if flat_ok  else "✗",
            "LONG":           "✓" if long_ok  else "✗",
            "SHORT":          "✓" if short_ok else "✗",
            "Q(FLAT)":        round(float(agent.q_table[sid, 0]), 5),
            "Q(LONG)":        round(float(agent.q_table[sid, 1]), 5),
            "Q(SHORT)":       round(float(agent.q_table[sid, 2]), 5),
        })

    if rows:
        def _color_allowed(val):
            if val == "✓": return "color: #00C853; font-weight: bold"
            if val == "✗": return "color: #D50000; font-weight: bold"
            return ""

        df_mask_detail = pd.DataFrame(rows)
        st.dataframe(
            df_mask_detail.style.map(_color_allowed, subset=["FLAT", "LONG", "SHORT"]),
            use_container_width=True, hide_index=True, height=450,
        )
        st.caption(f"{len(rows)} états affichés")
    else:
        st.success("Tous les états ont toutes les actions disponibles (masque vide).")


# ── Tab Contrôle ───────────────────────────────────────────────

def _run_cmd(args: list):
    """Lance main.py avec les args donnés et streame la sortie dans un st.code."""
    import subprocess
    output_lines = []
    out_area = st.empty()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "main.py")] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            output_lines.append(line)
            out_area.code("".join(output_lines), language="bash")
        proc.wait()
        return proc.returncode, "".join(output_lines)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return -1, str(e)


def tab_control():
    st.subheader("Panneau de contrôle — Pi*")
    st.caption("Pilotez l'ensemble du pipeline depuis l'interface sans passer par le terminal.")

    # ── Section 1 : Données ────────────────────────────────────
    with st.expander("1. Collecte de données (backfill)", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            days = st.slider("Jours d'historique à télécharger", 7, 180, 90, step=7)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_backfill = st.button("Lancer le backfill", use_container_width=True, key="btn_backfill")

        if do_backfill:
            st.info(f"Téléchargement de {days} jours depuis Deribit...")
            with st.spinner("Backfill en cours..."):
                rc, out = _run_cmd(["--backfill", "--days", str(days)])
            if rc == 0:
                st.success("Backfill terminé.")
                st.cache_data.clear()
            else:
                st.error("Erreur lors du backfill.")

    # ── Section 2 : Masques ────────────────────────────────────
    with st.expander("2. Génération des masques d'actions", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            ev_thr = st.slider(
                "Seuil EV (Q-valeur sous lequel une action est masquée)",
                min_value=-0.10, max_value=0.00, value=-0.01, step=0.005,
                format="%.3f", key="ctrl_ev",
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_masks = st.button("Générer les masques", use_container_width=True, key="btn_masks")

        st.caption("À relancer **après** chaque entraînement pour que les masques reflètent la nouvelle Q-table.")

        if do_masks:
            with st.spinner("Génération des masques..."):
                rc, out = _run_cmd(["--gen-masks", "--ev-threshold", str(ev_thr)])
            if rc == 0:
                st.success("Masques générés et sauvegardés dans db/action_mask.npy")
            else:
                st.error("Erreur lors de la génération des masques.")

    # ── Section 3 : Entraînement ───────────────────────────────
    with st.expander("3. Entraînement de l'agent", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            episodes = st.number_input(
                "Episodes", min_value=100, max_value=50000, value=5000, step=500,
            )
            alpha = st.number_input(
                "Alpha (taux d'apprentissage)", min_value=0.001, max_value=0.5,
                value=0.05, step=0.005, format="%.3f",
            )
        with col_b:
            epsilon_decay = st.number_input(
                "Epsilon decay (par épisode)", min_value=0.00005, max_value=0.01,
                value=0.0002, step=0.00005, format="%.5f",
            )
            use_masks = st.checkbox("Utiliser les masques pendant l'entraînement", value=True)
        with col_c:
            reset = st.checkbox(
                "Repartir d'un agent vierge (ignore le modèle existant)", value=False,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Résumé de la commande
        cmd_preview = ["--train",
                       f"--episodes {episodes}",
                       f"--alpha {alpha}",
                       f"--epsilon-decay {epsilon_decay}"]
        if reset:      cmd_preview.append("--reset")
        if use_masks:  cmd_preview.append("--use-masks")
        st.code("python main.py " + " ".join(cmd_preview), language="bash")

        do_train = st.button("Lancer l'entraînement", use_container_width=True,
                             type="primary", key="btn_train")

        if do_train:
            args = [
                "--train",
                "--episodes", str(episodes),
                "--alpha",    str(alpha),
                "--epsilon-decay", str(epsilon_decay),
            ]
            if reset:     args.append("--reset")
            if use_masks: args.append("--use-masks")

            st.info("Entraînement en cours — la sortie s'affiche en temps réel ci-dessous.")
            rc, out = _run_cmd(args)
            if rc == 0:
                st.success(f"Entraînement terminé ({episodes} épisodes).")
                st.cache_data.clear()
            else:
                st.error("L'entraînement s'est terminé avec une erreur.")

    # ── Section 4 : Backtest express ───────────────────────────
    with st.expander("4. Backtest rapide", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            bt_ratio = st.slider("Test set (%)", 5, 40, 20, step=5, key="ctrl_bt_ratio") / 100
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_bt = st.button("Lancer le backtest", use_container_width=True, key="btn_ctrl_bt")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            do_bt_csv = st.button("Backtest + export CSV", use_container_width=True, key="btn_ctrl_bt_csv")

        if do_bt or do_bt_csv:
            args = ["--backtest", "--test-ratio", str(bt_ratio)]
            if do_bt_csv: args.append("--export-csv")
            with st.spinner("Backtest en cours..."):
                rc, _ = _run_cmd(args)
            if rc == 0:
                st.success("Backtest terminé. Allez dans l'onglet **Backtest** pour visualiser.")
            else:
                st.error("Erreur lors du backtest.")

    # ── Section 5 : Infos modèle ───────────────────────────────
    with st.expander("5. Infos modèle & base", expanded=False):
        agent = load_agent()
        if agent is not None:
            visited = int((agent.q_table.sum(axis=1) != 0).sum())
            policy  = agent.policy_summary()
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("Épisodes entraînés", agent._episode_count)
            i2.metric("Epsilon actuel",     f"{agent.epsilon:.4f}")
            i3.metric("États visités",      f"{visited}/100")
            i4.metric("Alpha",              agent.alpha)
            i5.metric("Gamma",              agent.gamma)

            p1, p2, p3 = st.columns(3)
            p1.metric("FLAT dominant",  policy["FLAT"])
            p2.metric("LONG dominant",  policy["LONG"])
            p3.metric("SHORT dominant", policy["SHORT"])
        else:
            st.warning("Aucun modèle chargé.")

        st.divider()
        try:
            from data.storage import get_prices_conn
            with get_prices_conn() as conn:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM market_1m WHERE asset='BTC'"
                ).fetchone()[0]
                mn  = conn.execute(
                    "SELECT datetime(MIN(ts), 'unixepoch') FROM market_1m WHERE asset='BTC'"
                ).fetchone()[0]
                mx  = conn.execute(
                    "SELECT datetime(MAX(ts), 'unixepoch') FROM market_1m WHERE asset='BTC'"
                ).fetchone()[0]
            st.metric("Bougies 1min en base", f"{cnt:,}")
            st.caption(f"De {mn} à {mx}")
        except Exception as e:
            st.error(f"DB inaccessible : {e}")


# ── Tab Markov ─────────────────────────────────────────────────

def tab_markov(agent, df_1m):
    st.subheader("Chaîne de Markov — probabilités de transition")
    st.caption(
        "P(s'|s) : probabilité que le marché passe de l'état s à l'état s'. "
        "Se renforce à chaque entraînement et backtest."
    )

    mc = load_markov()

    # ── Bouton construction / mise à jour ──────────────────────
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        do_build = st.button("Construire / Mettre à jour Markov", use_container_width=True, key="btn_build_markov")
    with col_btn2:
        do_eval_acc = st.button("Évaluer précision", use_container_width=True, key="btn_eval_markov")

    if do_build:
        with st.spinner("Construction de la chaîne de Markov..."):
            rc, _ = _run_cmd(["--build-markov"])
        if rc == 0:
            st.success("Markov construit et sauvegardé dans db/markov.pkl")
            mc = load_markov()
        else:
            st.error("Erreur lors de la construction.")

    if mc is None:
        st.info("Aucune chaîne de Markov construite. Cliquez sur **Construire / Mettre à jour Markov**.")
        return

    # ── Métriques globales ─────────────────────────────────────
    stats = mc.summary()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("États visités",     stats["etats_visites"])
    m2.metric("États fiables (≥5)", stats["etats_fiables"])
    m3.metric("Transitions total",  f"{stats['transitions_tot']:,}")
    m4.metric("Transitions uniques", stats["transitions_uniq"])
    m5.metric("Entropie moyenne",   f"{stats['entropie_moy']:.3f}")

    st.divider()

    # ── Précision de prédiction ────────────────────────────────
    if do_eval_acc and not df_1m.empty:
        with st.spinner("Calcul de la précision sur le test set..."):
            env_tmp = TradingEnv(df_1m)
            n_test  = max(1, int(len(env_tmp.sessions) * 0.2))
            test_sessions = env_tmp.sessions[-n_test:]
            acc = mc.backtest_accuracy(test_sessions)
        st.session_state.markov_acc = acc

    if "markov_acc" in st.session_state:
        acc = st.session_state.markov_acc
        a1, a2, a3 = st.columns(3)
        a1.metric("Top-1 accuracy",      f"{acc['top1_accuracy']*100:.1f}%")
        a2.metric("Top-3 accuracy",      f"{acc['top3_accuracy']*100:.1f}%")
        a3.metric("Transitions évaluées", acc["n_evaluated"])
        st.divider()

    col_heat, col_pred = st.columns([3, 2])

    # ── Heatmap matrice de transition ─────────────────────────
    with col_heat:
        reliable_states = np.where(mc.confidence >= 5)[0]
        if len(reliable_states) == 0:
            st.warning("Aucun état fiable (moins de 5 transitions observées).")
        else:
            P_sub = mc.prob[np.ix_(reliable_states, reliable_states)]
            labels = [f"{s}" for s in reliable_states]

            fig_mc = go.Figure(go.Heatmap(
                z=P_sub,
                x=labels, y=labels,
                colorscale="Blues",
                zmin=0, zmax=P_sub.max() if P_sub.max() > 0 else 1,
                hovertemplate="De %{y} → %{x}<br>P = %{z:.4f}<extra></extra>",
                colorbar=dict(title="P(s'|s)", thickness=12),
            ))
            fig_mc.update_layout(
                title=f"Matrice de transition — {len(reliable_states)} états fiables",
                xaxis=dict(title="État suivant s'", tickfont=dict(size=8)),
                yaxis=dict(title="État courant s",  tickfont=dict(size=8), autorange="reversed"),
            )
            st.plotly_chart(_dark_fig(fig_mc, 420), use_container_width=True)

    # ── Prédictions top-3 par état ────────────────────────────
    with col_pred:
        st.markdown("**Prédictions top-3 par état (états fiables)**")

        state_options = {
            f"{s} — {SESSION_LABELS.get(s//25,'?')} | {VOLATILITY_LABELS.get((s%25)//5,'?')} | {PRICE_STRUCT_LABELS.get(s%5,'?')}": s
            for s in reliable_states
        }

        if state_options:
            sel_label = st.selectbox("Choisir un état source", options=list(state_options.keys()), key="markov_sel_state")
            sel_state = state_options[sel_label]

            preds = mc.predict(sel_state, top_k=5)
            if preds:
                pred_rows = []
                for p in preds:
                    s2 = p["state"]
                    pred_rows.append({
                        "État suivant": f"{s2} — {SESSION_LABELS.get(s2//25,'?')} | {VOLATILITY_LABELS.get((s2%25)//5,'?')} | {PRICE_STRUCT_LABELS.get(s2%5,'?')}",
                        "P(s'|s)":  round(p["prob"], 4),
                        "Occurrences": p["count"],
                        "Fiable": "✓" if p["reliable"] else "✗",
                    })
                df_preds = pd.DataFrame(pred_rows)

                fig_pred = go.Figure(go.Bar(
                    x=[r["État suivant"].split(" — ")[0] for r in pred_rows],
                    y=[r["P(s'|s)"] for r in pred_rows],
                    marker_color="#1565C0",
                    text=["{:.2%}".format(r["P(s'|s)"]) for r in pred_rows],
                    textposition="outside",
                    hovertext=[r["État suivant"] for r in pred_rows],
                    hoverinfo="text+y",
                ))
                fig_pred.update_layout(
                    title=f"Top-5 successeurs depuis état {sel_state}",
                    xaxis_title="État suivant",
                    yaxis_title="Probabilité",
                    yaxis=dict(range=[0, min(1.0, max(r["P(s'|s)"] for r in pred_rows) * 1.3)]),
                )
                st.plotly_chart(_dark_fig(fig_pred, 280), use_container_width=True)
                st.dataframe(df_preds, use_container_width=True, hide_index=True)

                # Structure de prix la plus probable
                struct_preds = mc.predict_structure(sel_state, top_k=3)
                if struct_preds:
                    st.markdown("**Structure de prix anticipée :**")
                    for sp in struct_preds:
                        pct = sp["prob"] * 100
                        st.progress(int(pct), text=f"{sp['structure']}  {pct:.1f}%")
            else:
                st.info("Aucune transition observée depuis cet état.")

    # ── Entropie par état ─────────────────────────────────────
    st.divider()
    st.subheader("Entropie par état (prévisibilité)")
    st.caption("Entropie basse = transitions déterministes. Haute = marché imprévisible depuis cet état.")

    entropy_rows = []
    for s in reliable_states:
        h = mc.transition_entropy(int(s))
        entropy_rows.append({
            "state_id": int(s),
            "Session":  SESSION_LABELS.get(s // 25, "?"),
            "Volatilité": VOLATILITY_LABELS.get((s % 25) // 5, "?"),
            "Structure":  PRICE_STRUCT_LABELS.get(s % 5, "?"),
            "Entropie":   round(h, 4),
            "Observations": int(mc.confidence[s]),
        })

    if entropy_rows:
        df_entropy = pd.DataFrame(entropy_rows).sort_values("Entropie")
        col_lo, col_hi = st.columns(2)
        with col_lo:
            st.markdown("**États les plus prévisibles (entropie basse)**")
            st.dataframe(df_entropy.head(10), use_container_width=True, hide_index=True)
        with col_hi:
            st.markdown("**États les moins prévisibles (entropie haute)**")
            st.dataframe(df_entropy.tail(10).sort_values("Entropie", ascending=False), use_container_width=True, hide_index=True)

        fig_ent = go.Figure(go.Bar(
            x=df_entropy["state_id"].astype(str),
            y=df_entropy["Entropie"],
            marker_color=px.colors.sequential.Blues[3:],
            hovertemplate="État %{x}<br>Entropie: %{y:.4f}<extra></extra>",
        ))
        fig_ent.update_layout(
            title="Entropie de transition par état (trié par state_id)",
            xaxis_title="state_id", yaxis_title="Entropie (bits)",
            xaxis=dict(tickfont=dict(size=9)),
        )
        df_ent_sorted = df_entropy.sort_values("state_id")
        fig_ent2 = go.Figure(go.Bar(
            x=df_ent_sorted["state_id"].astype(str),
            y=df_ent_sorted["Entropie"],
            marker_color=[
                f"rgb({int(50 + 180*(h/max(df_ent_sorted['Entropie'].max(), 0.001)))}, "
                f"{int(100 + 80*(1-h/max(df_ent_sorted['Entropie'].max(), 0.001)))}, 200)"
                for h in df_ent_sorted["Entropie"]
            ],
            hovertemplate="État %{x}<br>Entropie: %{y:.4f}<extra></extra>",
        ))
        fig_ent2.update_layout(
            title="Entropie de transition par état",
            xaxis_title="state_id", yaxis_title="Entropie (bits)",
            xaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(_dark_fig(fig_ent2, 250), use_container_width=True)


# ── Layout principal ───────────────────────────────────────────

def main():
    _init()

    st.title("₿ Pi* — BTC-PERPETUAL Dashboard")

    with st.sidebar:
        st.header("Parametres")
        n_candles = st.slider("Bougies affichees (5min)", 30, 288, 120, step=6)
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
        if auto_refresh:
            import time; time.sleep(0.1)
            st.rerun()
        st.divider()
        if st.button("Recharger les donnees", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.subheader("Backtest")
        test_ratio = st.slider("Test set (%)", 5, 40, 20, step=5) / 100
        run_btn    = st.button("Lancer le backtest", use_container_width=True)
        export_btn = st.button("Exporter trades CSV", use_container_width=True)
        st.divider()
        st.subheader("Base de donnees")
        try:
            from data.storage import get_prices_conn
            with get_prices_conn() as conn:
                market_count = conn.execute(
                    "SELECT COUNT(*) FROM market_1m WHERE asset='BTC'"
                ).fetchone()[0]
            st.metric("Bougies 1min (BTC)", f"{market_count:,}")
        except Exception as e:
            st.error(f"DB inaccessible : {e}")
        st.divider()
        st.subheader("Discord")
        _discord_url = (
            st.secrets.get("DISCORD_WEBHOOK_URL", "")
            if hasattr(st, "secrets") else ""
        ) or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if _discord_url:
            st.caption(f"Webhook : ...{_discord_url[-12:]}")
        else:
            st.caption("Webhook non configure")
        if st.button("Tester Discord", use_container_width=True):
            if not _discord_url:
                st.error("DISCORD_WEBHOOK_URL non defini.")
            else:
                try:
                    import requests as _req
                    from datetime import datetime as _dt
                    _msg = f"Pi* test -- {_dt.utcnow().strftime('%Y-%m-%d %H:%M')} UTC -- connexion OK"
                    _r = _req.post(_discord_url, json={"content": _msg}, timeout=8)
                    _r.raise_for_status()
                    st.success("Message envoye sur Discord.")
                except Exception as _e:
                    st.error(f"Echec : {_e}")

    with st.spinner("Chargement..."):
        df_1m = load_data()

    if df_1m.empty:
        st.warning("Aucune donnee en base. Lancez : `python main.py`")
        st.code("python main.py\npython main.py --train")
        return

    df5   = load_states()
    agent = load_agent()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Live", "Backtest", "États", "Masques", "Contrôle", "Markov"])

    with tab1:
        tab_live(df_1m, df5, agent, n_candles)

    with tab2:
        tab_backtest(df_1m, agent, test_ratio, run_btn, export_btn)

    with tab3:
        tab_states(agent)

    with tab4:
        tab_masks(agent, df_1m)

    with tab5:
        tab_control()

    with tab6:
        tab_markov(agent, df_1m)


if __name__ == "__main__":
    main()
