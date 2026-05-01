"""
live_dashboard.py — Dashboard Streamlit TA en temps réel.

Lancement local :
    streamlit run strategies/ta/live_dashboard.py

Sur Railway (2ème service) :
    streamlit run strategies/ta/live_dashboard.py --server.port $PORT --server.headless true
"""

import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.ta.features import _ema, _rsi, _atr, _stoch_k, _vwap_daily
from strategies.ta.live_runner import scan, load_live_configs, LIVE_MIN_N_OOS, LIVE_WR_DROP
from strategies.ta.config import SESSIONS_UTC, SYMBOL

# ─────────────────────────────────────────────────────────────────────────────
# Config Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TA Live — BTC/USDT",
    page_icon="📊",
    layout="wide",
)

REFRESH_SECS = 60   # auto-refresh toutes les 60 secondes

# ─────────────────────────────────────────────────────────────────────────────
# Cache données (évite de re-fetch à chaque widget interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_SECS, show_spinner=False)
def get_data(symbol: str = SYMBOL) -> dict:
    return scan(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def get_n_live_configs() -> int:
    return len(load_live_configs())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

def _regime_color(regime: str) -> str:
    return {"bull": "#00c853", "bear": "#d50000", "range": "#ff9800"}.get(regime, "#9e9e9e")


def _direction_color(direction: str) -> str:
    return "#00c853" if direction == "LONG" else "#d50000"


def _badge(label: str, value: str, color: str = "#444") -> str:
    return (
        f'<div style="display:inline-block;background:{color};color:white;'
        f'padding:4px 10px;border-radius:6px;margin:3px;font-size:13px;">'
        f'<b>{label}</b>: {value}</div>'
    )


def _session_label(dt: pd.Timestamp) -> str:
    h = dt.hour
    for start, end in SESSIONS_UTC:
        if start <= h < end:
            labels = {7: "London", 13: "NY Open"}
            return labels.get(start, f"{start:02d}-{end:02d} UTC")
    return "Hors session"


# ─────────────────────────────────────────────────────────────────────────────
# Chart principal
# ─────────────────────────────────────────────────────────────────────────────

def render_chart(result: dict) -> None:
    df15 = result["df15"]   # inclut la bougie courante

    # Indicateurs sur 15m pour overlay
    ema21  = _ema(df15["close"], 21)
    ema50  = _ema(df15["close"], 50)
    ema200_15m = _ema(df15["close"], 200)
    vwap   = _vwap_daily(df15)
    rsi14  = _rsi(df15["close"], 14)
    atr14  = _atr(df15["high"], df15["low"], df15["close"], 14)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=("BTC/USDT 15m", "RSI 14", "Volume"),
    )

    # ── Bougies ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df15.index,
        open=df15["open"], high=df15["high"],
        low=df15["low"],   close=df15["close"],
        name="BTC/USDT",
        increasing_line_color="#00c853",
        decreasing_line_color="#d50000",
        showlegend=False,
    ), row=1, col=1)

    # ── EMA overlays ─────────────────────────────────────────────────────────
    for ema_s, color, name in [
        (ema21,     "#ffeb3b", "EMA 21"),
        (ema50,     "#29b6f6", "EMA 50"),
        (ema200_15m, "#ce93d8", "EMA 200"),
    ]:
        fig.add_trace(go.Scatter(
            x=ema_s.index, y=ema_s.values,
            mode="lines", line=dict(color=color, width=1),
            name=name, opacity=0.85,
        ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vwap.index, y=vwap.values,
        mode="lines", line=dict(color="#ff7043", width=1, dash="dot"),
        name="VWAP", opacity=0.9,
    ), row=1, col=1)

    # ── Marqueur trigger ──────────────────────────────────────────────────────
    df_closed = result["df15_closed"]
    trigger   = result["trigger_raw"]
    if trigger:
        last_bar  = df_closed.index[-1]
        last_price = float(df_closed["close"].iloc[-1])
        symbol_m   = "triangle-up" if trigger == "LONG"  else "triangle-down"
        color_m    = "#00c853"    if trigger == "LONG"  else "#d50000"
        offset     = atr14.iloc[-1] * (1.5 if trigger == "LONG" else -1.5)
        fig.add_trace(go.Scatter(
            x=[last_bar],
            y=[last_price - offset if trigger == "LONG" else last_price + abs(offset)],
            mode="markers",
            marker=dict(symbol=symbol_m, size=16, color=color_m),
            name=f"Trigger {trigger}",
        ), row=1, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rsi14.index, y=rsi14.values,
        mode="lines", line=dict(color="#4fc3f7", width=1.5),
        name="RSI 14", showlegend=False,
    ), row=2, col=1)
    for level, color in [(70, "rgba(213,0,0,0.3)"), (30, "rgba(0,200,83,0.3)")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      opacity=0.7, row=2, col=1)
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(213,0,0,0.05)",
                  line_width=0, row=2, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    colors = ["#00c853" if c >= o else "#d50000"
              for c, o in zip(df15["close"], df15["open"])]
    fig.add_trace(go.Bar(
        x=df15.index, y=df15["volume"],
        marker_color=colors, name="Volume", showlegend=False,
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=600,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1e1e1e", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e1e1e", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panneau état courant
# ─────────────────────────────────────────────────────────────────────────────

def render_state_panel(result: dict) -> None:
    cs = result["current_state"]
    if not cs:
        st.warning("Etat courant indisponible.")
        return

    regime     = cs.get("regime", "?")
    ema_state  = cs.get("ema_state", "?")
    ema_slope  = cs.get("ema_slope", "?")
    swing      = cs.get("swing", "?")
    rsi_state  = cs.get("rsi_state", "?")
    stoch_state = cs.get("stoch_state", "?")
    atr_state  = cs.get("atr_state", "?")
    vwap_state = cs.get("vwap_state", "?")

    rsi_val    = cs.get("_rsi", "?")
    stoch_val  = cs.get("_stoch", "?")
    atr_ratio  = cs.get("_atr_ratio", "?")
    vwap_val   = cs.get("_vwap", "?")
    ema_4h     = cs.get("_ema_4h", "?")

    # Couleurs dynamiques
    regime_c   = _regime_color(regime)
    ema_c      = "#00c853" if ema_state == 1 else "#d50000"
    slope_c    = "#00c853" if ema_slope == 1 else ("#d50000" if ema_slope == -1 else "#9e9e9e")
    swing_c    = "#00c853" if swing == 1 else ("#d50000" if swing == -1 else "#9e9e9e")
    vwap_c     = "#00c853" if vwap_state == 1 else "#d50000"
    rsi_c      = {"oversold": "#00c853", "weak": "#9e9e9e",
                  "strong": "#ff9800", "overbought": "#d50000"}.get(rsi_state, "#9e9e9e")
    stoch_c    = rsi_c
    atr_c      = {"compression": "#29b6f6", "neutral": "#9e9e9e",
                  "expansion": "#ff9800"}.get(atr_state, "#9e9e9e")

    swing_label = {1: "+1 (uptrend)", -1: "-1 (downtrend)", 0: "0 (mixte)"}.get(swing, str(swing))
    ema_label   = f"{'above' if ema_state == 1 else 'below'} EMA{cs.get('_ema_4h','')} | slope {ema_slope:+d}"
    vwap_label  = f"{'above' if vwap_state == 1 else 'below'} {vwap_val}"

    badges_html = "".join([
        _badge("Regime",   regime,                   regime_c),
        _badge("EMA 4H",   ema_label,                ema_c),
        _badge("Swing 4H", swing_label,              swing_c),
        _badge(f"RSI({rsi_val})",   rsi_state,       rsi_c),
        _badge(f"Stoch({stoch_val})", stoch_state,   stoch_c),
        _badge(f"ATR ratio({atr_ratio})", atr_state, atr_c),
        _badge("VWAP",     vwap_label,               vwap_c),
    ])
    st.markdown(badges_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panneau signaux
# ─────────────────────────────────────────────────────────────────────────────

def render_signals(result: dict) -> None:
    trigger    = result["trigger"]
    trigger_raw = result["trigger_raw"]
    in_session = result["in_session"]
    matches    = result["matches"]

    if trigger_raw is None:
        st.info("Aucun trigger 2-barres sur la dernière bougie fermée.")
        return

    dir_color = _direction_color(trigger_raw)
    if not in_session:
        st.markdown(
            f'<div style="background:#333;padding:10px;border-radius:8px;">'
            f'Trigger <b style="color:{dir_color}">{trigger_raw}</b> détecté '
            f'— mais <b>hors session</b> (London 07-11 UTC / NY 13-17 UTC).</div>',
            unsafe_allow_html=True,
        )
        return

    if not matches:
        st.markdown(
            f'<div style="background:#d50000;padding:10px;border-radius:8px;">'
            f'Trigger <b>{trigger_raw}</b> en session — '
            f'<b>aucun match</b> dans les configs validées IS+OOS.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="background:#00695c;padding:12px;border-radius:8px;font-size:16px;">'
        f'🎯 <b>SIGNAL {trigger_raw}</b> — {len(matches)} config(s) validée(s) IS+OOS</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Table des matches
    rows = []
    for m in matches:
        rows.append({
            "Params":    m["params"],
            "Direction": m["direction"],
            "Regime":    m["regime"],
            "Swing":     m["swing"],
            "RSI":       m["rsi_state"],
            "Stoch":     m["stoch_state"],
            "ATR":       m["atr_state"],
            "n_IS":      m["n_IS"],
            "WR IS":     f"{m['wr_IS']:.1%}",
            "n_OOS":     m["n_OOS"],
            "WR OOS":    f"{m['wr_OOS']:.1%}",
            "Exp OOS":   f"{m['exp_R_OOS']:.2f}R",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.title("📊 TA Live — BTC/USDT 15m")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        manual_refresh = st.button("🔄 Refresh")

    if manual_refresh:
        st.cache_data.clear()

    # ── Fetch données ─────────────────────────────────────────────────────────
    with st.spinner("Chargement des données Binance..."):
        try:
            result = get_data()
        except Exception as e:
            st.error(f"Erreur de connexion Binance : {e}")
            st.stop()

    cs         = result["current_state"]
    regime     = cs.get("regime", "?")
    last_price = result["last_price"]
    last_bar   = result["last_bar_time"][:16].replace("T", " ")
    in_session = result["in_session"]
    trigger    = result["trigger"]

    # ── Métriques header ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("BTC/USDT", f"${last_price:,.2f}")
    with col2:
        st.metric("Dernière bougie (UTC)", last_bar)
    with col3:
        regime_color = _regime_color(regime)
        st.markdown(
            f'<div style="text-align:center;padding:8px;">'
            f'<div style="font-size:12px;color:#9e9e9e">Régime macro</div>'
            f'<div style="font-size:22px;color:{regime_color};font-weight:bold">'
            f'{regime.upper()}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        session_label = _session_label(
            pd.Timestamp(result["last_bar_time"]).tz_convert("UTC")
            if "+" in result["last_bar_time"] or "Z" in result["last_bar_time"]
            else pd.Timestamp(result["last_bar_time"], tz="UTC")
        )
        color_sess = "#00c853" if in_session else "#9e9e9e"
        st.markdown(
            f'<div style="text-align:center;padding:8px;">'
            f'<div style="font-size:12px;color:#9e9e9e">Session</div>'
            f'<div style="font-size:18px;color:{color_sess};font-weight:bold">'
            f'{session_label}</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        n_configs = get_n_live_configs()
        trigger_display = trigger if trigger else "—"
        trig_color = _direction_color(trigger) if trigger else "#9e9e9e"
        st.markdown(
            f'<div style="text-align:center;padding:8px;">'
            f'<div style="font-size:12px;color:#9e9e9e">Trigger ({n_configs} configs actives)</div>'
            f'<div style="font-size:22px;color:{trig_color};font-weight:bold">'
            f'{trigger_display}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────────
    render_chart(result)

    st.divider()

    # ── État courant + Signaux ────────────────────────────────────────────────
    col_state, col_signals = st.columns([1, 2])

    with col_state:
        st.subheader("Etat courant (EMA50 / RSI14 ref)")
        render_state_panel(result)

    with col_signals:
        st.subheader("Signaux")
        render_signals(result)

    st.divider()

    # ── Distribution des états sur tous les combos ───────────────────────────
    with st.expander("Distribution des états — 108 combos (dernière bougie)"):
        all_st = result["all_states"]
        if all_st:
            regimes = [v.get("regime") for v in all_st.values()]
            regime_counts = pd.Series(regimes).value_counts()

            rsi_states = [v.get("rsi_state") for v in all_st.values()]
            rsi_counts = pd.Series(rsi_states).value_counts()

            atr_states = [v.get("atr_state") for v in all_st.values()]
            atr_counts = pd.Series(atr_states).value_counts()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Régime (tous combos)")
                st.dataframe(regime_counts.rename("count").to_frame(),
                             use_container_width=True)
            with c2:
                st.caption("RSI state (tous combos)")
                st.dataframe(rsi_counts.rename("count").to_frame(),
                             use_container_width=True)
            with c3:
                st.caption("ATR state (tous combos)")
                st.dataframe(atr_counts.rename("count").to_frame(),
                             use_container_width=True)

    # ── Info timestamp + auto-refresh ────────────────────────────────────────
    st.caption(f"Dernier scan : {result['timestamp'][:19]} UTC — "
               f"Auto-refresh toutes les {REFRESH_SECS}s")

    # Auto-refresh : compte à rebours + rerun
    placeholder = st.empty()
    for remaining in range(REFRESH_SECS, 0, -1):
        placeholder.caption(f"Prochain refresh dans {remaining}s...")
        time.sleep(1)
    st.cache_data.clear()
    st.rerun()


if __name__ == "__main__":
    main()
