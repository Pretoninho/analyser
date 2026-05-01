"""
data/deribit.py — OHLCV, Funding Rate et Open Interest depuis Deribit BTC-PERPETUAL.
"""

from datetime import datetime
import time
import numpy as np
import requests
import pandas as pd
from config import ASSETS, SOURCES

# Deribit retourne au maximum ~5 000 bougies par appel
_DERIBIT_MAX_CANDLES = 5000

DERIBIT_INSTRUMENTS = {
    "BTC": "BTC-PERPETUAL",
}

DERIBIT_RESOLUTIONS = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "4h":  "240",
    "1d":  "1D",
}

_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}

# get_volatility_index_data utilise une resolution en secondes
# (ou "1D" pour daily), differente du format tradingview chart.
DVOL_RESOLUTIONS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": "1D",
}


def fetch_ohlcv(asset: str, timeframe: str = "1m", limit: int = 720) -> pd.DataFrame:
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    resolution = DERIBIT_RESOLUTIONS.get(timeframe)
    if not instrument or not resolution:
        print(f"[deribit] Actif ou timeframe inconnu : {asset} {timeframe}")
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - limit * _INTERVAL_SECONDS[timeframe] * 1000

    url = f"{SOURCES['deribit_base_url']}/get_tradingview_chart_data"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
        "resolution":      resolution,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau OHLCV : {e}")
        return pd.DataFrame()

    result = data.get("result", {})
    if result.get("status") != "ok" or not result.get("ticks"):
        print(f"[deribit] Reponse OHLCV invalide : {data.get('error', result.get('status'))}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "ts":     [t // 1000 for t in result["ticks"]],
        "open":   result["open"],
        "high":   result["high"],
        "low":    result["low"],
        "close":  result["close"],
        "volume": result["volume"],
    })

    df = df.iloc[:-1]  # exclure la derniere bougie (potentiellement incomplete)

    if limit and len(df) > limit:
        df = df.tail(limit)

    df = df.reset_index(drop=True)

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] OHLCV {asset} {timeframe} : {len(df)} bougies {first} -> {last} UTC")
    return df


def fetch_funding_rate(asset: str, limit: int = 200) -> pd.DataFrame:
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        print(f"[deribit] Actif inconnu : {asset}")
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - limit * 8 * 3600 * 1000  # funding toutes les 8h

    url = f"{SOURCES['deribit_base_url']}/get_funding_rate_history"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau funding : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API funding : {data['error']}")
        return pd.DataFrame()

    items = data.get("result", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["ts"]           = df["timestamp"].astype(int) // 1000
    df["funding_rate"] = df["interest_8h"].astype(float)
    df = df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] Funding {asset} : {len(df)} points {first} -> {last} UTC")
    return df


def fetch_ohlcv_historical(asset: str, timeframe: str = "1m", days: int = 30) -> pd.DataFrame:
    """
    Recupere l'historique complet sur N jours en paginant les appels Deribit.

    Deribit retourne ~5 000 bougies max par appel.
    Pour 30 jours de 1min : 43 200 bougies -> ~9 appels pagines.

    Args:
        asset     : "BTC"
        timeframe : "1m", "5m", "1h", etc.
        days      : nombre de jours a remonter (defaut 30)

    Returns:
        DataFrame consolide : ts, open, high, low, close, volume (trié ASC, sans doublons)
    """
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    resolution = DERIBIT_RESOLUTIONS.get(timeframe)
    if not instrument or not resolution:
        print(f"[deribit] Actif ou timeframe inconnu : {asset} {timeframe}")
        return pd.DataFrame()

    interval_ms  = _INTERVAL_SECONDS[timeframe] * 1000
    chunk_ms     = _DERIBIT_MAX_CANDLES * interval_ms
    end_ms       = int(time.time() * 1000)
    start_ms     = end_ms - days * 86400 * 1000

    url    = f"{SOURCES['deribit_base_url']}/get_tradingview_chart_data"
    chunks = []
    cursor = start_ms

    n_chunks = max(1, (end_ms - start_ms) // chunk_ms)
    print(f"[deribit] Backfill {asset} {timeframe} sur {days} jours (~{n_chunks} appels)...")

    try:
        while cursor < end_ms:
            chunk_end = min(cursor + chunk_ms, end_ms)
            done_pct  = (cursor - start_ms) / (end_ms - start_ms) * 100
            print(f"  {done_pct:5.1f}% — {pd.to_datetime(cursor, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')} UTC", end="\r")

            params = {
                "instrument_name": instrument,
                "start_timestamp": cursor,
                "end_timestamp":   chunk_end,
                "resolution":      resolution,
            }
            try:
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                print(f"\n[deribit] Erreur reseau (chunk) : {e}")
                time.sleep(2)
                continue

            result = data.get("result", {})
            if result.get("status") == "ok" and result.get("ticks"):
                chunks.append(pd.DataFrame({
                    "ts":     [t // 1000 for t in result["ticks"]],
                    "open":   result["open"],
                    "high":   result["high"],
                    "low":    result["low"],
                    "close":  result["close"],
                    "volume": result["volume"],
                }))

            cursor = chunk_end
            time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n[deribit] Interruption — {len(chunks)} chunks recuperes, sauvegarde partielle...")

    if not chunks:
        print("[deribit] Aucune donnee historique recuperee.")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df.iloc[:-1]  # exclure la derniere bougie incomplete

    first = pd.to_datetime(df["ts"].iloc[0],  unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last  = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] Historique {asset} {timeframe} : {len(df):,} bougies {first} -> {last} UTC")
    return df


def fetch_funding_rate_historical(asset: str, days: int = 30) -> pd.DataFrame:
    """Recupere l'historique du funding rate sur N jours (intervalles 8h)."""
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    url = f"{SOURCES['deribit_base_url']}/get_funding_rate_history"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau funding historique : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API funding : {data['error']}")
        return pd.DataFrame()

    items = data.get("result", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["ts"]           = df["timestamp"].astype(int) // 1000
    df["funding_rate"] = df["interest_8h"].astype(float)
    df = df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    print(f"[deribit] Funding historique {asset} : {len(df)} points sur {days} jours")
    return df


def fetch_open_interest(asset: str) -> pd.DataFrame:
    """
    Snapshot courant de l'OI depuis Deribit (pas d'historique REST disponible).
    Retourne un DataFrame avec un seul point (ts, open_interest).
    Le collector forward-fille cette valeur sur toute la grille 1min.
    """
    instrument = DERIBIT_INSTRUMENTS.get(asset)
    if not instrument:
        print(f"[deribit] Actif inconnu : {asset}")
        return pd.DataFrame()

    url = f"{SOURCES['deribit_base_url']}/get_book_summary_by_instrument"
    params = {"instrument_name": instrument}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau OI : {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"[deribit] Erreur API OI : {data['error']}")
        return pd.DataFrame()

    result = data.get("result", [])
    if not result:
        return pd.DataFrame()

    item = result[0]
    oi   = float(item.get("open_interest", 0))
    ts   = int(item.get("creation_timestamp", time.time() * 1000)) // 1000

    print(f"[deribit] OI {asset} : {oi:,.0f} BTC @ {pd.to_datetime(ts, unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC")
    return pd.DataFrame([{"ts": ts, "open_interest": oi}])


def fetch_dvol_history(asset: str = "BTC", timeframe: str = "1h", days: int = 30) -> pd.DataFrame:
    """
    Recupere l'historique du DVOL (Deribit Volatility Index).

    Args:
        asset: "BTC" ou "ETH"
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
        days: profondeur historique

    Returns:
        DataFrame avec colonnes: ts, dvol_open, dvol_high, dvol_low, dvol_close
        (ts en secondes unix, tri ascendant)
    """
    currency = asset.upper()
    resolution = DVOL_RESOLUTIONS.get(timeframe)
    if resolution is None:
        print(f"[deribit] Timeframe DVOL inconnu: {timeframe}")
        return pd.DataFrame()

    end_ts = int(time.time())
    start_ts = end_ts - int(days * 86400)

    url = f"{SOURCES['deribit_base_url']}/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": start_ts * 1000,
        "end_timestamp": end_ts * 1000,
        "resolution": resolution,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        print(f"[deribit] Erreur reseau DVOL: {e}")
        return pd.DataFrame()

    if "error" in payload:
        print(f"[deribit] Erreur API DVOL: {payload['error']}")
        return pd.DataFrame()

    result = payload.get("result", {})
    rows = result.get("data", []) if isinstance(result, dict) else []
    if not rows:
        print(f"[deribit] DVOL vide pour {currency}")
        return pd.DataFrame()

    parsed = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            ts_ms = int(item[0])
            d_open = float(item[1])
            d_high = float(item[2])
            d_low = float(item[3])
            d_close = float(item[4])
        except (TypeError, ValueError):
            continue

        parsed.append(
            {
                "ts": ts_ms // 1000,
                "dvol_open": d_open,
                "dvol_high": d_high,
                "dvol_low": d_low,
                "dvol_close": d_close,
            }
        )

    if not parsed:
        print(f"[deribit] DVOL parse vide pour {currency}")
        return pd.DataFrame()

    df = pd.DataFrame(parsed).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if len(df) > 1:
        # Exclut la derniere bougie potentiellement incomplete
        df = df.iloc[:-1].reset_index(drop=True)

    if df.empty:
        print(f"[deribit] DVOL apres nettoyage vide pour {currency}")
        return df

    first = pd.to_datetime(df["ts"].iloc[0], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    last = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    print(f"[deribit] DVOL {currency} {timeframe}: {len(df)} points {first} -> {last} UTC")
    return df


def _api_get(endpoint: str, params: dict, timeout: int = 15) -> dict:
    """Wrapper REST Deribit avec gestion d'erreurs homogène."""
    url = f"{SOURCES['deribit_base_url']}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Erreur reseau {endpoint}: {e}") from e

    if "error" in payload:
        raise RuntimeError(f"Erreur API {endpoint}: {payload['error']}")

    return payload.get("result", {})


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _weighted_avg(items: list[tuple[float, float]]) -> float | None:
    """Moyenne ponderee de tuples (valeur, poids)."""
    num = 0.0
    den = 0.0
    for value, weight in items:
        if value is None or weight is None or weight <= 0:
            continue
        num += value * weight
        den += weight
    if den <= 0:
        return None
    return num / den


def fetch_options_analytics(asset: str, max_books: int = 40) -> dict:
    """
    Recupere un snapshot options Deribit pour BTC avec :
      - IV ATM
      - Skew 25-delta (put_iv_25d - call_iv_25d)
      - Put/Call ratio
      - Term structure (1w/1m/3m/6m)
      - Max Pain (nearest expiry)
      - GEX proxy (somme gamma * OI * spot)

    Retourne un dict pret pour save_derivatives().
    """
    currency = asset.upper()
    now_ts = int(time.time())

    try:
        index_name = ASSETS.get(currency, {}).get("deribit_index", f"{currency.lower()}_usd")
        idx_res = _api_get("get_index_price", {"index_name": index_name}, timeout=10)
        index_price = _safe_float(idx_res.get("index_price"))
        if index_price is None or index_price <= 0:
            raise RuntimeError(f"index_price invalide pour {index_name}")

        summaries = _api_get(
            "get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
            timeout=20,
        )
        instruments = _api_get(
            "get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
            timeout=20,
        )
    except RuntimeError as e:
        print(f"[deribit] Snapshot options indisponible : {e}")
        return {}

    if not summaries or not instruments:
        print(f"[deribit] Snapshot options vide pour {currency}")
        return {}

    meta_by_name = {
        i.get("instrument_name"): {
            "expiry_ts": int(i.get("expiration_timestamp", 0)) // 1000,
            "strike": _safe_float(i.get("strike")),
            "option_type": i.get("option_type", "").lower(),
        }
        for i in instruments
        if i.get("instrument_name")
    }

    rows = []
    for s in summaries:
        name = s.get("instrument_name")
        if not name or name not in meta_by_name:
            continue
        m = meta_by_name[name]
        if m["expiry_ts"] <= now_ts:
            continue
        oi = _safe_float(s.get("open_interest"))
        iv = _safe_float(s.get("mark_iv"))
        vol = _safe_float(s.get("volume_usd"))
        if vol is None:
            vol = _safe_float(s.get("volume"))

        rows.append({
            "instrument_name": name,
            "expiry_ts": m["expiry_ts"],
            "strike": m["strike"],
            "option_type": m["option_type"],
            "open_interest": oi,
            "mark_iv": iv,
            "volume_metric": vol,
        })

    if not rows:
        print(f"[deribit] Aucun instrument option exploitable pour {currency}")
        return {}

    df = pd.DataFrame(rows)
    expiry_list = sorted(df["expiry_ts"].dropna().unique().tolist())
    if not expiry_list:
        return {}

    # Term structure d'IV : moyenne ponderee par OI sur chaque expiration.
    iv_by_expiry = []
    for exp in expiry_list:
        sub = df[df["expiry_ts"] == exp]
        iv_exp = _weighted_avg([(r.mark_iv, r.open_interest) for r in sub.itertuples(index=False)])
        if iv_exp is not None:
            iv_by_expiry.append((exp, iv_exp))

    def _term_value(target_days: int) -> float | None:
        if not iv_by_expiry:
            return None
        target_ts = now_ts + target_days * 86400
        best = min(iv_by_expiry, key=lambda x: abs(x[0] - target_ts))
        return float(best[1])

    term_1w = _term_value(7)
    term_1m = _term_value(30)
    term_3m = _term_value(90)
    term_6m = _term_value(180)

    nearest_expiry = expiry_list[0]
    near = df[df["expiry_ts"] == nearest_expiry].copy()

    # IV ATM : moyenne call/put la plus proche du spot, ponderee OI.
    near["abs_moneyness"] = (near["strike"] - index_price).abs()
    atm_slice = near.sort_values("abs_moneyness").head(8)
    iv_atm = _weighted_avg([(r.mark_iv, r.open_interest) for r in atm_slice.itertuples(index=False)])

    # Put/Call ratio : volume puts / volume calls (fallback OI si volume absent).
    call_metric = near[near["option_type"] == "call"]["volume_metric"].sum(min_count=1)
    put_metric = near[near["option_type"] == "put"]["volume_metric"].sum(min_count=1)
    if pd.isna(call_metric) or pd.isna(put_metric):
        call_metric = near[near["option_type"] == "call"]["open_interest"].sum(min_count=1)
        put_metric = near[near["option_type"] == "put"]["open_interest"].sum(min_count=1)
    put_call_ratio = None
    if call_metric and call_metric > 0 and put_metric is not None:
        put_call_ratio = float(put_metric / call_metric)

    # Max Pain sur l'expiration la plus proche (proxy OI-based).
    strike_df = (
        near.groupby(["strike", "option_type"], as_index=False)["open_interest"]
        .sum()
        .pivot(index="strike", columns="option_type", values="open_interest")
        .fillna(0.0)
    )
    max_pain = None
    if not strike_df.empty:
        strikes = sorted(strike_df.index.tolist())
        strike_index = strike_df.index.to_numpy(dtype=float)
        call_oi = strike_df["call"].to_numpy(dtype=float) if "call" in strike_df.columns else np.zeros(len(strike_df))
        put_oi = strike_df["put"].to_numpy(dtype=float) if "put" in strike_df.columns else np.zeros(len(strike_df))
        best_strike = None
        best_pain = None
        for s in strikes:
            call_pain = np.maximum(s - strike_index, 0.0).dot(call_oi)
            put_pain = np.maximum(strike_index - s, 0.0).dot(put_oi)
            pain = float(call_pain + put_pain)
            if best_pain is None or pain < best_pain:
                best_pain = pain
                best_strike = float(s)
        max_pain = best_strike

    # Greeks : subset sur nearest expiry pour skew delta et GEX proxy.
    near = near.dropna(subset=["strike"]).copy()
    near["dist"] = (near["strike"] - index_price).abs()
    greek_candidates = near.sort_values("dist").head(max_books)

    calls = []
    puts = []
    gex = 0.0

    for r in greek_candidates.itertuples(index=False):
        try:
            book = _api_get(
                "get_order_book",
                {"instrument_name": r.instrument_name, "depth": 1},
                timeout=8,
            )
        except RuntimeError:
            continue

        greeks = book.get("greeks", {}) or {}
        delta = _safe_float(greeks.get("delta"))
        gamma = _safe_float(greeks.get("gamma"))
        iv = _safe_float(book.get("mark_iv"))
        if iv is None:
            iv = r.mark_iv

        if delta is not None and iv is not None:
            if r.option_type == "call":
                calls.append((delta, iv))
            elif r.option_type == "put":
                puts.append((delta, iv))

        if gamma is not None and r.open_interest is not None:
            gex += gamma * float(r.open_interest) * index_price

    def _closest_iv(entries: list[tuple[float, float]], target_delta: float) -> float | None:
        if not entries:
            return None
        d, iv = min(entries, key=lambda x: abs(x[0] - target_delta))
        _ = d
        return float(iv)

    call_25 = _closest_iv(calls, 0.25)
    put_25 = _closest_iv(puts, -0.25)
    call_10 = _closest_iv(calls, 0.10)
    put_10 = _closest_iv(puts, -0.10)

    iv_skew_25d = None
    if call_25 is not None and put_25 is not None:
        iv_skew_25d = float(put_25 - call_25)

    iv_skew_10d = None
    if call_10 is not None and put_10 is not None:
        iv_skew_10d = float(put_10 - call_10)

    exp_label = datetime.utcfromtimestamp(nearest_expiry).strftime("%Y-%m-%d")
    print(
        f"[deribit] Options {currency} {exp_label} | "
        f"IV_ATM={iv_atm if iv_atm is not None else float('nan'):.2f} "
        f"PCR={put_call_ratio if put_call_ratio is not None else float('nan'):.2f} "
        f"Skew25={iv_skew_25d if iv_skew_25d is not None else float('nan'):+.2f} "
        f"MaxPain={max_pain if max_pain is not None else float('nan'):.0f} "
        f"GEX={gex:+.2f}"
    )

    return {
        "timestamp": now_ts,
        "iv_atm": iv_atm,
        "iv_skew_25d": iv_skew_25d,
        "iv_skew_10d": iv_skew_10d,
        "term_1w": term_1w,
        "term_1m": term_1m,
        "term_3m": term_3m,
        "term_6m": term_6m,
        "put_call_ratio": put_call_ratio,
        "index_price": index_price,
        "max_pain": max_pain,
        "gex": float(gex),
    }


def fetch_option_chain_snapshot(asset: str) -> pd.DataFrame:
    """
    Recupere un snapshot de la chaine options Deribit (non expirée).

    Colonnes de sortie:
      instrument_name, expiry_ts, strike, option_type,
      bid_price, ask_price, mark_price, mark_iv,
      open_interest, volume_usd, index_price, dte_days
    """
    currency = asset.upper()
    now_ts = int(time.time())

    try:
        summaries = _api_get(
            "get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
            timeout=20,
        )
        instruments = _api_get(
            "get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
            timeout=20,
        )
    except RuntimeError as e:
        print(f"[deribit] Option chain indisponible: {e}")
        return pd.DataFrame()

    if not summaries or not instruments:
        print(f"[deribit] Option chain vide pour {currency}")
        return pd.DataFrame()

    meta_by_name = {
        i.get("instrument_name"): {
            "expiry_ts": int(i.get("expiration_timestamp", 0)) // 1000,
            "strike": _safe_float(i.get("strike")),
            "option_type": i.get("option_type", "").lower(),
        }
        for i in instruments
        if i.get("instrument_name")
    }

    rows = []
    for s in summaries:
        name = s.get("instrument_name")
        if not name or name not in meta_by_name:
            continue

        m = meta_by_name[name]
        exp = m["expiry_ts"]
        if exp <= now_ts:
            continue

        index_price = _safe_float(s.get("underlying_price"))
        if index_price is None:
            index_price = _safe_float(s.get("index_price"))

        rows.append(
            {
                "instrument_name": name,
                "expiry_ts": exp,
                "strike": m["strike"],
                "option_type": m["option_type"],
                "bid_price": _safe_float(s.get("bid_price")),
                "ask_price": _safe_float(s.get("ask_price")),
                "mark_price": _safe_float(s.get("mark_price")),
                "mark_iv": _safe_float(s.get("mark_iv")),
                "open_interest": _safe_float(s.get("open_interest")),
                "volume_usd": _safe_float(s.get("volume_usd")),
                "index_price": index_price,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["strike", "index_price"]).copy()
    if df.empty:
        return df

    df["dte_days"] = (df["expiry_ts"] - now_ts) / 86400.0
    df = df[df["dte_days"] > 0].copy()
    df = df.sort_values(["expiry_ts", "strike"]).reset_index(drop=True)

    print(f"[deribit] Option chain {currency}: {len(df)} options actives")
    return df
