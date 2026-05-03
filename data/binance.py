"""
data/binance.py — Téléchargement et chargement des données historiques Binance.

Source  : data.binance.vision — BTCUSDT futures USDT-M, bougies 1min.
Format CSV Binance (sans header) :
  open_time, open, high, low, close, volume, close_time,
  quote_vol, trades, taker_buy_vol, taker_buy_quote_vol, ignore

Le fichier résultant est sauvegardé en Parquet pour chargement rapide.
"""

import io
import zipfile
from datetime import date
from pathlib import Path

import requests
import pandas as pd

# ── Chemins ────────────────────────────────────────────────────
import os
BINANCE_DIR = Path(__file__).parent.parent / "data_binance"

# ── Constantes Binance ─────────────────────────────────────────
_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
_SYMBOL   = os.getenv("BINANCE_SYMBOL", "BTCUSDT").upper()
_INTERVAL = "1m"

PARQUET_PATH = BINANCE_DIR / f"{_SYMBOL.lower()}_1m.parquet"

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades",
    "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]


# ── Téléchargement ─────────────────────────────────────────────

_BASE_URL_DAILY = "https://data.binance.vision/data/futures/um/daily/klines"


def _download_monthly(year: int, month: int, raw_dir: Path):
    tag      = f"{year}-{month:02d}"
    csv_path = raw_dir / f"{_SYMBOL}-{_INTERVAL}-{tag}.csv"
    if csv_path.exists():
        print(f"[binance] {tag} deja present.")
        return pd.read_csv(csv_path, header=None, names=_COLUMNS)
    url = f"{_BASE_URL}/{_SYMBOL}/{_INTERVAL}/{_SYMBOL}-{_INTERVAL}-{tag}.zip"
    print(f"[binance] {tag} (mensuel)...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_path.write_bytes(z.read(z.namelist()[0]))
        print("OK")
        return pd.read_csv(csv_path, header=None, names=_COLUMNS)
    except Exception as e:
        print(f"ERREUR ({e})")
        return None


def _download_daily(year: int, month: int, raw_dir: Path):
    from datetime import date, timedelta
    import calendar
    today    = date.today()
    day_dir  = raw_dir / "daily"
    day_dir.mkdir(exist_ok=True)
    frames   = []
    last_day = min(calendar.monthrange(year, month)[1],
                   today.day if (year, month) == (today.year, today.month) else 31)
    for d in range(1, last_day + 1):
        tag      = f"{year}-{month:02d}-{d:02d}"
        csv_path = day_dir / f"{_SYMBOL}-{_INTERVAL}-{tag}.csv"
        if not csv_path.exists():
            url = f"{_BASE_URL_DAILY}/{_SYMBOL}/{_INTERVAL}/{_SYMBOL}-{_INTERVAL}-{tag}.zip"
            print(f"[binance] {tag}...", end=" ", flush=True)
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    csv_path.write_bytes(z.read(z.namelist()[0]))
                print("OK")
            except Exception as e:
                print(f"skip ({e})")
                continue
        try:
            frames.append(pd.read_csv(csv_path, header=None, names=_COLUMNS))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


def download_binance_1m(
    start: tuple = (2020, 1),
    end:   tuple = None,
) -> Path:
    """
    Télécharge les fichiers Binance BTCUSDT futures 1m.
    Utilise les fichiers mensuels, et les fichiers journaliers pour le mois en cours.

    Args:
        start : (year, month) — début de l'historique
        end   : (year, month) — fin (défaut: mois courant)

    Returns:
        Path du fichier parquet résultant.
    """
    BINANCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = BINANCE_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    today = date.today()
    if end is None:
        end = (today.year, today.month)

    frames = []
    year, month = start
    end_year, end_month = end

    while (year, month) <= (end_year, end_month):
        is_current_month = (year == today.year and month == today.month)
        if is_current_month:
            df_m = _download_daily(year, month, raw_dir)
        else:
            df_m = _download_monthly(year, month, raw_dir)
            if df_m is None:
                year, month = _next_month(year, month)
                continue
        if df_m is not None and not df_m.empty:
            frames.append(df_m)
        year, month = _next_month(year, month)

    if not frames:
        raise RuntimeError("[binance] Aucune donnée téléchargée.")

    df_new = _clean(pd.concat(frames, ignore_index=True))

    # Fusionner avec le parquet existant si présent
    if PARQUET_PATH.exists():
        df_existing = pd.read_parquet(PARQUET_PATH)
        existing_end = df_existing["timestamp"].iloc[-1]
        new_start    = df_new["timestamp"].iloc[0]
        if new_start <= existing_end:
            # Supprimer le chevauchement dans l'existant
            df_existing = df_existing[df_existing["timestamp"] < new_start]
        df = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts")
    else:
        df = df_new

    df.to_parquet(PARQUET_PATH, index=False)

    y0 = df["timestamp"].iloc[0].strftime("%Y-%m")
    y1 = df["timestamp"].iloc[-1].strftime("%Y-%m")
    print(f"[binance] {len(df):,} bougies 1min ({y0} a {y1}) -> {PARQUET_PATH}")
    return PARQUET_PATH


def _next_month(year: int, month: int):
    month += 1
    if month > 12:
        month = 1
        year += 1
    return year, month


# ── Chargement ─────────────────────────────────────────────────

def load_binance_1m(path: Path = None) -> pd.DataFrame:
    """
    Charge le parquet Binance en DataFrame standardisé (même colonnes
    que les données Deribit pour compatibilité avec les fonctions existantes).
    """
    path = Path(path) if path else PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"[binance] Fichier introuvable : {path}\n"
            "Lancez d'abord : python main.py --download-binance"
        )
    df = pd.read_parquet(path)
    start = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    end   = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    print(f"[binance] {len(df):,} bougies chargees ({start} - {end})")
    return df


# ── Nettoyage interne ─────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    df["ts"]        = (df["open_time"] // 1000).astype("int64")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Colonnes standardisées — compatibles avec aggregate_5m / compute_features
    df = df[[
        "ts", "timestamp",
        "open", "high", "low", "close", "volume",
        "taker_buy_vol",
    ]].copy()

    # Funding rate absent des klines CSV → colonne à zéro
    df["funding_rate"]  = 0.0
    df["open_interest"] = 0.0

    df = (df
          .dropna(subset=["open", "high", "low", "close", "volume"])
          .drop_duplicates(subset=["ts"])
          .sort_values("ts")
          .reset_index(drop=True))
    return df
