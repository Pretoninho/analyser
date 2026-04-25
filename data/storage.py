"""
data/storage.py — Couche de persistance SQLite.

Toute sauvegarde et lecture de données passe par ce module.

Bases :
    prices.db   → ohlcv, derivatives  (données de marché brutes)
    results.db  → indicators          (résultats calculés)
"""

import sqlite3
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
from config import PRICES_DB_PATH, RESULTS_DB_PATH


# ── Connexions ─────────────────────────────────────────────────

@contextmanager
def _conn(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_prices_conn():
    return _conn(PRICES_DB_PATH)


def get_results_conn():
    return _conn(RESULTS_DB_PATH)


# ── Initialisation ─────────────────────────────────────────────

def init_db():
    """Crée toutes les tables dans les deux bases."""
    with get_prices_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                asset       TEXT    NOT NULL,
                timeframe   TEXT    NOT NULL,
                timestamp   INTEGER NOT NULL,
                open        REAL    NOT NULL,
                high        REAL    NOT NULL,
                low         REAL    NOT NULL,
                close       REAL    NOT NULL,
                volume      REAL    NOT NULL,
                vwap        REAL,
                created_at  TEXT    DEFAULT (datetime('now')),
                UNIQUE (asset, timeframe, timestamp)
            );

            CREATE TABLE IF NOT EXISTS derivatives (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                asset           TEXT    NOT NULL,
                timestamp       INTEGER NOT NULL,
                iv_atm          REAL,
                iv_skew_25d     REAL,
                iv_skew_10d     REAL,
                term_1w         REAL,
                term_1m         REAL,
                term_3m         REAL,
                term_6m         REAL,
                put_call_ratio  REAL,
                index_price     REAL,
                created_at      TEXT    DEFAULT (datetime('now')),
                UNIQUE (asset, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_asset_tf_ts
                ON ohlcv (asset, timeframe, timestamp);

            CREATE INDEX IF NOT EXISTS idx_derivatives_asset_ts
                ON derivatives (asset, timestamp);
        """)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS market_1m (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                asset           TEXT    NOT NULL,
                ts              INTEGER NOT NULL,
                open            REAL    NOT NULL,
                high            REAL    NOT NULL,
                low             REAL    NOT NULL,
                close           REAL    NOT NULL,
                volume          REAL,
                open_interest   REAL,
                funding_rate    REAL,
                UNIQUE(asset, ts)
            );

            CREATE INDEX IF NOT EXISTS idx_market_1m_asset_ts
                ON market_1m (asset, ts);
        """)
    print(f"[storage] prices.db initialisée : {PRICES_DB_PATH}")

    with get_results_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS indicators (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                asset           TEXT    NOT NULL,
                timeframe       TEXT    NOT NULL,
                timestamp       INTEGER NOT NULL,
                vol_realized    REAL,
                vol_annualized  REAL,
                atr             REAL,
                zscore          REAL,
                regime          TEXT,
                regime_score    REAL,
                vol_zscore      REAL,
                volume_zscore   REAL,
                price_zscore    REAL,
                liq_signal      INTEGER,
                created_at      TEXT    DEFAULT (datetime('now')),
                UNIQUE (asset, timeframe, timestamp)
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name         TEXT    NOT NULL,
                asset               TEXT    NOT NULL,
                timeframe           TEXT    NOT NULL,
                params_json         TEXT,
                capital_start       REAL,
                capital_end         REAL,
                total_return_pct    REAL,
                sharpe              REAL,
                max_drawdown_pct    REAL,
                win_rate            REAL,
                profit_factor       REAL,
                total_trades        INTEGER,
                winning_trades      INTEGER,
                losing_trades       INTEGER,
                avg_win_pct         REAL,
                avg_loss_pct        REAL,
                run_at              TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS backtest_trades (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER NOT NULL REFERENCES backtest_runs(id),
                signal_name         TEXT    NOT NULL,
                asset               TEXT    NOT NULL,
                timeframe           TEXT    NOT NULL,
                direction           INTEGER,
                entry_ts            TEXT,
                entry_price         REAL,
                exit_ts             TEXT,
                exit_price          REAL,
                qty                 REAL,
                capital_before      REAL,
                capital_after       REAL,
                pnl                 REAL,
                pnl_pct             REAL,
                fees                REAL,
                slippage_cost       REAL,
                exit_reason         TEXT,
                signal_label_entry  TEXT,
                signal_label_exit   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_indicators_asset_tf_ts
                ON indicators (asset, timeframe, timestamp);

            CREATE INDEX IF NOT EXISTS idx_bt_runs_signal
                ON backtest_runs (signal_name, asset, timeframe);

            CREATE INDEX IF NOT EXISTS idx_bt_trades_run
                ON backtest_trades (run_id);
        """)
    print(f"[storage] results.db initialisée : {RESULTS_DB_PATH}")

    # Migration : ajout des colonnes liquidation pour les DB existantes
    _migrate_indicators()


def _migrate_indicators():
    """Ajoute les colonnes liquidation à une DB indicators existante."""
    new_cols = [
        ("vol_zscore",    "REAL"),
        ("volume_zscore", "REAL"),
        ("price_zscore",  "REAL"),
        ("liq_signal",    "INTEGER"),
    ]
    with get_results_conn() as conn:
        for col, typ in new_cols:
            try:
                conn.execute(f"ALTER TABLE indicators ADD COLUMN {col} {typ}")
            except Exception:
                pass  # colonne déjà présente


# ── OHLCV ──────────────────────────────────────────────────────

def save_ohlcv(asset: str, timeframe: str, df: pd.DataFrame):
    """
    Sauvegarde un DataFrame OHLCV dans prices.db.
    Colonnes attendues : timestamp, open, high, low, close, volume, vwap (optionnel).
    INSERT OR IGNORE → pas de doublon.
    """
    if df.empty:
        return

    if "vwap" not in df.columns:
        df = df.copy()
        df["vwap"] = None

    rows = [
        (asset, timeframe, int(row.timestamp), float(row.open), float(row.high),
         float(row.low), float(row.close), float(row.volume), row.vwap)
        for row in df.itertuples(index=False)
    ]

    with get_prices_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO ohlcv
                (asset, timeframe, timestamp, open, high, low, close, volume, vwap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    print(f"[storage] OHLCV sauvegardé : {asset} {timeframe} — {len(rows)} bougies")


def load_ohlcv(asset: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Charge les N dernières bougies OHLCV."""
    with get_prices_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM ohlcv
            WHERE asset = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (asset, timeframe, limit)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","vwap"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_ohlcv_between(asset: str, timeframe: str,
                        start: datetime, end: datetime) -> pd.DataFrame:
    """Charge les bougies entre deux dates."""
    ts_start = int(start.timestamp())
    ts_end   = int(end.timestamp())

    with get_prices_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM ohlcv
            WHERE asset = ? AND timeframe = ?
              AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (asset, timeframe, ts_start, ts_end)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","vwap"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


# ── Dérivés ────────────────────────────────────────────────────

def save_derivatives(asset: str, data: dict):
    """
    Sauvegarde un snapshot de données dérivées dans prices.db.
    data doit contenir 'timestamp' (unix seconds) + les champs optionnels.
    INSERT OR REPLACE → un seul snapshot par (asset, timestamp).
    """
    fields = ["iv_atm", "iv_skew_25d", "iv_skew_10d",
              "term_1w", "term_1m", "term_3m", "term_6m",
              "put_call_ratio", "index_price"]

    with get_prices_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO derivatives
                (asset, timestamp, iv_atm, iv_skew_25d, iv_skew_10d,
                 term_1w, term_1m, term_3m, term_6m, put_call_ratio, index_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            asset,
            int(data.get("timestamp", datetime.utcnow().timestamp())),
            data.get("iv_atm"),
            data.get("iv_skew_25d"),
            data.get("iv_skew_10d"),
            data.get("term_1w"),
            data.get("term_1m"),
            data.get("term_3m"),
            data.get("term_6m"),
            data.get("put_call_ratio"),
            data.get("index_price"),
        ))

    print(f"[storage] Dérivés sauvegardés : {asset}")


def load_derivatives(asset: str, limit: int = 100) -> pd.DataFrame:
    """Charge les N derniers snapshots de dérivés."""
    with get_prices_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, iv_atm, iv_skew_25d, iv_skew_10d,
                   term_1w, term_1m, term_3m, term_6m,
                   put_call_ratio, index_price
            FROM derivatives
            WHERE asset = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (asset, limit)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "iv_atm", "iv_skew_25d", "iv_skew_10d",
        "term_1w", "term_1m", "term_3m", "term_6m",
        "put_call_ratio", "index_price"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Indicateurs ────────────────────────────────────────────────

def save_indicators(asset: str, timeframe: str, df: pd.DataFrame):
    """
    Sauvegarde les indicateurs calculés dans results.db.
    Colonnes attendues : timestamp + colonnes d'indicateurs.
    INSERT OR REPLACE → mise à jour si recalcul.
    """
    if df.empty:
        return

    cols = [
        "vol_realized", "vol_annualized", "atr", "zscore", "regime", "regime_score",
        "vol_zscore", "volume_zscore", "price_zscore", "liq_signal",
    ]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = None

    rows = [
        (asset, timeframe, int(row.timestamp.timestamp()),
         row.vol_realized, row.vol_annualized, row.atr,
         row.zscore, row.regime, row.regime_score,
         row.vol_zscore, row.volume_zscore, row.price_zscore, row.liq_signal)
        for row in df.itertuples(index=False)
    ]

    with get_results_conn() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO indicators
                (asset, timeframe, timestamp, vol_realized, vol_annualized,
                 atr, zscore, regime, regime_score,
                 vol_zscore, volume_zscore, price_zscore, liq_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    print(f"[storage] Indicateurs sauvegardés : {asset} {timeframe} — {len(rows)} lignes")


def load_indicators(asset: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """Charge les N derniers indicateurs calculés."""
    with get_results_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, vol_realized, vol_annualized, atr,
                   zscore, regime, regime_score,
                   vol_zscore, volume_zscore, price_zscore, liq_signal
            FROM indicators
            WHERE asset = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (asset, timeframe, limit)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "vol_realized", "vol_annualized",
        "atr", "zscore", "regime", "regime_score",
        "vol_zscore", "volume_zscore", "price_zscore", "liq_signal"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Backtest ───────────────────────────────────────────────────

def save_backtest_run(result) -> int:
    """
    Persiste un BacktestResult complet dans results.db.
    Retourne l'id du run inséré.
    """
    import json

    with get_results_conn() as conn:
        cur = conn.execute("""
            INSERT INTO backtest_runs
                (signal_name, asset, timeframe, params_json,
                 capital_start, capital_end, total_return_pct, sharpe,
                 max_drawdown_pct, win_rate, profit_factor, total_trades,
                 winning_trades, losing_trades, avg_win_pct, avg_loss_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.signal_name, result.asset, result.timeframe,
            json.dumps(result.params),
            result.capital_start, result.capital_end,
            result.total_return_pct, result.sharpe,
            result.max_drawdown_pct, result.win_rate,
            result.profit_factor, result.total_trades,
            result.winning_trades, result.losing_trades,
            result.avg_win_pct, result.avg_loss_pct,
        ))
        run_id = cur.lastrowid

        rows = [
            (run_id, result.signal_name, result.asset, result.timeframe,
             t.direction, str(t.entry_ts), t.entry_price,
             str(t.exit_ts), t.exit_price,
             t.qty, t.capital_before, t.capital_after,
             t.pnl, t.pnl_pct, t.fees, t.slippage_cost,
             t.exit_reason, t.signal_label_entry, t.signal_label_exit)
            for t in result.trades
        ]
        conn.executemany("""
            INSERT INTO backtest_trades
                (run_id, signal_name, asset, timeframe, direction,
                 entry_ts, entry_price, exit_ts, exit_price,
                 qty, capital_before, capital_after,
                 pnl, pnl_pct, fees, slippage_cost,
                 exit_reason, signal_label_entry, signal_label_exit)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    print(f"[storage] Backtest sauvegardé : run_id={run_id} — {result.signal_name} {result.asset} {result.timeframe}")
    return run_id


def load_backtest_runs(signal_name: str = None, asset: str = None,
                       timeframe: str = None, limit: int = 50) -> pd.DataFrame:
    """Charge l'historique des runs de backtest."""
    query  = "SELECT * FROM backtest_runs WHERE 1=1"
    params = []
    if signal_name:
        query += " AND signal_name = ?"; params.append(signal_name)
    if asset:
        query += " AND asset = ?";       params.append(asset)
    if timeframe:
        query += " AND timeframe = ?";   params.append(timeframe)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_results_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def load_backtest_trades(run_id: int) -> pd.DataFrame:
    """Charge les trades d'un run spécifique."""
    with get_results_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY id ASC",
            (run_id,)
        ).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ── market_1m ──────────────────────────────────────────────────

def save_market_1m(asset: str, df: pd.DataFrame):
    """
    Sauvegarde les données 1min fusionnées (OHLCV + OI + funding rate).
    Colonnes attendues : ts, open, high, low, close, volume, open_interest, funding_rate.
    INSERT OR IGNORE → idempotent, safe pour les mises à jour incrémentales.
    """
    if df.empty:
        return

    required = ["ts", "open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"[storage] save_market_1m : colonnes manquantes. Attendues : {required}")

    df = df.copy()
    for col in ["volume", "open_interest", "funding_rate"]:
        if col not in df.columns:
            df[col] = None

    rows = [
        (asset, int(row.ts),
         float(row.open), float(row.high), float(row.low), float(row.close),
         float(row.volume) if row.volume is not None else None,
         float(row.open_interest) if row.open_interest is not None else None,
         float(row.funding_rate)  if row.funding_rate  is not None else None)
        for row in df.itertuples(index=False)
    ]

    with get_prices_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO market_1m
                (asset, ts, open, high, low, close, volume, open_interest, funding_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    print(f"[storage] market_1m sauvegardé : {asset} — {len(rows)} bougies")


def load_market_1m(asset: str, limit: int = 1000) -> pd.DataFrame:
    """
    Charge les N dernières bougies 1min avec OI et funding rate.
    Retourne un DataFrame trié par ts croissant avec timestamp UTC.
    """
    with get_prices_conn() as conn:
        rows = conn.execute("""
            SELECT ts, open, high, low, close, volume, open_interest, funding_rate
            FROM market_1m
            WHERE asset = ?
            ORDER BY ts DESC
            LIMIT ?
        """, (asset, limit)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume", "open_interest", "funding_rate"
    ])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ── Utilitaires ────────────────────────────────────────────────

def db_summary() -> dict:
    """Retourne un résumé de l'état des deux bases."""
    summary = {}

    with get_prices_conn() as conn:
        for table in ["ohlcv", "derivatives"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            summary[table] = {"rows": count}

        assets = conn.execute(
            "SELECT DISTINCT asset, timeframe FROM ohlcv"
        ).fetchall()
        summary["assets"] = [dict(r) for r in assets]

        last_ohlcv = conn.execute("""
            SELECT asset, timeframe, MAX(timestamp) as last_ts
            FROM ohlcv GROUP BY asset, timeframe
        """).fetchall()
        summary["last_ohlcv"] = [dict(r) for r in last_ohlcv]

    with get_results_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
        summary["indicators"] = {"rows": count}

    return summary
