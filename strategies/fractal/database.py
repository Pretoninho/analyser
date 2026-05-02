"""
Database module for signal logging and persistence
Supports SQLite (development) and PostgreSQL (production)
"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
import os


class SignalDatabase:
    def __init__(self, db_type: str = "sqlite", db_path: str = None):
        """
        Initialise la base de données
        db_type: 'sqlite' ou 'postgresql'
        """
        self.db_type = db_type

        if db_type == "sqlite":
            self.db_path = db_path or "signals.db"
            self._init_sqlite()
        elif db_type == "postgresql":
            self.connection_string = os.getenv('DATABASE_URL')
            self._init_postgresql()
        else:
            raise ValueError(f"Unknown database type: {db_type}")

    def _init_sqlite(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                setup TEXT NOT NULL,
                day_date DATE NOT NULL,
                kz TEXT NOT NULL,
                pattern TEXT NOT NULL,
                entry_price REAL NOT NULL,
                confidence REAL NOT NULL,
                levels JSON NOT NULL,
                detected_at DATETIME NOT NULL,
                setup_tag TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                exit_price REAL,
                pnl REAL,
                exit_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_setup ON signals(setup)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_day_date ON signals(day_date)
        ''')

        conn.commit()
        conn.close()

    def _init_postgresql(self):
        """Initialise la base de données PostgreSQL"""
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                setup TEXT NOT NULL,
                day_date DATE NOT NULL,
                kz TEXT NOT NULL,
                pattern TEXT NOT NULL,
                entry_price DECIMAL NOT NULL,
                confidence DECIMAL NOT NULL,
                levels JSONB NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                setup_tag TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                exit_price DECIMAL,
                pnl DECIMAL,
                exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_setup ON signals(setup)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)
        ''')

        conn.commit()
        conn.close()

    def log_signal(self, signal: Dict) -> int:
        """Enregistre un signal dans la base de données"""
        if self.db_type == "sqlite":
            return self._log_signal_sqlite(signal)
        else:
            return self._log_signal_postgresql(signal)

    def _log_signal_sqlite(self, signal: Dict) -> int:
        """Enregistre un signal en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (
                timestamp, setup, day_date, kz, pattern, entry_price,
                confidence, levels, detected_at, setup_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['timestamp'],
            signal['setup'],
            signal['day_date'],
            signal['kz'],
            signal['pattern'],
            signal['entry_price'],
            signal['confidence'],
            json.dumps(signal['levels']),
            signal['detected_at'],
            signal['setup_tag']
        ))

        conn.commit()
        signal_id = cursor.lastrowid
        conn.close()

        return signal_id

    def _log_signal_postgresql(self, signal: Dict) -> int:
        """Enregistre un signal en PostgreSQL"""
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (
                timestamp, setup, day_date, kz, pattern, entry_price,
                confidence, levels, detected_at, setup_tag
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (
            signal['timestamp'],
            signal['setup'],
            signal['day_date'],
            signal['kz'],
            signal['pattern'],
            signal['entry_price'],
            signal['confidence'],
            json.dumps(signal['levels']),
            signal['detected_at'],
            signal['setup_tag']
        ))

        signal_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        return signal_id

    def get_signals(self, setup: Optional[str] = None, limit: int = 100,
                    status: str = 'open') -> List[Dict]:
        """Récupère les signaux de la base de données"""
        if self.db_type == "sqlite":
            return self._get_signals_sqlite(setup, limit, status)
        else:
            return self._get_signals_postgresql(setup, limit, status)

    def _get_signals_sqlite(self, setup: Optional[str] = None, limit: int = 100,
                            status: str = 'open') -> List[Dict]:
        """Récupère les signaux de SQLite"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if setup:
            cursor.execute('''
                SELECT * FROM signals
                WHERE setup = ? AND status = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (setup, status, limit))
        else:
            cursor.execute('''
                SELECT * FROM signals
                WHERE status = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (status, limit))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(self, setup: Optional[str] = None) -> Dict:
        """Retourne les statistiques des signaux"""
        if self.db_type == "sqlite":
            return self._get_statistics_sqlite(setup)
        else:
            return self._get_statistics_postgresql(setup)

    def _get_statistics_sqlite(self, setup: Optional[str] = None) -> Dict:
        """Récupère les statistiques de SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if setup:
            # Total signals
            cursor.execute('SELECT COUNT(*) FROM signals WHERE setup = ?', (setup,))
            total = cursor.fetchone()[0]

            # Win rate (PnL > 0)
            cursor.execute('''
                SELECT COUNT(*) FROM signals
                WHERE setup = ? AND status = 'closed' AND pnl > 0
            ''', (setup,))
            wins = cursor.fetchone()[0]

            # Avg PnL
            cursor.execute('''
                SELECT AVG(pnl) FROM signals
                WHERE setup = ? AND status = 'closed'
            ''', (setup,))
            avg_pnl = cursor.fetchone()[0] or 0

            # Total PnL
            cursor.execute('''
                SELECT SUM(pnl) FROM signals
                WHERE setup = ? AND status = 'closed'
            ''', (setup,))
            total_pnl = cursor.fetchone()[0] or 0
        else:
            cursor.execute('SELECT COUNT(*) FROM signals')
            total = cursor.fetchone()[0]

            cursor.execute('''
                SELECT COUNT(*) FROM signals
                WHERE status = 'closed' AND pnl > 0
            ''')
            wins = cursor.fetchone()[0]

            cursor.execute('''
                SELECT AVG(pnl) FROM signals WHERE status = 'closed'
            ''')
            avg_pnl = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT SUM(pnl) FROM signals WHERE status = 'closed'
            ''')
            total_pnl = cursor.fetchone()[0] or 0

        conn.close()

        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            'total_signals': total,
            'wins': wins,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        }

    def update_signal_exit(self, signal_id: int, exit_price: float,
                           pnl: float, exit_reason: str = None):
        """Met à jour l'exit d'un signal"""
        if self.db_type == "sqlite":
            self._update_signal_exit_sqlite(signal_id, exit_price, pnl, exit_reason)
        else:
            self._update_signal_exit_postgresql(signal_id, exit_price, pnl, exit_reason)

    def _update_signal_exit_sqlite(self, signal_id: int, exit_price: float,
                                   pnl: float, exit_reason: str = None):
        """Met à jour l'exit en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE signals
            SET exit_price = ?, pnl = ?, exit_reason = ?, status = 'closed'
            WHERE id = ?
        ''', (exit_price, pnl, exit_reason, signal_id))

        conn.commit()
        conn.close()
