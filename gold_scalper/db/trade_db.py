"""SQLite database for Gold scalper bot.

Stores candles, trades, signals, and strategy performance.
Same schema as BTC micro but in a separate DB file.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path("gold_trades/gold_scalper.db")


class GoldTradeDB:
    """SQLite-backed storage for gold scalper."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("GoldTradeDB initialized at %s", self._db_path)

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _create_tables(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL DEFAULT 0,
                    UNIQUE(symbol, timestamp)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_ts
                ON candles(symbol, timestamp DESC)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL DEFAULT 'XAU',
                    direction TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    size REAL,
                    notional_usd REAL,
                    pnl REAL,
                    hold_time_sec INTEGER,
                    confidence REAL,
                    strategy TEXT,
                    session TEXT,
                    trend TEXT,
                    atr REAL,
                    mode TEXT DEFAULT 'paper',
                    status TEXT,
                    meta TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ts
                ON trades(timestamp DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_strategy
                ON trades(strategy, status)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    session TEXT,
                    direction TEXT,
                    action TEXT NOT NULL,
                    confidence REAL,
                    price REAL,
                    reasons TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_ts
                ON signals(timestamp DESC)
            """)

    def insert_candles(self, symbol: str, candles: list[dict]):
        with self._cursor() as cur:
            cur.executemany(
                """INSERT OR IGNORE INTO candles
                   (symbol, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(symbol, c["timestamp"], c["open"], c["high"],
                  c["low"], c["close"], c.get("volume", 0))
                 for c in candles],
            )

    def get_candles(self, symbol: str, limit: int = 500, since_ts: float = 0) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT timestamp, open, high, low, close, volume
                   FROM candles WHERE symbol = ? AND timestamp > ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, since_ts, limit),
            )
            return [dict(r) for r in reversed(cur.fetchall())]

    def insert_trade(self, trade: dict) -> int:
        meta = trade.get("meta") or {}
        if isinstance(meta, dict):
            meta = json.dumps(meta, default=str)
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO trades
                   (timestamp, symbol, direction, signal, side,
                    entry_price, exit_price, stop_loss, take_profit,
                    size, notional_usd, pnl, hold_time_sec,
                    confidence, strategy, session, trend, atr, mode, status, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.get("timestamp", time.time()),
                    trade.get("symbol", "XAU"),
                    trade["direction"],
                    trade["signal"],
                    trade["side"],
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("stop_loss"),
                    trade.get("take_profit"),
                    trade.get("size"),
                    trade.get("notional_usd"),
                    trade.get("pnl"),
                    trade.get("hold_time_sec"),
                    trade.get("confidence"),
                    trade.get("strategy"),
                    trade.get("session"),
                    trade.get("trend"),
                    trade.get("atr"),
                    trade.get("mode", "paper"),
                    trade.get("status"),
                    meta,
                ),
            )
            return cur.lastrowid

    def insert_signal(self, signal: dict):
        reasons = signal.get("reasons")
        if isinstance(reasons, (list, dict)):
            reasons = json.dumps(reasons, default=str)
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO signals
                   (timestamp, strategy, session, direction, action,
                    confidence, price, reasons)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.get("timestamp", time.time()),
                    signal.get("strategy", "unknown"),
                    signal.get("session"),
                    signal.get("direction"),
                    signal["action"],
                    signal.get("confidence"),
                    signal.get("price"),
                    reasons,
                ),
            )

    def strategy_stats(self, strategy: str, hours: float = 24) -> dict:
        since = time.time() - hours * 3600
        with self._cursor() as cur:
            cur.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                     SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                     SUM(pnl) as total_pnl,
                     AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                     AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                     SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_win,
                     SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
                     AVG(hold_time_sec) as avg_hold
                   FROM trades
                   WHERE strategy = ? AND status = 'closed' AND timestamp > ?""",
                (strategy, since),
            )
            row = cur.fetchone()

        if not row or row["total"] == 0:
            return {"total": 0, "strategy": strategy}

        gross_loss = row["gross_loss"] or 0.001
        return {
            "strategy": strategy,
            "total": row["total"],
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "total_pnl": round(row["total_pnl"] or 0, 2),
            "win_rate": round((row["wins"] or 0) / row["total"] * 100, 1),
            "avg_win": round(row["avg_win"] or 0, 4),
            "avg_loss": round(row["avg_loss"] or 0, 4),
            "profit_factor": round((row["gross_win"] or 0) / gross_loss, 2),
            "avg_hold_sec": round(row["avg_hold"] or 0, 0),
        }

    def all_strategy_stats(self, hours: float = 24) -> list[dict]:
        since = time.time() - hours * 3600
        with self._cursor() as cur:
            cur.execute(
                "SELECT DISTINCT strategy FROM trades WHERE status = 'closed' AND timestamp > ?",
                (since,),
            )
            strategies = [r["strategy"] for r in cur.fetchall() if r["strategy"]]
        return [self.strategy_stats(s, hours) for s in strategies]

    def session_stats(self, hours: float = 168) -> list[dict]:
        """PnL by trading session."""
        since = time.time() - hours * 3600
        with self._cursor() as cur:
            cur.execute(
                """SELECT session,
                     COUNT(*) as total,
                     SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                     SUM(pnl) as total_pnl
                   FROM trades WHERE status = 'closed' AND timestamp > ?
                   GROUP BY session""",
                (since,),
            )
            return [dict(r) for r in cur.fetchall()]

    def cleanup_old_candles(self, keep_days: int = 30):
        cutoff = time.time() - keep_days * 86400
        with self._cursor() as cur:
            cur.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff,))
            if cur.rowcount:
                logger.info("Cleaned up %d old candles", cur.rowcount)

    def close(self):
        self._conn.close()
