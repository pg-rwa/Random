"""SQLite database for BTC micro-trading bot.

Stores candles, trades, signals, and strategy performance for analysis
and backtesting. Replaces JSON file logging with structured queries.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path("btc_trades/btc_micro.db")


class TradeDB:
    """SQLite-backed storage for candles, trades, and strategy signals."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("TradeDB initialized at %s", self._db_path)

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
            # 1-minute candle history
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

            # Trade log (replaces JSON files)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL DEFAULT 'BTC',
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

            # Strategy signals (every evaluation, not just trades)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT,
                    action TEXT NOT NULL,
                    confidence REAL,
                    price REAL,
                    indicators TEXT,
                    reasons TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_ts
                ON signals(timestamp DESC)
            """)

            # Strategy performance summary (updated periodically)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS strategy_perf (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    avg_hold_sec REAL,
                    period_hours REAL DEFAULT 24
                )
            """)

    # ------------------------------------------------------------------
    # Candle operations
    # ------------------------------------------------------------------

    def insert_candles(self, symbol: str, candles: list[dict]):
        """Bulk insert candles, ignoring duplicates."""
        with self._cursor() as cur:
            cur.executemany(
                """INSERT OR IGNORE INTO candles
                   (symbol, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(symbol, c["timestamp"], c["open"], c["high"],
                  c["low"], c["close"], c.get("volume", 0))
                 for c in candles],
            )

    def get_candles(
        self, symbol: str, limit: int = 500, since_ts: float = 0
    ) -> list[dict]:
        """Get recent candles ordered by time ascending."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT timestamp, open, high, low, close, volume
                   FROM candles
                   WHERE symbol = ? AND timestamp > ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, since_ts, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    def candle_count(self, symbol: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,)
            )
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Trade operations
    # ------------------------------------------------------------------

    def insert_trade(self, trade: dict) -> int:
        """Insert a trade record. Returns the row id."""
        meta = trade.get("meta") or {}
        if isinstance(meta, dict):
            meta = json.dumps(meta, default=str)
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO trades
                   (timestamp, symbol, direction, signal, side,
                    entry_price, exit_price, stop_loss, take_profit,
                    size, notional_usd, pnl, hold_time_sec,
                    confidence, strategy, trend, atr, mode, status, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.get("timestamp", time.time()),
                    trade.get("symbol", "BTC"),
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
                    trade.get("trend"),
                    trade.get("atr"),
                    trade.get("mode", "paper"),
                    trade.get("status"),
                    meta,
                ),
            )
            return cur.lastrowid

    def get_recent_trades(
        self, limit: int = 100, strategy: str | None = None,
        status: str | None = None, hours: float = 24,
    ) -> list[dict]:
        """Get recent trades with optional filters."""
        since = time.time() - hours * 3600
        query = "SELECT * FROM trades WHERE timestamp > ?"
        params: list = [since]

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_closed_trades(
        self, hours: float = 24, strategy: str | None = None
    ) -> list[dict]:
        """Get closed trades with PnL."""
        return self.get_recent_trades(
            limit=500, strategy=strategy, status="closed", hours=hours
        )

    # ------------------------------------------------------------------
    # Signal log operations
    # ------------------------------------------------------------------

    def insert_signal(self, signal: dict):
        """Log a strategy evaluation."""
        indicators = signal.get("indicators")
        if isinstance(indicators, dict):
            indicators = json.dumps(indicators, default=str)
        reasons = signal.get("reasons")
        if isinstance(reasons, (list, dict)):
            reasons = json.dumps(reasons, default=str)
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO signals
                   (timestamp, strategy, direction, action,
                    confidence, price, indicators, reasons)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.get("timestamp", time.time()),
                    signal.get("strategy", "unknown"),
                    signal.get("direction"),
                    signal["action"],
                    signal.get("confidence"),
                    signal.get("price"),
                    indicators,
                    reasons,
                ),
            )

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def strategy_stats(
        self, strategy: str, hours: float = 24
    ) -> dict:
        """Compute win rate, PnL, profit factor for a strategy."""
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
                   WHERE strategy = ? AND status = 'closed'
                     AND timestamp > ?""",
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
        """Stats for all strategies in the given period."""
        since = time.time() - hours * 3600
        with self._cursor() as cur:
            cur.execute(
                """SELECT DISTINCT strategy FROM trades
                   WHERE status = 'closed' AND timestamp > ?""",
                (since,),
            )
            strategies = [r["strategy"] for r in cur.fetchall() if r["strategy"]]
        return [self.strategy_stats(s, hours) for s in strategies]

    def best_strategy(self, hours: float = 24, min_trades: int = 5) -> str | None:
        """Return the strategy name with highest profit factor."""
        stats = self.all_strategy_stats(hours)
        valid = [s for s in stats if s["total"] >= min_trades]
        if not valid:
            return None
        best = max(valid, key=lambda s: s["profit_factor"])
        return best["strategy"] if best["profit_factor"] > 1.0 else None

    def direction_stats(
        self, hours: float = 24
    ) -> dict:
        """Win rate by direction."""
        since = time.time() - hours * 3600
        result = {}
        with self._cursor() as cur:
            for d in ("long", "short"):
                cur.execute(
                    """SELECT
                         COUNT(*) as total,
                         SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                         SUM(pnl) as total_pnl
                       FROM trades
                       WHERE direction = ? AND status = 'closed'
                         AND timestamp > ?""",
                    (d, since),
                )
                row = cur.fetchone()
                result[d] = {
                    "total": row["total"] or 0,
                    "wins": row["wins"] or 0,
                    "win_rate": round((row["wins"] or 0) / row["total"] * 100, 1) if row["total"] else 0,
                    "pnl": round(row["total_pnl"] or 0, 2),
                }
        return result

    def hourly_stats(self, hours: float = 168) -> list[dict]:
        """PnL and win rate by hour of day (for finding good/bad hours)."""
        since = time.time() - hours * 3600
        with self._cursor() as cur:
            cur.execute(
                """SELECT
                     CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) as hour,
                     COUNT(*) as total,
                     SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                     SUM(pnl) as total_pnl
                   FROM trades
                   WHERE status = 'closed' AND timestamp > ?
                   GROUP BY hour
                   ORDER BY hour""",
                (since,),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_old_candles(self, keep_days: int = 30):
        """Delete candles older than N days."""
        cutoff = time.time() - keep_days * 86400
        with self._cursor() as cur:
            cur.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
        if deleted:
            logger.info("Cleaned up %d old candles", deleted)

    def close(self):
        self._conn.close()
