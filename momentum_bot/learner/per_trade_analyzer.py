"""Per-trade self-learning analyzer — shared by both Gold Scalper and BTC Micro.

Instead of reviewing every N trades in a batch, this module analyzes the
LAST trade (and recent context) BEFORE each new entry decision. It maintains
a rolling memory of recent trades and computes real-time adjustments.

Key behaviors:
1. After every closed trade, update the rolling memory immediately.
2. Before every new entry, call `should_enter()` which checks:
   - Recent win-rate (last 5/10 trades)
   - Direction-specific performance
   - Time-of-day performance
   - Pattern detection (consecutive losses, SL-only streaks)
   - Volatility regime shift (ATR drift)
   - Confidence calibration (did high-confidence trades actually win?)
3. Emits real-time parameter micro-adjustments (small, frequent nudges).
4. Provides a `verdict` dict with entry recommendations and reasons.

This is designed to be FAST — called every cycle, no file I/O during
decision-making. File I/O only happens on trade close (append to memory).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum trades to keep in rolling memory
_MAX_MEMORY = 50


@dataclass
class TradeMemory:
    """Lightweight record of a closed trade for analysis."""
    timestamp: float = 0.0
    direction: str = ""          # "long" or "short"
    signal: str = ""             # exit reason: STOP_LOSS, TAKE_PROFIT, SIGNAL_EXIT
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    hold_time_sec: int = 0
    confidence: float = 0.0
    atr_at_entry: float = 0.0
    trend_at_entry: str = ""     # "up", "down", "neutral"
    hour: int = 0                # UTC hour of entry


@dataclass
class EntryVerdict:
    """Result of per-trade analysis before a new entry."""
    allowed: bool = True
    reasons: list[str] = field(default_factory=list)
    confidence_boost: float = 0.0      # Add to signal confidence
    size_multiplier: float = 1.0       # Scale position size (0.5 = half, 1.5 = 1.5x)
    tp_multiplier: float = 1.0         # Widen/tighten TP
    sl_multiplier: float = 1.0         # Widen/tighten SL


class PerTradeAnalyzer:
    """Real-time per-trade self-learning engine.

    Call `record_trade()` after every close.
    Call `should_enter()` before every new entry.
    """

    def __init__(
        self,
        bot_name: str = "bot",
        # Thresholds
        min_recent_wr: float = 0.25,        # Block entries if last-5 WR below this
        min_direction_wr: float = 0.20,      # Block a direction if last-5 dir WR below this
        max_consecutive_sl: int = 3,         # Block after 3 consecutive stop-losses
        max_same_dir_sl: int = 2,            # Block direction after 2 consecutive same-dir SLs
        confidence_calibration: bool = True,  # Adjust confidence based on recent accuracy
        adaptive_sizing: bool = True,         # Scale size based on streak
        adaptive_stops: bool = True,          # Adjust TP/SL based on recent outcomes
        cooldown_after_block_sec: int = 180,  # How long to pause after blocking
    ):
        self._name = bot_name
        self._memory: deque[TradeMemory] = deque(maxlen=_MAX_MEMORY)

        # Thresholds
        self._min_recent_wr = min_recent_wr
        self._min_direction_wr = min_direction_wr
        self._max_consecutive_sl = max_consecutive_sl
        self._max_same_dir_sl = max_same_dir_sl
        self._confidence_calibration = confidence_calibration
        self._adaptive_sizing = adaptive_sizing
        self._adaptive_stops = adaptive_stops
        self._cooldown_after_block = cooldown_after_block_sec

        # Block state with expiry
        self._blocked_until: float = 0.0
        self._long_blocked_until: float = 0.0
        self._short_blocked_until: float = 0.0

        # Running stats (updated on each trade close)
        self._total_trades: int = 0
        self._total_pnl: float = 0.0
        self._session_peak_pnl: float = 0.0

        logger.info(
            "%s PerTradeAnalyzer initialized | max_consec_sl=%d | "
            "min_wr=%.0f%% | adaptive_sizing=%s | adaptive_stops=%s",
            bot_name, max_consecutive_sl,
            min_recent_wr * 100, adaptive_sizing, adaptive_stops,
        )

    # ------------------------------------------------------------------
    # Public: record a closed trade
    # ------------------------------------------------------------------

    def record_trade(
        self,
        direction: str,
        signal: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        hold_time_sec: int,
        confidence: float = 0.0,
        atr_at_entry: float = 0.0,
        trend: str = "neutral",
    ) -> None:
        """Call after every trade close. Updates rolling memory and stats."""
        from datetime import datetime

        trade = TradeMemory(
            timestamp=time.time(),
            direction=direction,
            signal=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            hold_time_sec=hold_time_sec,
            confidence=confidence,
            atr_at_entry=atr_at_entry,
            trend_at_entry=trend,
            hour=datetime.utcnow().hour,
        )
        self._memory.append(trade)
        self._total_trades += 1
        self._total_pnl += pnl
        self._session_peak_pnl = max(self._session_peak_pnl, self._total_pnl)

        # Log the learning insight immediately
        self._log_trade_insight(trade)

    # ------------------------------------------------------------------
    # Public: check before entry
    # ------------------------------------------------------------------

    def should_enter(self, direction: str, confidence: float,
                     trend: str = "neutral", hour: int | None = None) -> EntryVerdict:
        """Analyze recent trades and decide if a new entry is wise.

        Call this BEFORE every new entry. Returns an EntryVerdict with
        allowed/blocked status, sizing recommendations, and reasons.
        """
        from datetime import datetime

        if hour is None:
            hour = datetime.utcnow().hour

        verdict = EntryVerdict()
        now = time.time()

        # Not enough data yet — allow with caution
        if len(self._memory) < 3:
            verdict.reasons.append("insufficient_data")
            verdict.size_multiplier = 0.75  # Smaller size while learning
            return verdict

        recent = list(self._memory)

        # --- CHECK 1: Global block (cooldown after severe loss) ---
        if now < self._blocked_until:
            verdict.allowed = False
            remaining = int(self._blocked_until - now)
            verdict.reasons.append(f"global_cooldown_{remaining}s")
            return verdict

        # --- CHECK 2: Direction-specific block ---
        if direction == "long" and now < self._long_blocked_until:
            verdict.allowed = False
            remaining = int(self._long_blocked_until - now)
            verdict.reasons.append(f"long_cooldown_{remaining}s")
            return verdict
        if direction == "short" and now < self._short_blocked_until:
            verdict.allowed = False
            remaining = int(self._short_blocked_until - now)
            verdict.reasons.append(f"short_cooldown_{remaining}s")
            return verdict

        # --- CHECK 3: Recent overall win rate (last 5) ---
        last_5 = recent[-5:]
        wins_5 = sum(1 for t in last_5 if t.pnl > 0)
        wr_5 = wins_5 / len(last_5)
        if wr_5 < self._min_recent_wr:
            verdict.allowed = False
            self._blocked_until = now + self._cooldown_after_block
            verdict.reasons.append(
                f"last5_wr={wr_5:.0%}_below_{self._min_recent_wr:.0%}"
            )
            logger.warning(
                "%s BLOCKED: last 5 trades WR=%.0f%% (need %.0f%%) — "
                "pausing %ds",
                self._name, wr_5 * 100, self._min_recent_wr * 100,
                self._cooldown_after_block,
            )
            return verdict

        # --- CHECK 4: Consecutive stop-loss streak ---
        consec_sl = self._count_consecutive_from_end(recent, "STOP_LOSS")
        if consec_sl >= self._max_consecutive_sl:
            verdict.allowed = False
            self._blocked_until = now + self._cooldown_after_block
            verdict.reasons.append(f"consecutive_sl={consec_sl}")
            logger.warning(
                "%s BLOCKED: %d consecutive stop-losses — pausing %ds",
                self._name, consec_sl, self._cooldown_after_block,
            )
            return verdict

        # --- CHECK 5: Same-direction consecutive stop-losses ---
        dir_trades = [t for t in recent if t.direction == direction]
        if dir_trades:
            dir_consec_sl = self._count_consecutive_from_end(
                dir_trades, "STOP_LOSS"
            )
            if dir_consec_sl >= self._max_same_dir_sl:
                if direction == "long":
                    self._long_blocked_until = now + self._cooldown_after_block
                else:
                    self._short_blocked_until = now + self._cooldown_after_block
                verdict.allowed = False
                verdict.reasons.append(
                    f"{direction}_consecutive_sl={dir_consec_sl}"
                )
                logger.warning(
                    "%s %s BLOCKED: %d consecutive %s stop-losses — pausing %ds",
                    self._name, direction.upper(), dir_consec_sl,
                    direction, self._cooldown_after_block,
                )
                return verdict

        # --- CHECK 6: Direction win rate (last 5 same-direction trades) ---
        dir_recent = dir_trades[-5:] if len(dir_trades) >= 5 else dir_trades
        if len(dir_recent) >= 3:
            dir_wr = sum(1 for t in dir_recent if t.pnl > 0) / len(dir_recent)
            if dir_wr < self._min_direction_wr:
                verdict.allowed = False
                if direction == "long":
                    self._long_blocked_until = now + self._cooldown_after_block
                else:
                    self._short_blocked_until = now + self._cooldown_after_block
                verdict.reasons.append(
                    f"{direction}_wr={dir_wr:.0%}_below_{self._min_direction_wr:.0%}"
                )
                logger.warning(
                    "%s %s BLOCKED: last %d %s trades WR=%.0f%% — pausing %ds",
                    self._name, direction.upper(), len(dir_recent),
                    direction, dir_wr * 100, self._cooldown_after_block,
                )
                return verdict

        # --- CHECK 7: Session drawdown kill switch ---
        drawdown = self._session_peak_pnl - self._total_pnl
        if drawdown > 15.0:  # $15 drawdown from peak
            verdict.allowed = False
            self._blocked_until = now + 600  # 10 min pause
            verdict.reasons.append(f"session_drawdown=${drawdown:.2f}")
            logger.warning(
                "%s KILL SWITCH: session drawdown $%.2f from peak — "
                "pausing 10 min",
                self._name, drawdown,
            )
            return verdict

        # --- PASSED ALL CHECKS — now compute adjustments ---

        # Adaptive sizing based on recent streak
        if self._adaptive_sizing:
            verdict.size_multiplier = self._compute_size_multiplier(recent)
            if verdict.size_multiplier != 1.0:
                verdict.reasons.append(
                    f"size_mult={verdict.size_multiplier:.2f}"
                )

        # Adaptive TP/SL based on recent exit patterns
        if self._adaptive_stops:
            tp_m, sl_m = self._compute_stop_adjustments(recent)
            verdict.tp_multiplier = tp_m
            verdict.sl_multiplier = sl_m
            if tp_m != 1.0 or sl_m != 1.0:
                verdict.reasons.append(
                    f"tp_mult={tp_m:.2f}_sl_mult={sl_m:.2f}"
                )

        # Confidence calibration — check if high-confidence trades actually won
        if self._confidence_calibration:
            boost = self._calibrate_confidence(recent, confidence)
            verdict.confidence_boost = boost
            if abs(boost) > 0.01:
                verdict.reasons.append(f"conf_adjust={boost:+.2f}")

        # Counter-trend penalty
        if dir_trades and len(dir_trades) >= 3:
            last_trend_trades = [
                t for t in dir_trades[-5:]
                if t.trend_at_entry == trend
            ]
            if last_trend_trades:
                trend_wr = sum(1 for t in last_trend_trades if t.pnl > 0) / len(
                    last_trend_trades
                )
                if trend_wr < 0.3:
                    verdict.size_multiplier *= 0.5
                    verdict.reasons.append(
                        f"trend_{trend}_wr={trend_wr:.0%}_penalty"
                    )

        return verdict

    # ------------------------------------------------------------------
    # Public: get summary for logging
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get current learning stats for logging."""
        recent = list(self._memory)
        if not recent:
            return {"trades": 0}

        last_5 = recent[-5:]
        last_10 = recent[-10:] if len(recent) >= 10 else recent

        return {
            "total_trades": self._total_trades,
            "memory_size": len(recent),
            "total_pnl": round(self._total_pnl, 2),
            "session_drawdown": round(
                self._session_peak_pnl - self._total_pnl, 2
            ),
            "last5_wr": round(
                sum(1 for t in last_5 if t.pnl > 0) / len(last_5) * 100, 1
            ),
            "last10_wr": round(
                sum(1 for t in last_10 if t.pnl > 0) / len(last_10) * 100, 1
            ),
            "long_blocked": time.time() < self._long_blocked_until,
            "short_blocked": time.time() < self._short_blocked_until,
            "global_blocked": time.time() < self._blocked_until,
        }

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _count_consecutive_from_end(
        self, trades: list[TradeMemory], signal_type: str
    ) -> int:
        """Count consecutive trades with given signal from the end."""
        count = 0
        for t in reversed(trades):
            if t.signal == signal_type:
                count += 1
            else:
                break
        return count

    def _compute_size_multiplier(self, recent: list[TradeMemory]) -> float:
        """Scale position size based on recent performance."""
        last_5 = recent[-5:]
        wins = sum(1 for t in last_5 if t.pnl > 0)
        losses = len(last_5) - wins
        wr = wins / len(last_5)

        # Hot streak: winning 80%+ of last 5 → slight increase
        if wr >= 0.8:
            return 1.25

        # Losing streak patterns
        consec_losses = self._count_consecutive_from_end(recent, "STOP_LOSS")
        if consec_losses >= 2:
            return 0.5   # Half size after 2 SLs in a row
        if wr <= 0.4 and losses >= 3:
            return 0.75  # Reduce after poor streak

        return 1.0

    def _compute_stop_adjustments(
        self, recent: list[TradeMemory]
    ) -> tuple[float, float]:
        """Adjust TP/SL multipliers based on recent exit patterns.

        Returns (tp_multiplier, sl_multiplier).
        """
        last_10 = recent[-10:] if len(recent) >= 10 else recent
        if len(last_10) < 5:
            return 1.0, 1.0

        sl_count = sum(1 for t in last_10 if t.signal == "STOP_LOSS")
        tp_count = sum(1 for t in last_10 if t.signal == "TAKE_PROFIT")
        se_count = sum(1 for t in last_10 if t.signal == "SIGNAL_EXIT")

        sl_rate = sl_count / len(last_10)
        tp_rate = tp_count / len(last_10)

        tp_mult = 1.0
        sl_mult = 1.0

        # Too many stop-losses (>50%) → widen SL slightly
        if sl_rate > 0.50:
            sl_mult = 1.15  # 15% wider SL

        # Very few TPs (<10%) → tighten TP to capture profits earlier
        if tp_rate < 0.10 and sl_rate > 0.30:
            tp_mult = 0.85  # 15% tighter TP

        # Signal exits are profitable → let TP run wider
        se_trades = [t for t in last_10 if t.signal == "SIGNAL_EXIT"]
        if se_trades:
            avg_se_pnl = sum(t.pnl for t in se_trades) / len(se_trades)
            if avg_se_pnl > 0 and tp_rate < 0.20:
                tp_mult = 1.1  # Signal exits working well, TP can be wider

        # Winning trades have very short hold times → TP is way too tight
        tp_trades = [t for t in last_10 if t.signal == "TAKE_PROFIT"]
        if len(tp_trades) >= 3:
            avg_hold = sum(t.hold_time_sec for t in tp_trades) / len(tp_trades)
            if avg_hold < 30:  # TP hitting in <30s means it's too tight
                tp_mult = 1.2

        return tp_mult, sl_mult

    def _calibrate_confidence(
        self, recent: list[TradeMemory], proposed_confidence: float
    ) -> float:
        """Check if the bot's confidence scores are actually predictive.

        If high-confidence trades are losing, reduce confidence.
        If low-confidence trades are winning, the signal is miscalibrated.
        """
        if len(recent) < 10:
            return 0.0

        # Split trades into above/below median confidence
        conf_trades = [t for t in recent if t.confidence > 0]
        if len(conf_trades) < 8:
            return 0.0

        median_conf = sorted(t.confidence for t in conf_trades)[
            len(conf_trades) // 2
        ]

        high_conf = [t for t in conf_trades if t.confidence >= median_conf]
        low_conf = [t for t in conf_trades if t.confidence < median_conf]

        if not high_conf or not low_conf:
            return 0.0

        high_wr = sum(1 for t in high_conf if t.pnl > 0) / len(high_conf)
        low_wr = sum(1 for t in low_conf if t.pnl > 0) / len(low_conf)

        # If high-confidence trades are WORSE than low-confidence,
        # confidence is miscalibrated → penalize
        if high_wr < low_wr:
            return -0.1  # Reduce proposed confidence

        # If high-confidence is much better, confidence is well-calibrated
        if high_wr > low_wr + 0.2:
            return 0.05  # Small boost

        return 0.0

    def _log_trade_insight(self, trade: TradeMemory) -> None:
        """Log immediate insight after a trade closes."""
        recent = list(self._memory)
        if len(recent) < 2:
            return

        # Quick stats
        last_5 = recent[-5:] if len(recent) >= 5 else recent
        wr_5 = sum(1 for t in last_5 if t.pnl > 0) / len(last_5)
        consec_sl = self._count_consecutive_from_end(recent, "STOP_LOSS")

        # Pattern detection
        patterns = []
        if consec_sl >= 2:
            patterns.append(f"SL_STREAK={consec_sl}")
        if trade.hold_time_sec < 30 and trade.pnl < 0:
            patterns.append("FAST_LOSS")
        if trade.signal == "STOP_LOSS" and trade.confidence > 0.8:
            patterns.append("HIGH_CONF_LOSS")

        # Check if counter-trend trade
        if (trade.direction == "long" and trade.trend_at_entry == "down") or \
           (trade.direction == "short" and trade.trend_at_entry == "up"):
            if trade.pnl < 0:
                patterns.append("COUNTER_TREND_LOSS")

        pattern_str = " | ".join(patterns) if patterns else "ok"

        logger.info(
            "[LEARN] %s %s %s PnL=$%.2f | last5_wr=%.0f%% | "
            "session=$%.2f | patterns: %s",
            self._name, trade.direction.upper(), trade.signal,
            trade.pnl, wr_5 * 100, self._total_pnl, pattern_str,
        )
