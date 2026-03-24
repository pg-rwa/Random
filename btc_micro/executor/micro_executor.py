"""BTC micro-trade executor — paper trading on mainnet prices.

v2: ATR-based dynamic stops, per-direction loss tracking, session drawdown
    circuit breaker, trailing stop support, micro position sizing.

Tracks dual long/short positions independently. Uses dynamic % TP/SL
tuned for BTC's volatility on 1m timeframe. Integrates with the
self-learner for adaptive parameter adjustment.
"""

import json
import logging
import time
from pathlib import Path

from momentum_bot.exchange.base import ExchangeBase, OrderSide
from momentum_bot.learner.per_trade_analyzer import PerTradeAnalyzer, EntryVerdict
from btc_micro.strategy.micro_scalper import MicroSignal, MicroResult

logger = logging.getLogger(__name__)


class MicroExecutor:
    """Executes BTC micro trades with dual long/short tracking."""

    def __init__(
        self,
        exchange: ExchangeBase,
        paper_trade: bool = True,
        tp_pct: float = 0.08,
        sl_pct: float = 0.06,
        position_size_usd: float = 50.0,
        leverage: int = 5,
        max_consecutive_losses: int = 4,
        cooldown_after_losses_sec: int = 180,
        daily_loss_limit_pct: float = 2.0,
        direction_cooldown_sec: float = 60.0,
        max_trades_per_hour: int = 20,
        log_dir: str = "btc_trades",
        learner=None,
        # v2: New params
        use_atr_stops: bool = True,
        atr_tp_mult: float = 1.5,
        atr_sl_mult: float = 1.0,
        trailing_stop_activate_pct: float = 0.05,
        trailing_stop_distance_pct: float = 0.03,
        session_drawdown_limit_pct: float = 3.0,
        per_direction_loss_limit: int = 3,
        per_direction_cooldown_sec: float = 120.0,
        scale_size_on_streak: bool = True,
    ):
        self.exchange = exchange
        self.paper_trade = paper_trade
        self.tp_pct = tp_pct / 100.0
        self.sl_pct = sl_pct / 100.0
        self.position_size_usd = position_size_usd
        self._base_position_size = position_size_usd  # v2: remember original
        self.leverage = leverage
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_after_losses_sec = cooldown_after_losses_sec
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_trades_per_hour = max_trades_per_hour

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log: list[dict] = []

        # Dual positions
        self._long_position: dict | None = None
        self._short_position: dict | None = None

        # Risk tracking
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 10000.0
        self._day_start: float = time.time()

        # Direction cooldowns
        self._last_long_close: float = 0.0
        self._last_short_close: float = 0.0
        self._direction_cooldown: float = direction_cooldown_sec

        # Hourly trade rate limiter
        self._hour_trade_timestamps: list[float] = []

        # Auto-learner
        self._learner = learner
        self._confidence_min: float = 0.0

        # v3: Per-trade self-learning analyzer
        self._per_trade = PerTradeAnalyzer(
            bot_name="BTC",
            min_recent_wr=0.30,
            min_direction_wr=0.25,
            max_consecutive_sl=3,
            max_same_dir_sl=2,
            cooldown_after_block_sec=120,
        )

        # v2: ATR-based dynamic stops
        self._use_atr_stops = use_atr_stops
        self._atr_tp_mult = atr_tp_mult
        self._atr_sl_mult = atr_sl_mult

        # v2: Trailing stop
        self._trailing_activate_pct = trailing_stop_activate_pct / 100.0
        self._trailing_distance_pct = trailing_stop_distance_pct / 100.0

        # v2: Session drawdown circuit breaker
        self._session_pnl: float = 0.0
        self._session_start_equity: float = 10000.0
        self._session_drawdown_limit = session_drawdown_limit_pct

        # v2: Per-direction loss tracking
        self._long_consecutive_losses: int = 0
        self._short_consecutive_losses: int = 0
        self._per_direction_loss_limit = per_direction_loss_limit
        self._per_direction_cooldown_sec = per_direction_cooldown_sec
        self._long_cooldown_until: float = 0.0
        self._short_cooldown_until: float = 0.0

        # v2: Scale down on losing streaks
        self._scale_size_on_streak = scale_size_on_streak

        mode = "PAPER" if paper_trade else "LIVE"
        logger.info(
            "MicroExecutor v2 initialized in %s mode | TP=%.2f%% SL=%.2f%% | "
            "Size=$%.0f | Leverage=%dx | Max %d trades/hr | ATR stops=%s",
            mode, tp_pct, sl_pct, position_size_usd, leverage,
            max_trades_per_hour, use_atr_stops,
        )

    @property
    def has_long(self) -> bool:
        return self._long_position is not None

    @property
    def has_short(self) -> bool:
        return self._short_position is not None

    def set_daily_equity(self, equity: float) -> None:
        now = time.time()
        if now - self._day_start > 86400:
            self._daily_pnl = 0.0
            self._day_start = now
        self._daily_start_equity = equity
        self._session_start_equity = equity

    def _is_in_cooldown(self) -> bool:
        if time.time() < self._cooldown_until:
            remaining = int(self._cooldown_until - time.time())
            logger.info(
                "Risk cooldown: %ds remaining after %d consecutive losses",
                remaining, self._consecutive_losses,
            )
            return True
        return False

    def _is_daily_limit_hit(self) -> bool:
        if self._daily_start_equity <= 0:
            return False
        loss_pct = abs(self._daily_pnl) / self._daily_start_equity * 100
        if self._daily_pnl < 0 and loss_pct >= self.daily_loss_limit_pct:
            logger.warning(
                "Daily loss limit hit: %.1f%% >= %.1f%%", loss_pct, self.daily_loss_limit_pct
            )
            return True
        return False

    def _is_session_drawdown_hit(self) -> bool:
        """v2: Session-level drawdown circuit breaker."""
        if self._session_start_equity <= 0:
            return False
        loss_pct = abs(self._session_pnl) / self._session_start_equity * 100
        if self._session_pnl < 0 and loss_pct >= self._session_drawdown_limit:
            logger.warning(
                "Session drawdown limit hit: %.1f%% >= %.1f%%",
                loss_pct, self._session_drawdown_limit,
            )
            return True
        return False

    def _is_rate_limited(self) -> bool:
        now = time.time()
        cutoff = now - 3600
        self._hour_trade_timestamps = [t for t in self._hour_trade_timestamps if t > cutoff]
        if len(self._hour_trade_timestamps) >= self.max_trades_per_hour:
            logger.info("Rate limited: %d trades in last hour", len(self._hour_trade_timestamps))
            return True
        return False

    def _is_direction_blocked(self, direction: str) -> bool:
        """v2: Per-direction loss tracking cooldown."""
        now = time.time()
        if direction == "long":
            if self._long_consecutive_losses >= self._per_direction_loss_limit:
                if now < self._long_cooldown_until:
                    remaining = int(self._long_cooldown_until - now)
                    logger.info("Long direction blocked: %ds remaining (%d losses)",
                                remaining, self._long_consecutive_losses)
                    return True
                # Cooldown expired, reset
                self._long_consecutive_losses = 0
        else:
            if self._short_consecutive_losses >= self._per_direction_loss_limit:
                if now < self._short_cooldown_until:
                    remaining = int(self._short_cooldown_until - now)
                    logger.info("Short direction blocked: %ds remaining (%d losses)",
                                remaining, self._short_consecutive_losses)
                    return True
                self._short_consecutive_losses = 0
        return False

    def _get_effective_size(self) -> float:
        """v2: Scale position size down during losing streaks."""
        if not self._scale_size_on_streak:
            return self._base_position_size

        if self._consecutive_losses >= 3:
            # Scale to 50% after 3 losses
            return self._base_position_size * 0.5
        elif self._consecutive_losses >= 2:
            # Scale to 75% after 2 losses
            return self._base_position_size * 0.75
        return self._base_position_size

    def _compute_dynamic_stops(
        self, direction: str, price: float, atr: float
    ) -> tuple[float, float]:
        """v2: ATR-based dynamic TP/SL instead of fixed %."""
        if self._use_atr_stops and atr > 0:
            tp_distance = atr * self._atr_tp_mult
            sl_distance = atr * self._atr_sl_mult

            # Clamp to reasonable bounds (min 0.05%, max 0.35%)
            # v3: widened from 0.03%-0.20% — BTC needs more room on 1m timeframe
            min_dist = price * 0.0005
            max_dist = price * 0.0035
            tp_distance = max(min_dist, min(max_dist, tp_distance))
            sl_distance = max(min_dist, min(max_dist, sl_distance))
        else:
            tp_distance = price * self.tp_pct
            sl_distance = price * self.sl_pct

        if direction == "long":
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        return stop_loss, take_profit

    async def check_stops(self, current_price: float) -> list[dict]:
        """Check SL/TP for both positions, including trailing stop updates."""
        closed = []

        if self._long_position:
            pos = self._long_position
            # v2: Update trailing stop
            if current_price > pos.get("best_price", pos["entry_price"]):
                pos["best_price"] = current_price
                # Activate trailing stop after hitting activate threshold
                pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                if pnl_pct >= self._trailing_activate_pct:
                    new_trail_sl = current_price * (1 - self._trailing_distance_pct)
                    if new_trail_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trail_sl

            if current_price <= pos["stop_loss"]:
                closed.append(self._close_paper_position("long", current_price, "STOP_LOSS"))
            elif current_price >= pos["take_profit"]:
                closed.append(self._close_paper_position("long", current_price, "TAKE_PROFIT"))

        if self._short_position:
            pos = self._short_position
            # v2: Update trailing stop for short
            if current_price < pos.get("best_price", pos["entry_price"]):
                pos["best_price"] = current_price
                pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
                if pnl_pct >= self._trailing_activate_pct:
                    new_trail_sl = current_price * (1 + self._trailing_distance_pct)
                    if new_trail_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trail_sl

            if current_price >= pos["stop_loss"]:
                closed.append(self._close_paper_position("short", current_price, "STOP_LOSS"))
            elif current_price <= pos["take_profit"]:
                closed.append(self._close_paper_position("short", current_price, "TAKE_PROFIT"))

        return closed

    def _close_paper_position(self, direction: str, exit_price: float, reason: str) -> dict:
        """Close a paper position and record PnL."""
        pos = self._long_position if direction == "long" else self._short_position

        if direction == "long":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]

        hold_sec = int(time.time() - pos["open_time"])

        trade_record = {
            "timestamp": time.time(),
            "symbol": "BTC",
            "direction": direction,
            "signal": reason,
            "side": "close",
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "size": pos["size"],
            "pnl": pnl,
            "hold_time_sec": hold_sec,
            "mode": "paper",
            "status": "closed",
            "atr_at_entry": pos.get("atr_at_entry", 0),
        }

        # Update risk tracking
        self._daily_pnl += pnl
        self._session_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
            # v2: Per-direction loss tracking
            if direction == "long":
                self._long_consecutive_losses += 1
                if self._long_consecutive_losses >= self._per_direction_loss_limit:
                    self._long_cooldown_until = time.time() + self._per_direction_cooldown_sec
                    logger.warning(
                        "Long hit %d losses -> direction cooldown %ds",
                        self._long_consecutive_losses, self._per_direction_cooldown_sec,
                    )
            else:
                self._short_consecutive_losses += 1
                if self._short_consecutive_losses >= self._per_direction_loss_limit:
                    self._short_cooldown_until = time.time() + self._per_direction_cooldown_sec
                    logger.warning(
                        "Short hit %d losses -> direction cooldown %ds",
                        self._short_consecutive_losses, self._per_direction_cooldown_sec,
                    )

            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until = time.time() + self.cooldown_after_losses_sec
                logger.warning(
                    "Hit %d consecutive losses -> cooldown %ds",
                    self._consecutive_losses, self.cooldown_after_losses_sec,
                )
        else:
            self._consecutive_losses = 0
            # v2: Reset per-direction losses on win
            if direction == "long":
                self._long_consecutive_losses = 0
            else:
                self._short_consecutive_losses = 0

        # Clear position and set direction cooldown
        if direction == "long":
            self._long_position = None
            self._last_long_close = time.time()
        else:
            self._short_position = None
            self._last_short_close = time.time()

        logger.info(
            "[BTC] %s CLOSE_%s %.6f @ %.2f -> %.2f | PnL=$%.2f | %s | hold=%ds",
            reason, direction.upper(), pos["size"], pos["entry_price"],
            exit_price, pnl, "WIN" if pnl > 0 else "LOSS", hold_sec,
        )

        self._trade_log.append(trade_record)
        self._persist_trade(trade_record)

        # v3: Record in per-trade analyzer for real-time learning
        self._per_trade.record_trade(
            direction=direction,
            signal=reason,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            hold_time_sec=hold_sec,
            confidence=pos.get("confidence", 0.0),
            atr_at_entry=pos.get("atr_at_entry", 0),
            trend=pos.get("trend", "neutral"),
        )

        if self._learner:
            self._learner.notify_trade_closed()

        return trade_record

    async def execute_signals(
        self, signals: MicroResult, current_price: float, symbol: str = "BTC",
        atr: float = 0.0,
    ) -> list[dict]:
        """Process micro signals."""
        executed = []

        for signal in signals.signals:
            if signal == MicroSignal.HOLD:
                continue

            # Close signals
            if signal == MicroSignal.CLOSE_LONG and self.has_long:
                trade = self._close_paper_position("long", current_price, "SIGNAL_EXIT")
                executed.append(trade)
                continue

            if signal == MicroSignal.CLOSE_SHORT and self.has_short:
                trade = self._close_paper_position("short", current_price, "SIGNAL_EXIT")
                executed.append(trade)
                continue

            # Entry risk checks
            if (self._is_in_cooldown() or self._is_daily_limit_hit()
                    or self._is_session_drawdown_hit() or self._is_rate_limited()):
                continue

            if signal == MicroSignal.OPEN_LONG and not self.has_long:
                if time.time() - self._last_long_close < self._direction_cooldown:
                    logger.debug("Long direction cooldown active")
                    continue
                if self._is_direction_blocked("long"):
                    continue
                if self._learner and not self._learner.is_entry_allowed("long"):
                    logger.info("Learner blocked LONG entry")
                    continue
                if self._confidence_min > 0 and signals.confidence < self._confidence_min:
                    logger.info(
                        "Learner: confidence %.3f < gate %.3f -- skipping LONG",
                        signals.confidence, self._confidence_min,
                    )
                    continue
                # v3: Per-trade self-learning check
                verdict = self._per_trade.should_enter(
                    "long", signals.confidence, signals.trend,
                )
                if not verdict.allowed:
                    logger.info("PerTrade BLOCKED LONG: %s", verdict.reasons)
                    continue
                trade = await self._open_position(
                    "long", current_price, symbol, signals, atr, verdict,
                )
                if trade:
                    executed.append(trade)

            elif signal == MicroSignal.OPEN_SHORT and not self.has_short:
                if time.time() - self._last_short_close < self._direction_cooldown:
                    logger.debug("Short direction cooldown active")
                    continue
                if self._is_direction_blocked("short"):
                    continue
                if self._learner and not self._learner.is_entry_allowed("short"):
                    logger.info("Learner blocked SHORT entry")
                    continue
                if self._confidence_min > 0 and signals.confidence < self._confidence_min:
                    logger.info(
                        "Learner: confidence %.3f < gate %.3f -- skipping SHORT",
                        signals.confidence, self._confidence_min,
                    )
                    continue
                # v3: Per-trade self-learning check
                verdict = self._per_trade.should_enter(
                    "short", signals.confidence, signals.trend,
                )
                if not verdict.allowed:
                    logger.info("PerTrade BLOCKED SHORT: %s", verdict.reasons)
                    continue
                trade = await self._open_position(
                    "short", current_price, symbol, signals, atr, verdict,
                )
                if trade:
                    executed.append(trade)

        return executed

    async def _open_position(
        self, direction: str, price: float, symbol: str, signals: MicroResult,
        atr: float = 0.0, verdict: EntryVerdict | None = None,
    ) -> dict | None:
        """Open a new micro position."""
        # v2: Dynamic position sizing on streak
        effective_size_usd = self._get_effective_size()
        # v3: Apply per-trade sizing adjustment
        if verdict and verdict.size_multiplier != 1.0:
            effective_size_usd *= verdict.size_multiplier
            logger.info("PerTrade sizing: $%.0f → $%.0f (×%.2f)",
                        self._get_effective_size(), effective_size_usd,
                        verdict.size_multiplier)
        size = (effective_size_usd * self.leverage) / price

        # v2: Dynamic ATR-based stops
        stop_loss, take_profit = self._compute_dynamic_stops(direction, price, atr)
        # v3: Apply per-trade stop adjustments
        if verdict and (verdict.tp_multiplier != 1.0 or verdict.sl_multiplier != 1.0):
            if direction == "long":
                sl_dist = price - stop_loss
                tp_dist = take_profit - price
                stop_loss = price - sl_dist * verdict.sl_multiplier
                take_profit = price + tp_dist * verdict.tp_multiplier
            else:
                sl_dist = stop_loss - price
                tp_dist = price - take_profit
                stop_loss = price + sl_dist * verdict.sl_multiplier
                take_profit = price - tp_dist * verdict.tp_multiplier

        position = {
            "entry_price": price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "open_time": time.time(),
            "best_price": price,  # v2: for trailing stop
            "atr_at_entry": atr,
            "confidence": signals.confidence,  # v3: for per-trade learning
            "trend": signals.trend,            # v3: for per-trade learning
        }

        trade_record = {
            "timestamp": time.time(),
            "symbol": symbol,
            "direction": direction,
            "signal": f"OPEN_{direction.upper()}",
            "side": direction,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "size": size,
            "notional_usd": effective_size_usd * self.leverage,
            "confidence": signals.confidence,
            "long_score": signals.long_score,
            "short_score": signals.short_score,
            "trend": signals.trend,
            "reasons": signals.reasons,
            "atr": atr,
            "mode": "paper" if self.paper_trade else "live",
            "status": "filled",
        }

        if self.paper_trade:
            if direction == "long":
                self._long_position = position
            else:
                self._short_position = position

            logger.info(
                "[BTC] OPEN_%s %.6f @ %.2f | SL=%.2f TP=%.2f | $%.0f notional | conf=%.2f | streak=%d",
                direction.upper(), size, price, stop_loss, take_profit,
                effective_size_usd * self.leverage, signals.confidence,
                self._consecutive_losses,
            )
        else:
            order_side = OrderSide.BUY if direction == "long" else OrderSide.SELL
            result = await self.exchange.place_order(
                symbol=symbol, side=order_side, size=size,
            )
            trade_record["order_result"] = {
                "success": result.success,
                "order_id": result.order_id,
                "filled_price": result.filled_price,
                "error": result.error,
            }
            if result.success:
                actual_price = result.filled_price or price
                position["entry_price"] = actual_price
                stop_loss, take_profit = self._compute_dynamic_stops(
                    direction, actual_price, atr
                )
                position["stop_loss"] = stop_loss
                position["take_profit"] = take_profit
                if direction == "long":
                    self._long_position = position
                else:
                    self._short_position = position
                trade_record["status"] = "filled"
                logger.info("[LIVE] OPEN_%s BTC %.6f @ %.2f", direction.upper(), size, actual_price)
            else:
                trade_record["status"] = "failed"
                logger.error("[LIVE] OPEN_%s FAILED: %s", direction.upper(), result.error)
                return None

        self._trade_log.append(trade_record)
        self._persist_trade(trade_record)
        self._hour_trade_timestamps.append(time.time())
        return trade_record

    def _persist_trade(self, trade: dict) -> None:
        date_str = time.strftime("%Y-%m-%d")
        log_file = self._log_dir / f"btc_micro_{date_str}.json"

        existing = []
        if log_file.exists():
            with open(log_file, "r") as f:
                existing = json.load(f)

        existing.append(trade)
        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def apply_learner_overrides(self, overrides: dict) -> None:
        """Apply parameter adjustments from the learner."""
        if "executor.tp_pct" in overrides:
            self.tp_pct = overrides["executor.tp_pct"] / 100.0
            logger.info("Learner override: tp_pct -> %.3f%%", overrides["executor.tp_pct"])
        if "executor.sl_pct" in overrides:
            self.sl_pct = overrides["executor.sl_pct"] / 100.0
            logger.info("Learner override: sl_pct -> %.3f%%", overrides["executor.sl_pct"])
        if "learner.confidence_min" in overrides:
            self._confidence_min = overrides["learner.confidence_min"]
            logger.info("Learner override: confidence_min -> %.3f", self._confidence_min)
        if "executor.direction_cooldown_sec.long" in overrides:
            self._direction_cooldown = max(
                self._direction_cooldown,
                overrides["executor.direction_cooldown_sec.long"],
            )
        if "executor.direction_cooldown_sec.short" in overrides:
            self._direction_cooldown = max(
                self._direction_cooldown,
                overrides["executor.direction_cooldown_sec.short"],
            )
        # v2: ATR multiplier overrides
        if "executor.atr_tp_mult" in overrides:
            self._atr_tp_mult = overrides["executor.atr_tp_mult"]
            logger.info("Learner override: atr_tp_mult -> %.2f", self._atr_tp_mult)
        if "executor.atr_sl_mult" in overrides:
            self._atr_sl_mult = overrides["executor.atr_sl_mult"]
            logger.info("Learner override: atr_sl_mult -> %.2f", self._atr_sl_mult)
        # v2: Kill switch — learner can pause all trading
        if overrides.get("learner.kill_switch"):
            self._cooldown_until = time.time() + 600  # 10 min pause
            logger.warning("KILL SWITCH activated by learner — pausing 10 minutes")

    def get_position_status(self) -> dict:
        return {
            "long": {
                "active": self.has_long,
                "entry": self._long_position["entry_price"] if self.has_long else None,
                "sl": self._long_position["stop_loss"] if self.has_long else None,
                "tp": self._long_position["take_profit"] if self.has_long else None,
            },
            "short": {
                "active": self.has_short,
                "entry": self._short_position["entry_price"] if self.has_short else None,
                "sl": self._short_position["stop_loss"] if self.has_short else None,
                "tp": self._short_position["take_profit"] if self.has_short else None,
            },
        }

    def get_trade_summary(self) -> dict:
        if not self._trade_log:
            return {"total_trades": 0}

        closed = [t for t in self._trade_log if "pnl" in t]
        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        hold_times = [t.get("hold_time_sec", 0) for t in closed if t.get("hold_time_sec")]

        return {
            "total_signals": len(self._trade_log),
            "closed_trades": len(closed),
            "total_pnl": round(sum(pnls), 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "session_pnl": round(self._session_pnl, 2),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "avg_hold_sec": round(sum(hold_times) / len(hold_times), 0) if hold_times else 0,
            "consecutive_losses": self._consecutive_losses,
            "long_streak_losses": self._long_consecutive_losses,
            "short_streak_losses": self._short_consecutive_losses,
            "cooldown_active": time.time() < self._cooldown_until,
            "trades_this_hour": len(self._hour_trade_timestamps),
            "position_size_usd": self._get_effective_size(),
            "per_trade_stats": self._per_trade.get_stats(),
        }
