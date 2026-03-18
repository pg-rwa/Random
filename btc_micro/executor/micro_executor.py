"""BTC micro-trade executor — paper trading on mainnet prices.

Tracks dual long/short positions independently. Uses fixed % TP/SL
tuned for BTC's volatility on 1m timeframe. Integrates with the
self-learner for adaptive parameter adjustment.
"""

import json
import logging
import time
from pathlib import Path

from momentum_bot.exchange.base import ExchangeBase, OrderSide
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
    ):
        self.exchange = exchange
        self.paper_trade = paper_trade
        self.tp_pct = tp_pct / 100.0
        self.sl_pct = sl_pct / 100.0
        self.position_size_usd = position_size_usd
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

        mode = "PAPER" if paper_trade else "LIVE"
        logger.info(
            "MicroExecutor initialized in %s mode | TP=%.2f%% SL=%.2f%% | "
            "Size=$%.0f | Leverage=%dx | Max %d trades/hr",
            mode, tp_pct, sl_pct, position_size_usd, leverage, max_trades_per_hour,
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

    def _is_rate_limited(self) -> bool:
        """Check if we've exceeded max trades per hour."""
        now = time.time()
        cutoff = now - 3600
        self._hour_trade_timestamps = [t for t in self._hour_trade_timestamps if t > cutoff]
        if len(self._hour_trade_timestamps) >= self.max_trades_per_hour:
            logger.info("Rate limited: %d trades in last hour", len(self._hour_trade_timestamps))
            return True
        return False

    async def check_stops(self, current_price: float) -> list[dict]:
        """Check SL/TP for both positions."""
        closed = []

        if self._long_position:
            pos = self._long_position
            if current_price <= pos["stop_loss"]:
                closed.append(self._close_paper_position("long", current_price, "STOP_LOSS"))
            elif current_price >= pos["take_profit"]:
                closed.append(self._close_paper_position("long", current_price, "TAKE_PROFIT"))

        if self._short_position:
            pos = self._short_position
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
        }

        # Update risk tracking
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until = time.time() + self.cooldown_after_losses_sec
                logger.warning(
                    "Hit %d consecutive losses -> cooldown %ds",
                    self._consecutive_losses, self.cooldown_after_losses_sec,
                )
        else:
            self._consecutive_losses = 0

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

        if self._learner:
            self._learner.notify_trade_closed()

        return trade_record

    async def execute_signals(
        self, signals: MicroResult, current_price: float, symbol: str = "BTC"
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
            if self._is_in_cooldown() or self._is_daily_limit_hit() or self._is_rate_limited():
                continue

            if signal == MicroSignal.OPEN_LONG and not self.has_long:
                if time.time() - self._last_long_close < self._direction_cooldown:
                    logger.debug("Long direction cooldown active")
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
                trade = await self._open_position("long", current_price, symbol, signals)
                if trade:
                    executed.append(trade)

            elif signal == MicroSignal.OPEN_SHORT and not self.has_short:
                if time.time() - self._last_short_close < self._direction_cooldown:
                    logger.debug("Short direction cooldown active")
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
                trade = await self._open_position("short", current_price, symbol, signals)
                if trade:
                    executed.append(trade)

        return executed

    async def _open_position(
        self, direction: str, price: float, symbol: str, signals: MicroResult
    ) -> dict | None:
        """Open a new micro position."""
        size = (self.position_size_usd * self.leverage) / price

        if direction == "long":
            stop_loss = price * (1 - self.sl_pct)
            take_profit = price * (1 + self.tp_pct)
        else:
            stop_loss = price * (1 + self.sl_pct)
            take_profit = price * (1 - self.tp_pct)

        position = {
            "entry_price": price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "open_time": time.time(),
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
            "notional_usd": self.position_size_usd * self.leverage,
            "confidence": signals.confidence,
            "long_score": signals.long_score,
            "short_score": signals.short_score,
            "reasons": signals.reasons,
            "mode": "paper" if self.paper_trade else "live",
            "status": "filled",
        }

        if self.paper_trade:
            if direction == "long":
                self._long_position = position
            else:
                self._short_position = position

            logger.info(
                "[BTC] OPEN_%s %.6f @ %.2f | SL=%.2f TP=%.2f | $%.0f notional | conf=%.2f",
                direction.upper(), size, price, stop_loss, take_profit,
                self.position_size_usd * self.leverage, signals.confidence,
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
                if direction == "long":
                    position["stop_loss"] = actual_price * (1 - self.sl_pct)
                    position["take_profit"] = actual_price * (1 + self.tp_pct)
                    self._long_position = position
                else:
                    position["stop_loss"] = actual_price * (1 + self.sl_pct)
                    position["take_profit"] = actual_price * (1 - self.tp_pct)
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
            # Only update if it's the worse direction
            self._direction_cooldown = max(
                self._direction_cooldown,
                overrides["executor.direction_cooldown_sec.long"],
            )
        if "executor.direction_cooldown_sec.short" in overrides:
            self._direction_cooldown = max(
                self._direction_cooldown,
                overrides["executor.direction_cooldown_sec.short"],
            )

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
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "avg_hold_sec": round(sum(hold_times) / len(hold_times), 0) if hold_times else 0,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active": time.time() < self._cooldown_until,
            "trades_this_hour": len(self._hour_trade_timestamps),
        }
