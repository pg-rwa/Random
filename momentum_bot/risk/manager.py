"""Risk management — position sizing, loss limits, and trade discipline."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    side: str
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    closed: bool = False


class RiskManager:
    """Enforces risk rules before any trade is taken.

    Rules:
        - Max risk per trade as % of account equity.
        - Max number of consecutive losses before cooldown.
        - Daily loss limit as % of starting equity.
        - Max open positions.
        - Trailing stop via ATR multiplier.
    """

    def __init__(self, config: dict):
        self.max_risk_pct: float = config.get("max_risk_per_trade_pct", 1.0)
        self.max_consecutive_losses: int = config.get("max_consecutive_losses", 3)
        self.cooldown_seconds: int = config.get("cooldown_seconds", 3600)
        self.daily_loss_limit_pct: float = config.get("daily_loss_limit_pct", 5.0)
        self.max_open_positions: int = config.get("max_open_positions", 3)
        self.trailing_stop_atr_mult: float = config.get("trailing_stop_atr_mult", 1.5)

        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._day_start: float = 0.0
        self._open_position_count: int = 0
        self._trades: list[TradeRecord] = []

    def set_daily_equity(self, equity: float) -> None:
        """Call at the start of each trading day (or bot startup)."""
        now = time.time()
        # Reset daily tracking if new day
        if now - self._day_start > 86400:
            self._daily_pnl = 0.0
            self._day_start = now
        self._daily_start_equity = equity

    def set_open_positions(self, count: int) -> None:
        self._open_position_count = count

    def can_trade(self) -> tuple[bool, str]:
        """Check if a new trade is allowed under current risk rules."""
        now = time.time()

        # Cooldown after consecutive losses
        if now < self._cooldown_until:
            remaining = int(self._cooldown_until - now)
            return False, f"Cooldown active ({remaining}s remaining after {self.max_consecutive_losses} consecutive losses)"

        # Daily loss limit
        if self._daily_start_equity > 0:
            daily_loss_pct = abs(self._daily_pnl) / self._daily_start_equity * 100
            if self._daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit_pct:
                return False, f"Daily loss limit hit ({daily_loss_pct:.1f}% >= {self.daily_loss_limit_pct}%)"

        # Max open positions
        if self._open_position_count >= self.max_open_positions:
            return False, f"Max open positions reached ({self._open_position_count}/{self.max_open_positions})"

        return True, "OK"

    def calculate_position_size(
        self, equity: float, entry_price: float, stop_price: float
    ) -> float:
        """Calculate position size based on risk per trade.

        Risk = (entry - stop) * size
        Max risk = equity * max_risk_pct / 100
        Size = max_risk / |entry - stop|
        """
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit == 0:
            return 0.0

        max_risk = equity * self.max_risk_pct / 100.0
        size = max_risk / risk_per_unit
        return size

    def calculate_stop_loss(
        self, entry_price: float, atr: float, side: str
    ) -> float:
        """Calculate stop loss price using ATR."""
        distance = atr * self.trailing_stop_atr_mult
        if side == "buy":
            return entry_price - distance
        else:
            return entry_price + distance

    def calculate_take_profit(
        self, entry_price: float, atr: float, side: str, reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit using reward:risk ratio based on ATR stop."""
        distance = atr * self.trailing_stop_atr_mult * reward_ratio
        if side == "buy":
            return entry_price + distance
        else:
            return entry_price - distance

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result and update risk state."""
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss recorded. Consecutive losses: %d/%d",
                self._consecutive_losses,
                self.max_consecutive_losses,
            )
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until = time.time() + self.cooldown_seconds
                logger.warning(
                    "Consecutive loss limit hit. Cooldown for %ds.",
                    self.cooldown_seconds,
                )
        else:
            self._consecutive_losses = 0

    def get_status(self) -> dict:
        """Get current risk manager status."""
        return {
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_start_equity": self._daily_start_equity,
            "cooldown_active": time.time() < self._cooldown_until,
            "open_positions": self._open_position_count,
        }
