"""Trade executor — bridges strategy signals to exchange orders with risk checks."""

import json
import logging
import os
import time
from pathlib import Path

from momentum_bot.exchange.base import ExchangeBase, OrderSide
from momentum_bot.risk.manager import RiskManager
from momentum_bot.strategy.momentum import Signal, SignalResult

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades based on strategy signals, subject to risk management rules.

    Supports:
        - Paper trading mode (logs trades without executing)
        - Live trading mode (executes on exchange)
        - Trade journaling to JSON file for review
    """

    def __init__(
        self,
        exchange: ExchangeBase,
        risk_manager: RiskManager,
        paper_trade: bool = True,
        log_dir: str = "trades",
    ):
        self.exchange = exchange
        self.risk = risk_manager
        self.paper_trade = paper_trade
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log: list[dict] = []
        # Track simulated positions in paper mode: {symbol: {side, size, entry_price}}
        self._paper_positions: dict[str, dict] = {}

        mode = "PAPER" if paper_trade else "LIVE"
        logger.info("TradeExecutor initialized in %s mode", mode)

    def get_paper_position(self, symbol: str) -> dict:
        """Return the tracked paper position for a symbol."""
        return self._paper_positions.get(symbol, {"size": 0, "side": "none", "entry_price": 0})

    async def execute_signal(
        self, symbol: str, signal: SignalResult, indicators: dict
    ) -> dict | None:
        """Process a strategy signal and execute if risk rules allow.

        Returns trade record dict if a trade was executed, None otherwise.
        """
        if signal.signal == Signal.HOLD:
            return None

        # Get current state
        equity = await self.exchange.get_balance() if not self.paper_trade else 10000.0
        position = await self.exchange.get_position(symbol) if not self.paper_trade else self.get_paper_position(symbol)
        current_price = indicators.get("close", 0)
        atr = indicators.get("atr", 0)

        # Handle close signals
        if signal.signal in (Signal.CLOSE_LONG, Signal.CLOSE_SHORT):
            return await self._close_position(symbol, position, signal, current_price)

        # Check risk rules for new trades
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.warning("Trade blocked by risk manager: %s", reason)
            return None

        # Determine side and calculate stop/position size
        side = "buy" if signal.signal == Signal.BUY else "sell"
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        if atr <= 0:
            logger.warning("ATR is zero, cannot calculate stop. Skipping trade.")
            return None

        stop_price = self.risk.calculate_stop_loss(current_price, atr, side)
        take_profit = self.risk.calculate_take_profit(current_price, atr, side)
        size = self.risk.calculate_position_size(equity, current_price, stop_price)

        if size <= 0:
            logger.warning("Calculated position size is 0. Skipping.")
            return None

        trade_record = {
            "timestamp": time.time(),
            "symbol": symbol,
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "reasons": signal.reasons,
            "side": side,
            "entry_price": current_price,
            "stop_loss": stop_price,
            "take_profit": take_profit,
            "size": size,
            "equity": equity,
            "atr": atr,
            "indicators": {k: v for k, v in indicators.items() if isinstance(v, (int, float))},
        }

        if self.paper_trade:
            trade_record["mode"] = "paper"
            trade_record["status"] = "filled"
            # Track simulated position so the strategy knows we're in a trade
            self._paper_positions[symbol] = {
                "side": side,
                "size": size,
                "entry_price": current_price,
            }
            logger.info(
                "[PAPER] %s %s %.4f @ %.2f | SL=%.2f TP=%.2f | %s",
                signal.signal.value, symbol, size, current_price,
                stop_price, take_profit, signal,
            )
        else:
            result = await self.exchange.place_order(
                symbol=symbol,
                side=order_side,
                size=size,
            )
            trade_record["mode"] = "live"
            trade_record["order_result"] = {
                "success": result.success,
                "order_id": result.order_id,
                "filled_price": result.filled_price,
                "error": result.error,
            }
            trade_record["status"] = "filled" if result.success else "failed"

            if result.success:
                logger.info(
                    "[LIVE] %s %s %.4f @ %.2f | SL=%.2f TP=%.2f",
                    signal.signal.value, symbol, size,
                    result.filled_price or current_price,
                    stop_price, take_profit,
                )
            else:
                logger.error("[LIVE] Order FAILED: %s", result.error)

        self._trade_log.append(trade_record)
        self._persist_trade(trade_record)
        return trade_record

    async def _close_position(
        self, symbol: str, position: dict, signal: SignalResult, current_price: float
    ) -> dict | None:
        """Close an existing position."""
        if position["size"] == 0:
            return None

        close_side = OrderSide.SELL if position["side"] == "buy" else OrderSide.BUY
        size = position["size"]

        trade_record = {
            "timestamp": time.time(),
            "symbol": symbol,
            "signal": signal.signal.value,
            "reasons": signal.reasons,
            "side": "close",
            "close_price": current_price,
            "size": size,
            "entry_price": position.get("entry_price", current_price),
            "pnl": position.get("unrealized_pnl", 0),
        }

        if self.paper_trade:
            pnl = (current_price - position.get("entry_price", current_price)) * size
            if position["side"] == "sell":
                pnl = -pnl
            trade_record["pnl"] = pnl
            trade_record["mode"] = "paper"
            trade_record["status"] = "closed"
            self.risk.record_trade_result(pnl)
            # Clear the simulated position
            self._paper_positions.pop(symbol, None)
            logger.info(
                "[PAPER] CLOSE %s %.4f @ %.2f | PnL=%.2f | %s",
                symbol, size, current_price, pnl, signal,
            )
        else:
            result = await self.exchange.place_order(
                symbol=symbol,
                side=close_side,
                size=size,
                reduce_only=True,
            )
            trade_record["mode"] = "live"
            if result.success:
                trade_record["status"] = "closed"
                pnl = position.get("unrealized_pnl", 0)
                self.risk.record_trade_result(pnl)
                trade_record["pnl"] = pnl
                logger.info("[LIVE] CLOSE %s @ %.2f | PnL=%.2f", symbol, current_price, pnl)
            else:
                trade_record["status"] = "close_failed"
                logger.error("[LIVE] Close FAILED: %s", result.error)

        self._trade_log.append(trade_record)
        self._persist_trade(trade_record)
        return trade_record

    def _persist_trade(self, trade: dict) -> None:
        """Append trade to daily JSON log file."""
        date_str = time.strftime("%Y-%m-%d")
        log_file = self._log_dir / f"trades_{date_str}.json"

        existing = []
        if log_file.exists():
            with open(log_file, "r") as f:
                existing = json.load(f)

        existing.append(trade)
        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def get_trade_summary(self) -> dict:
        """Summary statistics for current session."""
        if not self._trade_log:
            return {"total_trades": 0}

        pnls = [t.get("pnl", 0) for t in self._trade_log if "pnl" in t]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_trades": len(self._trade_log),
            "closed_trades": len(pnls),
            "total_pnl": sum(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
        }
