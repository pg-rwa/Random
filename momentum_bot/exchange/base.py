"""Abstract base class for exchange integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Candle:
    timestamp: int  # ms epoch
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)


class ExchangeBase(ABC):
    """Abstract interface for any exchange. Implement this to add new exchanges."""

    @abstractmethod
    async def get_candles(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list[Candle]:
        """Fetch historical candles.

        Args:
            symbol: Trading pair (e.g. "BTC", "ETH", "GOLD").
            interval: Candle interval (e.g. "5m", "15m", "1h").
            limit: Number of candles to fetch.
        """
        ...

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """Get current mid price for a symbol."""
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a market or limit order.

        Args:
            symbol: Trading pair.
            side: Buy or sell.
            size: Position size in units of the asset.
            price: Limit price. None for market order.
            reduce_only: If True, only reduces existing position.
        """
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> dict:
        """Get current position for a symbol.

        Returns dict with keys: size, side, entry_price, unrealized_pnl, leverage.
        Size of 0 means no position.
        """
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Get account equity / balance in USD."""
        ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        ...

    @abstractmethod
    async def get_all_positions(self) -> list[dict]:
        """Get all open positions."""
        ...
