"""Hyperliquid exchange integration."""

import logging
import time
from typing import Optional

import requests
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from .base import ExchangeBase, Candle, OrderSide, OrderResult

logger = logging.getLogger(__name__)

# Map generic interval strings to Hyperliquid candle intervals
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class HyperliquidExchange(ExchangeBase):
    """Hyperliquid perpetual futures exchange."""

    def __init__(self, secret_key: str, wallet_address: str, testnet: bool = True):
        self._wallet = wallet_address
        self._base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self._secret_key = secret_key
        self._testnet = testnet
        # Lazy-initialized to avoid network calls during construction
        self.__info: Optional[Info] = None
        self.__exchange: Optional[Exchange] = None
        logger.info(
            "Hyperliquid exchange initialized (testnet=%s, wallet=%s...%s)",
            testnet,
            wallet_address[:6] if wallet_address else "N/A",
            wallet_address[-4:] if wallet_address else "N/A",
        )

    @property
    def _info(self) -> Info:
        if self.__info is None:
            # Load all builder perp DEXes so HIP-3 assets (XAU, XAG, etc.) are available
            try:
                resp = requests.post(
                    self._base_url + "/info",
                    json={"type": "perpDexs"},
                    timeout=10,
                )
                dex_list = resp.json()
                perp_dex_names = [""] + [d["name"] for d in dex_list]
                logger.info("Loading perp DEXes: %s", perp_dex_names)
                self.__info = Info(self._base_url, skip_ws=True, perp_dexs=perp_dex_names)
            except Exception as e:
                logger.warning("Failed to load builder DEXes, using default: %s", e)
                self.__info = Info(self._base_url, skip_ws=True)
        return self.__info

    @property
    def _exchange(self) -> Exchange:
        if self.__exchange is None:
            self.__exchange = Exchange(
                wallet=None,
                base_url=self._base_url,
                account_address=self._wallet,
            )
        return self.__exchange

    async def get_candles(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list[Candle]:
        hl_interval = INTERVAL_MAP.get(interval, interval)
        end_time = int(time.time() * 1000)
        start_time = end_time - limit * 60 * 1000  # rough estimate

        try:
            raw = self._info.candles_snapshot(symbol, hl_interval, start_time, end_time)
        except KeyError:
            # Symbol not in SDK's name_to_coin — use direct API call (HIP-3 assets)
            logger.info("Symbol %s not in SDK mapping, using direct API", symbol)
            raw = self._candles_direct(symbol, hl_interval, start_time, end_time)

        candles = []
        for c in raw:
            candles.append(
                Candle(
                    timestamp=c["t"],
                    open=float(c["o"]),
                    high=float(c["h"]),
                    low=float(c["l"]),
                    close=float(c["c"]),
                    volume=float(c["v"]),
                )
            )
        return candles

    def _candles_direct(self, coin: str, interval: str, start_time: int, end_time: int) -> list:
        """Fetch candles via direct API POST for assets not in SDK mapping."""
        resp = requests.post(
            self._base_url + "/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                },
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_price(self, symbol: str) -> float:
        all_mids = self._info.all_mids()
        if symbol in all_mids:
            return float(all_mids[symbol])
        raise ValueError(f"Symbol {symbol} not found on Hyperliquid")

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        try:
            is_buy = side == OrderSide.BUY

            if price is None:
                # Market order — use aggressive limit price to simulate market
                mid = await self.get_price(symbol)
                slippage = 0.001  # 0.1% slippage tolerance
                price = mid * (1 + slippage) if is_buy else mid * (1 - slippage)

            result = self._exchange.order(
                symbol,
                is_buy,
                size,
                price,
                {"limit": {"tif": "Ioc"}},  # Immediate or cancel for market-like
                reduce_only=reduce_only,
            )

            if result["status"] == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                order_id = None
                filled_price = price
                if statuses and "resting" in statuses[0]:
                    order_id = str(statuses[0]["resting"]["oid"])
                elif statuses and "filled" in statuses[0]:
                    order_id = str(statuses[0]["filled"]["oid"])
                    filled_price = float(statuses[0]["filled"]["avgPx"])

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    filled_price=filled_price,
                    filled_qty=size,
                    raw=result,
                )
            else:
                return OrderResult(
                    success=False,
                    error=str(result),
                    raw=result,
                )

        except Exception as e:
            logger.error("Order failed: %s", e)
            return OrderResult(success=False, error=str(e))

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            result = self._exchange.cancel(symbol, int(order_id))
            return result.get("status") == "ok"
        except Exception as e:
            logger.error("Cancel failed: %s", e)
            return False

    async def get_position(self, symbol: str) -> dict:
        user_state = self._info.user_state(self._wallet)
        positions = user_state.get("assetPositions", [])

        for pos in positions:
            p = pos.get("position", {})
            if p.get("coin") == symbol:
                size = float(p.get("szi", 0))
                return {
                    "size": abs(size),
                    "side": "buy" if size > 0 else "sell" if size < 0 else "none",
                    "entry_price": float(p.get("entryPx", 0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                    "leverage": float(p.get("leverage", {}).get("value", 1)),
                }

        return {
            "size": 0,
            "side": "none",
            "entry_price": 0,
            "unrealized_pnl": 0,
            "leverage": 1,
        }

    async def get_balance(self) -> float:
        user_state = self._info.user_state(self._wallet)
        return float(user_state.get("marginSummary", {}).get("accountValue", 0))

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self._exchange.update_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error("Set leverage failed: %s", e)
            return False

    async def get_all_positions(self) -> list[dict]:
        user_state = self._info.user_state(self._wallet)
        positions = []
        for pos in user_state.get("assetPositions", []):
            p = pos.get("position", {})
            size = float(p.get("szi", 0))
            if size != 0:
                positions.append({
                    "symbol": p.get("coin"),
                    "size": abs(size),
                    "side": "buy" if size > 0 else "sell",
                    "entry_price": float(p.get("entryPx", 0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                })
        return positions
