"""Main bot runner — connects all components and runs the trading loop."""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from momentum_bot.exchange.hyperliquid import HyperliquidExchange
from momentum_bot.indicators.technical import TechnicalIndicators
from momentum_bot.strategy.momentum import MomentumStrategy
from momentum_bot.risk.manager import RiskManager
from momentum_bot.executor.trade_executor import TradeExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("momentum_bot")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def run_bot(config: dict, paper_trade: bool = True) -> None:
    """Main trading loop."""
    # Initialize exchange
    secret_key = os.getenv("HL_SECRET_KEY", "")
    wallet = os.getenv("HL_WALLET_ADDRESS", "")
    testnet = os.getenv("HL_TESTNET", "true").lower() == "true"

    if not paper_trade and (not secret_key or not wallet):
        logger.error("HL_SECRET_KEY and HL_WALLET_ADDRESS required for live trading. Set in .env file.")
        sys.exit(1)

    exchange = HyperliquidExchange(secret_key, wallet, testnet=testnet or paper_trade)

    # Initialize components
    strategy = MomentumStrategy(config.get("strategy", {}))
    risk_mgr = RiskManager(config.get("risk", {}))
    reentry_cooldown = config.get("strategy", {}).get("reentry_cooldown_seconds", 300)
    executor = TradeExecutor(
        exchange, risk_mgr, paper_trade=paper_trade,
        reentry_cooldown_seconds=reentry_cooldown,
    )

    symbols = config.get("symbols", ["BTC"])
    interval = config.get("interval", "5m")
    candle_limit = config.get("candle_limit", 100)
    loop_interval = config.get("loop_interval_seconds", 30)
    leverage = config.get("leverage", 1)

    logger.info("=" * 60)
    logger.info("Momentum Bot Starting")
    logger.info("  Mode: %s", "PAPER" if paper_trade else "LIVE")
    logger.info("  Symbols: %s", symbols)
    logger.info("  Interval: %s", interval)
    logger.info("  Loop every: %ds", loop_interval)
    logger.info("  Leverage: %dx", leverage)
    logger.info("=" * 60)

    # Set leverage for all symbols
    if not paper_trade:
        for sym in symbols:
            await exchange.set_leverage(sym, leverage)

    # Set initial equity for risk manager
    if not paper_trade:
        equity = await exchange.get_balance()
        risk_mgr.set_daily_equity(equity)
    else:
        risk_mgr.set_daily_equity(10000.0)

    # Main loop
    cycle = 0
    while True:
        cycle += 1
        logger.info("----- Cycle %d -----", cycle)

        for symbol in symbols:
            try:
                # 1. Fetch candles
                candles = await exchange.get_candles(symbol, interval, candle_limit)
                if len(candles) < 30:
                    logger.warning("[%s] Not enough candles (%d). Skipping.", symbol, len(candles))
                    continue

                # 2. Compute indicators
                indicators = TechnicalIndicators.compute_all(
                    candles, config.get("indicators", {})
                )
                if not indicators:
                    logger.warning("[%s] Indicator computation failed. Skipping.", symbol)
                    continue

                # 3. Check stop-loss / take-profit for paper positions
                if paper_trade:
                    stop_trade = await executor.check_stops(symbol, indicators["close"])
                    if stop_trade:
                        logger.info("[%s] Position closed by %s", symbol, stop_trade["signal"])

                # 4. Get current position
                if not paper_trade:
                    pos = await exchange.get_position(symbol)
                    current_side = pos["side"]
                    positions = await exchange.get_all_positions()
                    risk_mgr.set_open_positions(len(positions))
                else:
                    paper_pos = executor.get_paper_position(symbol)
                    current_side = paper_pos["side"]
                    # Track open paper positions for risk manager
                    open_count = sum(
                        1 for p in (executor.get_paper_position(s) for s in symbols)
                        if p["size"] > 0
                    )
                    risk_mgr.set_open_positions(open_count)

                # 5. Generate signal
                signal = strategy.evaluate(indicators, current_side)

                logger.info(
                    "[%s] Price=%.2f | EMA(f)=%.2f EMA(s)=%.2f | RSI=%.1f | MACD-H=%.4f | Vol=%.1fx | Signal: %s",
                    symbol,
                    indicators["close"],
                    indicators["ema_fast"],
                    indicators["ema_slow"],
                    indicators["rsi"],
                    indicators["macd_histogram"],
                    indicators["volume_ratio"],
                    signal,
                )

                # 6. Execute if there's a trade signal
                trade = await executor.execute_signal(symbol, signal, indicators)
                if trade:
                    logger.info("[%s] Trade executed: %s", symbol, trade.get("status"))

            except Exception as e:
                logger.error("[%s] Error in cycle: %s", symbol, e, exc_info=True)

        # Print session summary periodically
        if cycle % 10 == 0:
            summary = executor.get_trade_summary()
            logger.info("Session summary: %s", summary)
            logger.info("Risk status: %s", risk_mgr.get_status())

        await asyncio.sleep(loop_interval)


def main():
    parser = argparse.ArgumentParser(description="Momentum Trading Bot")
    parser.add_argument(
        "-c", "--config",
        default="momentum_bot/config/btc.yaml",
        help="Path to asset config YAML file",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (default: paper trade)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbols from config (e.g. --symbols BTC ETH)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.symbols:
        config["symbols"] = args.symbols

    if args.live:
        logger.warning("!!! LIVE TRADING MODE — REAL MONEY AT RISK !!!")
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            logger.info("Live trading cancelled.")
            sys.exit(0)

    asyncio.run(run_bot(config, paper_trade=not args.live))


if __name__ == "__main__":
    main()
