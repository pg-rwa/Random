"""Gold Scalper Bot — main entry point.

v5: Session-aware pattern-based strategies + SQLite database.
    Replaces Bollinger Band mean-reversion with Asian Range Breakout,
    Liquidity Sweep, Session VWAP, and EMA Pullback.
    Self-learner removed — strategies have fixed, proven rules.

Usage:
    python -m gold_scalper.main                      # paper trading (default)
    python -m gold_scalper.main --live                # live trading
    python -m gold_scalper.main -c path/to/config.yaml
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import numpy as np
import yaml
from dotenv import load_dotenv

from momentum_bot.exchange.hyperliquid import HyperliquidExchange
from momentum_bot.exchange.base import Candle
from momentum_bot.indicators.technical import TechnicalIndicators
from gold_scalper.strategy.scalper import BollingerScalper, ScalpSignal, current_session
from gold_scalper.executor.scalp_executor import ScalpExecutor
from gold_scalper.db.trade_db import GoldTradeDB

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_scalper")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_scalp_indicators(candles: list[Candle], config: dict) -> dict:
    """Compute indicators for the session-aware strategies."""
    if len(candles) < 2:
        return {}

    closes = np.array([c.close for c in candles])
    ti = TechnicalIndicators

    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    rsi_period = config.get("rsi_period", 14)
    ema_fast_period = config.get("ema_fast", 9)
    ema_slow_period = config.get("ema_slow", 21)
    ema_trend_period = config.get("ema_trend", 50)

    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(closes, bb_period, bb_std)
    rsi_vals = ti.rsi(closes, rsi_period)
    ema_fast = ti.ema(closes, ema_fast_period)
    ema_slow = ti.ema(closes, ema_slow_period)
    ema_trend = ti.ema(closes, ema_trend_period)
    atr_vals = ti.atr(candles, 14)
    adx_vals = ti.adx(candles, 14)
    vwap_vals = ti.vwap(candles)
    vol_sma = ti.volume_sma(candles, 20)
    current_volume = candles[-1].volume
    avg_volume = vol_sma[-1] if not np.isnan(vol_sma[-1]) else current_volume

    return {
        "close": closes[-1],
        "prev_close": closes[-2] if len(closes) > 1 else closes[-1],
        "bb_upper": bb_upper[-1] if not np.isnan(bb_upper[-1]) else 0,
        "bb_middle": bb_middle[-1] if not np.isnan(bb_middle[-1]) else 0,
        "bb_lower": bb_lower[-1] if not np.isnan(bb_lower[-1]) else 0,
        "bb_width_pct": ((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100)
            if not np.isnan(bb_middle[-1]) and bb_middle[-1] > 0 else 0,
        "rsi": rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50.0,
        "rsi_prev": rsi_vals[-2] if len(rsi_vals) > 1 and not np.isnan(rsi_vals[-2]) else 50.0,
        "ema_fast": ema_fast[-1],
        "ema_slow": ema_slow[-1],
        "ema_trend": ema_trend[-1],
        "atr": atr_vals[-1] if not np.isnan(atr_vals[-1]) else 0,
        "adx": adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0,
        "vwap": vwap_vals[-1],
        "volume": current_volume,
        "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1.0,
    }


def candles_to_dicts(candles: list[Candle]) -> list[dict]:
    """Convert Candle objects to dicts for strategy consumption."""
    return [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]


async def run_scalper(config: dict, paper_trade: bool = True) -> None:
    """Main scalping loop."""
    secret_key = os.getenv("HL_SECRET_KEY", "")
    wallet = os.getenv("HL_WALLET_ADDRESS", "")
    testnet_env = os.getenv("HL_TESTNET")
    testnet = (testnet_env or "false").lower() == "true"
    if paper_trade:
        testnet = False

    if not paper_trade and (not secret_key or not wallet):
        logger.error("HL_SECRET_KEY and HL_WALLET_ADDRESS required for live trading.")
        sys.exit(1)

    exchange = HyperliquidExchange(secret_key, wallet, testnet=testnet)

    configured_symbol = config.get("symbol", "XAU")
    interval = config.get("interval", "1m")

    search_terms = list(set(["XAU", "GOLD", "@GOLD", "@XAU", configured_symbol]))
    discovered = exchange.discover_symbol(search_terms)
    if discovered:
        symbol = discovered
        logger.info("Gold symbol discovered: '%s' (configured: '%s')", symbol, configured_symbol)
    else:
        symbol = configured_symbol
        logger.warning("Could not auto-discover gold symbol, using configured: '%s'", symbol)

    candle_limit = config.get("candle_limit", 100)
    loop_interval = config.get("loop_interval_seconds", 60)
    leverage = config.get("leverage", 10)

    strategy_cfg = config.get("strategy", {})
    executor_cfg = config.get("executor", {})

    # Initialize SQLite database
    db = GoldTradeDB(config.get("db_path", "gold_trades/gold_scalper.db"))

    strategy = BollingerScalper(strategy_cfg)
    executor = ScalpExecutor(
        exchange=exchange,
        paper_trade=paper_trade,
        tp_pct=executor_cfg.get("tp_pct", 0.14),
        sl_pct=executor_cfg.get("sl_pct", 0.14),
        position_size_usd=executor_cfg.get("position_size_usd", 100.0),
        leverage=leverage,
        max_consecutive_losses=executor_cfg.get("max_consecutive_losses", 5),
        cooldown_after_losses_sec=executor_cfg.get("cooldown_after_losses_sec", 300),
        daily_loss_limit_pct=executor_cfg.get("daily_loss_limit_pct", 3.0),
        direction_cooldown_sec=executor_cfg.get("direction_cooldown_sec", 60.0),
        learner=None,
        db=db,
        use_atr_stops=executor_cfg.get("use_atr_stops", True),
        atr_sl_multiplier=executor_cfg.get("atr_sl_multiplier", 1.5),
        atr_tp_multiplier=executor_cfg.get("atr_tp_multiplier", 2.0),
        max_dir_consecutive_losses=executor_cfg.get("max_dir_consecutive_losses", 3),
        dir_cooldown_after_losses_sec=executor_cfg.get("dir_cooldown_after_losses_sec", 300),
        session_max_drawdown_usd=executor_cfg.get("session_max_drawdown_usd", 20.0),
    )

    logger.info("=" * 60)
    logger.info("Gold Scalper Bot v5 Starting")
    logger.info("  Mode: %s", "PAPER" if paper_trade else "LIVE")
    logger.info("  Symbol: %s", symbol)
    logger.info("  Interval: %s | Loop: %ds", interval, loop_interval)
    logger.info("  Leverage: %dx", leverage)
    logger.info("  TP: %.2f%% | SL: %.2f%% | ATR stops: %s",
                executor_cfg.get("tp_pct", 0.14), executor_cfg.get("sl_pct", 0.14),
                executor_cfg.get("use_atr_stops", True))
    logger.info("  Position: $%.0f x %dx = $%.0f notional",
                executor_cfg.get("position_size_usd", 100), leverage,
                executor_cfg.get("position_size_usd", 100) * leverage)
    logger.info("  Strategies: Asian Breakout, Liquidity Sweep, Session VWAP, EMA Pullback")
    logger.info("  Database: SQLite at %s", config.get("db_path", "gold_trades/gold_scalper.db"))
    logger.info("  Session: %s", current_session().upper())
    logger.info("=" * 60)

    if not paper_trade:
        await exchange.set_leverage(symbol, leverage)
        equity = await exchange.get_balance()
        executor.set_daily_equity(equity)
    else:
        executor.set_daily_equity(10000.0)

    cycle = 0
    while True:
        cycle += 1
        try:
            candles = await exchange.get_candles(symbol, interval, candle_limit)
            if len(candles) < 25:
                logger.warning("Not enough candles (%d). Waiting...", len(candles))
                await asyncio.sleep(loop_interval)
                continue

            # Store candles in database
            candle_dicts = candles_to_dicts(candles)
            db.insert_candles(symbol, candle_dicts)

            indicators = compute_scalp_indicators(candles, strategy_cfg)
            if not indicators:
                logger.warning("Indicator computation failed. Skipping.")
                await asyncio.sleep(loop_interval)
                continue

            live_price = await exchange.get_price(symbol)
            indicators["close"] = live_price

            if indicators.get("atr", 0) > 0:
                executor.update_atr(indicators["atr"])

            closed_trades = await executor.check_stops(live_price)
            for ct in closed_trades:
                logger.info("  Position closed: %s %s PnL=$%.2f [%s]",
                            ct["direction"], ct["signal"], ct["pnl"],
                            ct.get("strategy", ""))

            # Generate signals with candle data for pattern detection
            session = current_session()
            signals = strategy.evaluate(
                indicators, executor.has_long, executor.has_short,
                candles_data=candle_dicts,
            )

            # Log signal to database
            if signals.strategy:
                db.insert_signal({
                    "strategy": signals.strategy,
                    "session": session,
                    "direction": "long" if ScalpSignal.OPEN_LONG in signals.signals else
                                 "short" if ScalpSignal.OPEN_SHORT in signals.signals else None,
                    "action": signals.signals[0].value if signals.signals else "HOLD",
                    "confidence": signals.confidence,
                    "price": live_price,
                    "reasons": signals.reasons,
                })

            executed = await executor.execute_signals(signals, live_price, symbol)

            pos_status = executor.get_position_status()
            long_str = f"LONG@{pos_status['long']['entry']:.2f}" if pos_status["long"]["active"] else "---"
            short_str = f"SHORT@{pos_status['short']['entry']:.2f}" if pos_status["short"]["active"] else "---"

            log_signals = [s.value for s in signals.signals if s != ScalpSignal.HOLD]
            signal_str = ", ".join(log_signals) if log_signals else "HOLD"
            strat_str = f"[{signals.strategy}]" if signals.strategy else ""

            logger.info(
                "[%d] $%.2f | RSI=%.1f | VWAP=%.2f | ADX=%.0f | trend=%s | "
                "session=%s | %s | %s | %s %s",
                cycle, live_price, indicators["rsi"], indicators.get("vwap", 0),
                indicators.get("adx", 0), signals.trend,
                session.upper(), long_str, short_str, signal_str, strat_str,
            )

            if cycle % 10 == 0:
                summary = executor.get_trade_summary()
                logger.info(
                    "--- Summary: %d trades | PnL=$%.2f | Session=$%.2f | WR=%.0f%% | "
                    "Avg hold=%ds | Consec losses=%d ---",
                    summary.get("closed_trades", 0),
                    summary.get("total_pnl", 0),
                    summary.get("session_pnl", 0),
                    summary.get("win_rate", 0),
                    summary.get("avg_hold_sec", 0),
                    summary.get("consecutive_losses", 0),
                )

                all_stats = db.all_strategy_stats(hours=24)
                for st in all_stats:
                    if st["total"] > 0:
                        logger.info(
                            "  [%s] %d trades | PnL=$%.2f | WR=%.0f%% | PF=%.2f",
                            st["strategy"], st["total"], st["total_pnl"],
                            st["win_rate"], st["profit_factor"],
                        )

            if cycle % 1000 == 0:
                db.cleanup_old_candles(keep_days=30)

        except Exception as e:
            logger.error("Error in cycle %d: %s", cycle, e, exc_info=True)

        await asyncio.sleep(loop_interval)


def main():
    parser = argparse.ArgumentParser(description="Gold Scalper Bot v5")
    parser.add_argument(
        "-c", "--config",
        default="gold_scalper/config/gold_scalp.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (default: paper)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.live:
        logger.warning("!!! LIVE TRADING MODE — REAL MONEY AT RISK !!!")
        confirm = input("Type 'YES' to confirm live gold scalping: ")
        if confirm != "YES":
            logger.info("Cancelled.")
            sys.exit(0)

    asyncio.run(run_scalper(config, paper_trade=not args.live))


if __name__ == "__main__":
    main()
