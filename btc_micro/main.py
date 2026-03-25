"""BTC Micro-Trading Bot — main entry point.

v5: Pattern-based strategies + SQLite database.
    Replaces weighted signal scorer with VWAP Reclaim, FVG, BOS, EMA Pullback.
    All candles and trades stored in SQLite for analysis and backtesting.
    Self-learner removed — strategies have fixed, proven rules.

Usage:
    python -m btc_micro                          # paper trading (default)
    python -m btc_micro --live                   # live trading
    python -m btc_micro -c path/to/config.yaml
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
from btc_micro.strategy.micro_scalper import MicroScalper, MicroSignal
from btc_micro.executor.micro_executor import MicroExecutor
from btc_micro.db.trade_db import TradeDB

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("btc_micro")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_micro_indicators(candles: list[Candle], config: dict) -> dict:
    """Compute all indicators for the pattern-based strategies."""
    if len(candles) < 2:
        return {}

    closes = np.array([c.close for c in candles])
    ti = TechnicalIndicators

    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    rsi_period = config.get("rsi_period", 14)
    ema_fast_period = config.get("ema_fast", 8)
    ema_slow_period = config.get("ema_slow", 21)
    ema_trend_period = config.get("ema_trend", 50)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(closes, bb_period, bb_std)

    # RSI
    rsi_vals = ti.rsi(closes, rsi_period)

    # EMAs
    ema_fast = ti.ema(closes, ema_fast_period)
    ema_slow = ti.ema(closes, ema_slow_period)
    ema_trend = ti.ema(closes, ema_trend_period)

    # MACD
    macd_line, signal_line, histogram = ti.macd(closes, 12, 26, 9)

    # ATR
    atr_vals = ti.atr(candles, 14)

    # ADX
    adx_vals = ti.adx(candles, 14)

    # VWAP
    vwap_vals = ti.vwap(candles)

    # Volume
    vol_sma = ti.volume_sma(candles, 20)
    current_volume = candles[-1].volume
    avg_volume = vol_sma[-1] if not np.isnan(vol_sma[-1]) else current_volume

    # BB width
    bb_width_pct = (
        (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100
        if not np.isnan(bb_middle[-1]) and bb_middle[-1] > 0
        else 0
    )

    return {
        "close": closes[-1],
        "prev_close": closes[-2] if len(closes) > 1 else closes[-1],
        "bb_upper": bb_upper[-1] if not np.isnan(bb_upper[-1]) else 0,
        "bb_middle": bb_middle[-1] if not np.isnan(bb_middle[-1]) else 0,
        "bb_lower": bb_lower[-1] if not np.isnan(bb_lower[-1]) else 0,
        "bb_width_pct": bb_width_pct,
        "rsi": rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50.0,
        "rsi_prev": rsi_vals[-2] if len(rsi_vals) > 1 and not np.isnan(rsi_vals[-2]) else 50.0,
        "ema_fast": ema_fast[-1],
        "ema_slow": ema_slow[-1],
        "ema_fast_prev": ema_fast[-2] if len(ema_fast) > 1 else ema_fast[-1],
        "ema_slow_prev": ema_slow[-2] if len(ema_slow) > 1 else ema_slow[-1],
        "ema_trend": ema_trend[-1] if not np.isnan(ema_trend[-1]) else 0,
        "macd_line": macd_line[-1],
        "macd_signal": signal_line[-1],
        "macd_histogram": histogram[-1],
        "macd_histogram_prev": histogram[-2] if len(histogram) > 1 else 0,
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


async def run_micro_bot(config: dict, paper_trade: bool = True) -> None:
    """Main micro-trading loop."""
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

    symbol = config.get("symbol", "BTC")
    interval = config.get("interval", "1m")
    candle_limit = config.get("candle_limit", 100)
    loop_interval = config.get("loop_interval_seconds", 20)
    leverage = config.get("leverage", 5)

    strategy_cfg = config.get("strategy", {})
    executor_cfg = config.get("executor", {})

    # Initialize SQLite database
    db = TradeDB(config.get("db_path", "btc_trades/btc_micro.db"))

    strategy = MicroScalper(strategy_cfg)
    executor = MicroExecutor(
        exchange=exchange,
        paper_trade=paper_trade,
        tp_pct=executor_cfg.get("tp_pct", 0.15),
        sl_pct=executor_cfg.get("sl_pct", 0.12),
        position_size_usd=executor_cfg.get("position_size_usd", 40.0),
        leverage=leverage,
        max_consecutive_losses=executor_cfg.get("max_consecutive_losses", 4),
        cooldown_after_losses_sec=executor_cfg.get("cooldown_after_losses_sec", 120),
        daily_loss_limit_pct=executor_cfg.get("daily_loss_limit_pct", 2.0),
        direction_cooldown_sec=executor_cfg.get("direction_cooldown_sec", 45.0),
        max_trades_per_hour=executor_cfg.get("max_trades_per_hour", 30),
        learner=None,
        use_atr_stops=executor_cfg.get("use_atr_stops", True),
        atr_tp_mult=executor_cfg.get("atr_tp_mult", 2.5),
        atr_sl_mult=executor_cfg.get("atr_sl_mult", 1.8),
        trailing_stop_activate_pct=executor_cfg.get("trailing_stop_activate_pct", 0.10),
        trailing_stop_distance_pct=executor_cfg.get("trailing_stop_distance_pct", 0.06),
        session_drawdown_limit_pct=executor_cfg.get("session_drawdown_limit_pct", 3.0),
        per_direction_loss_limit=executor_cfg.get("per_direction_loss_limit", 3),
        per_direction_cooldown_sec=executor_cfg.get("per_direction_cooldown_sec", 120.0),
        scale_size_on_streak=executor_cfg.get("scale_size_on_streak", True),
        db=db,
    )

    logger.info("=" * 60)
    logger.info("BTC Micro-Trading Bot v5 Starting")
    logger.info("  Mode: %s", "PAPER" if paper_trade else "LIVE")
    logger.info("  Symbol: %s", symbol)
    logger.info("  Interval: %s | Loop: %ds", interval, loop_interval)
    logger.info("  Leverage: %dx", leverage)
    logger.info("  TP: %.2f%% | SL: %.2f%% | ATR stops: %s",
                executor_cfg.get("tp_pct", 0.15), executor_cfg.get("sl_pct", 0.12),
                executor_cfg.get("use_atr_stops", True))
    logger.info("  Position: $%.0f x %dx = $%.0f notional",
                executor_cfg.get("position_size_usd", 40), leverage,
                executor_cfg.get("position_size_usd", 40) * leverage)
    logger.info("  Strategies: VWAP Reclaim, FVG, Structure Break, EMA Pullback")
    logger.info("  Database: SQLite at %s", config.get("db_path", "btc_trades/btc_micro.db"))
    logger.info("  Trailing stop: activate=%.2f%% trail=%.2f%%",
                executor_cfg.get("trailing_stop_activate_pct", 0.10),
                executor_cfg.get("trailing_stop_distance_pct", 0.06))
    logger.info("=" * 60)

    # Set leverage and equity
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
            # 1. Fetch 1m candles
            candles = await exchange.get_candles(symbol, interval, candle_limit)
            if len(candles) < 25:
                logger.warning("Not enough candles (%d). Waiting...", len(candles))
                await asyncio.sleep(loop_interval)
                continue

            # 2. Store candles in database
            candle_dicts = candles_to_dicts(candles)
            db.insert_candles(symbol, candle_dicts)

            # 3. Compute indicators
            indicators = compute_micro_indicators(candles, strategy_cfg)
            if not indicators:
                logger.warning("Indicator computation failed. Skipping.")
                await asyncio.sleep(loop_interval)
                continue

            # 4. Get live price
            live_price = await exchange.get_price(symbol)
            indicators["close"] = live_price
            # prev_close stays from candle data

            # 5. Check SL/TP on existing positions
            closed_trades = await executor.check_stops(live_price)
            for ct in closed_trades:
                logger.info("  Position closed: %s %s PnL=$%.2f [%s]",
                            ct["direction"], ct["signal"], ct["pnl"],
                            ct.get("strategy", ""))

            # 6. Generate signals (pass candle dicts for FVG/BOS pattern detection)
            signals = strategy.evaluate(
                indicators, executor.has_long, executor.has_short,
                candles_data=candle_dicts,
            )

            # 7. Log signal to database
            if signals.strategy:
                db.insert_signal({
                    "strategy": signals.strategy,
                    "direction": "long" if MicroSignal.OPEN_LONG in signals.signals else
                                 "short" if MicroSignal.OPEN_SHORT in signals.signals else None,
                    "action": signals.signals[0].value if signals.signals else "HOLD",
                    "confidence": signals.confidence,
                    "price": live_price,
                    "reasons": signals.reasons,
                })

            # 8. Execute signals
            atr = indicators.get("atr", 0)
            executed = await executor.execute_signals(signals, live_price, symbol, atr=atr)

            # Log cycle info
            pos_status = executor.get_position_status()
            long_str = (f"LONG@{pos_status['long']['entry']:.2f}"
                        if pos_status["long"]["active"] else "---")
            short_str = (f"SHORT@{pos_status['short']['entry']:.2f}"
                         if pos_status["short"]["active"] else "---")

            log_signals = [s.value for s in signals.signals if s != MicroSignal.HOLD]
            signal_str = ", ".join(log_signals) if log_signals else "HOLD"
            strat_str = f"[{signals.strategy}]" if signals.strategy else ""

            logger.info(
                "[%d] $%.2f | RSI=%.1f | VWAP=%.2f | EMA=%s | ADX=%.0f | "
                "trend=%s | %s | %s | %s %s",
                cycle, live_price, indicators["rsi"], indicators["vwap"],
                "UP" if indicators["ema_fast"] > indicators["ema_slow"] else "DN",
                indicators.get("adx", 0),
                signals.trend, long_str, short_str, signal_str, strat_str,
            )

            # Summary every 10 cycles
            if cycle % 10 == 0:
                summary = executor.get_trade_summary()
                logger.info(
                    "--- Summary: %d trades | PnL=$%.2f | Session=$%.2f | WR=%.0f%% | "
                    "Avg hold=%ds | Losses streak=%d | Trades/hr=%d ---",
                    summary.get("closed_trades", 0),
                    summary.get("total_pnl", 0),
                    summary.get("session_pnl", 0),
                    summary.get("win_rate", 0),
                    summary.get("avg_hold_sec", 0),
                    summary.get("consecutive_losses", 0),
                    summary.get("trades_this_hour", 0),
                )

                # Log per-strategy performance from DB
                all_stats = db.all_strategy_stats(hours=24)
                for st in all_stats:
                    if st["total"] > 0:
                        logger.info(
                            "  [%s] %d trades | PnL=$%.2f | WR=%.0f%% | PF=%.2f",
                            st["strategy"], st["total"], st["total_pnl"],
                            st["win_rate"], st["profit_factor"],
                        )

            # Periodic DB cleanup
            if cycle % 1000 == 0:
                db.cleanup_old_candles(keep_days=30)

        except Exception as e:
            logger.error("Error in cycle %d: %s", cycle, e, exc_info=True)

        await asyncio.sleep(loop_interval)


def main():
    parser = argparse.ArgumentParser(description="BTC Micro-Trading Bot v5")
    parser.add_argument(
        "-c", "--config",
        default="btc_micro/config/btc_micro.yaml",
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
        confirm = input("Type 'YES' to confirm live BTC micro-trading: ")
        if confirm != "YES":
            logger.info("Cancelled.")
            sys.exit(0)

    asyncio.run(run_micro_bot(config, paper_trade=not args.live))


if __name__ == "__main__":
    main()
