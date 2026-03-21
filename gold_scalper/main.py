"""Gold Scalper Bot — main entry point.

Runs a Bollinger Band mean-reversion scalper on GOLD perpetual futures.
Trades every ~60 seconds with 10X leverage, quick TP/SL, dual long+short.

Usage:
    python -m gold_scalper.main                      # paper trading (default)
    python -m gold_scalper.main --live                # live trading
    python -m gold_scalper.main -c path/to/config.yaml
"""

import argparse
import asyncio
import logging
import math
import os
import sys
import time

import numpy as np
import yaml
from dotenv import load_dotenv

from momentum_bot.exchange.hyperliquid import HyperliquidExchange
from momentum_bot.exchange.base import Candle
from momentum_bot.indicators.technical import TechnicalIndicators
from gold_scalper.strategy.scalper import BollingerScalper, ScalpSignal
from gold_scalper.executor.scalp_executor import ScalpExecutor
from gold_scalper.learner.trade_learner import TradeLearner

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
    """Compute indicators needed for the scalping strategy.

    Uses Bollinger Bands + RSI on 1-minute candles.
    v2: Added EMA50 (trend filter), ADX (trend strength), enhanced ATR.
    """
    if len(candles) < 2:
        return {}

    closes = np.array([c.close for c in candles])
    ti = TechnicalIndicators

    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    rsi_period = config.get("rsi_period", 14)
    ema_fast_period = config.get("ema_fast", 9)
    ema_slow_period = config.get("ema_slow", 21)
    ema_trend_period = config.get("ema_trend", 50)  # v2: longer EMA for trend

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(closes, bb_period, bb_std)

    # RSI
    rsi_vals = ti.rsi(closes, rsi_period)

    # EMAs for trend context
    ema_fast = ti.ema(closes, ema_fast_period)
    ema_slow = ti.ema(closes, ema_slow_period)
    ema_trend = ti.ema(closes, ema_trend_period)  # v2: EMA50 trend filter

    # ATR for volatility context
    atr_vals = ti.atr(candles, 14)

    # v2: ADX for trend strength
    adx_vals = ti.adx(candles, 14)

    # Volume
    vol_sma = ti.volume_sma(candles, 20)
    current_volume = candles[-1].volume
    avg_volume = vol_sma[-1] if not np.isnan(vol_sma[-1]) else current_volume

    return {
        "close": closes[-1],
        "bb_upper": bb_upper[-1] if not np.isnan(bb_upper[-1]) else 0,
        "bb_middle": bb_middle[-1] if not np.isnan(bb_middle[-1]) else 0,
        "bb_lower": bb_lower[-1] if not np.isnan(bb_lower[-1]) else 0,
        "bb_width_pct": ((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100)
        if not np.isnan(bb_middle[-1]) and bb_middle[-1] > 0
        else 0,
        "rsi": rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50.0,
        "rsi_prev": rsi_vals[-2] if len(rsi_vals) > 1 and not np.isnan(rsi_vals[-2]) else 50.0,
        "ema_fast": ema_fast[-1],
        "ema_slow": ema_slow[-1],
        "ema_trend": ema_trend[-1],  # v2: EMA50
        "atr": atr_vals[-1] if not np.isnan(atr_vals[-1]) else 0,
        "adx": adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0,  # v2: ADX
        "volume": current_volume,
        "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1.0,
    }


async def run_scalper(config: dict, paper_trade: bool = True) -> None:
    """Main scalping loop."""
    # Initialize exchange
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

    # Load strategy & executor config
    configured_symbol = config.get("symbol", "XAU")
    interval = config.get("interval", "1m")

    # Auto-discover gold symbol — HIP-3 assets may have different names
    # Try multiple possible names: XAU (standard), GOLD, @GOLD (hyperp prefix)
    search_terms = list(set(["XAU", "GOLD", "@GOLD", "@XAU", configured_symbol]))
    discovered = exchange.discover_symbol(search_terms)
    if discovered:
        symbol = discovered
        logger.info("Gold symbol discovered: '%s' (configured: '%s')", symbol, configured_symbol)
    else:
        symbol = configured_symbol
        logger.warning("Could not auto-discover gold symbol, using configured: '%s'", symbol)
        logger.warning("Available coins in SDK: %s",
                        list(exchange._info.name_to_coin.keys())[:50])
    candle_limit = config.get("candle_limit", 100)
    loop_interval = config.get("loop_interval_seconds", 60)
    leverage = config.get("leverage", 10)

    strategy_cfg = config.get("strategy", {})
    executor_cfg = config.get("executor", {})
    learner_cfg = config.get("learner", {})

    # Initialize auto-learner (can be disabled via config)
    learner = None
    if learner_cfg.get("enabled", True):
        learner = TradeLearner(
            log_dir=learner_cfg.get("log_dir", "gold_trades"),
            lookback_days=learner_cfg.get("lookback_days", 7),
            review_every_n_trades=learner_cfg.get("review_every_n_trades", 10),
            min_trades_to_learn=learner_cfg.get("min_trades_to_learn", 15),
            learning_rate=learner_cfg.get("learning_rate", 0.25),
            enable_time_filter=learner_cfg.get("enable_time_filter", True),
            enable_direction_filter=learner_cfg.get("enable_direction_filter", True),
            enable_tp_sl_tuning=learner_cfg.get("enable_tp_sl_tuning", True),
            enable_confidence_gate=learner_cfg.get("enable_confidence_gate", True),
        )
        # Run initial review on startup if we have enough historical data
        overrides = learner.review_and_adapt(config)
        logger.info("Learner initial review: %d overrides", len(overrides))

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
        direction_cooldown_sec=executor_cfg.get("direction_cooldown_sec", 120.0),
        learner=learner,
        # v2 parameters
        use_atr_stops=executor_cfg.get("use_atr_stops", True),
        atr_sl_multiplier=executor_cfg.get("atr_sl_multiplier", 1.5),
        atr_tp_multiplier=executor_cfg.get("atr_tp_multiplier", 2.0),
        max_dir_consecutive_losses=executor_cfg.get("max_dir_consecutive_losses", 3),
        dir_cooldown_after_losses_sec=executor_cfg.get("dir_cooldown_after_losses_sec", 600),
        session_max_drawdown_usd=executor_cfg.get("session_max_drawdown_usd", 20.0),
    )

    # Apply any initial overrides from learner
    if learner and overrides:
        executor.apply_learner_overrides(overrides)

    logger.info("=" * 60)
    logger.info("Gold Scalper Bot v2 Starting")
    logger.info("  Mode: %s", "PAPER" if paper_trade else "LIVE")
    logger.info("  Symbol: %s", symbol)
    logger.info("  Interval: %s (scalp candles)", interval)
    logger.info("  Loop every: %ds", loop_interval)
    logger.info("  Leverage: %dx", leverage)
    logger.info("  TP: %.2f%% | SL: %.2f%%", executor_cfg.get("tp_pct", 0.14), executor_cfg.get("sl_pct", 0.14))
    logger.info("  ATR stops: %s (SL×%.1f TP×%.1f)",
                executor_cfg.get("use_atr_stops", True),
                executor_cfg.get("atr_sl_multiplier", 1.5),
                executor_cfg.get("atr_tp_multiplier", 2.0))
    logger.info("  Position size: $%.0f × %dx = $%.0f notional",
                executor_cfg.get("position_size_usd", 100), leverage,
                executor_cfg.get("position_size_usd", 100) * leverage)
    logger.info("  Trend filter: %s (ADX range<%d trend>%d)",
                strategy_cfg.get("use_trend_filter", True),
                strategy_cfg.get("adx_range_threshold", 25),
                strategy_cfg.get("adx_trend_threshold", 30))
    logger.info("  Dir loss breaker: %d consecutive → %ds pause",
                executor_cfg.get("max_dir_consecutive_losses", 3),
                executor_cfg.get("dir_cooldown_after_losses_sec", 600))
    logger.info("  Session max drawdown: $%.0f", executor_cfg.get("session_max_drawdown_usd", 20.0))
    logger.info("  Dual mode: LONG + SHORT simultaneously")
    logger.info("  Auto-learner: %s", "ENABLED" if learner else "DISABLED")
    if learner:
        logger.info("    Review every: %d trades", learner_cfg.get("review_every_n_trades", 10))
        logger.info("    Blocked hours: %s", sorted(learner.blocked_hours) or "none yet")
    logger.info("=" * 60)

    # Set leverage
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
            # 1. Fetch 1-minute candles
            candles = await exchange.get_candles(symbol, interval, candle_limit)
            if len(candles) < 25:
                logger.warning("Not enough candles (%d). Waiting...", len(candles))
                await asyncio.sleep(loop_interval)
                continue

            # 2. Compute indicators
            indicators = compute_scalp_indicators(candles, strategy_cfg)
            if not indicators:
                logger.warning("Indicator computation failed. Skipping.")
                await asyncio.sleep(loop_interval)
                continue

            # 3. Get live price for accurate paper trading
            live_price = await exchange.get_price(symbol)
            indicators["close"] = live_price

            # v2: Feed ATR to executor for dynamic stops
            if indicators.get("atr", 0) > 0:
                executor.update_atr(indicators["atr"])

            # 4. Check SL/TP on existing positions
            closed_trades = await executor.check_stops(live_price)
            for ct in closed_trades:
                logger.info("  Position closed: %s %s PnL=$%.2f",
                            ct["direction"], ct["signal"], ct["pnl"])

            # 5. Generate scalp signals (can produce both long + short)
            signals = strategy.evaluate(indicators, executor.has_long, executor.has_short)

            # 6. Execute signals
            executed = await executor.execute_signals(signals, live_price, symbol)

            # Log cycle info
            pos_status = executor.get_position_status()
            long_str = f"LONG@{pos_status['long']['entry']:.2f}" if pos_status["long"]["active"] else "---"
            short_str = f"SHORT@{pos_status['short']['entry']:.2f}" if pos_status["short"]["active"] else "---"

            log_signals = [s.value for s in signals.signals if s != ScalpSignal.HOLD]
            signal_str = ", ".join(log_signals) if log_signals else "HOLD"

            logger.info(
                "[%d] Price=%.2f | BB=[%.2f / %.2f / %.2f] w=%.2f%% | RSI=%.1f | "
                "ADX=%.0f | Trend=%s | ATR=%.2f | Pos: %s | %s | Signal: %s",
                cycle, live_price,
                indicators["bb_lower"], indicators["bb_middle"], indicators["bb_upper"],
                indicators["bb_width_pct"], indicators["rsi"],
                indicators.get("adx", 0), signals.trend,
                indicators.get("atr", 0),
                long_str, short_str, signal_str,
            )

            # Print summary every 10 cycles
            if cycle % 10 == 0:
                summary = executor.get_trade_summary()
                logger.info(
                    "--- Summary: %d trades | PnL=$%.2f (session=$%.2f) | WR=%.0f%% | "
                    "Avg hold=%ds | Consec losses=%d | L_streak=%d S_streak=%d ---",
                    summary.get("closed_trades", 0),
                    summary.get("total_pnl", 0),
                    summary.get("session_pnl", 0),
                    summary.get("win_rate", 0),
                    summary.get("avg_hold_sec", 0),
                    summary.get("consecutive_losses", 0),
                    summary.get("long_consec_losses", 0),
                    summary.get("short_consec_losses", 0),
                )

            # Auto-learner: periodic review and parameter adaptation
            if learner and learner.should_review():
                overrides = learner.review_and_adapt(config)
                if overrides:
                    executor.apply_learner_overrides(overrides)

        except Exception as e:
            logger.error("Error in cycle %d: %s", cycle, e, exc_info=True)

        await asyncio.sleep(loop_interval)


def main():
    parser = argparse.ArgumentParser(description="Gold Scalper Bot — BB Mean Reversion")
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
