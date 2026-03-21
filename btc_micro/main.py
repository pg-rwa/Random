"""BTC Micro-Trading Bot — main entry point.

v2: Added ADX, EMA50 trend indicators, ATR passthrough to executor,
    faster loop interval, session drawdown tracking.

Runs a multi-signal weighted micro-scalper on BTC perpetual futures.
Trades every ~20 seconds with 5x leverage, tight TP/SL, dual long+short.
Self-learning module adapts parameters from trade history.

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
from btc_micro.learner.btc_learner import BTCLearner

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
    """Compute all indicators for the multi-signal micro strategy."""
    if len(candles) < 2:
        return {}

    closes = np.array([c.close for c in candles])
    ti = TechnicalIndicators

    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    rsi_period = config.get("rsi_period", 14)
    ema_fast_period = config.get("ema_fast", 8)
    ema_slow_period = config.get("ema_slow", 21)
    ema_trend_period = config.get("ema_trend", 50)  # v2: EMA50 for trend

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(closes, bb_period, bb_std)

    # RSI
    rsi_vals = ti.rsi(closes, rsi_period)

    # EMAs
    ema_fast = ti.ema(closes, ema_fast_period)
    ema_slow = ti.ema(closes, ema_slow_period)
    ema_trend = ti.ema(closes, ema_trend_period)  # v2

    # MACD
    macd_line, signal_line, histogram = ti.macd(closes, 12, 26, 9)

    # ATR
    atr_vals = ti.atr(candles, 14)

    # v2: ADX for trend strength
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
        "ema_trend": ema_trend[-1] if not np.isnan(ema_trend[-1]) else 0,  # v2
        "macd_line": macd_line[-1],
        "macd_signal": signal_line[-1],
        "macd_histogram": histogram[-1],
        "macd_histogram_prev": histogram[-2] if len(histogram) > 1 else 0,
        "atr": atr_vals[-1] if not np.isnan(atr_vals[-1]) else 0,
        "adx": adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0,  # v2
        "vwap": vwap_vals[-1],
        "volume": current_volume,
        "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1.0,
    }


async def run_micro_bot(config: dict, paper_trade: bool = True) -> None:
    """Main micro-trading loop."""
    # Initialize exchange
    secret_key = os.getenv("HL_SECRET_KEY", "")
    wallet = os.getenv("HL_WALLET_ADDRESS", "")
    testnet_env = os.getenv("HL_TESTNET")
    testnet = (testnet_env or "false").lower() == "true"
    if paper_trade:
        testnet = False  # Use mainnet prices for paper trading

    if not paper_trade and (not secret_key or not wallet):
        logger.error("HL_SECRET_KEY and HL_WALLET_ADDRESS required for live trading.")
        sys.exit(1)

    exchange = HyperliquidExchange(secret_key, wallet, testnet=testnet)

    symbol = config.get("symbol", "BTC")
    interval = config.get("interval", "1m")
    candle_limit = config.get("candle_limit", 100)
    loop_interval = config.get("loop_interval_seconds", 20)  # v2: faster default
    leverage = config.get("leverage", 5)

    strategy_cfg = config.get("strategy", {})
    executor_cfg = config.get("executor", {})
    learner_cfg = config.get("learner", {})

    # Initialize self-learner
    learner = None
    if learner_cfg.get("enabled", True):
        learner = BTCLearner(
            log_dir=learner_cfg.get("log_dir", "btc_trades"),
            lookback_days=learner_cfg.get("lookback_days", 7),
            review_every_n_trades=learner_cfg.get("review_every_n_trades", 10),
            min_trades_to_learn=learner_cfg.get("min_trades_to_learn", 15),
            learning_rate=learner_cfg.get("learning_rate", 0.25),
            enable_time_filter=learner_cfg.get("enable_time_filter", True),
            enable_direction_filter=learner_cfg.get("enable_direction_filter", True),
            enable_tp_sl_tuning=learner_cfg.get("enable_tp_sl_tuning", True),
            enable_confidence_gate=learner_cfg.get("enable_confidence_gate", True),
            enable_weight_tuning=learner_cfg.get("enable_weight_tuning", True),
            enable_volatility_regime=learner_cfg.get("enable_volatility_regime", True),
            enable_kill_switch=learner_cfg.get("enable_kill_switch", True),
            enable_atr_stop_tuning=learner_cfg.get("enable_atr_stop_tuning", True),
            kill_switch_loss_streak=learner_cfg.get("kill_switch_loss_streak", 6),
            kill_switch_drawdown_pct=learner_cfg.get("kill_switch_drawdown_pct", 1.5),
        )
        overrides = learner.review_and_adapt(config)
        logger.info("Learner initial review: %d overrides", len(overrides))

    strategy = MicroScalper(strategy_cfg)
    executor = MicroExecutor(
        exchange=exchange,
        paper_trade=paper_trade,
        tp_pct=executor_cfg.get("tp_pct", 0.08),
        sl_pct=executor_cfg.get("sl_pct", 0.06),
        position_size_usd=executor_cfg.get("position_size_usd", 40.0),
        leverage=leverage,
        max_consecutive_losses=executor_cfg.get("max_consecutive_losses", 4),
        cooldown_after_losses_sec=executor_cfg.get("cooldown_after_losses_sec", 120),
        daily_loss_limit_pct=executor_cfg.get("daily_loss_limit_pct", 2.0),
        direction_cooldown_sec=executor_cfg.get("direction_cooldown_sec", 45.0),
        max_trades_per_hour=executor_cfg.get("max_trades_per_hour", 30),
        learner=learner,
        # v2 params
        use_atr_stops=executor_cfg.get("use_atr_stops", True),
        atr_tp_mult=executor_cfg.get("atr_tp_mult", 1.5),
        atr_sl_mult=executor_cfg.get("atr_sl_mult", 1.0),
        trailing_stop_activate_pct=executor_cfg.get("trailing_stop_activate_pct", 0.05),
        trailing_stop_distance_pct=executor_cfg.get("trailing_stop_distance_pct", 0.03),
        session_drawdown_limit_pct=executor_cfg.get("session_drawdown_limit_pct", 3.0),
        per_direction_loss_limit=executor_cfg.get("per_direction_loss_limit", 3),
        per_direction_cooldown_sec=executor_cfg.get("per_direction_cooldown_sec", 120.0),
        scale_size_on_streak=executor_cfg.get("scale_size_on_streak", True),
    )

    # Apply initial learner overrides
    if learner and overrides:
        executor.apply_learner_overrides(overrides)

    logger.info("=" * 60)
    logger.info("BTC Micro-Trading Bot v2 Starting")
    logger.info("  Mode: %s", "PAPER" if paper_trade else "LIVE")
    logger.info("  Symbol: %s", symbol)
    logger.info("  Interval: %s | Loop: %ds", interval, loop_interval)
    logger.info("  Leverage: %dx", leverage)
    logger.info("  TP: %.2f%% | SL: %.2f%% | ATR stops: %s",
                executor_cfg.get("tp_pct", 0.08), executor_cfg.get("sl_pct", 0.06),
                executor_cfg.get("use_atr_stops", True))
    logger.info("  Position: $%.0f x %dx = $%.0f notional",
                executor_cfg.get("position_size_usd", 40), leverage,
                executor_cfg.get("position_size_usd", 40) * leverage)
    logger.info("  Dual mode: LONG + SHORT simultaneously")
    logger.info("  Max trades/hr: %d", executor_cfg.get("max_trades_per_hour", 30))
    logger.info("  Trailing stop: activate=%.2f%% trail=%.2f%%",
                executor_cfg.get("trailing_stop_activate_pct", 0.05),
                executor_cfg.get("trailing_stop_distance_pct", 0.03))
    logger.info("  Self-learner: %s", "ENABLED" if learner else "DISABLED")
    if learner:
        logger.info("    Review every: %d trades", learner_cfg.get("review_every_n_trades", 10))
        logger.info("    Kill switch: %s (streak=%d)",
                     "ON" if learner_cfg.get("enable_kill_switch", True) else "OFF",
                     learner_cfg.get("kill_switch_loss_streak", 6))
        logger.info("    Signal weight tuning: %s",
                     "ON" if learner_cfg.get("enable_weight_tuning", True) else "OFF")
        logger.info("    Volatility regime: %s",
                     "ON" if learner_cfg.get("enable_volatility_regime", True) else "OFF")
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

            # 2. Compute indicators (with v2 ADX + EMA trend)
            indicators = compute_micro_indicators(candles, strategy_cfg)
            if not indicators:
                logger.warning("Indicator computation failed. Skipping.")
                await asyncio.sleep(loop_interval)
                continue

            # 3. Get live price
            live_price = await exchange.get_price(symbol)
            indicators["close"] = live_price

            # 4. Check SL/TP on existing positions (with trailing stops)
            closed_trades = await executor.check_stops(live_price)
            for ct in closed_trades:
                logger.info("  Position closed: %s %s PnL=$%.2f",
                            ct["direction"], ct["signal"], ct["pnl"])

            # 5. Generate micro signals
            signals = strategy.evaluate(indicators, executor.has_long, executor.has_short)

            # 6. Execute signals (v2: pass ATR for dynamic stops)
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

            logger.info(
                "[%d] $%.2f | RSI=%.1f | VWAP=%.2f | EMA=%s | ADX=%.0f | "
                "L=%.2f S=%.2f | %s | %s | %s",
                cycle, live_price, indicators["rsi"], indicators["vwap"],
                "UP" if indicators["ema_fast"] > indicators["ema_slow"] else "DN",
                indicators.get("adx", 0),
                signals.long_score, signals.short_score,
                long_str, short_str, signal_str,
            )

            # Summary every 10 cycles
            if cycle % 10 == 0:
                summary = executor.get_trade_summary()
                logger.info(
                    "--- Summary: %d trades | PnL=$%.2f | Session=$%.2f | WR=%.0f%% | "
                    "Avg hold=%ds | Losses streak=%d | Trades/hr=%d | Size=$%.0f ---",
                    summary.get("closed_trades", 0),
                    summary.get("total_pnl", 0),
                    summary.get("session_pnl", 0),
                    summary.get("win_rate", 0),
                    summary.get("avg_hold_sec", 0),
                    summary.get("consecutive_losses", 0),
                    summary.get("trades_this_hour", 0),
                    summary.get("position_size_usd", 40),
                )

            # Auto-learner: periodic review
            if learner and learner.should_review():
                overrides = learner.review_and_adapt(config)
                if overrides:
                    executor.apply_learner_overrides(overrides)

        except Exception as e:
            logger.error("Error in cycle %d: %s", cycle, e, exc_info=True)

        await asyncio.sleep(loop_interval)


def main():
    parser = argparse.ArgumentParser(description="BTC Micro-Trading Bot v2")
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
