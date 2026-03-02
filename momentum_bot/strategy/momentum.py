"""Momentum strategy engine — generates buy/sell/hold signals from indicators."""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class SignalResult:
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reasons: list[str]

    def __str__(self) -> str:
        return f"{self.signal.value} (conf={self.confidence:.0%}) — {', '.join(self.reasons)}"


class MomentumStrategy:
    """Configurable momentum strategy using EMA, RSI, MACD, Volume, and VWAP.

    Entry rules (all conditions scored, threshold determines trade):
        - Price above/below fast EMA (trend direction)
        - Fast EMA above/below slow EMA (trend confirmation)
        - Volume above average (conviction)
        - RSI in momentum zone (not exhausted)
        - MACD histogram increasing (acceleration)
        - Price above/below VWAP (institutional bias)

    Exit rules (any triggers close):
        - Price crosses EMA against position
        - RSI exits momentum zone
        - MACD histogram reversal
    """

    def __init__(self, config: dict):
        # Entry thresholds
        self.rsi_buy_min: float = config.get("rsi_buy_min", 50.0)
        self.rsi_buy_max: float = config.get("rsi_buy_max", 75.0)
        self.rsi_sell_min: float = config.get("rsi_sell_min", 25.0)
        self.rsi_sell_max: float = config.get("rsi_sell_max", 50.0)
        self.volume_threshold: float = config.get("volume_threshold", 1.3)
        self.min_confidence: float = config.get("min_confidence", 0.6)

        # Exit thresholds
        self.rsi_exit_long_below: float = config.get("rsi_exit_long_below", 45.0)
        self.rsi_exit_short_above: float = config.get("rsi_exit_short_above", 55.0)

        # Weight for each signal component (must sum to ~1.0)
        self.weights = config.get("weights", {
            "ema_trend": 0.20,
            "ema_cross": 0.15,
            "rsi_zone": 0.20,
            "macd_accel": 0.20,
            "volume": 0.15,
            "vwap": 0.10,
        })

    def evaluate(self, indicators: dict, current_position: str = "none") -> SignalResult:
        """Evaluate all momentum conditions and produce a signal.

        Args:
            indicators: Dict from TechnicalIndicators.compute_all().
            current_position: "buy" (long), "sell" (short), or "none".

        Returns:
            SignalResult with signal, confidence, and reasoning.
        """
        if not indicators:
            return SignalResult(Signal.HOLD, 0.0, ["Insufficient data"])

        import math
        for key in ["close", "ema_fast", "ema_slow", "rsi", "macd_histogram", "volume_ratio", "vwap"]:
            if key not in indicators or (isinstance(indicators[key], float) and math.isnan(indicators[key])):
                return SignalResult(Signal.HOLD, 0.0, [f"Missing indicator: {key}"])

        # Check exit conditions first if we have a position
        if current_position == "buy":
            exit_signal = self._check_exit_long(indicators)
            if exit_signal:
                return exit_signal
        elif current_position == "sell":
            exit_signal = self._check_exit_short(indicators)
            if exit_signal:
                return exit_signal

        # Score buy conditions
        buy_score, buy_reasons = self._score_buy(indicators)
        sell_score, sell_reasons = self._score_sell(indicators)

        # Only signal if we don't already have a position in that direction
        if buy_score >= self.min_confidence and current_position != "buy":
            return SignalResult(Signal.BUY, buy_score, buy_reasons)
        elif sell_score >= self.min_confidence and current_position != "sell":
            return SignalResult(Signal.SELL, sell_score, sell_reasons)

        return SignalResult(Signal.HOLD, max(buy_score, sell_score), ["No clear momentum signal"])

    def _score_buy(self, ind: dict) -> tuple[float, list[str]]:
        """Score bullish momentum conditions."""
        score = 0.0
        reasons = []
        w = self.weights

        # 1. Price above fast EMA
        if ind["close"] > ind["ema_fast"]:
            score += w["ema_trend"]
            reasons.append(f"Price above EMA-fast ({ind['close']:.2f} > {ind['ema_fast']:.2f})")

        # 2. Fast EMA above slow EMA (uptrend)
        if ind["ema_fast"] > ind["ema_slow"]:
            score += w["ema_cross"]
            reasons.append("Fast EMA > Slow EMA (uptrend)")
        # Bonus: EMA just crossed over
        if ind["ema_fast_prev"] <= ind["ema_slow_prev"] and ind["ema_fast"] > ind["ema_slow"]:
            score += 0.05
            reasons.append("Bullish EMA crossover just occurred")

        # 3. RSI in bullish momentum zone
        if self.rsi_buy_min <= ind["rsi"] <= self.rsi_buy_max:
            score += w["rsi_zone"]
            reasons.append(f"RSI in buy zone ({ind['rsi']:.1f})")

        # 4. MACD histogram positive and increasing
        if ind["macd_histogram"] > 0 and ind["macd_histogram"] > ind["macd_histogram_prev"]:
            score += w["macd_accel"]
            reasons.append("MACD histogram positive & increasing")
        elif ind["macd_histogram"] > 0:
            score += w["macd_accel"] * 0.5
            reasons.append("MACD histogram positive")

        # 5. Volume above average
        if ind["volume_ratio"] >= self.volume_threshold:
            score += w["volume"]
            reasons.append(f"Volume {ind['volume_ratio']:.1f}x above average")

        # 6. Price above VWAP
        if ind["close"] > ind["vwap"]:
            score += w["vwap"]
            reasons.append("Price above VWAP")

        return score, reasons

    def _score_sell(self, ind: dict) -> tuple[float, list[str]]:
        """Score bearish momentum conditions."""
        score = 0.0
        reasons = []
        w = self.weights

        if ind["close"] < ind["ema_fast"]:
            score += w["ema_trend"]
            reasons.append(f"Price below EMA-fast ({ind['close']:.2f} < {ind['ema_fast']:.2f})")

        if ind["ema_fast"] < ind["ema_slow"]:
            score += w["ema_cross"]
            reasons.append("Fast EMA < Slow EMA (downtrend)")
        if ind["ema_fast_prev"] >= ind["ema_slow_prev"] and ind["ema_fast"] < ind["ema_slow"]:
            score += 0.05
            reasons.append("Bearish EMA crossover just occurred")

        if self.rsi_sell_min <= ind["rsi"] <= self.rsi_sell_max:
            score += w["rsi_zone"]
            reasons.append(f"RSI in sell zone ({ind['rsi']:.1f})")

        if ind["macd_histogram"] < 0 and ind["macd_histogram"] < ind["macd_histogram_prev"]:
            score += w["macd_accel"]
            reasons.append("MACD histogram negative & decreasing")
        elif ind["macd_histogram"] < 0:
            score += w["macd_accel"] * 0.5
            reasons.append("MACD histogram negative")

        if ind["volume_ratio"] >= self.volume_threshold:
            score += w["volume"]
            reasons.append(f"Volume {ind['volume_ratio']:.1f}x above average")

        if ind["close"] < ind["vwap"]:
            score += w["vwap"]
            reasons.append("Price below VWAP")

        return score, reasons

    def _check_exit_long(self, ind: dict) -> SignalResult | None:
        """Check if we should exit a long position."""
        reasons = []

        if ind["close"] < ind["ema_slow"]:
            reasons.append("Price broke below slow EMA")
        if ind["rsi"] < self.rsi_exit_long_below:
            reasons.append(f"RSI dropped below {self.rsi_exit_long_below} ({ind['rsi']:.1f})")
        if ind["macd_histogram"] < 0 and ind["macd_histogram"] < ind["macd_histogram_prev"]:
            reasons.append("MACD histogram turned negative & decreasing")

        if reasons:
            return SignalResult(Signal.CLOSE_LONG, 0.8, reasons)
        return None

    def _check_exit_short(self, ind: dict) -> SignalResult | None:
        """Check if we should exit a short position."""
        reasons = []

        if ind["close"] > ind["ema_slow"]:
            reasons.append("Price broke above slow EMA")
        if ind["rsi"] > self.rsi_exit_short_above:
            reasons.append(f"RSI rose above {self.rsi_exit_short_above} ({ind['rsi']:.1f})")
        if ind["macd_histogram"] > 0 and ind["macd_histogram"] > ind["macd_histogram_prev"]:
            reasons.append("MACD histogram turned positive & increasing")

        if reasons:
            return SignalResult(Signal.CLOSE_SHORT, 0.8, reasons)
        return None
