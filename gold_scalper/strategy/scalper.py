"""Bollinger Band mean-reversion scalping strategy for gold.

Strategy logic:
- LONG when price dips to/below lower Bollinger Band + RSI confirms oversold
- SHORT when price spikes to/above upper Bollinger Band + RSI confirms overbought
- Both long and short can be active simultaneously (dual-position mode)
- Quick profit targets at mid-band or fixed % — whichever is tighter
- Tight stops to limit downside on each micro-trade
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScalpSignal(Enum):
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


@dataclass
class ScalpResult:
    signals: list[ScalpSignal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return f"ScalpResult(signals={sigs}, conf={self.confidence:.2f}, reasons={self.reasons})"


class BollingerScalper:
    """Bollinger Band + RSI mean-reversion scalper.

    Designed for low-volatility assets like gold with tight spreads.
    Can generate both long AND short signals simultaneously.
    """

    def __init__(self, config: dict):
        self.bb_period: int = config.get("bb_period", 20)
        self.bb_std: float = config.get("bb_std", 2.0)
        self.rsi_period: int = config.get("rsi_period", 14)
        self.rsi_oversold: float = config.get("rsi_oversold", 35.0)
        self.rsi_overbought: float = config.get("rsi_overbought", 65.0)
        self.rsi_exit_long: float = config.get("rsi_exit_long", 55.0)
        self.rsi_exit_short: float = config.get("rsi_exit_short", 45.0)
        # How close to the band (as % of band width) to trigger entry
        self.band_touch_pct: float = config.get("band_touch_pct", 0.15)
        # Use EMA trend filter — only long in uptrend, short in downtrend, or both in range
        self.use_trend_filter: bool = config.get("use_trend_filter", False)

    def evaluate(self, indicators: dict, has_long: bool, has_short: bool) -> ScalpResult:
        """Evaluate current market state and generate scalp signals.

        Can return multiple signals (e.g., OPEN_LONG + CLOSE_SHORT) in one call.

        Args:
            indicators: Dict with bb_upper, bb_middle, bb_lower, rsi, close, etc.
            has_long: Whether we currently hold a long position.
            has_short: Whether we currently hold a short position.
        """
        result = ScalpResult()
        price = indicators.get("close", 0)
        bb_upper = indicators.get("bb_upper", 0)
        bb_middle = indicators.get("bb_middle", 0)
        bb_lower = indicators.get("bb_lower", 0)
        rsi = indicators.get("rsi", 50)

        # Guard against NaN
        for val in [price, bb_upper, bb_middle, bb_lower, rsi]:
            if isinstance(val, float) and math.isnan(val):
                result.signals.append(ScalpSignal.HOLD)
                result.reasons.append("NaN in indicators")
                return result

        if bb_upper == 0 or bb_lower == 0 or bb_middle == 0:
            result.signals.append(ScalpSignal.HOLD)
            result.reasons.append("Bollinger Bands not ready")
            return result

        band_width = bb_upper - bb_lower
        if band_width <= 0:
            result.signals.append(ScalpSignal.HOLD)
            result.reasons.append("Band width zero")
            return result

        # Distance from bands as % of band width
        dist_to_lower = (price - bb_lower) / band_width
        dist_to_upper = (bb_upper - price) / band_width

        # --- EXIT LOGIC (check first) ---

        # Close long: price reached mid-band or RSI no longer favorable
        if has_long:
            if price >= bb_middle:
                result.signals.append(ScalpSignal.CLOSE_LONG)
                result.reasons.append(f"Long TP: price {price:.2f} >= mid-band {bb_middle:.2f}")
            elif rsi >= self.rsi_exit_long:
                result.signals.append(ScalpSignal.CLOSE_LONG)
                result.reasons.append(f"Long exit: RSI {rsi:.1f} >= {self.rsi_exit_long}")

        # Close short: price reached mid-band or RSI no longer favorable
        if has_short:
            if price <= bb_middle:
                result.signals.append(ScalpSignal.CLOSE_SHORT)
                result.reasons.append(f"Short TP: price {price:.2f} <= mid-band {bb_middle:.2f}")
            elif rsi <= self.rsi_exit_short:
                result.signals.append(ScalpSignal.CLOSE_SHORT)
                result.reasons.append(f"Short exit: RSI {rsi:.1f} <= {self.rsi_exit_short}")

        # --- ENTRY LOGIC ---

        # Open long: price near/below lower band + RSI oversold
        if not has_long and dist_to_lower <= self.band_touch_pct:
            if rsi <= self.rsi_oversold:
                conf = self._calc_confidence(dist_to_lower, rsi, "long")
                result.signals.append(ScalpSignal.OPEN_LONG)
                result.reasons.append(
                    f"Long entry: price {price:.2f} near lower BB {bb_lower:.2f} "
                    f"(dist={dist_to_lower:.2%}), RSI={rsi:.1f}"
                )
                result.confidence = max(result.confidence, conf)

        # Open short: price near/above upper band + RSI overbought
        if not has_short and dist_to_upper <= self.band_touch_pct:
            if rsi >= self.rsi_overbought:
                conf = self._calc_confidence(dist_to_upper, rsi, "short")
                result.signals.append(ScalpSignal.OPEN_SHORT)
                result.reasons.append(
                    f"Short entry: price {price:.2f} near upper BB {bb_upper:.2f} "
                    f"(dist={dist_to_upper:.2%}), RSI={rsi:.1f}"
                )
                result.confidence = max(result.confidence, conf)

        # Default hold if no signals
        if not result.signals:
            result.signals.append(ScalpSignal.HOLD)

        return result

    def _calc_confidence(self, band_dist: float, rsi: float, side: str) -> float:
        """Calculate confidence 0-1 based on how extreme the setup is."""
        # Closer to band = higher confidence
        band_score = max(0, 1.0 - band_dist / self.band_touch_pct)

        # More extreme RSI = higher confidence
        if side == "long":
            rsi_score = max(0, (self.rsi_oversold - rsi) / self.rsi_oversold)
        else:
            rsi_score = max(0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))

        return 0.5 * band_score + 0.5 * rsi_score + 0.3  # base 0.3
