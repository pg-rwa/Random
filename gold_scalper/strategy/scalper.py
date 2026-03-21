"""Bollinger Band mean-reversion scalping strategy for gold.

Strategy logic:
- LONG when price dips to/below lower Bollinger Band + RSI confirms oversold
- SHORT when price spikes to/above upper Bollinger Band + RSI confirms overbought
- Both long and short can be active simultaneously (dual-position mode)
- Quick profit targets at mid-band or fixed % — whichever is tighter
- Tight stops to limit downside on each micro-trade

v2 additions:
- EMA trend filter: skip counter-trend trades (no longs in downtrend, no shorts in uptrend)
- ADX regime: mean-reversion only in ranging markets (ADX < threshold)
- RSI divergence: require RSI to be turning (not just oversold/overbought)
- Volatility gate: skip entries when ATR is too high (choppy/news)
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
    trend: str = "neutral"  # "up", "down", "neutral"

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return f"ScalpResult(signals={sigs}, conf={self.confidence:.2f}, trend={self.trend}, reasons={self.reasons})"


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
        # Minimum BB width (%) to allow entries — below this, bands are too tight (breakout zone)
        self.min_bb_width_pct: float = config.get("min_bb_width_pct", 0.25)

        # --- v2: Trend filter ---
        self.use_trend_filter: bool = config.get("use_trend_filter", True)
        # ADX below this = ranging (mean-reversion friendly)
        self.adx_range_threshold: float = config.get("adx_range_threshold", 25.0)
        # ADX above this = strong trend (only trade with trend, never against)
        self.adx_trend_threshold: float = config.get("adx_trend_threshold", 30.0)
        # Max ATR as % of price — skip entries when volatility is extreme
        self.max_atr_pct: float = config.get("max_atr_pct", 0.5)
        # Require RSI to be turning (prev RSI more extreme than current)
        self.require_rsi_turn: bool = config.get("require_rsi_turn", True)

    def _detect_trend(self, indicators: dict) -> str:
        """Determine trend direction using EMA50 and EMA crossover."""
        price = indicators.get("close", 0)
        ema_fast = indicators.get("ema_fast", 0)
        ema_slow = indicators.get("ema_slow", 0)
        ema_trend = indicators.get("ema_trend", 0)  # EMA50

        if ema_trend == 0 or ema_fast == 0 or ema_slow == 0:
            return "neutral"

        # Strong trend: price on one side of EMA50 AND fast EMA confirms
        if price > ema_trend and ema_fast > ema_slow:
            return "up"
        elif price < ema_trend and ema_fast < ema_slow:
            return "down"
        return "neutral"

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
        rsi_prev = indicators.get("rsi_prev", rsi)
        adx = indicators.get("adx", 0)
        atr = indicators.get("atr", 0)

        # Detect trend for this cycle
        trend = self._detect_trend(indicators)
        result.trend = trend

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

        # --- EXIT LOGIC (check first — always allow exits) ---

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

        # --- ENTRY FILTERS ---

        # Skip entries when BB width is too narrow (breakout zone, not mean reversion)
        bb_width_pct = band_width / bb_middle * 100 if bb_middle > 0 else 0
        if bb_width_pct < self.min_bb_width_pct:
            if not result.signals:
                result.signals.append(ScalpSignal.HOLD)
                result.reasons.append(f"BB width {bb_width_pct:.2f}% < {self.min_bb_width_pct}% min")
            return result

        # v2: Volatility gate — skip entries when ATR is too high (news/chaos)
        atr_pct = (atr / price * 100) if price > 0 and atr > 0 else 0
        if atr_pct > self.max_atr_pct:
            if not result.signals:
                result.signals.append(ScalpSignal.HOLD)
                result.reasons.append(f"ATR {atr_pct:.2f}% > {self.max_atr_pct}% — too volatile")
            return result

        # v2: Determine what's allowed based on trend and ADX
        allow_long = True
        allow_short = True

        if self.use_trend_filter:
            # Strong trend (ADX > threshold): only trade WITH the trend
            if adx > self.adx_trend_threshold:
                if trend == "down":
                    allow_long = False  # Don't buy in a downtrend
                elif trend == "up":
                    allow_short = False  # Don't short in an uptrend

            # Moderate trend: be cautious with counter-trend trades
            elif adx > self.adx_range_threshold:
                if trend == "down":
                    allow_long = False  # Still no longs in downtrend
                elif trend == "up":
                    allow_short = False

        # --- ENTRY LOGIC ---

        # Open long: price near/below lower band + RSI oversold
        if not has_long and allow_long and dist_to_lower <= self.band_touch_pct:
            if rsi <= self.rsi_oversold:
                # v2: Require RSI to be turning up (not diving deeper)
                rsi_turning = not self.require_rsi_turn or rsi > rsi_prev
                if rsi_turning:
                    conf = self._calc_confidence(dist_to_lower, rsi, "long", adx, trend)
                    result.signals.append(ScalpSignal.OPEN_LONG)
                    result.reasons.append(
                        f"Long entry: price {price:.2f} near lower BB {bb_lower:.2f} "
                        f"(dist={dist_to_lower:.2%}), RSI={rsi:.1f}, trend={trend}, ADX={adx:.0f}"
                    )
                    result.confidence = max(result.confidence, conf)
                else:
                    if not result.signals:
                        result.reasons.append(
                            f"Long skip: RSI still falling ({rsi_prev:.1f}→{rsi:.1f})"
                        )

        # Open short: price near/above upper band + RSI overbought
        if not has_short and allow_short and dist_to_upper <= self.band_touch_pct:
            if rsi >= self.rsi_overbought:
                # v2: Require RSI to be turning down
                rsi_turning = not self.require_rsi_turn or rsi < rsi_prev
                if rsi_turning:
                    conf = self._calc_confidence(dist_to_upper, rsi, "short", adx, trend)
                    result.signals.append(ScalpSignal.OPEN_SHORT)
                    result.reasons.append(
                        f"Short entry: price {price:.2f} near upper BB {bb_upper:.2f} "
                        f"(dist={dist_to_upper:.2%}), RSI={rsi:.1f}, trend={trend}, ADX={adx:.0f}"
                    )
                    result.confidence = max(result.confidence, conf)
                else:
                    if not result.signals:
                        result.reasons.append(
                            f"Short skip: RSI still rising ({rsi_prev:.1f}→{rsi:.1f})"
                        )

        # Default hold if no signals
        if not result.signals:
            result.signals.append(ScalpSignal.HOLD)

        return result

    def _calc_confidence(self, band_dist: float, rsi: float, side: str,
                         adx: float = 0, trend: str = "neutral") -> float:
        """Calculate confidence 0-3 based on how extreme the setup is.

        v2: Boost confidence when trading with trend, penalize counter-trend.
        """
        # Closer to band = higher confidence
        band_score = max(0, 1.0 - band_dist / self.band_touch_pct)

        # More extreme RSI = higher confidence
        if side == "long":
            rsi_score = max(0, (self.rsi_oversold - rsi) / self.rsi_oversold)
        else:
            rsi_score = max(0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))

        base = 0.5 * band_score + 0.5 * rsi_score + 0.3  # base 0.3

        # v2: Trend alignment bonus/penalty
        if side == "long" and trend == "up":
            base += 0.3  # With-trend bonus
        elif side == "long" and trend == "down":
            base -= 0.2  # Counter-trend penalty
        elif side == "short" and trend == "down":
            base += 0.3
        elif side == "short" and trend == "up":
            base -= 0.2

        # v2: Low ADX (ranging) is ideal for mean-reversion
        if adx > 0 and adx < 20:
            base += 0.15  # Ranging market bonus

        return max(0.0, base)
