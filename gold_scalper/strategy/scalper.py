"""Gold scalping strategy — session-aware, pattern-based strategies.

v5: Complete rewrite. Replaces Bollinger Band mean-reversion with
proven gold-specific strategies used by successful XAU traders:

1. Asian Range Breakout  — Gold consolidates in Asia, London breaks it
2. Liquidity Sweep       — ICT/SMC stop-hunt at round numbers then reversal
3. Session VWAP Reclaim  — Institutional mean-reversion on VWAP cross
4. EMA Pullback          — Trend-following pullback to dynamic S/R

Gold is session-driven. Each strategy knows WHAT TIME IT IS and only
fires during its optimal session window. No more trading blind.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    trend: str = "neutral"
    strategy: str = ""

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return (f"ScalpResult(signals={sigs}, conf={self.confidence:.2f}, "
                f"strategy={self.strategy}, trend={self.trend})")


# ======================================================================
# Session helpers
# ======================================================================

def current_session() -> str:
    """Determine current trading session based on UTC time.

    Asian:  00:00 - 08:00 UTC (consolidation)
    London: 08:00 - 13:00 UTC (breakout)
    NY:     13:00 - 21:00 UTC (continuation/reversal)
    Late:   21:00 - 00:00 UTC (low volume)
    """
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 21:
        return "ny"
    return "late"


def is_round_number(price: float, increment: float = 5.0) -> bool:
    """Check if price is near a round number ($5 or $10 increment)."""
    return abs(price % increment) < 0.50 or abs(price % increment - increment) < 0.50


def nearest_round(price: float, increment: float = 5.0) -> float:
    """Return the nearest round number."""
    return round(price / increment) * increment


# ======================================================================
# Individual Strategy Detectors
# ======================================================================

class AsianRangeBreakout:
    """Asian Range Breakout — gold's #1 proven session strategy.

    Gold consolidates during Asian session (00:00-08:00 UTC) because
    London physical gold market is closed. When London opens (08:00 UTC),
    institutional orders break the Asian range with 65-70% directional accuracy.

    Entry Rules:
    - Track Asian session high/low from candle data
    - At London open (08:00-10:00 UTC), enter on breakout above/below range
    - Require candle body close outside range (not just a wick)
    - Volume must confirm the break (> 1.1x average)
    - EMA trend filter: only take breaks WITH the higher-timeframe trend

    Exit: SL at opposite side of Asian range. TP = 2x range width.
    """

    def __init__(self, config: dict):
        self.min_range_pct = config.get("asian_min_range_pct", 0.05)
        self.max_range_pct = config.get("asian_max_range_pct", 0.40)
        self.breakout_buffer_pct = config.get("asian_breakout_buffer_pct", 0.01)
        self.volume_confirm = config.get("asian_volume_confirm", 1.1)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        prev_close = ind.get("prev_close", 0)
        volume_ratio = ind.get("volume_ratio", 1.0)
        session = current_session()

        if price <= 0 or len(candles_data) < 20:
            return None

        # Only fire during London session (08:00-10:00 UTC window)
        if session != "london":
            return None

        hour = datetime.now(timezone.utc).hour
        if hour > 10:
            return None  # Past the breakout window

        # Find Asian range from candle data (candles with timestamps in 00:00-08:00 UTC)
        asian_high = 0
        asian_low = float("inf")
        asian_candle_count = 0

        for c in candles_data:
            ts = c.get("timestamp", 0)
            if ts > 0:
                try:
                    candle_hour = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts,
                                                         tz=timezone.utc).hour
                except (ValueError, OSError):
                    continue
                if 0 <= candle_hour < 8:
                    asian_high = max(asian_high, c["high"])
                    asian_low = min(asian_low, c["low"])
                    asian_candle_count += 1

        if asian_candle_count < 5 or asian_low >= asian_high:
            return None

        range_size = asian_high - asian_low
        range_pct = range_size / price * 100

        if range_pct < self.min_range_pct or range_pct > self.max_range_pct:
            return None

        buffer = price * (self.breakout_buffer_pct / 100)

        # LONG: Price breaks above Asian high
        if price > asian_high + buffer and prev_close <= asian_high + buffer:
            if volume_ratio >= self.volume_confirm:
                # Trend confirmation
                ema_fast = ind.get("ema_fast", 0)
                ema_trend = ind.get("ema_trend", 0)
                if ema_trend > 0 and price < ema_trend * 0.998:
                    return None  # Against major trend

                confidence = min(0.90, 0.60 + range_pct * 1.5 + (volume_ratio - 1) * 0.2)
                return {
                    "direction": "long",
                    "signal": ScalpSignal.OPEN_LONG,
                    "confidence": confidence,
                    "strategy": "asian_breakout",
                    "reason": (f"Asian breakout long: broke {asian_high:.2f} "
                               f"(range {asian_low:.2f}-{asian_high:.2f}, {range_pct:.2f}%), "
                               f"vol={volume_ratio:.1f}x"),
                    "asian_high": asian_high,
                    "asian_low": asian_low,
                }

        # SHORT: Price breaks below Asian low
        if price < asian_low - buffer and prev_close >= asian_low - buffer:
            if volume_ratio >= self.volume_confirm:
                ema_fast = ind.get("ema_fast", 0)
                ema_trend = ind.get("ema_trend", 0)
                if ema_trend > 0 and price > ema_trend * 1.002:
                    return None

                confidence = min(0.90, 0.60 + range_pct * 1.5 + (volume_ratio - 1) * 0.2)
                return {
                    "direction": "short",
                    "signal": ScalpSignal.OPEN_SHORT,
                    "confidence": confidence,
                    "strategy": "asian_breakout",
                    "reason": (f"Asian breakout short: broke {asian_low:.2f} "
                               f"(range {asian_low:.2f}-{asian_high:.2f}, {range_pct:.2f}%), "
                               f"vol={volume_ratio:.1f}x"),
                    "asian_high": asian_high,
                    "asian_low": asian_low,
                }

        return None


class LiquiditySweep:
    """Liquidity Sweep + Reversal — gold's stop-hunting behavior.

    Gold consistently sweeps above/below round numbers ($5/$10 increments)
    to trigger retail stops, then reverses. This is institutional order flow.

    Entry Rules:
    - Price wicks through a round number ($5 increment: 4395, 4400, 4405)
    - Candle closes back on the other side (sweep, not breakout)
    - Volume confirms the sweep (institutions filling orders)
    - Swing structure confirms reversal direction

    Exit: SL beyond the sweep wick. TP = 1:2 R:R to opposing level.

    Best during: London and NY sessions (institutional activity).
    """

    def __init__(self, config: dict):
        self.round_increment = config.get("sweep_round_increment", 5.0)
        self.wick_min_pct = config.get("sweep_wick_min_pct", 0.02)
        self.lookback = config.get("sweep_lookback", 5)
        self.volume_confirm = config.get("sweep_volume_confirm", 1.15)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        session = current_session()

        if price <= 0 or len(candles_data) < self.lookback + 2:
            return None

        # Only during London and NY sessions
        if session not in ("london", "ny"):
            return None

        volume_ratio = ind.get("volume_ratio", 1.0)

        # Check recent candles for sweep pattern
        recent = candles_data[-(self.lookback + 1):]

        for i in range(len(recent) - 1):
            candle = recent[i]
            next_candle = recent[i + 1] if i + 1 < len(recent) else None
            if not next_candle:
                continue

            c_high = candle["high"]
            c_low = candle["low"]
            c_close = candle["close"]
            c_open = candle["open"]

            # Find nearest round number
            nearest_above = nearest_round(max(c_high, c_open, c_close), self.round_increment)
            nearest_below = nearest_round(min(c_low, c_open, c_close), self.round_increment)

            # Bullish sweep: wick below round number, close above it
            if c_low < nearest_below and c_close > nearest_below:
                sweep_depth = nearest_below - c_low
                sweep_pct = sweep_depth / price * 100

                if sweep_pct >= self.wick_min_pct:
                    # Next candle should confirm bullish (close higher)
                    if next_candle["close"] > c_close and volume_ratio >= self.volume_confirm:
                        # Current price must still be above the swept level
                        if price > nearest_below:
                            confidence = min(0.85, 0.55 + sweep_pct * 5 + (volume_ratio - 1) * 0.2)
                            return {
                                "direction": "long",
                                "signal": ScalpSignal.OPEN_LONG,
                                "confidence": confidence,
                                "strategy": "liquidity_sweep",
                                "reason": (f"Liquidity sweep long: swept below ${nearest_below:.0f} "
                                           f"(wick to {c_low:.2f}, {sweep_pct:.2f}%), "
                                           f"vol={volume_ratio:.1f}x"),
                                "swept_level": nearest_below,
                                "sweep_low": c_low,
                            }

            # Bearish sweep: wick above round number, close below it
            if c_high > nearest_above and c_close < nearest_above:
                sweep_depth = c_high - nearest_above
                sweep_pct = sweep_depth / price * 100

                if sweep_pct >= self.wick_min_pct:
                    if next_candle["close"] < c_close and volume_ratio >= self.volume_confirm:
                        if price < nearest_above:
                            confidence = min(0.85, 0.55 + sweep_pct * 5 + (volume_ratio - 1) * 0.2)
                            return {
                                "direction": "short",
                                "signal": ScalpSignal.OPEN_SHORT,
                                "confidence": confidence,
                                "strategy": "liquidity_sweep",
                                "reason": (f"Liquidity sweep short: swept above ${nearest_above:.0f} "
                                           f"(wick to {c_high:.2f}, {sweep_pct:.2f}%), "
                                           f"vol={volume_ratio:.1f}x"),
                                "swept_level": nearest_above,
                                "sweep_high": c_high,
                            }

        return None


class SessionVWAP:
    """Session VWAP Reclaim — institutional mean-reversion.

    Gold respects VWAP as an institutional execution benchmark even more
    than crypto. When price crosses VWAP with volume, it signals
    institutional order flow direction.

    Entry Rules:
    - Price crosses from below VWAP to above (long) or above to below (short)
    - Volume confirms (> 1.2x average)
    - Not during Asian session (low volume = unreliable VWAP)
    - EMA trend not strongly opposed

    Exit: ATR-based TP/SL.
    """

    def __init__(self, config: dict):
        self.min_volume_ratio = config.get("vwap_min_volume_ratio", 1.2)
        self.max_deviation_pct = config.get("vwap_max_deviation_pct", 0.12)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        vwap = ind.get("vwap", 0)
        prev_close = ind.get("prev_close", 0)
        volume_ratio = ind.get("volume_ratio", 1.0)
        session = current_session()

        if vwap <= 0 or price <= 0 or prev_close <= 0:
            return None

        # Skip Asian session — VWAP unreliable with low volume
        if session == "asian" or session == "late":
            return None

        dev_pct = abs(price - vwap) / vwap * 100
        if dev_pct > self.max_deviation_pct:
            return None

        if volume_ratio < self.min_volume_ratio:
            return None

        # LONG: crossed from below VWAP to above
        if prev_close < vwap and price > vwap:
            ema_fast = ind.get("ema_fast", 0)
            ema_slow = ind.get("ema_slow", 0)
            if ema_fast > 0 and ema_slow > 0 and ema_fast < ema_slow * 0.998:
                return None

            confidence = min(0.85, 0.50 + (volume_ratio - 1.0) * 0.3)
            return {
                "direction": "long",
                "signal": ScalpSignal.OPEN_LONG,
                "confidence": confidence,
                "strategy": "session_vwap",
                "reason": (f"VWAP reclaim long: crossed above {vwap:.2f}, "
                           f"session={session}, vol={volume_ratio:.1f}x"),
            }

        # SHORT: crossed from above VWAP to below
        if prev_close > vwap and price < vwap:
            ema_fast = ind.get("ema_fast", 0)
            ema_slow = ind.get("ema_slow", 0)
            if ema_fast > 0 and ema_slow > 0 and ema_fast > ema_slow * 1.002:
                return None

            confidence = min(0.85, 0.50 + (volume_ratio - 1.0) * 0.3)
            return {
                "direction": "short",
                "signal": ScalpSignal.OPEN_SHORT,
                "confidence": confidence,
                "strategy": "session_vwap",
                "reason": (f"VWAP reclaim short: crossed below {vwap:.2f}, "
                           f"session={session}, vol={volume_ratio:.1f}x"),
            }

        return None


class GoldEMAPullback:
    """EMA Pullback — trend-following pullback to dynamic S/R.

    Gold trends cleanly during London and NY because institutional flows
    are directional. EMA21 acts as the re-entry zone.

    Entry Rules:
    - LONG: Uptrend (EMA9 > EMA21 > EMA50). Price pulls back to EMA21,
            then bounces above EMA9. RSI not overbought.
    - SHORT: Mirror for downtrend.
    - Only during London and NY sessions.

    Exit: TP = 1.5x distance from EMA21 to entry. SL = below EMA50.
    """

    def __init__(self, config: dict):
        self.touch_tolerance_pct = config.get("ema_touch_tolerance_pct", 0.04)
        self.rsi_upper = config.get("ema_rsi_upper", 70)
        self.rsi_lower = config.get("ema_rsi_lower", 30)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        ema_trend = ind.get("ema_trend", 0)
        rsi = ind.get("rsi", 50)
        prev_close = ind.get("prev_close", 0)
        session = current_session()

        if price <= 0 or ema_fast <= 0 or ema_slow <= 0 or ema_trend <= 0:
            return None

        # Only during active sessions
        if session not in ("london", "ny"):
            return None

        tolerance = ema_slow * (self.touch_tolerance_pct / 100)

        # LONG: Uptrend + pullback to EMA21 + bounce
        if ema_fast > ema_slow > ema_trend:
            touched = False
            if len(candles_data) >= 4:
                for c in candles_data[-4:-1]:
                    if abs(c["low"] - ema_slow) <= tolerance or c["low"] <= ema_slow:
                        touched = True
                        break

            if touched and price > ema_fast and rsi < self.rsi_upper:
                if prev_close <= ema_fast * 1.001:
                    confidence = 0.70
                    if rsi < 45:
                        confidence = 0.80
                    return {
                        "direction": "long",
                        "signal": ScalpSignal.OPEN_LONG,
                        "confidence": confidence,
                        "strategy": "ema_pullback",
                        "reason": (f"EMA pullback long: bounced off EMA21 {ema_slow:.2f}, "
                                   f"RSI={rsi:.0f}, session={session}"),
                    }

        # SHORT: Downtrend + rally to EMA21 + rejection
        if ema_fast < ema_slow < ema_trend:
            touched = False
            if len(candles_data) >= 4:
                for c in candles_data[-4:-1]:
                    if abs(c["high"] - ema_slow) <= tolerance or c["high"] >= ema_slow:
                        touched = True
                        break

            if touched and price < ema_fast and rsi > self.rsi_lower:
                if prev_close >= ema_fast * 0.999:
                    confidence = 0.70
                    if rsi > 55:
                        confidence = 0.80
                    return {
                        "direction": "short",
                        "signal": ScalpSignal.OPEN_SHORT,
                        "confidence": confidence,
                        "strategy": "ema_pullback",
                        "reason": (f"EMA pullback short: rejected at EMA21 {ema_slow:.2f}, "
                                   f"RSI={rsi:.0f}, session={session}"),
                    }

        return None


# ======================================================================
# Strategy Engine
# ======================================================================

class BollingerScalper:
    """Session-aware gold strategy engine. Runs all strategies, picks the best."""

    def __init__(self, config: dict):
        self.strategies = {
            "asian_breakout": AsianRangeBreakout(config),
            "liquidity_sweep": LiquiditySweep(config),
            "session_vwap": SessionVWAP(config),
            "ema_pullback": GoldEMAPullback(config),
        }

        self.enabled = {
            "asian_breakout": config.get("enable_asian_breakout", True),
            "liquidity_sweep": config.get("enable_liquidity_sweep", True),
            "session_vwap": config.get("enable_session_vwap", True),
            "ema_pullback": config.get("enable_ema_pullback", True),
        }

        self.min_confidence = config.get("min_confidence", 0.50)
        self.max_atr_pct = config.get("max_atr_pct", 0.5)

        # Exit thresholds
        self.exit_rsi_long = config.get("exit_rsi_long", 75)
        self.exit_rsi_short = config.get("exit_rsi_short", 25)

        enabled_names = [k for k, v in self.enabled.items() if v]
        logger.info("Gold Strategy v5 initialized | strategies: %s | min_conf=%.2f",
                     enabled_names, self.min_confidence)

    def evaluate(
        self, indicators: dict, has_long: bool, has_short: bool,
        candles_data: list[dict] | None = None,
    ) -> ScalpResult:
        """Evaluate all strategies, pick the best signal."""
        result = ScalpResult()
        price = indicators.get("close", 0)
        atr = indicators.get("atr", 0)
        rsi = indicators.get("rsi", 50)

        result.trend = self._detect_trend(indicators)

        # Guard NaN
        for key in ["close", "rsi", "bb_upper", "bb_lower", "bb_middle"]:
            val = indicators.get(key, 0)
            if isinstance(val, float) and math.isnan(val):
                result.signals.append(ScalpSignal.HOLD)
                result.reasons.append(f"NaN in {key}")
                return result

        if price <= 0:
            result.signals.append(ScalpSignal.HOLD)
            return result

        # ATR volatility gate
        atr_pct = (atr / price * 100) if price > 0 and atr > 0 else 0
        entry_blocked = atr_pct > self.max_atr_pct

        # --- EXIT LOGIC ---
        if has_long:
            exit_reason = self._should_exit_long(indicators)
            if exit_reason:
                result.signals.append(ScalpSignal.CLOSE_LONG)
                result.reasons.append(exit_reason)

        if has_short:
            exit_reason = self._should_exit_short(indicators)
            if exit_reason:
                result.signals.append(ScalpSignal.CLOSE_SHORT)
                result.reasons.append(exit_reason)

        if entry_blocked:
            if not result.signals:
                result.signals.append(ScalpSignal.HOLD)
                result.reasons.append(f"ATR {atr_pct:.2f}% > {self.max_atr_pct}%")
            return result

        # --- ENTRY LOGIC ---
        candles = candles_data or []
        candidates = []

        for name, strategy in self.strategies.items():
            if not self.enabled.get(name, True):
                continue
            try:
                sig = strategy.evaluate(indicators, candles)
                if sig and sig.get("confidence", 0) >= self.min_confidence:
                    candidates.append(sig)
            except Exception as e:
                logger.warning("Strategy %s error: %s", name, e)

        if candidates:
            filtered = []
            for c in candidates:
                if c["direction"] == "long" and has_long:
                    continue
                if c["direction"] == "short" and has_short:
                    continue
                filtered.append(c)

            if filtered:
                best = max(filtered, key=lambda c: c["confidence"])
                result.signals.append(best["signal"])
                result.reasons.append(best["reason"])
                result.confidence = best["confidence"]
                result.strategy = best.get("strategy", "")

        if not result.signals:
            result.signals.append(ScalpSignal.HOLD)

        return result

    def _detect_trend(self, indicators: dict) -> str:
        price = indicators.get("close", 0)
        ema_fast = indicators.get("ema_fast", 0)
        ema_slow = indicators.get("ema_slow", 0)
        ema_trend = indicators.get("ema_trend", 0)

        if ema_trend == 0 or ema_fast == 0 or ema_slow == 0:
            return "neutral"

        if price > ema_trend and ema_fast > ema_slow:
            return "up"
        elif price < ema_trend and ema_fast < ema_slow:
            return "down"
        return "neutral"

    def _should_exit_long(self, ind: dict) -> str | None:
        rsi = ind.get("rsi", 50)
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        price = ind.get("close", 0)

        if rsi > self.exit_rsi_long:
            return f"Exit long: RSI {rsi:.0f} > {self.exit_rsi_long}"

        if ema_fast > 0 and ema_slow > 0:
            if ema_fast < ema_slow * 0.999 and price < ema_fast:
                return f"Exit long: EMA bearish cross, price below EMA9"

        return None

    def _should_exit_short(self, ind: dict) -> str | None:
        rsi = ind.get("rsi", 50)
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        price = ind.get("close", 0)

        if rsi < self.exit_rsi_short:
            return f"Exit short: RSI {rsi:.0f} < {self.exit_rsi_short}"

        if ema_fast > 0 and ema_slow > 0:
            if ema_fast > ema_slow * 1.001 and price > ema_fast:
                return f"Exit short: EMA bullish cross, price above EMA9"

        return None
