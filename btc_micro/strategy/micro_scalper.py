"""BTC micro-trading strategy — proven pattern-based strategies.

v5: Complete rewrite. Replaces the weighted signal scorer with discrete,
rule-based strategies used by successful traders:

1. VWAP Reclaim    — Institutional mean-reversion when price reclaims VWAP
2. Fair Value Gap  — ICT/SMC 3-candle imbalance fill
3. Structure Break — Break of Structure (BOS) momentum continuation
4. EMA Pullback    — Trend pullback to dynamic support/resistance

Each strategy produces independent signals. The engine picks the highest
confidence signal when multiple strategies agree. No arbitrary scoring —
every entry has a specific market structure reason.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MicroSignal(Enum):
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


@dataclass
class MicroResult:
    signals: list[MicroSignal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    long_score: float = 0.0
    short_score: float = 0.0
    trend: str = "neutral"
    strategy: str = ""

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return (f"MicroResult(signals={sigs}, conf={self.confidence:.2f}, "
                f"strategy={self.strategy}, trend={self.trend})")


# ======================================================================
# Individual Strategy Detectors
# ======================================================================

class VWAPReclaim:
    """VWAP Reclaim strategy — institutional mean-reversion.

    Entry Rules:
    - LONG: Price was below VWAP, crosses back above with volume > 1.2x avg.
            Confirms institutions defending VWAP as support.
    - SHORT: Price was above VWAP, crosses back below with volume > 1.2x avg.
            Confirms institutions using VWAP as resistance.

    Exit: ATR-based TP/SL. TP = 1.5x ATR, SL = 1.0x ATR.
    Why it works: VWAP is the benchmark institutional traders use.
    Price reclaiming VWAP signals institutional order flow direction.
    """

    def __init__(self, config: dict):
        self.min_volume_ratio = config.get("vwap_min_volume_ratio", 1.2)
        self.min_deviation_pct = config.get("vwap_min_deviation_pct", 0.03)
        self.max_deviation_pct = config.get("vwap_max_deviation_pct", 0.15)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        """Returns signal dict or None."""
        price = ind.get("close", 0)
        vwap = ind.get("vwap", 0)
        volume_ratio = ind.get("volume_ratio", 1.0)
        prev_close = ind.get("prev_close", 0)

        if vwap <= 0 or price <= 0 or prev_close <= 0:
            return None

        dev_pct = abs(price - vwap) / vwap * 100

        # Must be near VWAP (not too stretched)
        if dev_pct > self.max_deviation_pct:
            return None

        # Need volume confirmation
        if volume_ratio < self.min_volume_ratio:
            return None

        # LONG: Price crossed from below VWAP to above
        if prev_close < vwap and price > vwap:
            # Confirm with EMA trend not strongly bearish
            ema_fast = ind.get("ema_fast", 0)
            ema_slow = ind.get("ema_slow", 0)
            if ema_fast > 0 and ema_slow > 0 and ema_fast < ema_slow * 0.998:
                return None  # Strong downtrend, skip

            confidence = min(0.9, 0.5 + (volume_ratio - 1.0) * 0.3)
            return {
                "direction": "long",
                "signal": MicroSignal.OPEN_LONG,
                "confidence": confidence,
                "strategy": "vwap_reclaim",
                "reason": f"VWAP reclaim long: price crossed above VWAP {vwap:.0f}, vol={volume_ratio:.1f}x",
            }

        # SHORT: Price crossed from above VWAP to below
        if prev_close > vwap and price < vwap:
            ema_fast = ind.get("ema_fast", 0)
            ema_slow = ind.get("ema_slow", 0)
            if ema_fast > 0 and ema_slow > 0 and ema_fast > ema_slow * 1.002:
                return None  # Strong uptrend, skip

            confidence = min(0.9, 0.5 + (volume_ratio - 1.0) * 0.3)
            return {
                "direction": "short",
                "signal": MicroSignal.OPEN_SHORT,
                "confidence": confidence,
                "strategy": "vwap_reclaim",
                "reason": f"VWAP reclaim short: price crossed below VWAP {vwap:.0f}, vol={volume_ratio:.1f}x",
            }

        return None


class FairValueGap:
    """Fair Value Gap (FVG) — ICT/SMC imbalance fill strategy.

    A Fair Value Gap is a 3-candle pattern where candle 2 creates a gap
    between candle 1's high and candle 3's low (bullish) or candle 1's
    low and candle 3's high (bearish).

    Entry Rules:
    - LONG: Bullish FVG detected (gap between candle1.high and candle3.low),
            price pulls back into the gap zone. Enter when price touches
            the gap's midpoint with trend confirmation.
    - SHORT: Bearish FVG detected, price rallies into gap zone.

    Exit: TP at opposite side of gap. SL at gap invalidation (below gap
    for bullish, above for bearish).

    Why it works: FVGs represent institutional order flow imbalance.
    Price tends to "fill" these gaps as unfilled orders get matched.
    """

    def __init__(self, config: dict):
        self.min_gap_pct = config.get("fvg_min_gap_pct", 0.03)
        self.max_gap_pct = config.get("fvg_max_gap_pct", 0.25)
        self.lookback = config.get("fvg_lookback", 15)

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        if price <= 0 or len(candles_data) < self.lookback + 3:
            return None

        # Scan recent candles for FVG formations
        recent = candles_data[-(self.lookback + 3):]
        best_signal = None
        best_distance = float("inf")

        for i in range(len(recent) - 2):
            c1 = recent[i]
            c2 = recent[i + 1]
            c3 = recent[i + 2]

            # Bullish FVG: candle3.low > candle1.high (gap up)
            if c3["low"] > c1["high"]:
                gap_size = c3["low"] - c1["high"]
                gap_pct = gap_size / price * 100
                if self.min_gap_pct <= gap_pct <= self.max_gap_pct:
                    gap_mid = c1["high"] + gap_size / 2
                    # Price must be near or in the gap (pullback to fill)
                    if c1["high"] <= price <= c3["low"] * 1.001:
                        distance = abs(price - gap_mid)
                        if distance < best_distance:
                            best_distance = distance
                            # Trend confirmation: recent close above ema
                            ema_fast = ind.get("ema_fast", 0)
                            if ema_fast > 0 and price < ema_fast * 0.997:
                                continue  # Counter-trend, skip
                            confidence = min(0.85, 0.5 + gap_pct * 2)
                            best_signal = {
                                "direction": "long",
                                "signal": MicroSignal.OPEN_LONG,
                                "confidence": confidence,
                                "strategy": "fvg",
                                "reason": f"Bullish FVG fill: gap {c1['high']:.0f}-{c3['low']:.0f} ({gap_pct:.2f}%)",
                                "fvg_low": c1["high"],
                                "fvg_high": c3["low"],
                            }

            # Bearish FVG: candle1.low > candle3.high (gap down)
            if c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                gap_pct = gap_size / price * 100
                if self.min_gap_pct <= gap_pct <= self.max_gap_pct:
                    gap_mid = c3["high"] + gap_size / 2
                    if c3["high"] * 0.999 <= price <= c1["low"]:
                        distance = abs(price - gap_mid)
                        if distance < best_distance:
                            best_distance = distance
                            ema_fast = ind.get("ema_fast", 0)
                            if ema_fast > 0 and price > ema_fast * 1.003:
                                continue
                            confidence = min(0.85, 0.5 + gap_pct * 2)
                            best_signal = {
                                "direction": "short",
                                "signal": MicroSignal.OPEN_SHORT,
                                "confidence": confidence,
                                "strategy": "fvg",
                                "reason": f"Bearish FVG fill: gap {c3['high']:.0f}-{c1['low']:.0f} ({gap_pct:.2f}%)",
                                "fvg_low": c3["high"],
                                "fvg_high": c1["low"],
                            }

        return best_signal


class StructureBreak:
    """Break of Structure (BOS) — momentum continuation.

    Detects when price breaks a recent swing high/low, signaling
    trend continuation. Based on ICT/SMC market structure concepts.

    Entry Rules:
    - LONG: Price breaks above the most recent swing high (higher high)
            after making a higher low. Volume confirms the break.
    - SHORT: Price breaks below recent swing low (lower low) after
            making a lower high.

    Exit: Trail stop below the broken level. TP = 2x the swing range.

    Why it works: Breaking structure means order flow has shifted.
    Institutional money drives these breaks, and momentum follows.
    """

    def __init__(self, config: dict):
        self.swing_lookback = config.get("bos_swing_lookback", 10)
        self.min_break_pct = config.get("bos_min_break_pct", 0.02)
        self.max_break_pct = config.get("bos_max_break_pct", 0.15)
        self.volume_confirm = config.get("bos_volume_confirm", 1.1)

    def _find_swing_highs(self, candles: list[dict], n: int = 3) -> list[dict]:
        """Find swing highs (local maxima with n candles on each side)."""
        swings = []
        for i in range(n, len(candles) - n):
            high = candles[i]["high"]
            is_swing = all(candles[i - j]["high"] <= high for j in range(1, n + 1))
            is_swing = is_swing and all(candles[i + j]["high"] <= high for j in range(1, n + 1))
            if is_swing:
                swings.append({"price": high, "index": i})
        return swings

    def _find_swing_lows(self, candles: list[dict], n: int = 3) -> list[dict]:
        """Find swing lows (local minima)."""
        swings = []
        for i in range(n, len(candles) - n):
            low = candles[i]["low"]
            is_swing = all(candles[i - j]["low"] >= low for j in range(1, n + 1))
            is_swing = is_swing and all(candles[i + j]["low"] >= low for j in range(1, n + 1))
            if is_swing:
                swings.append({"price": low, "index": i})
        return swings

    def evaluate(self, ind: dict, candles_data: list[dict]) -> dict | None:
        price = ind.get("close", 0)
        prev_close = ind.get("prev_close", 0)
        volume_ratio = ind.get("volume_ratio", 1.0)

        if price <= 0 or len(candles_data) < self.swing_lookback + 6:
            return None

        recent = candles_data[-(self.swing_lookback + 6):-1]  # Exclude current candle
        swing_highs = self._find_swing_highs(recent)
        swing_lows = self._find_swing_lows(recent)

        # LONG BOS: Current price breaks above most recent swing high
        if swing_highs:
            last_sh = swing_highs[-1]
            break_pct = (price - last_sh["price"]) / last_sh["price"] * 100

            if self.min_break_pct <= break_pct <= self.max_break_pct:
                if prev_close <= last_sh["price"]:  # Fresh break
                    if volume_ratio >= self.volume_confirm:
                        # Confirm higher low exists (structure intact)
                        if len(swing_lows) >= 2:
                            if swing_lows[-1]["price"] > swing_lows[-2]["price"]:
                                confidence = min(0.85, 0.5 + break_pct * 3 + (volume_ratio - 1) * 0.2)
                                return {
                                    "direction": "long",
                                    "signal": MicroSignal.OPEN_LONG,
                                    "confidence": confidence,
                                    "strategy": "bos",
                                    "reason": f"BOS long: broke swing high {last_sh['price']:.0f} by {break_pct:.2f}%, vol={volume_ratio:.1f}x",
                                    "broken_level": last_sh["price"],
                                }

        # SHORT BOS: Current price breaks below most recent swing low
        if swing_lows:
            last_sl = swing_lows[-1]
            break_pct = (last_sl["price"] - price) / last_sl["price"] * 100

            if self.min_break_pct <= break_pct <= self.max_break_pct:
                if prev_close >= last_sl["price"]:
                    if volume_ratio >= self.volume_confirm:
                        if len(swing_highs) >= 2:
                            if swing_highs[-1]["price"] < swing_highs[-2]["price"]:
                                confidence = min(0.85, 0.5 + break_pct * 3 + (volume_ratio - 1) * 0.2)
                                return {
                                    "direction": "short",
                                    "signal": MicroSignal.OPEN_SHORT,
                                    "confidence": confidence,
                                    "strategy": "bos",
                                    "reason": f"BOS short: broke swing low {last_sl['price']:.0f} by {break_pct:.2f}%, vol={volume_ratio:.1f}x",
                                    "broken_level": last_sl["price"],
                                }

        return None


class EMAPullback:
    """EMA Pullback — trend-following pullback to dynamic S/R.

    The most reliable scalping strategy: trade pullbacks to EMA in
    an established trend. Works because EMAs act as dynamic support/
    resistance that institutional algos monitor.

    Entry Rules:
    - LONG: Trend is up (EMA8 > EMA21 > EMA50). Price pulls back to
            touch EMA21 (within 0.02%) then bounces (current candle
            closes above EMA8). RSI is not overbought.
    - SHORT: Trend is down. Price rallies to touch EMA21 then rejects.

    Exit: TP = 1.5x distance from EMA21 to entry. SL = below EMA50.

    Why it works: In trends, the 21 EMA acts as institutional re-entry
    zone. Pullbacks to this level offer the best risk:reward for
    trend continuation.
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

        if price <= 0 or ema_fast <= 0 or ema_slow <= 0 or ema_trend <= 0:
            return None

        tolerance = ema_slow * (self.touch_tolerance_pct / 100)

        # LONG: Uptrend + pullback to EMA21 + bounce
        if ema_fast > ema_slow > ema_trend:
            # Price recently touched EMA21 (within last 3 candles)
            touched = False
            if len(candles_data) >= 4:
                for c in candles_data[-4:-1]:
                    if abs(c["low"] - ema_slow) <= tolerance or c["low"] <= ema_slow:
                        touched = True
                        break

            if touched and price > ema_fast and rsi < self.rsi_upper:
                # Current candle bounced above EMA8 = confirmation
                if prev_close <= ema_fast * 1.001:
                    confidence = 0.7
                    # Bonus: RSI showing oversold-ish in uptrend = strong
                    if rsi < 45:
                        confidence = 0.8
                    return {
                        "direction": "long",
                        "signal": MicroSignal.OPEN_LONG,
                        "confidence": confidence,
                        "strategy": "ema_pullback",
                        "reason": f"EMA pullback long: bounced off EMA21 {ema_slow:.0f}, RSI={rsi:.0f}",
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
                    confidence = 0.7
                    if rsi > 55:
                        confidence = 0.8
                    return {
                        "direction": "short",
                        "signal": MicroSignal.OPEN_SHORT,
                        "confidence": confidence,
                        "strategy": "ema_pullback",
                        "reason": f"EMA pullback short: rejected at EMA21 {ema_slow:.0f}, RSI={rsi:.0f}",
                    }

        return None


# ======================================================================
# Strategy Engine — orchestrates all strategies
# ======================================================================

class MicroScalper:
    """Runs all strategies and picks the best signal."""

    def __init__(self, config: dict):
        self.strategies = {
            "vwap_reclaim": VWAPReclaim(config),
            "fvg": FairValueGap(config),
            "bos": StructureBreak(config),
            "ema_pullback": EMAPullback(config),
        }

        # Which strategies are enabled
        self.enabled = {
            "vwap_reclaim": config.get("enable_vwap_reclaim", True),
            "fvg": config.get("enable_fvg", True),
            "bos": config.get("enable_bos", True),
            "ema_pullback": config.get("enable_ema_pullback", True),
        }

        # Minimum confidence to act on a signal
        self.min_confidence = config.get("min_confidence", 0.50)

        # ATR volatility gate
        self.max_atr_pct = config.get("max_atr_pct", 0.3)

        # Exit threshold — when trend weakens, close positions
        self.exit_rsi_long = config.get("exit_rsi_long", 75)
        self.exit_rsi_short = config.get("exit_rsi_short", 25)

        enabled_names = [k for k, v in self.enabled.items() if v]
        logger.info("MicroScalper v5 initialized | strategies: %s | min_conf=%.2f",
                     enabled_names, self.min_confidence)

    def evaluate(
        self, indicators: dict, has_long: bool, has_short: bool,
        candles_data: list[dict] | None = None,
    ) -> MicroResult:
        """Evaluate all strategies, pick the best signal."""
        result = MicroResult()
        price = indicators.get("close", 0)
        atr = indicators.get("atr", 0)
        rsi = indicators.get("rsi", 50)

        # Detect trend for logging
        result.trend = self._detect_trend(indicators)

        # Guard NaN
        for key in ["close", "rsi", "vwap", "ema_fast", "ema_slow"]:
            val = indicators.get(key, 0)
            if isinstance(val, float) and math.isnan(val):
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(f"NaN in {key}")
                return result

        if price <= 0:
            result.signals.append(MicroSignal.HOLD)
            return result

        # ATR volatility gate — skip entries when too volatile
        atr_pct = (atr / price * 100) if price > 0 and atr > 0 else 0
        entry_blocked_by_vol = atr_pct > self.max_atr_pct

        # --- EXIT LOGIC (always evaluate first) ---
        if has_long:
            should_exit = self._should_exit_long(indicators)
            if should_exit:
                result.signals.append(MicroSignal.CLOSE_LONG)
                result.reasons.append(should_exit)

        if has_short:
            should_exit = self._should_exit_short(indicators)
            if should_exit:
                result.signals.append(MicroSignal.CLOSE_SHORT)
                result.reasons.append(should_exit)

        # --- ENTRY LOGIC ---
        if entry_blocked_by_vol:
            if not result.signals:
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(f"ATR {atr_pct:.2f}% > {self.max_atr_pct}% — too volatile")
            return result

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

        # Pick the highest-confidence signal
        if candidates:
            # Filter: don't open a direction we already have
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

                # Set scores for logging compatibility
                if best["direction"] == "long":
                    result.long_score = best["confidence"]
                    result.short_score = 0.0
                else:
                    result.short_score = best["confidence"]
                    result.long_score = 0.0

        if not result.signals:
            result.signals.append(MicroSignal.HOLD)

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
        """Check if we should exit a long position based on market structure."""
        rsi = ind.get("rsi", 50)
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        price = ind.get("close", 0)

        # RSI overbought — momentum exhaustion
        if rsi > self.exit_rsi_long:
            return f"Exit long: RSI overbought {rsi:.0f} > {self.exit_rsi_long}"

        # EMA cross against position — trend reversal signal
        if ema_fast > 0 and ema_slow > 0:
            if ema_fast < ema_slow * 0.999 and price < ema_fast:
                return f"Exit long: EMA bearish cross, price below EMA8"

        return None

    def _should_exit_short(self, ind: dict) -> str | None:
        rsi = ind.get("rsi", 50)
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        price = ind.get("close", 0)

        if rsi < self.exit_rsi_short:
            return f"Exit short: RSI oversold {rsi:.0f} < {self.exit_rsi_short}"

        if ema_fast > 0 and ema_slow > 0:
            if ema_fast > ema_slow * 1.001 and price > ema_fast:
                return f"Exit short: EMA bullish cross, price above EMA8"

        return None
