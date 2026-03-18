"""BTC micro-trading strategy — multi-signal weighted scorer.

Unlike the gold scalper (pure BB mean-reversion), this uses a composite
of 6 micro-signals each contributing a weighted score. When the aggregate
score crosses a threshold, we enter a micro trade with tight TP/SL.

Signals:
1. RSI extreme bounce    — RSI dipping into oversold/overbought then reversing
2. VWAP deviation        — price stretched away from VWAP snapping back
3. EMA micro-cross       — fast EMA crossing slow EMA on 1m
4. Volume spike          — sudden volume surge confirming direction
5. MACD micro-momentum   — histogram flipping or accelerating
6. BB squeeze breakout   — BB contracting then price breaking out

Each signal scores 0.0–1.0, multiplied by its weight. Total score
determines entry direction and confidence.
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

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return (f"MicroResult(signals={sigs}, conf={self.confidence:.2f}, "
                f"L={self.long_score:.2f}, S={self.short_score:.2f})")


class MicroScalper:
    """Multi-signal BTC micro-trade strategy on 1m candles."""

    def __init__(self, config: dict):
        # Signal weights (must sum to ~1.0 for normalized scoring)
        self.w_rsi = config.get("weight_rsi", 0.20)
        self.w_vwap = config.get("weight_vwap", 0.15)
        self.w_ema = config.get("weight_ema", 0.15)
        self.w_volume = config.get("weight_volume", 0.15)
        self.w_macd = config.get("weight_macd", 0.20)
        self.w_bb = config.get("weight_bb", 0.15)

        # Thresholds
        self.entry_threshold = config.get("entry_threshold", 0.55)
        self.exit_threshold = config.get("exit_threshold", 0.30)

        # RSI params
        self.rsi_oversold = config.get("rsi_oversold", 30.0)
        self.rsi_overbought = config.get("rsi_overbought", 70.0)

        # VWAP deviation threshold (% from VWAP to consider stretched)
        self.vwap_dev_pct = config.get("vwap_deviation_pct", 0.15)

        # Volume spike multiplier
        self.volume_spike_mult = config.get("volume_spike_mult", 1.5)

        # BB squeeze detection
        self.bb_squeeze_width_pct = config.get("bb_squeeze_width_pct", 0.30)
        self.min_bb_width_pct = config.get("min_bb_width_pct", 0.10)

    def evaluate(
        self, indicators: dict, has_long: bool, has_short: bool
    ) -> MicroResult:
        """Score all micro-signals and decide entry/exit."""
        result = MicroResult()
        price = indicators.get("close", 0)

        # Guard NaN
        for key in ["close", "rsi", "vwap", "ema_fast", "ema_slow",
                     "macd_histogram", "bb_upper", "bb_lower", "bb_middle"]:
            val = indicators.get(key, 0)
            if isinstance(val, float) and math.isnan(val):
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(f"NaN in {key}")
                return result

        if price <= 0:
            result.signals.append(MicroSignal.HOLD)
            return result

        # --- Compute individual signal scores ---
        long_scores = {}
        short_scores = {}

        # 1. RSI extreme bounce
        l, s = self._score_rsi(indicators)
        long_scores["rsi"] = l * self.w_rsi
        short_scores["rsi"] = s * self.w_rsi

        # 2. VWAP deviation (mean-reversion toward VWAP)
        l, s = self._score_vwap(indicators)
        long_scores["vwap"] = l * self.w_vwap
        short_scores["vwap"] = s * self.w_vwap

        # 3. EMA micro-cross
        l, s = self._score_ema(indicators)
        long_scores["ema"] = l * self.w_ema
        short_scores["ema"] = s * self.w_ema

        # 4. Volume spike
        l, s = self._score_volume(indicators)
        long_scores["volume"] = l * self.w_volume
        short_scores["volume"] = s * self.w_volume

        # 5. MACD micro-momentum
        l, s = self._score_macd(indicators)
        long_scores["macd"] = l * self.w_macd
        short_scores["macd"] = s * self.w_macd

        # 6. BB position
        l, s = self._score_bb(indicators)
        long_scores["bb"] = l * self.w_bb
        short_scores["bb"] = s * self.w_bb

        total_long = sum(long_scores.values())
        total_short = sum(short_scores.values())
        result.long_score = round(total_long, 3)
        result.short_score = round(total_short, 3)

        # --- EXIT LOGIC ---
        if has_long:
            # Exit long if short score dominates or long score collapsed
            if total_short > self.entry_threshold or total_long < self.exit_threshold:
                result.signals.append(MicroSignal.CLOSE_LONG)
                result.reasons.append(
                    f"Close long: L={total_long:.2f} < {self.exit_threshold} "
                    f"or S={total_short:.2f} > {self.entry_threshold}"
                )

        if has_short:
            if total_long > self.entry_threshold or total_short < self.exit_threshold:
                result.signals.append(MicroSignal.CLOSE_SHORT)
                result.reasons.append(
                    f"Close short: S={total_short:.2f} < {self.exit_threshold} "
                    f"or L={total_long:.2f} > {self.entry_threshold}"
                )

        # --- ENTRY LOGIC ---
        bb_width_pct = indicators.get("bb_width_pct", 0)
        if bb_width_pct < self.min_bb_width_pct:
            if not result.signals:
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(
                    f"BB too tight ({bb_width_pct:.3f}% < {self.min_bb_width_pct}%)"
                )
            return result

        if not has_long and total_long >= self.entry_threshold:
            # Only enter long if long clearly beats short
            if total_long > total_short + 0.10:
                result.signals.append(MicroSignal.OPEN_LONG)
                top_signals = sorted(long_scores.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
                result.reasons.append(
                    f"Long entry: score={total_long:.2f} [{top_str}]"
                )
                result.confidence = total_long

        if not has_short and total_short >= self.entry_threshold:
            if total_short > total_long + 0.10:
                result.signals.append(MicroSignal.OPEN_SHORT)
                top_signals = sorted(short_scores.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
                result.reasons.append(
                    f"Short entry: score={total_short:.2f} [{top_str}]"
                )
                result.confidence = total_short

        if not result.signals:
            result.signals.append(MicroSignal.HOLD)

        return result

    # ------------------------------------------------------------------
    # Signal scorers — each returns (long_score, short_score) in [0, 1]
    # ------------------------------------------------------------------

    def _score_rsi(self, ind: dict) -> tuple[float, float]:
        """RSI extreme bounce detection."""
        rsi = ind.get("rsi", 50)
        rsi_prev = ind.get("rsi_prev", 50)
        long_score = 0.0
        short_score = 0.0

        # Long: RSI was oversold and now turning up
        if rsi < self.rsi_oversold:
            long_score = (self.rsi_oversold - rsi) / self.rsi_oversold
            if rsi > rsi_prev:  # Turning up — stronger signal
                long_score = min(1.0, long_score * 1.5)

        # Short: RSI was overbought and now turning down
        if rsi > self.rsi_overbought:
            short_score = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            if rsi < rsi_prev:  # Turning down
                short_score = min(1.0, short_score * 1.5)

        return long_score, short_score

    def _score_vwap(self, ind: dict) -> tuple[float, float]:
        """VWAP deviation — price far below VWAP = long, far above = short."""
        price = ind.get("close", 0)
        vwap = ind.get("vwap", 0)
        if vwap <= 0:
            return 0.0, 0.0

        dev_pct = (price - vwap) / vwap * 100
        threshold = self.vwap_dev_pct

        long_score = 0.0
        short_score = 0.0

        if dev_pct < -threshold * 0.5:
            # Price below VWAP — bullish reversion
            long_score = min(1.0, abs(dev_pct) / threshold)
        elif dev_pct > threshold * 0.5:
            # Price above VWAP — bearish reversion
            short_score = min(1.0, dev_pct / threshold)

        return long_score, short_score

    def _score_ema(self, ind: dict) -> tuple[float, float]:
        """EMA cross on micro timeframe."""
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        ema_fast_prev = ind.get("ema_fast_prev", 0)
        ema_slow_prev = ind.get("ema_slow_prev", 0)

        if ema_slow <= 0:
            return 0.0, 0.0

        # Current spread as % of price
        spread = (ema_fast - ema_slow) / ema_slow * 100
        prev_spread = (ema_fast_prev - ema_slow_prev) / ema_slow_prev * 100 if ema_slow_prev > 0 else 0

        long_score = 0.0
        short_score = 0.0

        # Bullish: fast crossing above slow, or spread widening upward
        if ema_fast > ema_slow:
            long_score = min(1.0, abs(spread) * 10)  # Scale small % to 0-1
            if prev_spread < 0:  # Fresh cross
                long_score = min(1.0, long_score * 1.5)
        elif ema_fast < ema_slow:
            short_score = min(1.0, abs(spread) * 10)
            if prev_spread > 0:  # Fresh bearish cross
                short_score = min(1.0, short_score * 1.5)

        return long_score, short_score

    def _score_volume(self, ind: dict) -> tuple[float, float]:
        """Volume spike confirmation — high volume confirms the direction of the candle."""
        volume_ratio = ind.get("volume_ratio", 1.0)
        price = ind.get("close", 0)
        ema_fast = ind.get("ema_fast", 0)

        if volume_ratio < self.volume_spike_mult:
            return 0.0, 0.0

        # Spike detected — score based on magnitude
        spike_score = min(1.0, (volume_ratio - 1.0) / (self.volume_spike_mult - 1.0))

        # Direction: if price > ema_fast, volume confirms bullish; else bearish
        if price > ema_fast:
            return spike_score, 0.0
        elif price < ema_fast:
            return 0.0, spike_score
        return 0.0, 0.0

    def _score_macd(self, ind: dict) -> tuple[float, float]:
        """MACD histogram micro-momentum."""
        hist = ind.get("macd_histogram", 0)
        hist_prev = ind.get("macd_histogram_prev", 0)

        if isinstance(hist, float) and math.isnan(hist):
            return 0.0, 0.0

        long_score = 0.0
        short_score = 0.0

        # Histogram positive and accelerating = bullish
        if hist > 0:
            long_score = 0.5
            if hist > hist_prev:  # Accelerating
                long_score = 0.8
        elif hist < 0:
            short_score = 0.5
            if hist < hist_prev:  # Accelerating bearish
                short_score = 0.8

        # Histogram flip is a strong signal
        if hist > 0 and hist_prev <= 0:
            long_score = 1.0
        elif hist < 0 and hist_prev >= 0:
            short_score = 1.0

        return long_score, short_score

    def _score_bb(self, ind: dict) -> tuple[float, float]:
        """Bollinger Band position — mean-reversion near bands."""
        price = ind.get("close", 0)
        bb_upper = ind.get("bb_upper", 0)
        bb_lower = ind.get("bb_lower", 0)
        bb_middle = ind.get("bb_middle", 0)

        if bb_upper <= bb_lower or bb_middle <= 0:
            return 0.0, 0.0

        band_width = bb_upper - bb_lower
        dist_to_lower = (price - bb_lower) / band_width
        dist_to_upper = (bb_upper - price) / band_width

        long_score = 0.0
        short_score = 0.0

        # Near lower band — bullish
        if dist_to_lower < 0.25:
            long_score = 1.0 - dist_to_lower / 0.25

        # Near upper band — bearish
        if dist_to_upper < 0.25:
            short_score = 1.0 - dist_to_upper / 0.25

        return long_score, short_score
