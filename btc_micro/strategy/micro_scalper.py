"""BTC micro-trading strategy — multi-signal weighted scorer.

v2: Lowered entry thresholds, added trend filter (EMA50+ADX),
    ATR volatility gate, RSI momentum confirmation, reduced
    signal-spread requirement so the bot actually trades.
v4: Trend dampening — penalise counter-trend mean-reversion signals
    (RSI, VWAP, BB) so the bot stops going long in downtrends.
    Stronger trend filter (ADX 22), higher score-spread (0.10),
    and smarter exit logic that holds winners longer.

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
    trend: str = "neutral"  # v2: "up", "down", "neutral"

    def __str__(self):
        sigs = [s.value for s in self.signals]
        return (f"MicroResult(signals={sigs}, conf={self.confidence:.2f}, "
                f"L={self.long_score:.2f}, S={self.short_score:.2f}, trend={self.trend})")


class MicroScalper:
    """Multi-signal BTC micro-trade strategy on 1m candles."""

    def __init__(self, config: dict):
        # Signal weights (must sum to ~1.0 for normalized scoring)
        self.w_rsi = config.get("weight_rsi", 0.20)
        self.w_vwap = config.get("weight_vwap", 0.15)
        self.w_ema = config.get("weight_ema", 0.15)
        self.w_volume = config.get("weight_volume", 0.10)
        self.w_macd = config.get("weight_macd", 0.25)
        self.w_bb = config.get("weight_bb", 0.15)

        # v2: Lowered thresholds — the old 0.55 was nearly impossible to reach
        self.entry_threshold = config.get("entry_threshold", 0.38)
        self.exit_threshold = config.get("exit_threshold", 0.20)

        # v2: Reduced spread requirement — old 0.10 was too restrictive
        self.min_score_spread = config.get("min_score_spread", 0.05)

        # RSI params
        self.rsi_oversold = config.get("rsi_oversold", 35.0)
        self.rsi_overbought = config.get("rsi_overbought", 65.0)

        # VWAP deviation threshold (% from VWAP to consider stretched)
        self.vwap_dev_pct = config.get("vwap_deviation_pct", 0.10)

        # Volume spike multiplier
        self.volume_spike_mult = config.get("volume_spike_mult", 1.3)

        # BB squeeze detection
        self.bb_squeeze_width_pct = config.get("bb_squeeze_width_pct", 0.30)
        self.min_bb_width_pct = config.get("min_bb_width_pct", 0.08)

        # v2: Trend filter
        self.use_trend_filter = config.get("use_trend_filter", True)
        self.adx_range_threshold = config.get("adx_range_threshold", 25.0)
        self.adx_trend_threshold = config.get("adx_trend_threshold", 30.0)
        self.max_atr_pct = config.get("max_atr_pct", 0.3)

    def _detect_trend(self, indicators: dict) -> str:
        """Determine trend using EMA50 and EMA crossover."""
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

    def evaluate(
        self, indicators: dict, has_long: bool, has_short: bool
    ) -> MicroResult:
        """Score all micro-signals and decide entry/exit."""
        result = MicroResult()
        price = indicators.get("close", 0)
        adx = indicators.get("adx", 0)
        atr = indicators.get("atr", 0)

        # v2: Detect trend
        trend = self._detect_trend(indicators)
        result.trend = trend

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

        # v2: Volatility gate — skip when ATR is extreme
        atr_pct = (atr / price * 100) if price > 0 and atr > 0 else 0
        if atr_pct > self.max_atr_pct:
            if not has_long and not has_short:
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(f"ATR {atr_pct:.2f}% > {self.max_atr_pct}% — too volatile")
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

        # v4: Trend dampening — penalise counter-trend mean-reversion signals.
        # In a downtrend, RSI/VWAP/BB produce strong "buy the dip" long scores
        # that cause the bot to catch falling knives. Dampen them heavily.
        if trend == "down":
            # Dampen long mean-reversion signals (keep EMA/MACD/volume intact)
            dampen = 0.35  # keep only 35% of counter-trend score
            for key in ("rsi", "vwap", "bb"):
                if key in long_scores and long_scores[key] > 0:
                    reduction = long_scores[key] * (1 - dampen)
                    long_scores[key] *= dampen
                    total_long -= reduction
            # Boost short scores slightly when trend confirms
            total_short *= 1.10
        elif trend == "up":
            dampen = 0.35
            for key in ("rsi", "vwap", "bb"):
                if key in short_scores and short_scores[key] > 0:
                    reduction = short_scores[key] * (1 - dampen)
                    short_scores[key] *= dampen
                    total_short -= reduction
            total_long *= 1.10

        total_long = max(0.0, total_long)
        total_short = max(0.0, total_short)
        result.long_score = round(total_long, 3)
        result.short_score = round(total_short, 3)

        # --- EXIT LOGIC (always allow exits) ---
        # v4: Require stronger opposing signal to exit a winning position.
        # Old logic exited at entry_threshold (0.38) — too early, cut winners short.
        # Now require opposing score to beat current direction + spread (signal flip).
        exit_flip_threshold = self.entry_threshold + self.min_score_spread  # ~0.48

        if has_long:
            # Exit if long score collapsed OR short score dominates
            if total_long < self.exit_threshold or total_short > exit_flip_threshold:
                result.signals.append(MicroSignal.CLOSE_LONG)
                result.reasons.append(
                    f"Close long: L={total_long:.2f} < {self.exit_threshold} "
                    f"or S={total_short:.2f} > {exit_flip_threshold:.2f}"
                )

        if has_short:
            if total_short < self.exit_threshold or total_long > exit_flip_threshold:
                result.signals.append(MicroSignal.CLOSE_SHORT)
                result.reasons.append(
                    f"Close short: S={total_short:.2f} < {self.exit_threshold} "
                    f"or L={total_long:.2f} > {exit_flip_threshold:.2f}"
                )

        # --- ENTRY FILTERS ---
        bb_width_pct = indicators.get("bb_width_pct", 0)
        if bb_width_pct < self.min_bb_width_pct:
            if not result.signals:
                result.signals.append(MicroSignal.HOLD)
                result.reasons.append(
                    f"BB too tight ({bb_width_pct:.3f}% < {self.min_bb_width_pct}%)"
                )
            return result

        # v4: Trend filter — block counter-trend entries more aggressively.
        # Old ADX threshold of 30 missed most downtrends (ADX 20-28 = unfiltered).
        # Now block at adx_trend_threshold (22) AND also block when EMA trend is
        # clearly established even at lower ADX.
        allow_long = True
        allow_short = True
        if self.use_trend_filter:
            if adx > self.adx_trend_threshold:
                if trend == "down":
                    allow_long = False
                elif trend == "up":
                    allow_short = False
            # v4: Even in "ranging" ADX, if EMA trend is clear, block counter-trend
            elif adx > self.adx_range_threshold and trend != "neutral":
                if trend == "down":
                    allow_long = False
                elif trend == "up":
                    allow_short = False

        # --- ENTRY LOGIC ---
        if not has_long and allow_long and total_long >= self.entry_threshold:
            if total_long > total_short + self.min_score_spread:
                # v2: Trend bonus for confidence
                conf = total_long
                if trend == "up":
                    conf += 0.1
                result.signals.append(MicroSignal.OPEN_LONG)
                top_signals = sorted(long_scores.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
                result.reasons.append(
                    f"Long entry: score={total_long:.2f} [{top_str}] trend={trend} ADX={adx:.0f}"
                )
                result.confidence = conf

        if not has_short and allow_short and total_short >= self.entry_threshold:
            if total_short > total_long + self.min_score_spread:
                conf = total_short
                if trend == "down":
                    conf += 0.1
                result.signals.append(MicroSignal.OPEN_SHORT)
                top_signals = sorted(short_scores.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
                result.reasons.append(
                    f"Short entry: score={total_short:.2f} [{top_str}] trend={trend} ADX={adx:.0f}"
                )
                result.confidence = conf

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

        # Long: RSI oversold
        if rsi < self.rsi_oversold:
            long_score = (self.rsi_oversold - rsi) / self.rsi_oversold
            if rsi > rsi_prev:  # Turning up — stronger
                long_score = min(1.0, long_score * 1.5)
        # v2: Partial credit for approaching oversold
        elif rsi < 40:
            long_score = (40 - rsi) / 20.0  # 0 at 40, 0.5 at 30

        # Short: RSI overbought
        if rsi > self.rsi_overbought:
            short_score = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            if rsi < rsi_prev:  # Turning down
                short_score = min(1.0, short_score * 1.5)
        elif rsi > 60:
            short_score = (rsi - 60) / 20.0

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

        # v2: Start scoring at 30% of threshold instead of 50%
        if dev_pct < -threshold * 0.3:
            long_score = min(1.0, abs(dev_pct) / threshold)
        elif dev_pct > threshold * 0.3:
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

        spread = (ema_fast - ema_slow) / ema_slow * 100
        prev_spread = (ema_fast_prev - ema_slow_prev) / ema_slow_prev * 100 if ema_slow_prev > 0 else 0

        long_score = 0.0
        short_score = 0.0

        if ema_fast > ema_slow:
            long_score = min(1.0, abs(spread) * 15)  # v2: more sensitive (was 10)
            if prev_spread < 0:  # Fresh cross
                long_score = min(1.0, long_score * 1.5)
        elif ema_fast < ema_slow:
            short_score = min(1.0, abs(spread) * 15)
            if prev_spread > 0:
                short_score = min(1.0, short_score * 1.5)

        return long_score, short_score

    def _score_volume(self, ind: dict) -> tuple[float, float]:
        """Volume spike confirmation."""
        volume_ratio = ind.get("volume_ratio", 1.0)
        price = ind.get("close", 0)
        ema_fast = ind.get("ema_fast", 0)

        if volume_ratio < self.volume_spike_mult:
            return 0.0, 0.0

        spike_score = min(1.0, (volume_ratio - 1.0) / (self.volume_spike_mult - 1.0))

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

        # v2: Wider zone (was 0.25, now 0.35) for more signals
        if dist_to_lower < 0.35:
            long_score = 1.0 - dist_to_lower / 0.35

        if dist_to_upper < 0.35:
            short_score = 1.0 - dist_to_upper / 0.35

        return long_score, short_score
