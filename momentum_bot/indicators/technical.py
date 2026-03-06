"""Technical indicator calculations using numpy for performance."""

import numpy as np

from momentum_bot.exchange.base import Candle


class TechnicalIndicators:
    """Stateless indicator calculator. Feed it candles, get back indicator values."""

    @staticmethod
    def ema(closes: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(closes)
        ema[0] = closes[0]
        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def sma(closes: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        sma = np.full_like(closes, np.nan)
        if len(closes) < period:
            return sma
        cumsum = np.cumsum(closes)
        sma[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0], cumsum[:-period]))) / period
        return sma

    @staticmethod
    def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.empty(len(closes))
        avg_loss = np.empty(len(closes))
        rsi = np.full(len(closes), np.nan)

        avg_gain[0] = np.nan
        avg_loss[0] = np.nan

        if len(gains) < period:
            return rsi

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        for i in range(period, len(closes)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def macd(
        closes: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD line, signal line, and histogram."""
        ti = TechnicalIndicators
        fast_ema = ti.ema(closes, fast_period)
        slow_ema = ti.ema(closes, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ti.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def atr(candles: list[Candle], period: int = 14) -> np.ndarray:
        """Average True Range — measures volatility."""
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        tr = np.empty(len(candles))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(candles)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def adx(candles: list[Candle], period: int = 14) -> np.ndarray:
        """Average Directional Index — measures trend strength (0-100).

        ADX > 25 indicates a strong trend; ADX < 20 indicates ranging/choppy.
        """
        n = len(candles)
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        adx_out = np.full(n, np.nan)
        if n < period * 2 + 1:
            return adx_out

        # True Range, +DM, -DM
        tr = np.empty(n)
        plus_dm = np.empty(n)
        minus_dm = np.empty(n)
        tr[0] = highs[0] - lows[0]
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0

        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

        # Smoothed TR, +DM, -DM using Wilder's method
        atr_s = np.mean(tr[1 : period + 1])
        plus_dm_s = np.mean(plus_dm[1 : period + 1])
        minus_dm_s = np.mean(minus_dm[1 : period + 1])

        dx_vals = []
        for i in range(period + 1, n):
            atr_s = atr_s - (atr_s / period) + tr[i]
            plus_dm_s = plus_dm_s - (plus_dm_s / period) + plus_dm[i]
            minus_dm_s = minus_dm_s - (minus_dm_s / period) + minus_dm[i]

            plus_di = 100 * plus_dm_s / atr_s if atr_s > 0 else 0
            minus_di = 100 * minus_dm_s / atr_s if atr_s > 0 else 0
            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
            dx_vals.append((i, dx))

        # ADX = smoothed DX over `period`
        if len(dx_vals) >= period:
            adx_val = np.mean([d[1] for d in dx_vals[:period]])
            adx_out[dx_vals[period - 1][0]] = adx_val
            for j in range(period, len(dx_vals)):
                adx_val = (adx_val * (period - 1) + dx_vals[j][1]) / period
                adx_out[dx_vals[j][0]] = adx_val

        return adx_out

    @staticmethod
    def vwap(candles: list[Candle]) -> np.ndarray:
        """Volume Weighted Average Price (cumulative from start of candle list)."""
        typical_prices = np.array([(c.high + c.low + c.close) / 3.0 for c in candles])
        volumes = np.array([c.volume for c in candles])

        cum_tp_vol = np.cumsum(typical_prices * volumes)
        cum_vol = np.cumsum(volumes)

        # Avoid division by zero
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_prices)
        return vwap

    @staticmethod
    def volume_sma(candles: list[Candle], period: int = 20) -> np.ndarray:
        """Simple moving average of volume."""
        volumes = np.array([c.volume for c in candles])
        return TechnicalIndicators.sma(volumes, period)

    @staticmethod
    def bollinger_bands(
        closes: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands — middle, upper, lower."""
        middle = TechnicalIndicators.sma(closes, period)
        std = np.full_like(closes, np.nan)
        for i in range(period - 1, len(closes)):
            std[i] = np.std(closes[i - period + 1 : i + 1])
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def compute_all(candles: list[Candle], config: dict) -> dict:
        """Compute all indicators needed by the momentum strategy.

        Args:
            candles: List of candle data.
            config: Strategy config with indicator parameters.

        Returns:
            Dict of indicator name -> latest value(s).
        """
        if len(candles) < 2:
            return {}

        closes = np.array([c.close for c in candles])
        ti = TechnicalIndicators

        ema_fast_period = config.get("ema_fast", 9)
        ema_slow_period = config.get("ema_slow", 21)
        rsi_period = config.get("rsi_period", 14)
        macd_fast = config.get("macd_fast", 12)
        macd_slow = config.get("macd_slow", 26)
        macd_signal = config.get("macd_signal", 9)
        atr_period = config.get("atr_period", 14)
        vol_sma_period = config.get("volume_sma_period", 20)

        ema_fast = ti.ema(closes, ema_fast_period)
        ema_slow = ti.ema(closes, ema_slow_period)
        rsi_vals = ti.rsi(closes, rsi_period)
        macd_line, signal_line, histogram = ti.macd(
            closes, macd_fast, macd_slow, macd_signal
        )
        atr_vals = ti.atr(candles, atr_period)
        adx_vals = ti.adx(candles, atr_period)
        vwap_vals = ti.vwap(candles)
        vol_sma = ti.volume_sma(candles, vol_sma_period)

        current_volume = candles[-1].volume
        avg_volume = vol_sma[-1] if not np.isnan(vol_sma[-1]) else current_volume

        return {
            "close": closes[-1],
            "ema_fast": ema_fast[-1],
            "ema_slow": ema_slow[-1],
            "ema_fast_prev": ema_fast[-2],
            "ema_slow_prev": ema_slow[-2],
            "rsi": rsi_vals[-1],
            "rsi_prev": rsi_vals[-2],
            "macd_line": macd_line[-1],
            "macd_signal": signal_line[-1],
            "macd_histogram": histogram[-1],
            "macd_histogram_prev": histogram[-2],
            "atr": atr_vals[-1],
            "adx": adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0.0,
            "vwap": vwap_vals[-1],
            "volume": current_volume,
            "volume_sma": avg_volume,
            "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1.0,
        }
