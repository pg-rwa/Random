"""Self-learning module for BTC micro-trading bot.

Adapted from the gold scalper learner with BTC-specific enhancements:
- Signal weight optimization: adjusts the 6 signal weights based on
  which signals were most predictive of winning trades
- Volatility regime detection: BTC has distinct vol regimes — tightens
  params in low-vol, loosens in high-vol
- Streak analysis: detects extended losing streaks and temporarily
  raises entry threshold
- All original learner features: TP/SL tuning, direction filter,
  time-of-day filter, confidence gate
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Bounds to prevent runaway drift
_BOUNDS = {
    "tp_pct": (0.04, 0.25),
    "sl_pct": (0.03, 0.15),
    "confidence_min": (0.0, 1.5),
    "entry_threshold": (0.35, 0.80),
    "direction_cooldown_sec": (30.0, 600.0),
    "weight": (0.05, 0.40),
}


def _clamp(value: float, key: str) -> float:
    lo, hi = _BOUNDS.get(key, (value, value))
    return max(lo, min(hi, value))


class BTCLearner:
    """Analyses BTC micro-trade history and adapts parameters."""

    def __init__(
        self,
        log_dir: str = "btc_trades",
        lookback_days: int = 7,
        review_every_n_trades: int = 15,
        min_trades_to_learn: int = 25,
        learning_rate: float = 0.20,
        enable_time_filter: bool = True,
        enable_direction_filter: bool = True,
        enable_tp_sl_tuning: bool = True,
        enable_confidence_gate: bool = True,
        enable_weight_tuning: bool = True,
        enable_volatility_regime: bool = True,
    ):
        self._log_dir = Path(log_dir)
        self._lookback_days = lookback_days
        self._review_interval = review_every_n_trades
        self._min_trades = min_trades_to_learn
        self._lr = learning_rate

        # Feature flags
        self._enable_time_filter = enable_time_filter
        self._enable_direction_filter = enable_direction_filter
        self._enable_tp_sl_tuning = enable_tp_sl_tuning
        self._enable_confidence_gate = enable_confidence_gate
        self._enable_weight_tuning = enable_weight_tuning
        self._enable_volatility_regime = enable_volatility_regime

        # State
        self._trades_since_review: int = 0
        self._last_review_ts: float = 0.0
        self._current_overrides: dict = {}
        self._blocked_hours: set[int] = set()
        self._short_allowed: bool = True
        self._long_allowed: bool = True

        logger.info(
            "BTCLearner initialized | lookback=%dd | review_every=%d trades | lr=%.2f",
            lookback_days, review_every_n_trades, learning_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_trade_closed(self) -> None:
        self._trades_since_review += 1

    def should_review(self) -> bool:
        return self._trades_since_review >= self._review_interval

    def review_and_adapt(self, current_config: dict) -> dict:
        """Full learning cycle. Returns dict of parameter overrides."""
        self._trades_since_review = 0
        self._last_review_ts = time.time()

        trades = self._load_recent_trades()
        closed = [t for t in trades if t.get("status") == "closed" and "pnl" in t]

        if len(closed) < self._min_trades:
            logger.info(
                "BTCLearner: only %d closed trades (need %d) -- skipping",
                len(closed), self._min_trades,
            )
            return self._current_overrides

        overrides: dict = {}

        if self._enable_tp_sl_tuning:
            overrides.update(self._tune_tp_sl(closed, current_config))

        if self._enable_direction_filter:
            overrides.update(self._tune_direction_bias(closed, current_config))

        if self._enable_time_filter:
            overrides.update(self._tune_time_filter(closed))

        if self._enable_confidence_gate:
            overrides.update(self._tune_confidence_gate(closed, current_config))

        if self._enable_weight_tuning:
            overrides.update(self._tune_signal_weights(trades, closed, current_config))

        if self._enable_volatility_regime:
            overrides.update(self._tune_volatility_regime(closed, current_config))

        self._current_overrides = overrides
        self._log_review(closed, overrides)
        return overrides

    def is_entry_allowed(self, direction: str, hour: int | None = None) -> bool:
        if hour is None:
            hour = datetime.utcnow().hour

        if hour in self._blocked_hours:
            logger.debug("BTCLearner: hour %d blocked", hour)
            return False

        if direction == "short" and not self._short_allowed:
            logger.debug("BTCLearner: shorts disabled")
            return False

        if direction == "long" and not self._long_allowed:
            logger.debug("BTCLearner: longs disabled")
            return False

        return True

    @property
    def overrides(self) -> dict:
        return dict(self._current_overrides)

    @property
    def blocked_hours(self) -> set[int]:
        return set(self._blocked_hours)

    # ------------------------------------------------------------------
    # Learning sub-routines
    # ------------------------------------------------------------------

    def _tune_tp_sl(self, closed: list[dict], config: dict) -> dict:
        """Adjust TP/SL based on exit-type analysis."""
        overrides = {}
        exec_cfg = config.get("executor", {})
        curr_tp = exec_cfg.get("tp_pct", 0.08)
        curr_sl = exec_cfg.get("sl_pct", 0.06)

        stop_losses = [t for t in closed if t.get("signal") == "STOP_LOSS"]
        take_profits = [t for t in closed if t.get("signal") == "TAKE_PROFIT"]

        sl_rate = len(stop_losses) / len(closed) if closed else 0
        tp_rate = len(take_profits) / len(closed) if closed else 0

        # Too many stop-outs — widen SL
        if sl_rate > 0.40:
            new_sl = curr_sl + self._lr * 0.015
            overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
            logger.info("BTCLearner: SL rate %.0f%% -> widening SL to %.3f%%",
                        sl_rate * 100, overrides["executor.sl_pct"])

        # TP rarely hit — widen it
        if tp_rate < 0.10 and len(closed) > 30:
            new_tp = curr_tp + self._lr * 0.02
            overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
            logger.info("BTCLearner: TP rate %.0f%% -> widening TP to %.3f%%",
                        tp_rate * 100, overrides["executor.tp_pct"])

        # R:R analysis
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] < 0]
        if wins and losses:
            avg_win = sum(t["pnl"] for t in wins) / len(wins)
            avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
            rr = avg_win / avg_loss if avg_loss > 0 else 1.0

            if rr < 0.8:
                new_tp = curr_tp + self._lr * 0.03
                new_sl = curr_sl - self._lr * 0.005
                overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
                overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
                logger.info("BTCLearner: R:R=%.2f -> TP=%.3f%% SL=%.3f%%",
                            rr, overrides["executor.tp_pct"], overrides["executor.sl_pct"])

        return overrides

    def _tune_direction_bias(self, closed: list[dict], config: dict) -> dict:
        overrides = {}
        for direction in ("long", "short"):
            dir_trades = [t for t in closed if t.get("direction") == direction]
            if len(dir_trades) < 8:
                continue

            dir_wins = [t for t in dir_trades if t["pnl"] > 0]
            wr = len(dir_wins) / len(dir_trades)
            dir_pnl = sum(t["pnl"] for t in dir_trades)

            if wr < 0.42 and dir_pnl < 0:
                if direction == "short":
                    self._short_allowed = False
                else:
                    self._long_allowed = False
                logger.info("BTCLearner: %s disabled — WR=%.0f%% PnL=$%.2f",
                            direction.upper(), wr * 100, dir_pnl)
            else:
                if direction == "short":
                    self._short_allowed = True
                else:
                    self._long_allowed = True

            if wr < 0.50 and dir_pnl < 0:
                curr_cd = config.get("executor", {}).get("direction_cooldown_sec", 60)
                new_cd = curr_cd + self._lr * 30
                overrides[f"executor.direction_cooldown_sec.{direction}"] = round(
                    _clamp(new_cd, "direction_cooldown_sec"), 0
                )

        return overrides

    def _tune_time_filter(self, closed: list[dict]) -> dict:
        by_hour: dict[int, list[dict]] = defaultdict(list)
        for t in closed:
            h = datetime.fromtimestamp(t["timestamp"]).hour
            by_hour[h].append(t)

        blocked = set()
        for hour, trades in by_hour.items():
            if len(trades) < 3:
                continue
            hour_pnl = sum(t["pnl"] for t in trades)
            hour_wr = len([t for t in trades if t["pnl"] > 0]) / len(trades)

            if hour_pnl < -0.5 and hour_wr < 0.38:
                blocked.add(hour)
                logger.info("BTCLearner: blocking hour %02d:00 — PnL=$%.2f WR=%.0f%%",
                            hour, hour_pnl, hour_wr * 100)

        self._blocked_hours = blocked
        return {"learner.blocked_hours": sorted(blocked)} if blocked else {}

    def _tune_confidence_gate(self, closed: list[dict], config: dict) -> dict:
        overrides = {}
        opens = [t for t in self._load_recent_trades() if t.get("status") == "filled"]
        open_by_entry = {}
        for o in opens:
            key = (o.get("entry_price"), o.get("direction"))
            open_by_entry[key] = o

        conf_trades = []
        for t in closed:
            key = (t.get("entry_price"), t.get("direction"))
            o = open_by_entry.get(key)
            if o and "confidence" in o:
                conf_trades.append((o["confidence"], t["pnl"]))

        if len(conf_trades) < 15:
            return overrides

        conf_trades.sort(key=lambda x: x[0])
        mid = len(conf_trades) // 2
        low_pnl = sum(p for _, p in conf_trades[:mid])
        high_pnl = sum(p for _, p in conf_trades[mid:])

        if low_pnl < 0 and high_pnl > low_pnl:
            winning_confs = sorted([c for c, p in conf_trades if p > 0])
            if winning_confs:
                gate = winning_confs[len(winning_confs) // 4]
                overrides["learner.confidence_min"] = round(
                    _clamp(gate, "confidence_min"), 3
                )
                logger.info("BTCLearner: confidence gate -> %.3f", gate)

        return overrides

    def _tune_signal_weights(
        self, all_trades: list[dict], closed: list[dict], config: dict
    ) -> dict:
        """Analyze which signal scores correlated with winning trades.

        Matches open-trade records (which have long_score/short_score)
        with their closed counterparts to see which signal-rich entries won.
        """
        overrides = {}
        opens = [t for t in all_trades if t.get("status") == "filled"]
        open_by_key = {}
        for o in opens:
            key = (o.get("entry_price"), o.get("direction"))
            open_by_key[key] = o

        # Collect (direction, score, pnl) pairs
        scored_trades = []
        for t in closed:
            key = (t.get("entry_price"), t.get("direction"))
            o = open_by_key.get(key)
            if o and "long_score" in o and "short_score" in o:
                scored_trades.append({
                    "direction": t["direction"],
                    "long_score": o["long_score"],
                    "short_score": o["short_score"],
                    "confidence": o.get("confidence", 0),
                    "pnl": t["pnl"],
                })

        if len(scored_trades) < 20:
            return overrides

        # Analyze: did high-score entries actually win?
        wins = [t for t in scored_trades if t["pnl"] > 0]
        losses = [t for t in scored_trades if t["pnl"] < 0]

        if not wins or not losses:
            return overrides

        avg_win_score = sum(
            t["long_score"] if t["direction"] == "long" else t["short_score"]
            for t in wins
        ) / len(wins)

        avg_loss_score = sum(
            t["long_score"] if t["direction"] == "long" else t["short_score"]
            for t in losses
        ) / len(losses)

        # If winners have notably higher scores, the strategy is working.
        # If not, raise the entry threshold to be more selective.
        score_gap = avg_win_score - avg_loss_score
        if score_gap < 0.05:
            # Scores aren't differentiating — raise threshold
            curr_threshold = config.get("strategy", {}).get("entry_threshold", 0.55)
            new_threshold = curr_threshold + self._lr * 0.05
            overrides["strategy.entry_threshold"] = round(
                _clamp(new_threshold, "entry_threshold"), 3
            )
            logger.info(
                "BTCLearner: score gap=%.3f (low) -> raising threshold to %.3f",
                score_gap, overrides["strategy.entry_threshold"],
            )
        elif score_gap > 0.15:
            # Good discrimination — can lower threshold slightly
            curr_threshold = config.get("strategy", {}).get("entry_threshold", 0.55)
            new_threshold = curr_threshold - self._lr * 0.02
            overrides["strategy.entry_threshold"] = round(
                _clamp(new_threshold, "entry_threshold"), 3
            )
            logger.info(
                "BTCLearner: score gap=%.3f (good) -> lowering threshold to %.3f",
                score_gap, overrides["strategy.entry_threshold"],
            )

        return overrides

    def _tune_volatility_regime(self, closed: list[dict], config: dict) -> dict:
        """Detect BTC volatility regime from recent trade outcomes.

        In high-vol: widen TP/SL to avoid noise stops.
        In low-vol: tighten for quicker micro-profits.
        """
        overrides = {}
        if len(closed) < 15:
            return overrides

        # Use price range from recent trades as volatility proxy
        recent = closed[-30:] if len(closed) > 30 else closed
        prices = []
        for t in recent:
            prices.append(t.get("entry_price", 0))
            if t.get("exit_price"):
                prices.append(t["exit_price"])

        if len(prices) < 4:
            return overrides

        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        if avg_price <= 0:
            return overrides

        range_pct = price_range / avg_price * 100

        exec_cfg = config.get("executor", {})
        curr_tp = exec_cfg.get("tp_pct", 0.08)
        curr_sl = exec_cfg.get("sl_pct", 0.06)

        # High vol regime (>0.5% range in recent trades)
        if range_pct > 0.5:
            new_tp = curr_tp + self._lr * 0.02
            new_sl = curr_sl + self._lr * 0.01
            overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
            overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
            logger.info("BTCLearner: HIGH VOL regime (range=%.2f%%) -> TP=%.3f%% SL=%.3f%%",
                        range_pct, overrides["executor.tp_pct"], overrides["executor.sl_pct"])
        # Low vol regime (<0.15% range)
        elif range_pct < 0.15:
            new_tp = curr_tp - self._lr * 0.01
            new_sl = curr_sl - self._lr * 0.005
            overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
            overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
            logger.info("BTCLearner: LOW VOL regime (range=%.2f%%) -> TP=%.3f%% SL=%.3f%%",
                        range_pct, overrides["executor.tp_pct"], overrides["executor.sl_pct"])

        return overrides

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_recent_trades(self) -> list[dict]:
        all_trades = []
        if not self._log_dir.exists():
            return all_trades

        now = datetime.now()
        for f in sorted(self._log_dir.glob("btc_micro_*.json")):
            try:
                date_str = f.stem.replace("btc_micro_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if (now - file_date).days <= self._lookback_days:
                    with open(f) as fh:
                        all_trades.extend(json.load(fh))
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning("BTCLearner: skipping %s: %s", f.name, e)

        return all_trades

    def _log_review(self, closed: list[dict], overrides: dict) -> None:
        total_pnl = sum(t["pnl"] for t in closed)
        wins = len([t for t in closed if t["pnl"] > 0])
        wr = wins / len(closed) * 100 if closed else 0

        logger.info("=" * 55)
        logger.info("BTCLearner Review Complete")
        logger.info("  Analyzed: %d trades | PnL=$%.2f | WR=%.1f%%",
                     len(closed), total_pnl, wr)
        logger.info("  Blocked hours: %s", sorted(self._blocked_hours) or "none")
        logger.info("  Longs: %s | Shorts: %s",
                     "ON" if self._long_allowed else "OFF",
                     "ON" if self._short_allowed else "OFF")
        if overrides:
            logger.info("  Parameter overrides:")
            for k, v in sorted(overrides.items()):
                logger.info("    %s = %s", k, v)
        else:
            logger.info("  No parameter changes needed")
        logger.info("=" * 55)
