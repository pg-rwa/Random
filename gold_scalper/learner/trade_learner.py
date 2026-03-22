"""Auto-learning module for the gold scalper bot.

Reads historical trade logs and dynamically adjusts strategy/executor
parameters based on observed performance patterns. Runs a review every
N trades (configurable) and writes adjusted overrides that the executor
and strategy read on the next cycle.

Key adaptations:
1. TP/SL ratio — widens TP when wins are being cut short, tightens SL
   when stop-outs dominate losses.
2. Direction bias — reduces short sizing or skips shorts entirely when
   short win-rate is poor (and vice versa).
3. Time-of-day filter — learns which hours are unprofitable and blocks
   entries during those windows.
4. Confidence threshold — raises the minimum confidence required to enter
   when recent win-rate is low.
5. Volatility regime — adjusts band_touch_pct and BB width filter based
   on recent ATR readings.

v2 additions:
6. Streak analysis — detects losing streaks and triggers protective mode.
7. Recent performance weighting — gives more weight to last 10 trades.
8. Kill switch — if cumulative PnL is deeply negative, recommend stopping.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Bounds to prevent runaway parameter drift
_BOUNDS = {
    "tp_pct": (0.08, 0.35),
    "sl_pct": (0.08, 0.25),
    "confidence_min": (0.0, 1.5),
    "band_touch_pct": (0.05, 0.35),
    "rsi_oversold": (20.0, 40.0),
    "rsi_overbought": (60.0, 80.0),
    "direction_cooldown_sec": (30.0, 600.0),
}


def _clamp(value: float, key: str) -> float:
    lo, hi = _BOUNDS.get(key, (value, value))
    return max(lo, min(hi, value))


class TradeLearner:
    """Analyses recent trades and emits parameter adjustments."""

    def __init__(
        self,
        log_dir: str = "gold_trades",
        lookback_days: int = 7,
        review_every_n_trades: int = 10,
        min_trades_to_learn: int = 15,
        learning_rate: float = 0.25,
        enable_time_filter: bool = True,
        enable_direction_filter: bool = True,
        enable_tp_sl_tuning: bool = True,
        enable_confidence_gate: bool = True,
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

        # State
        self._trades_since_review: int = 0
        self._last_review_ts: float = 0.0
        self._current_overrides: dict = {}
        self._blocked_hours: set[int] = set()
        self._short_allowed: bool = True
        self._long_allowed: bool = True

        # v2: Kill switch state
        self._kill_switch_active: bool = False

        logger.info(
            "TradeLearner v2 initialized | lookback=%dd | review_every=%d trades | lr=%.2f",
            lookback_days, review_every_n_trades, learning_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_trade_closed(self) -> None:
        """Call after every closed trade to track review cadence."""
        self._trades_since_review += 1

    def should_review(self) -> bool:
        """Returns True when it's time to re-analyze and update params."""
        return self._trades_since_review >= self._review_interval

    def review_and_adapt(self, current_config: dict) -> dict:
        """Run the full learning cycle. Returns a dict of parameter overrides.

        The caller merges these overrides into the live config.
        Overrides use flat dotted keys, e.g.:
            {"executor.tp_pct": 0.18, "strategy.rsi_oversold": 28.0}
        """
        self._trades_since_review = 0
        self._last_review_ts = time.time()

        trades = self._load_recent_trades()
        closed = [t for t in trades if t.get("status") == "closed" and "pnl" in t]

        if len(closed) < self._min_trades:
            logger.info(
                "TradeLearner: only %d closed trades (need %d) — skipping review",
                len(closed), self._min_trades,
            )
            return self._current_overrides

        overrides: dict = {}

        # v2: Check kill switch first — if recent performance is terrible, stop everything
        self._check_kill_switch(closed)

        if self._enable_tp_sl_tuning:
            overrides.update(self._tune_tp_sl(closed, current_config))

        if self._enable_direction_filter:
            overrides.update(self._tune_direction_bias(closed, current_config))

        if self._enable_time_filter:
            overrides.update(self._tune_time_filter(closed))

        if self._enable_confidence_gate:
            overrides.update(self._tune_confidence_gate(closed, current_config))

        # v2: Analyze recent streak to tighten/loosen parameters
        overrides.update(self._analyze_recent_streak(closed, current_config))

        self._current_overrides = overrides
        self._log_review(closed, overrides)
        return overrides

    def is_entry_allowed(self, direction: str, hour: int | None = None) -> bool:
        """Quick check called before each entry to enforce learned filters.

        Args:
            direction: "long" or "short"
            hour: Current UTC hour (0-23). If None, uses system clock.
        """
        # v2: Kill switch — block ALL entries when deeply negative
        if self._kill_switch_active:
            logger.warning("TradeLearner: KILL SWITCH active — all entries blocked")
            return False

        if hour is None:
            hour = datetime.utcnow().hour

        if hour in self._blocked_hours:
            logger.debug("TradeLearner: hour %d blocked by time filter", hour)
            return False

        if direction == "short" and not self._short_allowed:
            logger.debug("TradeLearner: shorts disabled by direction filter")
            return False

        if direction == "long" and not self._long_allowed:
            logger.debug("TradeLearner: longs disabled by direction filter")
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
        """Adjust TP/SL percentages based on exit-type analysis."""
        overrides = {}

        exec_cfg = config.get("executor", {})
        curr_tp = exec_cfg.get("tp_pct", 0.14)
        curr_sl = exec_cfg.get("sl_pct", 0.14)

        # Split by exit signal
        stop_losses = [t for t in closed if t.get("signal") == "STOP_LOSS"]
        take_profits = [t for t in closed if t.get("signal") == "TAKE_PROFIT"]
        signal_exits = [t for t in closed if t.get("signal") == "SIGNAL_EXIT"]

        sl_rate = len(stop_losses) / len(closed) if closed else 0
        tp_rate = len(take_profits) / len(closed) if closed else 0

        # If stop-losses dominate (>40% of exits), widen SL slightly to give
        # trades more room, but also widen TP to maintain edge
        if sl_rate > 0.40:
            new_sl = curr_sl + self._lr * 0.02  # nudge SL wider
            overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
            logger.info("TradeLearner: SL rate %.0f%% too high → widening SL to %.2f%%",
                        sl_rate * 100, overrides["executor.sl_pct"])

        # If TP hits are rare (<10%), our TP is too aggressive — widen it
        if tp_rate < 0.10 and len(closed) > 40:
            new_tp = curr_tp + self._lr * 0.03
            overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
            logger.info("TradeLearner: TP rate only %.0f%% → widening TP to %.2f%%",
                        tp_rate * 100, overrides["executor.tp_pct"])

        # If signal exits are profitable on average, lean into them more
        # by widening TP (let signal exits do the work instead of fixed TP)
        if signal_exits:
            avg_sig_pnl = sum(t["pnl"] for t in signal_exits) / len(signal_exits)
            if avg_sig_pnl > 0 and tp_rate < 0.20:
                new_tp = curr_tp + self._lr * 0.02
                overrides["executor.tp_pct"] = round(
                    _clamp(new_tp, "tp_pct"), 4
                )

        # If we're winning a lot but wins are tiny vs losses, asymmetric TP/SL
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] < 0]
        if wins and losses:
            avg_win = sum(t["pnl"] for t in wins) / len(wins)
            avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
            rr = avg_win / avg_loss if avg_loss > 0 else 1.0

            # R:R below 0.7 — need bigger wins or smaller losses
            if rr < 0.7:
                new_tp = curr_tp + self._lr * 0.04  # widen TP more aggressively
                new_sl = curr_sl - self._lr * 0.01  # tighten SL slightly
                overrides["executor.tp_pct"] = round(_clamp(new_tp, "tp_pct"), 4)
                overrides["executor.sl_pct"] = round(_clamp(new_sl, "sl_pct"), 4)
                logger.info("TradeLearner: R:R=%.2f too low → TP=%.2f%% SL=%.2f%%",
                            rr, overrides["executor.tp_pct"], overrides["executor.sl_pct"])

        return overrides

    def _tune_direction_bias(self, closed: list[dict], config: dict) -> dict:
        """Disable or throttle a direction that's consistently losing."""
        overrides = {}

        for direction in ("long", "short"):
            dir_trades = [t for t in closed if t.get("direction") == direction]
            if len(dir_trades) < 10:
                continue

            dir_wins = [t for t in dir_trades if t["pnl"] > 0]
            wr = len(dir_wins) / len(dir_trades)
            dir_pnl = sum(t["pnl"] for t in dir_trades)

            # If win rate < 35% AND net negative, disable this direction
            # v3: lowered from 45% — was too aggressive, disabled directions too fast
            if wr < 0.35 and dir_pnl < -2.0:
                if direction == "short":
                    self._short_allowed = False
                else:
                    self._long_allowed = False
                logger.info(
                    "TradeLearner: %s disabled — WR=%.0f%% PnL=$%.2f",
                    direction.upper(), wr * 100, dir_pnl,
                )
            else:
                if direction == "short":
                    self._short_allowed = True
                else:
                    self._long_allowed = True

            # If one direction is much worse, increase its cooldown
            if wr < 0.50 and dir_pnl < 0:
                curr_cd = config.get("executor", {}).get("direction_cooldown_sec", 120)
                new_cd = curr_cd + self._lr * 60  # add some cooldown
                overrides[f"executor.direction_cooldown_sec.{direction}"] = round(
                    _clamp(new_cd, "direction_cooldown_sec"), 0
                )

        return overrides

    def _tune_time_filter(self, closed: list[dict]) -> dict:
        """Identify and block unprofitable hours."""
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

            # Block hours that are net negative with WR < 40%
            if hour_pnl < -1.0 and hour_wr < 0.40:
                blocked.add(hour)
                logger.info(
                    "TradeLearner: blocking hour %02d:00 — PnL=$%.2f WR=%.0f%% (%d trades)",
                    hour, hour_pnl, hour_wr * 100, len(trades),
                )

        self._blocked_hours = blocked
        return {"learner.blocked_hours": sorted(blocked)} if blocked else {}

    def _tune_confidence_gate(self, closed: list[dict], config: dict) -> dict:
        """Raise minimum confidence when overall performance is weak."""
        overrides = {}

        # Match closed trades to their opening records to get confidence
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

        if len(conf_trades) < 20:
            return overrides

        # Split into low/high confidence at median
        conf_trades.sort(key=lambda x: x[0])
        mid = len(conf_trades) // 2
        low_half = conf_trades[:mid]
        high_half = conf_trades[mid:]

        low_pnl = sum(p for _, p in low_half)
        high_pnl = sum(p for _, p in high_half)

        # If low-confidence trades are net losers, raise the gate
        if low_pnl < 0 and high_pnl > low_pnl:
            median_conf = conf_trades[mid][0]
            # Set minimum confidence to ~25th percentile of winning trades
            winning_confs = sorted([c for c, p in conf_trades if p > 0])
            if winning_confs:
                gate = winning_confs[len(winning_confs) // 4]
                overrides["learner.confidence_min"] = round(
                    _clamp(gate, "confidence_min"), 3
                )
                logger.info(
                    "TradeLearner: low-conf PnL=$%.2f vs high-conf $%.2f → "
                    "confidence gate=%.3f (median=%.3f)",
                    low_pnl, high_pnl, gate, median_conf,
                )

        return overrides

    # ------------------------------------------------------------------
    # v2 learning sub-routines
    # ------------------------------------------------------------------

    def _check_kill_switch(self, closed: list[dict]) -> None:
        """If the last N trades are deeply negative, activate kill switch.

        Kill switch pauses ALL trading until the next review cycle
        finds improved conditions or the operator manually restarts.
        """
        # Check last 10 trades
        recent = closed[-10:] if len(closed) >= 10 else closed
        recent_pnl = sum(t["pnl"] for t in recent)
        recent_losses = sum(1 for t in recent if t["pnl"] < 0)

        # Kill if last 10 trades lost >$8 or 8+ out of 10 are losses
        if recent_pnl < -8.0 or (len(recent) >= 10 and recent_losses >= 8):
            if not self._kill_switch_active:
                self._kill_switch_active = True
                logger.warning(
                    "!!! KILL SWITCH ACTIVATED !!! Last %d trades: PnL=$%.2f, %d losses — "
                    "ALL ENTRIES BLOCKED until next review finds improvement",
                    len(recent), recent_pnl, recent_losses,
                )
        else:
            if self._kill_switch_active:
                logger.info("Kill switch DEACTIVATED — recent performance improved")
            self._kill_switch_active = False

    def _analyze_recent_streak(self, closed: list[dict], config: dict) -> dict:
        """Analyze the most recent trades with higher weight.

        If the last 10 trades are losing, raise confidence gate aggressively.
        If the last 10 are winning, relax slightly.
        """
        overrides = {}
        if len(closed) < 10:
            return overrides

        last_10 = closed[-10:]
        last_10_pnl = sum(t["pnl"] for t in last_10)
        last_10_wins = sum(1 for t in last_10 if t["pnl"] > 0)
        last_10_wr = last_10_wins / len(last_10)

        # If last 10 trades are net negative with <40% win rate, get defensive
        if last_10_pnl < -2.0 and last_10_wr < 0.40:
            # Raise confidence gate significantly
            current_gate = self._current_overrides.get("learner.confidence_min", 0.0)
            new_gate = max(current_gate, 0.9)  # At least 0.9
            overrides["learner.confidence_min"] = round(_clamp(new_gate, "confidence_min"), 3)
            logger.info(
                "TradeLearner STREAK ALERT: last 10 trades PnL=$%.2f WR=%.0f%% → "
                "confidence gate raised to %.3f",
                last_10_pnl, last_10_wr * 100, new_gate,
            )

        # Check for large individual losses (slippage/gaps)
        big_losses = [t for t in last_10 if t["pnl"] < -3.0]
        if big_losses:
            logger.warning(
                "TradeLearner: %d outsized losses (>$3) in last 10 trades — "
                "consider reducing position size",
                len(big_losses),
            )

        return overrides

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_recent_trades(self) -> list[dict]:
        """Load trade logs from the last N days."""
        all_trades = []
        if not self._log_dir.exists():
            return all_trades

        now = datetime.now()
        for f in sorted(self._log_dir.glob("gold_scalp_*.json")):
            try:
                date_str = f.stem.replace("gold_scalp_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if (now - file_date).days <= self._lookback_days:
                    with open(f) as fh:
                        all_trades.extend(json.load(fh))
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning("TradeLearner: skipping %s: %s", f.name, e)

        return all_trades

    def _log_review(self, closed: list[dict], overrides: dict) -> None:
        """Log the review results for transparency."""
        total_pnl = sum(t["pnl"] for t in closed)
        wins = len([t for t in closed if t["pnl"] > 0])
        wr = wins / len(closed) * 100 if closed else 0

        logger.info("=" * 50)
        logger.info("TradeLearner Review Complete")
        logger.info("  Analyzed: %d trades | PnL=$%.2f | WR=%.1f%%",
                     len(closed), total_pnl, wr)
        logger.info("  Blocked hours: %s", sorted(self._blocked_hours) or "none")
        logger.info("  Shorts allowed: %s | Longs allowed: %s",
                     self._short_allowed, self._long_allowed)
        if overrides:
            logger.info("  Parameter overrides:")
            for k, v in sorted(overrides.items()):
                logger.info("    %s = %s", k, v)
        else:
            logger.info("  No parameter changes needed")
        logger.info("=" * 50)
