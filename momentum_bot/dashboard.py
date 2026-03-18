"""Simple web dashboard for monitoring the momentum bot."""

import json
import logging
import os
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TRADES_DIR = Path("trades")
GOLD_TRADES_DIR = Path("gold_trades")
BTC_TRADES_DIR = Path("btc_trades")
WALLET = os.getenv("HL_WALLET_ADDRESS", "")
TESTNET = os.getenv("HL_TESTNET", "true").lower() == "true"

_info = None


def _get_info():
    """Lazily create the Hyperliquid Info client."""
    global _info
    if _info is None:
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            base_url = constants.TESTNET_API_URL if TESTNET else constants.MAINNET_API_URL
            _info = Info(base_url, skip_ws=True)
        except Exception as e:
            logger.warning("Could not connect to Hyperliquid: %s", e)
            return None
    return _info


def _load_trades(days: int = 7) -> list[dict]:
    """Load trade logs from the last N days."""
    trades = []
    if not TRADES_DIR.exists():
        return trades
    files = sorted(TRADES_DIR.glob("trades_*.json"), reverse=True)[:days]
    for f in files:
        try:
            with open(f) as fh:
                trades.extend(json.load(fh))
        except (json.JSONDecodeError, IOError):
            continue
    trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
    return trades


def _load_gold_trades(days: int = 7) -> list[dict]:
    """Load gold scalper trade logs from the last N days."""
    trades = []
    if not GOLD_TRADES_DIR.exists():
        return trades
    files = sorted(GOLD_TRADES_DIR.glob("gold_scalp_*.json"), reverse=True)[:days]
    for f in files:
        try:
            with open(f) as fh:
                trades.extend(json.load(fh))
        except (json.JSONDecodeError, IOError):
            continue
    trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
    return trades


def _load_btc_trades(days: int = 7) -> list[dict]:
    """Load BTC micro-trade logs from the last N days."""
    trades = []
    if not BTC_TRADES_DIR.exists():
        return trades
    files = sorted(BTC_TRADES_DIR.glob("btc_micro_*.json"), reverse=True)[:days]
    for f in files:
        try:
            with open(f) as fh:
                trades.extend(json.load(fh))
        except (json.JSONDecodeError, IOError):
            continue
    trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
    return trades


def _derive_paper_positions(momentum_trades: list[dict], gold_trades: list[dict], btc_trades: list[dict] | None = None) -> list[dict]:
    """Derive open paper positions from trade logs (for paper mode display)."""
    positions = []

    # Momentum: group by symbol, check if last trade is an open (status=filled, side!=close)
    symbol_last: dict[str, dict] = {}
    for t in sorted(momentum_trades, key=lambda x: x.get("timestamp", 0)):
        sym = t.get("symbol", "")
        if not sym:
            continue
        if t.get("status") == "closed" or t.get("side") == "close":
            symbol_last.pop(sym, None)
        elif t.get("status") == "filled" and t.get("side") in ("buy", "sell"):
            symbol_last[sym] = t

    for sym, t in symbol_last.items():
        positions.append({
            "symbol": sym,
            "size": t.get("size", 0),
            "side": "LONG" if t.get("side") == "buy" else "SHORT",
            "entry_price": t.get("entry_price", 0),
            "mark_price": 0,
            "unrealized_pnl": 0,
            "leverage": "3",
            "stop_loss": t.get("stop_loss", 0),
            "take_profit": t.get("take_profit", 0),
            "source": "paper",
        })

    # Gold: group by direction (long/short), check if last trade per direction is open
    for direction in ("long", "short"):
        last_open = None
        for t in sorted(gold_trades, key=lambda x: x.get("timestamp", 0)):
            d = t.get("direction", t.get("side", ""))
            if d != direction and t.get("side") != "close":
                continue
            if t.get("side") == "close" and t.get("direction") == direction:
                last_open = None
            elif t.get("status") == "filled" and d == direction:
                last_open = t

        if last_open:
            positions.append({
                "symbol": "GOLD",
                "size": last_open.get("size", 0),
                "side": direction.upper(),
                "entry_price": last_open.get("entry_price", 0),
                "mark_price": 0,
                "unrealized_pnl": 0,
                "leverage": "10",
                "stop_loss": last_open.get("stop_loss", 0),
                "take_profit": last_open.get("take_profit", 0),
                "source": "paper",
            })

    # BTC Micro: same dual-position logic as gold
    if btc_trades:
        for direction in ("long", "short"):
            last_open = None
            for t in sorted(btc_trades, key=lambda x: x.get("timestamp", 0)):
                d = t.get("direction", t.get("side", ""))
                if d != direction and t.get("side") != "close":
                    continue
                if t.get("side") == "close" and t.get("direction") == direction:
                    last_open = None
                elif t.get("status") == "closed" and t.get("direction") == direction:
                    last_open = None
                elif t.get("status") == "filled" and d == direction:
                    last_open = t

            if last_open:
                positions.append({
                    "symbol": "BTC (micro)",
                    "size": last_open.get("size", 0),
                    "side": direction.upper(),
                    "entry_price": last_open.get("entry_price", 0),
                    "mark_price": 0,
                    "unrealized_pnl": 0,
                    "leverage": "5",
                    "stop_loss": last_open.get("stop_loss", 0),
                    "take_profit": last_open.get("take_profit", 0),
                    "source": "paper",
                })

    return positions


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html")


async def handle_api_status(request: web.Request) -> web.Response:
    """Return account status + positions + recent trades."""
    data: dict = {"wallet": WALLET, "testnet": TESTNET, "positions": [], "balance": 0}

    try:
        info = _get_info()
        if WALLET and info:
            state = info.user_state(WALLET)
            margin = state.get("marginSummary", {})
            data["balance"] = float(margin.get("accountValue", 0))
            data["margin_used"] = float(margin.get("totalMarginUsed", 0))
            data["withdrawable"] = float(margin.get("totalRawUsd", 0))

            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                size = float(p.get("szi", 0))
                if size != 0:
                    data["positions"].append({
                        "symbol": p.get("coin"),
                        "size": float(p.get("szi", 0)),
                        "side": "LONG" if size > 0 else "SHORT",
                        "entry_price": float(p.get("entryPx", 0)),
                        "mark_price": float(p.get("positionValue", 0)) / abs(size) if size else 0,
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                        "leverage": p.get("leverage", {}).get("value", "1"),
                    })
    except Exception as e:
        data["error"] = str(e)

    # Recent trades from journal (momentum bot)
    trades = _load_trades(days=7)
    data["trades"] = trades[:50]  # last 50 trades

    # Summary stats (momentum bot)
    pnls = [t.get("pnl", 0) for t in trades if "pnl" in t and t.get("status") == "closed"]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    data["summary"] = {
        "total_trades": len(trades),
        "closed": len(pnls),
        "total_pnl": round(sum(pnls), 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
    }

    # Gold Scalper data
    gold_trades = _load_gold_trades(days=7)
    gold_closed = [t for t in gold_trades if "pnl" in t and t.get("status") == "closed"]
    gold_pnls = [t["pnl"] for t in gold_closed]
    gold_wins = [p for p in gold_pnls if p > 0]
    gold_losses = [p for p in gold_pnls if p < 0]
    gold_hold_times = [t.get("hold_time_sec", 0) for t in gold_closed if t.get("hold_time_sec")]

    data["gold_scalper"] = {
        "trades": gold_trades[:50],
        "summary": {
            "total_trades": len(gold_trades),
            "closed": len(gold_closed),
            "total_pnl": round(sum(gold_pnls), 2),
            "wins": len(gold_wins),
            "losses": len(gold_losses),
            "win_rate": round(len(gold_wins) / len(gold_pnls) * 100, 1) if gold_pnls else 0,
            "avg_win": round(sum(gold_wins) / len(gold_wins), 2) if gold_wins else 0,
            "avg_loss": round(sum(gold_losses) / len(gold_losses), 2) if gold_losses else 0,
            "avg_hold_sec": round(sum(gold_hold_times) / len(gold_hold_times), 0) if gold_hold_times else 0,
        },
    }

    # BTC Micro-Trading data
    btc_trades = _load_btc_trades(days=7)
    btc_closed = [t for t in btc_trades if "pnl" in t and t.get("status") == "closed"]
    btc_pnls = [t["pnl"] for t in btc_closed]
    btc_wins = [p for p in btc_pnls if p > 0]
    btc_losses = [p for p in btc_pnls if p < 0]
    btc_hold_times = [t.get("hold_time_sec", 0) for t in btc_closed if t.get("hold_time_sec")]

    data["btc_micro"] = {
        "trades": btc_trades[:50],
        "summary": {
            "total_trades": len(btc_trades),
            "closed": len(btc_closed),
            "total_pnl": round(sum(btc_pnls), 2),
            "wins": len(btc_wins),
            "losses": len(btc_losses),
            "win_rate": round(len(btc_wins) / len(btc_pnls) * 100, 1) if btc_pnls else 0,
            "avg_win": round(sum(btc_wins) / len(btc_wins), 2) if btc_wins else 0,
            "avg_loss": round(sum(btc_losses) / len(btc_losses), 2) if btc_losses else 0,
            "avg_hold_sec": round(sum(btc_hold_times) / len(btc_hold_times), 0) if btc_hold_times else 0,
        },
    }

    # Add paper positions if no exchange positions detected
    if not data["positions"]:
        paper_pos = _derive_paper_positions(trades, gold_trades, btc_trades)
        if paper_pos:
            data["positions"] = paper_pos

    return web.json_response(data)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.5em; }
  .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card .label { color: #8b949e; font-size: 0.8em; text-transform: uppercase; margin-bottom: 4px; }
  .card .value { font-size: 1.6em; font-weight: bold; }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  .neutral { color: #8b949e; }
  h2 { color: #58a6ff; margin: 24px 0 12px; font-size: 1.2em; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  th { background: #1c2128; color: #8b949e; text-align: left; padding: 10px 12px; font-size: 0.8em; text-transform: uppercase; }
  td { padding: 10px 12px; border-top: 1px solid #21262d; font-size: 0.9em; }
  tr:hover { background: #1c2128; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold; }
  .badge-long { background: #0d4429; color: #3fb950; }
  .badge-short { background: #4a1518; color: #f85149; }
  .badge-buy { background: #0d4429; color: #3fb950; }
  .badge-sell { background: #4a1518; color: #f85149; }
  .badge-close { background: #2a2000; color: #d29922; }
  .badge-paper { background: #1c2333; color: #58a6ff; }
  .badge-live { background: #2a1500; color: #d29922; }
  .badge-open_long { background: #0d4429; color: #3fb950; }
  .badge-open_short { background: #4a1518; color: #f85149; }
  .badge-stop_loss { background: #4a1518; color: #f85149; }
  .badge-take_profit { background: #0d4429; color: #3fb950; }
  .badge-signal_exit { background: #2a2000; color: #d29922; }
  .refresh-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .refresh-bar span { color: #8b949e; font-size: 0.85em; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-green { background: #3fb950; }
  .dot-red { background: #f85149; }
  .empty { text-align: center; padding: 40px; color: #8b949e; }
  .conf-bar { background: #21262d; border-radius: 4px; height: 6px; width: 80px; display: inline-block; vertical-align: middle; }
  .conf-fill { background: #58a6ff; height: 100%; border-radius: 4px; }
  .section-divider { border: none; border-top: 2px solid #30363d; margin: 40px 0 32px; }
  .section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .section-header h2 { margin: 0; }
  .section-tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.7em; font-weight: bold; text-transform: uppercase; }
  .tag-momentum { background: #1c2333; color: #58a6ff; }
  .tag-scalper { background: #2a1c00; color: #d29922; }
  .gold-accent { color: #d29922; }
  .grid-gold .card { border-color: #3d3020; }
  .conf-fill-gold { background: #d29922; }
  .btc-accent { color: #f0883e; }
  .grid-btc .card { border-color: #3d2a1a; }
  .tag-micro { background: #2a1c00; color: #f0883e; }
</style>
</head>
<body>

<h1>Trading Dashboard</h1>
<div class="refresh-bar">
  <span class="dot" id="statusDot"></span>
  <span id="statusText">Connecting...</span>
  <span style="margin-left:auto" id="lastUpdate"></span>
</div>

<!-- ===== PNL SUMMARY (TOP) ===== -->
<div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); margin-bottom: 16px;">
  <div class="card" style="border-left: 3px solid #f0883e;">
    <div class="label">BTC Micro PnL</div>
    <div class="value" id="topBtcPnl">--</div>
    <div style="margin-top:8px; font-size:0.8em; color:#8b949e;">
      <span id="topBtcWinRate">--</span> win rate &bull; <span id="topBtcTrades">0</span> trades &bull;
      <span class="positive" id="topBtcWins">0</span>W / <span class="negative" id="topBtcLosses">0</span>L
    </div>
  </div>
  <div class="card" style="border-left: 3px solid #d29922;">
    <div class="label">Gold Scalper PnL</div>
    <div class="value" id="topGoldPnl">--</div>
    <div style="margin-top:8px; font-size:0.8em; color:#8b949e;">
      <span id="topGoldWinRate">--</span> win rate &bull; <span id="topGoldTrades">0</span> trades &bull;
      <span class="positive" id="topGoldWins">0</span>W / <span class="negative" id="topGoldLosses">0</span>L
    </div>
  </div>
  <div class="card" style="border-left: 3px solid #58a6ff;">
    <div class="label">Momentum PnL</div>
    <div class="value" id="topMomentumPnl">--</div>
    <div style="margin-top:8px; font-size:0.8em; color:#8b949e;">
      <span id="topMomWinRate">--</span> win rate &bull; <span id="topMomTrades">0</span> trades &bull;
      <span class="positive" id="topMomWins">0</span>W / <span class="negative" id="topMomLosses">0</span>L
    </div>
  </div>
</div>

<!-- ===== ACCOUNT OVERVIEW ===== -->
<div class="grid">
  <div class="card"><div class="label">Account Balance</div><div class="value" id="balance">--</div></div>
  <div class="card"><div class="label">Combined PnL</div><div class="value" id="combinedPnl">--</div></div>
</div>

<h2>Open Positions</h2>
<table>
  <thead><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>SL / TP</th><th>uPnL</th><th>Lev</th><th>Mode</th></tr></thead>
  <tbody id="positionsBody"><tr><td colspan="8" class="empty">No open positions</td></tr></tbody>
</table>

<!-- ===== BTC MICRO-TRADING SECTION ===== -->
<hr class="section-divider">
<div class="section-header">
  <h2 class="btc-accent">BTC Micro-Trading Bot</h2>
  <span class="section-tag tag-micro">Micro &bull; 5X &bull; 1m &bull; Self-Learning</span>
</div>

<div class="grid grid-btc" id="btcSummaryCards">
  <div class="card"><div class="label">BTC Micro PnL</div><div class="value" id="btcPnl">--</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value" id="btcWinRate">--</div></div>
  <div class="card"><div class="label">Trades</div><div class="value" id="btcTotalTrades">--</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value" id="btcWinLoss">--</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value" id="btcAvgWinLoss">--</div></div>
  <div class="card"><div class="label">Avg Hold Time</div><div class="value" id="btcAvgHold">--</div></div>
</div>

<h2 class="btc-accent">BTC Micro Trades</h2>
<table>
  <thead><tr><th>Time</th><th>Direction</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Size</th><th>Scores (L/S)</th><th>Hold</th><th>PnL</th></tr></thead>
  <tbody id="btcTradesBody"><tr><td colspan="9" class="empty">No BTC micro trades yet</td></tr></tbody>
</table>

<!-- ===== MOMENTUM BOT SECTION ===== -->
<hr class="section-divider">
<div class="section-header">
  <h2>Momentum Bot</h2>
  <span class="section-tag tag-momentum">Momentum</span>
</div>

<div class="grid" id="summaryCards">
  <div class="card"><div class="label">Momentum PnL</div><div class="value" id="totalPnl">--</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value" id="winRate">--</div></div>
  <div class="card"><div class="label">Trades</div><div class="value" id="totalTrades">--</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value" id="winLoss">--</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value" id="avgWinLoss">--</div></div>
</div>

<h2>Momentum Trades</h2>
<table>
  <thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Side</th><th>Price</th><th>Size</th><th>Confidence</th><th>PnL</th><th>Mode</th></tr></thead>
  <tbody id="tradesBody"><tr><td colspan="9" class="empty">No trades yet</td></tr></tbody>
</table>

<!-- ===== GOLD SCALPER SECTION ===== -->
<hr class="section-divider">
<div class="section-header">
  <h2 class="gold-accent">Gold Scalper Bot</h2>
  <span class="section-tag tag-scalper">Scalper &bull; 10X &bull; 1m</span>
</div>

<div class="grid grid-gold" id="goldSummaryCards">
  <div class="card"><div class="label">Gold PnL</div><div class="value" id="goldPnl">--</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value" id="goldWinRate">--</div></div>
  <div class="card"><div class="label">Trades</div><div class="value" id="goldTotalTrades">--</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value" id="goldWinLoss">--</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value" id="goldAvgWinLoss">--</div></div>
  <div class="card"><div class="label">Avg Hold Time</div><div class="value" id="goldAvgHold">--</div></div>
</div>

<h2 class="gold-accent">Gold Scalp Trades</h2>
<table>
  <thead><tr><th>Time</th><th>Direction</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Size</th><th>Hold</th><th>PnL</th><th>Mode</th></tr></thead>
  <tbody id="goldTradesBody"><tr><td colspan="9" class="empty">No gold trades yet</td></tr></tbody>
</table>

<script>
const $ = id => document.getElementById(id);

function pnlClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral'; }
function fmt(v, d=2) { return v != null ? Number(v).toFixed(d) : '--'; }
function fmtTime(ts) {
  if (!ts) return '--';
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
function fmtHold(sec) {
  if (!sec) return '--';
  if (sec < 60) return sec + 's';
  if (sec < 3600) return Math.floor(sec/60) + 'm ' + (sec%60) + 's';
  return Math.floor(sec/3600) + 'h ' + Math.floor((sec%3600)/60) + 'm';
}

function badgeFor(text) {
  const t = (text || '').toLowerCase();
  let cls = 'badge-close';
  if (t === 'long' || t === 'buy' || t === 'open_long') cls = 'badge-long';
  else if (t === 'short' || t === 'sell' || t === 'open_short') cls = 'badge-short';
  else if (t === 'take_profit') cls = 'badge-take_profit';
  else if (t === 'stop_loss') cls = 'badge-stop_loss';
  else if (t === 'signal_exit' || t === 'close' || t.startsWith('close')) cls = 'badge-close';
  else if (t === 'paper') cls = 'badge-paper';
  else if (t === 'live') cls = 'badge-live';
  return `<span class="badge ${cls}">${text}</span>`;
}

function confBar(conf) {
  if (conf == null) return '--';
  const pct = Math.round(conf * 100);
  return `<span class="conf-bar"><span class="conf-fill" style="width:${pct}%"></span></span> ${pct}%`;
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const s = d.summary || {};
    const g = (d.gold_scalper || {}).summary || {};
    const b = (d.btc_micro || {}).summary || {};

    $('statusDot').className = 'dot dot-green';
    $('statusText').textContent = (d.testnet ? 'Testnet' : 'Mainnet') + ' \\u2022 ' + (d.wallet ? d.wallet.slice(0,6) + '...' + d.wallet.slice(-4) : 'No wallet');
    $('lastUpdate').textContent = 'Updated ' + new Date().toLocaleTimeString();

    // Account overview
    $('balance').textContent = '$' + fmt(d.balance);
    const combinedPnl = (s.total_pnl || 0) + (g.total_pnl || 0) + (b.total_pnl || 0);
    $('combinedPnl').innerHTML = `<span class="${pnlClass(combinedPnl)}">$${fmt(combinedPnl)}</span>`;

    // ===== TOP PNL SUMMARY =====
    $('topBtcPnl').innerHTML = `<span class="${pnlClass(b.total_pnl)}">$${fmt(b.total_pnl)}</span>`;
    $('topBtcWinRate').textContent = fmt(b.win_rate,1) + '%';
    $('topBtcTrades').textContent = b.closed || 0;
    $('topBtcWins').textContent = b.wins || 0;
    $('topBtcLosses').textContent = b.losses || 0;

    $('topMomentumPnl').innerHTML = `<span class="${pnlClass(s.total_pnl)}">$${fmt(s.total_pnl)}</span>`;
    $('topMomWinRate').textContent = fmt(s.win_rate,1) + '%';
    $('topMomTrades').textContent = s.closed || 0;
    $('topMomWins').textContent = s.wins || 0;
    $('topMomLosses').textContent = s.losses || 0;

    $('topGoldPnl').innerHTML = `<span class="${pnlClass(g.total_pnl)}">$${fmt(g.total_pnl)}</span>`;
    $('topGoldWinRate').textContent = fmt(g.win_rate,1) + '%';
    $('topGoldTrades').textContent = g.closed || 0;
    $('topGoldWins').textContent = g.wins || 0;
    $('topGoldLosses').textContent = g.losses || 0;

    // ===== MOMENTUM BOT =====
    $('totalPnl').innerHTML = `<span class="${pnlClass(s.total_pnl)}">$${fmt(s.total_pnl)}</span>`;
    $('winRate').innerHTML = `<span class="${s.win_rate >= 50 ? 'positive' : s.win_rate > 0 ? 'negative' : 'neutral'}">${fmt(s.win_rate,1)}%</span>`;
    $('totalTrades').textContent = s.total_trades || 0;
    $('winLoss').innerHTML = `<span class="positive">${s.wins||0}</span> / <span class="negative">${s.losses||0}</span>`;
    $('avgWinLoss').innerHTML = `<span class="positive">$${fmt(s.avg_win)}</span> / <span class="negative">$${fmt(s.avg_loss)}</span>`;

    // Positions (exchange + paper)
    const pb = $('positionsBody');
    if (d.positions && d.positions.length) {
      pb.innerHTML = d.positions.map(p => {
        const isPaper = p.source === 'paper';
        const slTp = p.stop_loss ? `$${fmt(p.stop_loss)} / $${fmt(p.take_profit)}` : '--';
        return `<tr>
        <td><strong>${p.symbol}</strong></td>
        <td>${badgeFor(p.side)}</td>
        <td>${fmt(Math.abs(p.size), 4)}</td>
        <td>$${fmt(p.entry_price)}</td>
        <td style="font-size:0.85em">${slTp}</td>
        <td class="${pnlClass(p.unrealized_pnl)}">$${fmt(p.unrealized_pnl)}</td>
        <td>${p.leverage}x</td>
        <td>${isPaper ? badgeFor('paper') : badgeFor('live')}</td>
      </tr>`;
      }).join('');
    } else {
      pb.innerHTML = '<tr><td colspan="8" class="empty">No open positions</td></tr>';
    }

    // Momentum trades
    const tb = $('tradesBody');
    if (d.trades && d.trades.length) {
      tb.innerHTML = d.trades.map(t => `<tr>
        <td>${fmtTime(t.timestamp)}</td>
        <td><strong>${t.symbol || '--'}</strong></td>
        <td>${badgeFor(t.signal || '--')}</td>
        <td>${badgeFor(t.side || '--')}</td>
        <td>$${fmt(t.side === 'close' ? t.close_price : t.entry_price)}</td>
        <td>${fmt(t.size, 4)}</td>
        <td>${confBar(t.confidence)}</td>
        <td class="${pnlClass(t.pnl)}">${t.pnl != null ? '$'+fmt(t.pnl) : '--'}</td>
        <td>${badgeFor(t.mode || '--')}</td>
      </tr>`).join('');
    } else {
      tb.innerHTML = '<tr><td colspan="9" class="empty">No trades yet</td></tr>';
    }

    // ===== GOLD SCALPER =====
    $('goldPnl').innerHTML = `<span class="${pnlClass(g.total_pnl)}">$${fmt(g.total_pnl)}</span>`;
    $('goldWinRate').innerHTML = `<span class="${g.win_rate >= 50 ? 'positive' : g.win_rate > 0 ? 'negative' : 'neutral'}">${fmt(g.win_rate,1)}%</span>`;
    $('goldTotalTrades').textContent = g.total_trades || 0;
    $('goldWinLoss').innerHTML = `<span class="positive">${g.wins||0}</span> / <span class="negative">${g.losses||0}</span>`;
    $('goldAvgWinLoss').innerHTML = `<span class="positive">$${fmt(g.avg_win)}</span> / <span class="negative">$${fmt(g.avg_loss)}</span>`;
    $('goldAvgHold').textContent = fmtHold(g.avg_hold_sec);

    // Gold trades
    const gt = $('goldTradesBody');
    const goldTrades = (d.gold_scalper || {}).trades || [];
    if (goldTrades.length) {
      gt.innerHTML = goldTrades.map(t => `<tr>
        <td>${fmtTime(t.timestamp)}</td>
        <td>${badgeFor(t.direction || t.side || '--')}</td>
        <td>${badgeFor(t.signal || '--')}</td>
        <td>$${fmt(t.entry_price)}</td>
        <td>${t.exit_price ? '$'+fmt(t.exit_price) : '--'}</td>
        <td>${fmt(t.size, 4)}</td>
        <td>${fmtHold(t.hold_time_sec)}</td>
        <td class="${pnlClass(t.pnl)}">${t.pnl != null ? '$'+fmt(t.pnl) : '--'}</td>
        <td>${badgeFor(t.mode || '--')}</td>
      </tr>`).join('');
    } else {
      gt.innerHTML = '<tr><td colspan="9" class="empty">No gold trades yet</td></tr>';
    }

    // ===== BTC MICRO =====
    $('btcPnl').innerHTML = `<span class="${pnlClass(b.total_pnl)}">$${fmt(b.total_pnl)}</span>`;
    $('btcWinRate').innerHTML = `<span class="${b.win_rate >= 50 ? 'positive' : b.win_rate > 0 ? 'negative' : 'neutral'}">${fmt(b.win_rate,1)}%</span>`;
    $('btcTotalTrades').textContent = b.total_trades || 0;
    $('btcWinLoss').innerHTML = `<span class="positive">${b.wins||0}</span> / <span class="negative">${b.losses||0}</span>`;
    $('btcAvgWinLoss').innerHTML = `<span class="positive">$${fmt(b.avg_win)}</span> / <span class="negative">$${fmt(b.avg_loss)}</span>`;
    $('btcAvgHold').textContent = fmtHold(b.avg_hold_sec);

    const bt = $('btcTradesBody');
    const btcTrades = (d.btc_micro || {}).trades || [];
    if (btcTrades.length) {
      bt.innerHTML = btcTrades.map(t => `<tr>
        <td>${fmtTime(t.timestamp)}</td>
        <td>${badgeFor(t.direction || t.side || '--')}</td>
        <td>${badgeFor(t.signal || '--')}</td>
        <td>$${fmt(t.entry_price)}</td>
        <td>${t.exit_price ? '$'+fmt(t.exit_price) : '--'}</td>
        <td>${fmt(t.size, 6)}</td>
        <td style="font-size:0.85em">${t.long_score != null ? 'L='+fmt(t.long_score,2)+' S='+fmt(t.short_score,2) : '--'}</td>
        <td>${fmtHold(t.hold_time_sec)}</td>
        <td class="${pnlClass(t.pnl)}">${t.pnl != null ? '$'+fmt(t.pnl) : '--'}</td>
      </tr>`).join('');
    } else {
      bt.innerHTML = '<tr><td colspan="9" class="empty">No BTC micro trades yet</td></tr>';
    }

  } catch(e) {
    $('statusDot').className = 'dot dot-red';
    $('statusText').textContent = 'Error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_api_status)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Momentum Bot Dashboard")
    parser.add_argument("-p", "--port", type=int, default=5050, help="Port (default: 5050)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Dashboard starting at http://{args.host}:{args.port}")
    web.run_app(create_app(), host=args.host, port=args.port)
