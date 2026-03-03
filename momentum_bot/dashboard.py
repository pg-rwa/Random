"""Simple web dashboard for monitoring the momentum bot."""

import json
import os
import time
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.utils import constants

load_dotenv()

TRADES_DIR = Path("trades")
WALLET = os.getenv("HL_WALLET_ADDRESS", "")
TESTNET = os.getenv("HL_TESTNET", "true").lower() == "true"

base_url = constants.TESTNET_API_URL if TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True)


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


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html")


async def handle_api_status(request: web.Request) -> web.Response:
    """Return account status + positions + recent trades."""
    data: dict = {"wallet": WALLET, "testnet": TESTNET, "positions": [], "balance": 0}

    try:
        if WALLET:
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

    # Recent trades from journal
    trades = _load_trades(days=7)
    data["trades"] = trades[:50]  # last 50 trades

    # Summary stats
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

    return web.json_response(data)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Momentum Bot Dashboard</title>
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
  .refresh-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .refresh-bar span { color: #8b949e; font-size: 0.85em; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-green { background: #3fb950; }
  .dot-red { background: #f85149; }
  .empty { text-align: center; padding: 40px; color: #8b949e; }
  .conf-bar { background: #21262d; border-radius: 4px; height: 6px; width: 80px; display: inline-block; vertical-align: middle; }
  .conf-fill { background: #58a6ff; height: 100%; border-radius: 4px; }
</style>
</head>
<body>

<h1>Momentum Bot Dashboard</h1>
<div class="refresh-bar">
  <span class="dot" id="statusDot"></span>
  <span id="statusText">Connecting...</span>
  <span style="margin-left:auto" id="lastUpdate"></span>
</div>

<div class="grid" id="summaryCards">
  <div class="card"><div class="label">Balance</div><div class="value" id="balance">--</div></div>
  <div class="card"><div class="label">Total PnL</div><div class="value" id="totalPnl">--</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value" id="winRate">--</div></div>
  <div class="card"><div class="label">Trades</div><div class="value" id="totalTrades">--</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value" id="winLoss">--</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value" id="avgWinLoss">--</div></div>
</div>

<h2>Open Positions</h2>
<table id="positionsTable">
  <thead><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>uPnL</th><th>Leverage</th></tr></thead>
  <tbody id="positionsBody"><tr><td colspan="6" class="empty">No open positions</td></tr></tbody>
</table>

<h2>Recent Trades</h2>
<table id="tradesTable">
  <thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Side</th><th>Price</th><th>Size</th><th>Confidence</th><th>PnL</th><th>Mode</th></tr></thead>
  <tbody id="tradesBody"><tr><td colspan="9" class="empty">No trades yet</td></tr></tbody>
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

function badgeFor(text) {
  const t = (text || '').toLowerCase();
  let cls = 'badge-close';
  if (t === 'long' || t === 'buy') cls = 'badge-long';
  else if (t === 'short' || t === 'sell') cls = 'badge-short';
  else if (t === 'close' || t.startsWith('close')) cls = 'badge-close';
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

    $('statusDot').className = 'dot dot-green';
    $('statusText').textContent = (d.testnet ? 'Testnet' : 'Mainnet') + ' \u2022 ' + (d.wallet ? d.wallet.slice(0,6) + '...' + d.wallet.slice(-4) : 'No wallet');
    $('lastUpdate').textContent = 'Updated ' + new Date().toLocaleTimeString();

    $('balance').textContent = '$' + fmt(d.balance);
    $('totalPnl').innerHTML = `<span class="${pnlClass(s.total_pnl)}">$${fmt(s.total_pnl)}</span>`;
    $('winRate').innerHTML = `<span class="${s.win_rate >= 50 ? 'positive' : s.win_rate > 0 ? 'negative' : 'neutral'}">${fmt(s.win_rate,1)}%</span>`;
    $('totalTrades').textContent = s.total_trades || 0;
    $('winLoss').innerHTML = `<span class="positive">${s.wins||0}</span> / <span class="negative">${s.losses||0}</span>`;
    $('avgWinLoss').innerHTML = `<span class="positive">$${fmt(s.avg_win)}</span> / <span class="negative">$${fmt(s.avg_loss)}</span>`;

    // Positions
    const pb = $('positionsBody');
    if (d.positions && d.positions.length) {
      pb.innerHTML = d.positions.map(p => `<tr>
        <td><strong>${p.symbol}</strong></td>
        <td>${badgeFor(p.side)}</td>
        <td>${fmt(Math.abs(p.size), 4)}</td>
        <td>$${fmt(p.entry_price)}</td>
        <td class="${pnlClass(p.unrealized_pnl)}">$${fmt(p.unrealized_pnl)}</td>
        <td>${p.leverage}x</td>
      </tr>`).join('');
    } else {
      pb.innerHTML = '<tr><td colspan="6" class="empty">No open positions</td></tr>';
    }

    // Trades
    const tb = $('tradesBody');
    if (d.trades && d.trades.length) {
      tb.innerHTML = d.trades.map(t => `<tr>
        <td>${fmtTime(t.timestamp)}</td>
        <td><strong>${t.symbol || '--'}</strong></td>
        <td>${badgeFor(t.signal || '--')}</td>
        <td>${badgeFor(t.side || '--')}</td>
        <td>$${fmt(t.entry_price || t.close_price)}</td>
        <td>${fmt(t.size, 4)}</td>
        <td>${confBar(t.confidence)}</td>
        <td class="${pnlClass(t.pnl)}">${t.pnl != null ? '$'+fmt(t.pnl) : '--'}</td>
        <td>${badgeFor(t.mode || '--')}</td>
      </tr>`).join('');
    } else {
      tb.innerHTML = '<tr><td colspan="9" class="empty">No trades yet</td></tr>';
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
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Dashboard starting at http://{args.host}:{args.port}")
    web.run_app(create_app(), host=args.host, port=args.port)
