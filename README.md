# Momentum Trading Bot

A rule-based momentum trading bot for **Hyperliquid** perpetual futures. Generic enough to trade any asset — crypto large caps, altcoins, commodities (gold, silver, crude oil), or any token listed on the exchange.

## Architecture

```
momentum_bot/
├── exchange/          # Exchange abstraction layer
│   ├── base.py        # Abstract interface (implement for new exchanges)
│   └── hyperliquid.py # Hyperliquid integration
├── indicators/        # Technical indicator calculations
│   └── technical.py   # EMA, RSI, MACD, ATR, VWAP, Bollinger Bands
├── strategy/          # Trading strategy logic
│   └── momentum.py    # Momentum strategy with configurable scoring
├── risk/              # Risk management
│   └── manager.py     # Position sizing, loss limits, cooldowns
├── executor/          # Trade execution
│   └── trade_executor.py  # Paper + live trading, trade journaling
├── config/            # Asset-specific configurations
│   ├── btc.yaml       # BTC config
│   ├── large_caps.yaml    # BTC + ETH + SOL
│   ├── commodities.yaml   # Gold, Silver, Oil
│   └── alt_tokens.yaml    # DOGE, AVAX, LINK, ARB
└── main.py            # Bot runner and CLI entry point
```

## How It Works

### Momentum Strategy

The bot scores 6 momentum conditions (each weighted), and trades when total confidence exceeds a threshold:

| Condition | Weight | Long Signal | Short Signal |
|-----------|--------|-------------|--------------|
| EMA Trend | 20% | Price > Fast EMA | Price < Fast EMA |
| EMA Cross | 15% | Fast EMA > Slow EMA | Fast EMA < Slow EMA |
| RSI Zone | 20% | RSI 50-75 | RSI 25-50 |
| MACD Acceleration | 20% | Histogram +  & rising | Histogram - & falling |
| Volume | 15% | Volume > 1.3x avg | Volume > 1.3x avg |
| VWAP | 10% | Price > VWAP | Price < VWAP |

**Exit Rules** (any triggers close):
- Price breaks below slow EMA (for longs)
- RSI drops below 45 (for longs) / rises above 55 (for shorts)
- MACD histogram reverses direction

### Risk Management

- **Position sizing**: Based on ATR stop-loss distance and max risk % per trade
- **Consecutive loss cooldown**: Pauses trading after N consecutive losses
- **Daily loss limit**: Stops trading if daily losses exceed threshold
- **Max open positions**: Limits concurrent exposure
- **Trailing stops**: ATR-based stop-loss calculation

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your Hyperliquid credentials
```

## Usage

### Paper Trading (default — no real money)

```bash
# Trade BTC with default config
python -m momentum_bot.main

# Trade multiple large caps
python -m momentum_bot.main -c momentum_bot/config/large_caps.yaml

# Trade commodities
python -m momentum_bot.main -c momentum_bot/config/commodities.yaml

# Trade altcoins
python -m momentum_bot.main -c momentum_bot/config/alt_tokens.yaml

# Override symbols from CLI
python -m momentum_bot.main --symbols BTC ETH SOL
```

### Live Trading

```bash
# Requires HL_SECRET_KEY and HL_WALLET_ADDRESS in .env
# Will prompt for confirmation before starting
python -m momentum_bot.main --live -c momentum_bot/config/btc.yaml
```

## Configuration

All parameters are configurable via YAML. Create a new YAML file for any asset:

```yaml
symbols:
  - BTC
interval: "5m"            # Candle interval
loop_interval_seconds: 30 # How often to check
leverage: 3

indicators:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14

strategy:
  min_confidence: 0.60    # Minimum score to trigger trade
  volume_threshold: 1.3   # Volume must be 1.3x average

risk:
  max_risk_per_trade_pct: 1.0
  daily_loss_limit_pct: 5.0
  trailing_stop_atr_mult: 1.5
```

## Adding a New Exchange

Implement the `ExchangeBase` abstract class in `exchange/base.py`:

```python
from momentum_bot.exchange.base import ExchangeBase

class MyExchange(ExchangeBase):
    async def get_candles(self, symbol, interval, limit=100):
        ...
    async def get_price(self, symbol):
        ...
    async def place_order(self, symbol, side, size, price=None, reduce_only=False):
        ...
    # ... implement all abstract methods
```

## Trade Logs

All trades are logged to `trades/trades_YYYY-MM-DD.json` with full indicator snapshots for post-session review.

## Disclaimer

This is a trading tool, not financial advice. Use paper trading mode to validate strategies before risking real capital. Cryptocurrency and commodity trading involves substantial risk of loss.
