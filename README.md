# Prophet

Live Heikin Ashi + WaveTrend chart dashboard. A self-hosted, free replacement
for a paid TradingView setup: HA candles, WaveTrend oscillator, money flow,
EMAs, and a signal backtest, refreshed automatically throughout the day and
served as a web page.

Started as a HBAR/USD reconstruction of a TradingView "Market Cipher B"-style
chart; built to add more symbols over time.

## How it works

- `app/strategy.py` -- data fetch (free OHLCV via `ccxt`, no API key) +
  indicators (Heikin Ashi, WaveTrend, money flow, EMAs, ATR) + backtester.
  Fetches from Coinbase by default; if Coinbase's listing history doesn't
  cover the requested start date, it automatically checks other exchanges
  (Binance, Kraken, Bitfinex, OKX, Bybit, KuCoin, Huobi, Poloniex) for one
  with deeper history and switches the whole series to that single exchange
  (no stitching across sources, to avoid a price discontinuity at the seam).
- `app/chart.py` -- builds the Plotly figure (candles, EMAs, WaveTrend pane,
  money flow, equity curve) as an embeddable HTML fragment.
- `app/main.py` -- FastAPI app. A background thread refreshes every symbol/
  timeframe on an interval and caches the rendered chart HTML in memory; the
  `/` route just serves the latest cached render, so page loads are instant
  and the page itself auto-refreshes on a meta tag to pick up new data.

## Strategy spec

**Signal source:** Heikin Ashi values computed from real OHLC. Signals come
from HA values; **fills always use real prices** -- a backtest that fills at
HA prices is fake.

- `ha_close = (o+h+l+c)/4`
- `ha_open = (prev_ha_open + prev_ha_close)/2`, seeded on bar 0 with `(o+c)/2`
- `ha_high = max(h, ha_open, ha_close)`, `ha_low = min(l, ha_open, ha_close)`

**WaveTrend** (defaults: channel 9, average 12, MA 3):

- `ap = hlc3`, `esa = EMA(ap, 9)`, `d = EMA(|ap-esa|, 9)`
- `ci = (ap - esa) / (0.015*d)`, `wt1 = EMA(ci, 12)`, `wt2 = SMA(wt1, 3)`
- Green dot: wt1 crosses above wt2 while wt2 <= buy_zone (default 0)
- Red dot: wt1 crosses below wt2 while wt2 >= sell_zone (default 0)

**Money flow:** `SMA(((c-o)/(h-l))*150, 60) - 2.5`, on HA values.

**Entry (long only):** green dot + close above slow EMA (21) by default.
**Exit:** red dot (default), or an ATR(14) x 2 stop off the signal bar's
close, whichever comes first. Fees: 0.1% per side, full equity per trade, no
pyramiding, no lookahead -- signal at bar close, fill at next bar's real open.

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `SYMBOLS` | `HBAR/USD` | Comma-separated list, e.g. `HBAR/USD,BTC/USD,ETH/USD` |
| `TIMEFRAMES` | `1d,1w` | Comma-separated: `1d`, `3d`, `1w`, `1M` |
| `SINCE` | `2021-01-01` | Backtest/chart start date |
| `EXCHANGE` | `coinbase` | Primary exchange; auto-backfills from others if it doesn't cover `SINCE` |
| `REFRESH_SECONDS` | `900` | How often the background job recomputes and the page auto-reloads |

## Deploy (Railway)

Service start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
