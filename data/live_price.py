"""
Live price helper.
Single cached Alpaca data client, batch quote support.
Priority: latest quote mid-price → latest 1-min bar → daily bar fallback.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    """Cache one StockHistoricalDataClient instead of rebuilding per call."""
    global _client
    if _client is not None:
        return _client
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or api_key.startswith("your_"):
        return None
    from alpaca.data.historical import StockHistoricalDataClient
    _client = StockHistoricalDataClient(api_key, secret_key)
    return _client


def get_live_prices(symbols: list[str], data_provider=None) -> dict[str, float]:
    """
    Batch fetch current prices for multiple symbols in ONE request.
    Returns {symbol: price}. Missing symbols fall back to latest bar,
    then to the daily data provider.
    """
    prices: dict[str, float] = {}
    if not symbols:
        return prices

    client = _get_client()

    # 1. Batch latest quotes (mid-price)
    if client is not None:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
            quotes = client.get_stock_latest_quote(req)
            for sym in symbols:
                q = quotes.get(sym)
                if q is None:
                    continue
                bid = float(q.bid_price or 0)
                ask = float(q.ask_price or 0)
                if bid > 0 and ask > 0:
                    prices[sym] = round((bid + ask) / 2, 2)
                elif ask > 0:
                    prices[sym] = round(ask, 2)
        except Exception as e:
            print(f"  [price] batch quote failed: {e}")

        # 2. Batch latest bars for anything the quotes missed
        missing = [s for s in symbols if s not in prices]
        if missing:
            try:
                from alpaca.data.requests import StockLatestBarRequest
                from alpaca.data.enums import DataFeed
                req = StockLatestBarRequest(symbol_or_symbols=missing, feed=DataFeed.IEX)
                bars = client.get_stock_latest_bar(req)
                for sym in missing:
                    if sym in bars:
                        prices[sym] = round(float(bars[sym].close), 2)
            except Exception as e:
                print(f"  [price] batch bar failed: {e}")

    # 3. Daily-bar fallback (mock mode / no credentials)
    if data_provider is not None:
        for sym in symbols:
            if sym not in prices:
                try:
                    bar = data_provider.fetch_latest_bar(sym)
                    if bar and bar.get("close"):
                        prices[sym] = bar["close"]
                except Exception:
                    pass

    return prices


def get_live_price(symbol: str, data_provider=None) -> Optional[float]:
    """Single-symbol convenience wrapper."""
    return get_live_prices([symbol], data_provider).get(symbol)
