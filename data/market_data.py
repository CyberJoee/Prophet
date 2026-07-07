"""
Data provider architecture.
- DataProvider: abstract base — all agents talk to this interface
- AlpacaDataProvider: production (uses credentials, real market data)
- MockDataProvider: testing / CI (deterministic fake data, no network needed)

Set DATA_PROVIDER=alpaca in .env for production, leave unset for mock.
"""
import os
import random
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    if "Volume" in df.columns:
        df["VWAP_D"] = (
            ((df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]).cumsum()
            / df["Volume"].cumsum()
        )
    return df


def _row_to_dict(symbol: str, row: pd.Series, ts: datetime) -> dict:
    def _f(*keys):
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                return float(row[k])
        return None
    return {
        "symbol":      symbol,
        "timestamp":   ts,
        "open":        _f("Open","open"),
        "high":        _f("High","high"),
        "low":         _f("Low","low"),
        "close":       _f("Close","close"),
        "volume":      _f("Volume","volume"),
        "vwap":        _f("VWAP_D","vwap"),
        "rsi_14":      _f("RSI_14"),
        "macd":        _f("MACD_12_26_9"),
        "macd_signal": _f("MACDs_12_26_9"),
        "bb_upper":    _f("BBU_20_2.0_2.0", "BBU_20_2.0"),
        "bb_lower":    _f("BBL_20_2.0_2.0", "BBL_20_2.0"),
        "atr_14":      _f("ATRr_14", "ATR_14"),
    }


# ─── Abstract base ─────────────────────────────────────────────────────────────

class DataProvider(ABC):

    @abstractmethod
    def fetch_bars(self, symbol: str, days: int = 60, interval: str = "1d") -> list[dict]:
        """OHLCV + indicator bars, oldest first."""

    @abstractmethod
    def fetch_latest_bar(self, symbol: str) -> Optional[dict]:
        """Most recent completed bar with indicators."""

    @abstractmethod
    def fetch_intraday(self, symbol: str, days_back: int = 5, interval: str = "5m") -> list[dict]:
        """Intraday bars."""

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> dict:
        """Sector, market cap, P/E, 52w range, beta."""

    @abstractmethod
    def fetch_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """Recent news with sentiment placeholders."""

    def scan_watchlist(self, symbols: list[str]) -> list[dict]:
        """Enrich all watchlist symbols. Main entry point for research agent."""
        results = []
        for sym in symbols:
            try:
                bar = self.fetch_latest_bar(sym)
                if bar:
                    fund = self.fetch_fundamentals(sym)
                    bar.update({k: v for k, v in fund.items() if k not in bar})
                    results.append(bar)
            except Exception as e:
                print(f"  [data] scan_watchlist skipping {sym}: {e}")
        return sorted(results, key=lambda x: x.get("rsi_14") or 50)


# ─── Mock provider ─────────────────────────────────────────────────────────────

_MOCK_META = {
    "NVDA":  {"base": 120.0,  "sector": "Technology",  "beta": 1.8},
    "AAPL":  {"base": 200.0,  "sector": "Technology",  "beta": 1.2},
    "TSLA":  {"base": 270.0,  "sector": "Automotive",  "beta": 2.2},
    "MSFT":  {"base": 430.0,  "sector": "Technology",  "beta": 1.1},
    "SPY":   {"base": 560.0,  "sector": "ETF",         "beta": 1.0},
    "QQQ":   {"base": 480.0,  "sector": "ETF",         "beta": 1.2},
    "AMZN":  {"base": 205.0,  "sector": "Consumer",    "beta": 1.4},
    "META":  {"base": 600.0,  "sector": "Technology",  "beta": 1.5},
    "GOOGL": {"base": 175.0,  "sector": "Technology",  "beta": 1.3},
    "JPM":   {"base": 225.0,  "sector": "Financial",   "beta": 1.1},
}


def _default_meta(symbol: str) -> dict:
    return _MOCK_META.get(symbol, {"base": 100.0, "sector": "Unknown", "beta": 1.0})


class MockDataProvider(DataProvider):
    """
    Deterministic fake data using seeded random walk.
    seed = hash(symbol) so each symbol always gets the same price history.
    No network calls — safe for CI and testing.
    """

    def _generate_ohlcv(self, symbol: str, n_bars: int, interval_minutes: int = 1440) -> pd.DataFrame:
        meta = _default_meta(symbol)
        seed = abs(hash(symbol)) % 100_000
        rng = random.Random(seed)
        price = meta["base"]
        daily_vol = 0.015 * meta["beta"]

        rows = []
        now = datetime.utcnow().replace(second=0, microsecond=0)
        for i in range(n_bars):
            dt = now - timedelta(minutes=interval_minutes * (n_bars - i))
            change = rng.gauss(0, daily_vol * math.sqrt(interval_minutes / 1440))
            price *= (1 + change)
            h = price * (1 + abs(rng.gauss(0, 0.003)))
            l = price * (1 - abs(rng.gauss(0, 0.003)))
            o = price * (1 + rng.gauss(0, 0.001))
            vol = abs(rng.gauss(meta["base"] * 500_000, meta["base"] * 100_000))
            rows.append({"Datetime": dt, "Open": o, "High": h, "Low": l, "Close": price, "Volume": vol})

        df = pd.DataFrame(rows).set_index("Datetime")
        return df

    def fetch_bars(self, symbol: str, days: int = 60, interval: str = "1d") -> list[dict]:
        interval_map = {"1d": 1440, "1h": 60, "30m": 30, "15m": 15, "5m": 5, "1m": 1}
        interval_min = interval_map.get(interval, 1440)
        n = min((days * 1440) // interval_min, 500)
        df = self._generate_ohlcv(symbol, n, interval_min)
        df = _compute_indicators(df)
        return [_row_to_dict(symbol, row, ts) for ts, row in df.iterrows()]

    def fetch_latest_bar(self, symbol: str) -> Optional[dict]:
        bars = self.fetch_bars(symbol, days=60, interval="1d")
        return bars[-1] if bars else None

    def fetch_intraday(self, symbol: str, days_back: int = 5, interval: str = "5m") -> list[dict]:
        return self.fetch_bars(symbol, days=days_back, interval=interval)

    def fetch_fundamentals(self, symbol: str) -> dict:
        meta = _default_meta(symbol)
        seed = abs(hash(symbol + "fund")) % 100_000
        rng = random.Random(seed)
        base = meta["base"]
        return {
            "symbol":       symbol,
            "company_name": f"{symbol} Corp",
            "sector":       meta["sector"],
            "market_cap":   int(base * rng.uniform(5e8, 5e10)),
            "pe_ratio":     round(rng.uniform(12, 45), 1),
            "52w_high":     round(base * rng.uniform(1.05, 1.6), 2),
            "52w_low":      round(base * rng.uniform(0.5, 0.85), 2),
            "avg_volume":   int(base * rng.uniform(200_000, 2_000_000)),
            "beta":         meta["beta"],
        }

    def fetch_news(self, symbol: str, limit: int = 10) -> list[dict]:
        templates = [
            (f"{symbol} beats earnings estimates by 12%", 0.75, "bullish"),
            (f"Analysts raise {symbol} price target to new high", 0.65, "bullish"),
            (f"{symbol} announces strategic partnership", 0.55, "bullish"),
            (f"Institutional investors increase {symbol} holdings", 0.45, "bullish"),
            (f"{symbol} faces increased competition in core market", -0.3, "bearish"),
        ]
        return [
            {
                "symbol":          symbol,
                "headline":        templates[i % len(templates)][0],
                "published_at":    datetime.utcnow() - timedelta(hours=i * 6),
                "sentiment_score": templates[i % len(templates)][1],
                "sentiment_label": templates[i % len(templates)][2],
                "url":             None,
            }
            for i in range(min(limit, 5))
        ]


# ─── Alpaca provider (production) ─────────────────────────────────────────────

class AlpacaDataProvider(DataProvider):
    """Production provider — requires ALPACA_API_KEY + ALPACA_SECRET_KEY."""

    def __init__(self):
        from alpaca.data.historical import StockHistoricalDataClient
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)

    def _interval_to_timeframe(self, interval: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        return {
            "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
            "1d":  TimeFrame(1,  TimeFrameUnit.Day),
        }.get(interval, TimeFrame(1, TimeFrameUnit.Day))

    def fetch_bars(self, symbol: str, days: int = 60, interval: str = "1d") -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import DataFeed
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._interval_to_timeframe(interval),
                start=datetime.utcnow() - timedelta(days=days),
                end=datetime.utcnow(),
                adjustment="split",
                feed=DataFeed.IEX,
            )
            bars = self.stock_client.get_stock_bars(req)
            raw = bars.data.get(symbol, [])
            if not raw:
                return []
            df = pd.DataFrame([
                {"Datetime": b.timestamp, "Open": b.open, "High": b.high,
                 "Low": b.low, "Close": b.close, "Volume": b.volume}
                for b in raw
            ]).set_index("Datetime")
            df = _compute_indicators(df)
            return [_row_to_dict(symbol, row, ts) for ts, row in df.iterrows()]
        except Exception as e:
            print(f"  [data] fetch_bars({symbol}) failed: {e}")
            return []

    def fetch_latest_bar(self, symbol: str) -> Optional[dict]:
        bars = self.fetch_bars(symbol, days=5, interval="1d")
        return bars[-1] if bars else None

    def fetch_intraday(self, symbol: str, days_back: int = 5, interval: str = "5m") -> list[dict]:
        return self.fetch_bars(symbol, days=days_back, interval=interval)

    def fetch_fundamentals(self, symbol: str) -> dict:
        # Alpaca doesn't supply fundamentals; add Polygon.io key for full data
        return {"symbol": symbol, "sector": None, "market_cap": None, "pe_ratio": None}

    def fetch_news(self, symbol: str, limit: int = 10,
                   max_age_hours: int = 18) -> list[dict]:
        """
        Fetch recent news for a symbol. Only articles published within
        max_age_hours qualify — for an intraday system, stale news is worse
        than no news: a 4-day-old headline presented alongside this
        morning's looks identical to the LLM and gets re-traded.
        """
        try:
            from datetime import datetime, timedelta, timezone
            from alpaca.data.historical import NewsClient
            from alpaca.data.requests import NewsRequest
            client = NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            # NewsRequest expects symbols as a string, not a list
            req = NewsRequest(symbols=symbol, limit=limit, start=cutoff)
            news = client.get_news(req)
            return [
                {
                    "symbol":          symbol,
                    "headline":        n.headline,
                    "summary":         getattr(n, "summary", None),
                    "published_at":    n.created_at,
                    "url":             n.url,
                }
                for n in getattr(news, "news", [])
            ]
        except Exception as e:
            print(f"  [data] fetch_news({symbol}) failed: {e} — continuing without news")
            return []


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_provider() -> DataProvider:
    mode = os.getenv("DATA_PROVIDER", "mock").lower()
    if mode == "alpaca":
        return AlpacaDataProvider()
    return MockDataProvider()
