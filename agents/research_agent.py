"""
Research Agent
Runs at 9:45 AM ET each market day.
- Pulls latest bars + technicals for every watchlist symbol
- Fetches recent news
- Calls Claude to produce a structured market briefing
- Saves snapshot to DB and logs its decision
"""
import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


RESEARCH_SYSTEM_PROMPT = """You are a professional market research analyst for a quantitative trading firm.
You receive a watchlist of stocks with their latest technical indicators and news headlines.
Your job is to produce a structured market briefing.

Respond ONLY with a valid JSON object — no preamble, no markdown, no explanation.
Structure:
{
  "market_mood": "bullish" | "bearish" | "neutral" | "mixed",
  "market_mood_reason": "<1 sentence>",
  "opportunities": [
    {
      "symbol": "NVDA",
      "direction": "long" | "short",
      "confidence": 0.0-1.0,
      "setup_type": "momentum" | "orb" | "vwap_bounce" | "reversal" | "options_play" | "earnings",
      "thesis": "<2-3 sentences: why this setup, what the technicals say>",
      "key_level": <float or null>,
      "risk_note": "<1 sentence on what would invalidate the setup>"
    }
  ],
  "avoid": ["TSLA", "..."],
  "avoid_reason": "<brief reason to avoid each symbol listed>"
}
Return at most 5 opportunities. Only include high-conviction setups.
"""


def _age_label(published_at) -> str:
    """
    '2h ago' / '35m ago' — recency is real information for an intraday
    decision; the old '(None)' sentiment placeholder was not.
    Tolerates datetime (naive or tz-aware), ISO string, or missing.
    """
    from datetime import datetime, timezone
    if published_at is None:
        return "time unknown"
    try:
        if isinstance(published_at, str):
            published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        mins = max(0, int((now - published_at).total_seconds() // 60))
        if mins < 60:
            return f"{mins}m ago"
        hours = mins / 60
        if hours < 24:
            return f"{hours:.0f}h ago"
        return f"{hours/24:.0f}d ago"
    except Exception:
        return "time unknown"


def _build_research_prompt(watchlist_data: list[dict], news: dict) -> str:
    lines = ["WATCHLIST SCAN RESULTS:\n"]
    for d in watchlist_data:
        sym = d["symbol"]
        rsi = d.get("rsi_14")
        macd = d.get("macd")
        macd_sig = d.get("macd_signal")
        close = d.get("close")
        vwap = d.get("vwap")
        bb_u = d.get("bb_upper")
        bb_l = d.get("bb_lower")
        atr = d.get("atr_14")
        sector = d.get("sector", "Unknown")

        # MACD crossover signal
        macd_signal = ""
        if macd is not None and macd_sig is not None:
            if macd > macd_sig:
                macd_signal = "MACD bullish cross"
            else:
                macd_signal = "MACD bearish cross"

        # Price vs VWAP
        vwap_signal = ""
        if close and vwap:
            vwap_signal = "above VWAP" if close > vwap else "below VWAP"

        # Bollinger band position
        bb_signal = ""
        if close and bb_u and bb_l:
            if close > bb_u:
                bb_signal = "above upper BB (overbought)"
            elif close < bb_l:
                bb_signal = "below lower BB (oversold)"
            else:
                mid = (bb_u + bb_l) / 2
                bb_signal = "upper half of BB" if close > mid else "lower half of BB"

        sym_news = news.get(sym, [])
        news_line = " | ".join([f"\"{n['headline']}\" ({_age_label(n.get('published_at'))})"
                                 for n in sym_news[:2]]) if sym_news else "No news in last 18h"

        if close is None:
            continue
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        atr_str = f"{atr:.2f}" if atr is not None else "N/A"
        lines.append(
            f"{sym} ({sector}): close=${close:.2f}  rsi={rsi_str}  {macd_signal}  "
            f"{vwap_signal}  {bb_signal}  atr={atr_str}\n"
            f"  News: {news_line}"
        )

    return "\n".join(lines)


class ResearchAgent:

    def __init__(self, data_provider=None, db_session=None):
        """
        data_provider: instance of DataProvider (MockDataProvider or AlpacaDataProvider)
        db_session: SQLAlchemy session for logging
        """
        if data_provider is None:
            from data.market_data import get_provider
            data_provider = get_provider()
        self.data = data_provider
        self.db = db_session

    def _call_llm(self, prompt: str) -> dict:
        from agents.llm_client import call_llm
        return call_llm(RESEARCH_SYSTEM_PROMPT, prompt)

    def run(self, symbols: list[str] = None) -> dict:
        """
        Execute a full research scan.
        Returns the structured briefing dict.
        """
        # 1. Get watchlist if not provided
        if symbols is None:
            if self.db:
                from db.operations import get_watchlist
                symbols = get_watchlist(self.db)
            else:
                symbols = ["NVDA","AAPL","TSLA","MSFT","SPY"]

        # 2. Fetch market data
        watchlist_data = self.data.scan_watchlist(symbols)

        # 3. Fetch news for each symbol (failures are non-fatal)
        news = {}
        for sym in symbols:
            try:
                news[sym] = self.data.fetch_news(sym, limit=3)
            except Exception as e:
                print(f"  [research] news fetch skipped for {sym}: {e}")
                news[sym] = []

        # 4. Persist snapshots to DB
        if self.db:
            from db.operations import save_snapshot
            for bar in watchlist_data:
                try:
                    save_snapshot(self.db, bar)
                except Exception:
                    pass  # Don't fail the scan if a snapshot fails

        # 5. Build prompt and call Claude
        prompt = _build_research_prompt(watchlist_data, news)
        briefing = self._call_llm(prompt)
        briefing["scan_time"] = datetime.utcnow().isoformat()
        briefing["symbols_scanned"] = symbols

        # 6. Log the decision
        if self.db:
            from db.operations import log_decision
            log_decision(
                self.db,
                agent="research",
                decision_type="scan",
                reasoning=f"Scanned {len(symbols)} symbols. Market mood: {briefing.get('market_mood')}",
                inputs={"symbols": symbols},
                output=briefing,
            )

        return briefing

    def run_mock(self, symbols: list[str] = None) -> dict:
        """
        Run without Claude API — returns a hardcoded mock briefing for testing.
        Always uses MockDataProvider so it never hits live APIs.
        """
        if symbols is None:
            if self.db:
                from db.operations import get_watchlist
                symbols = get_watchlist(self.db)
            else:
                symbols = ["NVDA","AAPL","TSLA","MSFT","SPY"]

        # Always use mock data here — this is the safe fallback path
        from data.market_data import MockDataProvider
        mock_dp = MockDataProvider()
        watchlist_data = mock_dp.scan_watchlist(symbols)
        news = {sym: mock_dp.fetch_news(sym, limit=2) for sym in symbols}
        prompt = _build_research_prompt(watchlist_data, news)

        # Return mock briefing shaped exactly like the real one
        return {
            "market_mood": "bullish",
            "market_mood_reason": "Tech sector momentum strong, NVDA above VWAP with bullish MACD",
            "opportunities": [
                {
                    "symbol": "NVDA",
                    "direction": "long",
                    "confidence": 0.78,
                    "setup_type": "momentum",
                    "thesis": "NVDA showing strength above VWAP with RSI at 57 — not overbought. MACD bullish crossover. AI sector tailwind.",
                    "key_level": watchlist_data[0]["close"] if watchlist_data else 120.0,
                    "risk_note": "Close below VWAP or RSI drop below 45 would invalidate."
                },
                {
                    "symbol": "AAPL",
                    "direction": "long",
                    "confidence": 0.62,
                    "setup_type": "vwap_bounce",
                    "thesis": "AAPL holding VWAP support. Earnings beat catalyst still in play. Low ATR means defined risk.",
                    "key_level": watchlist_data[1]["close"] if len(watchlist_data) > 1 else 200.0,
                    "risk_note": "Break below 50-day MA invalidates the setup."
                }
            ],
            "avoid": ["TSLA"],
            "avoid_reason": "TSLA showing elevated volatility without clear direction; ATR too high for current risk budget.",
            "scan_time": datetime.utcnow().isoformat(),
            "symbols_scanned": symbols,
            "_mock": True,
        }
