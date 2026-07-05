"""
Earnings Guard — don't hold an "intraday momentum trade" into an earnings
print by accident.

Prophet previously had zero earnings awareness: it could buy NVDA at 9:45 AM
on earnings day and treat the resulting gap as a strategy outcome. Earnings
moves are lottery tickets, not setups — a day-trading system should simply
not be in the name.

Data source: yfinance calendar (best-effort). This module FAILS OPEN — if
yfinance is missing, rate-limited, or wrong, the symbol trades normally and
the failure is logged. A guard that can halt the bot on a data hiccup is
worse than no guard.

Set EARNINGS_GUARD=off to disable entirely.
"""
import os
from datetime import datetime, timedelta, date
from typing import Optional


def _earnings_date_for(symbol: str) -> Optional[date]:
    """Best-effort next earnings date via yfinance. None on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        # yfinance has returned this as a dict (newer) and DataFrame (older)
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            if dates:
                d = dates[0]
                return d if isinstance(d, date) else None
        elif cal is not None and hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            d = cal.loc["Earnings Date"][0]
            return d.date() if hasattr(d, "date") else None
    except Exception:
        return None
    return None


def filter_earnings_risk(symbols: list[str], within_days: int = 2) -> tuple[list[str], dict]:
    """
    Split symbols into (safe, blocked) where blocked maps symbol → earnings
    date for anything reporting within `within_days` calendar days
    (today counts — day-of is the most dangerous).

    Fail-open: lookup failures are treated as safe.
    """
    if os.getenv("EARNINGS_GUARD", "on").lower() in ("off", "0", "false"):
        return symbols, {}

    today = datetime.utcnow().date()
    horizon = today + timedelta(days=within_days)

    safe, blocked = [], {}
    for sym in symbols:
        edate = _earnings_date_for(sym)
        if edate is not None and today <= edate <= horizon:
            blocked[sym] = edate.isoformat()
            print(f"  [earnings] {sym} reports {edate} — blocked from new entries")
        else:
            safe.append(sym)
    return safe, blocked
