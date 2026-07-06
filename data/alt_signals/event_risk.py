"""
Event Risk Collector — macro event awareness via Polymarket's public API.

Prediction markets aggregate information faster than most feeds, and their
public API is free. We use them for one job: know when a macro landmine is
about to go off. A day-trading system holding through an FOMC decision or
CPI print is gambling, not trading — the same logic as the earnings guard,
at market level.

  GET https://gamma-api.polymarket.com/markets
      ?closed=false&order=volume24hr&ascending=false&limit=100

We scan high-volume open markets whose question matches macro keywords and
whose end date falls within the risk horizon. Output:

  event_risk       none | elevated | high
  events           [{question, ends, prob_yes, volume24h}]
  suggested_scale  1.0 / 0.5 — the scheduler multiplies this into regime
                   risk scaling (event risk is enforced in code, not left
                   to the LLM's discretion)

Fails open, like every collector.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

MACRO_KEYWORDS = (
    "fed ", "fomc", "rate cut", "rate hike", "interest rate", "cpi",
    "inflation", "recession", "jobs report", "nonfarm", "payrolls",
    "government shutdown", "debt ceiling", "tariff",
)
MIN_VOLUME_24H = 50_000       # ignore thin markets
RISK_HORIZON_HOURS = 48


def _is_macro(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in MACRO_KEYWORDS)


def _parse_markets(payload: list, now: datetime) -> list[dict]:
    horizon = now + timedelta(hours=RISK_HORIZON_HOURS)
    events = []
    for mkt in payload:
        try:
            question = mkt.get("question") or ""
            if not _is_macro(question):
                continue
            vol24 = float(mkt.get("volume24hr") or 0)
            if vol24 < MIN_VOLUME_24H:
                continue
            end_raw = mkt.get("endDate") or mkt.get("end_date_iso")
            if not end_raw:
                continue
            ends = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
            if ends.tzinfo is None:
                ends = ends.replace(tzinfo=timezone.utc)
            if not (now <= ends <= horizon):
                continue
            prob = None
            prices = mkt.get("outcomePrices")
            if prices:
                if isinstance(prices, str):
                    import json as _json
                    prices = _json.loads(prices)
                if isinstance(prices, list) and prices:
                    prob = round(float(prices[0]), 3)
            events.append({
                "question": question[:140],
                "ends": ends.isoformat(),
                "hours_away": round((ends - now).total_seconds() / 3600, 1),
                "prob_yes": prob,
                "volume24h": int(vol24),
            })
        except Exception:
            continue
    events.sort(key=lambda e: e["hours_away"])
    return events[:5]


def collect_event_risk() -> dict:
    """Market-wide macro event risk snapshot. Fail-open default: no risk."""
    default = {"event_risk": "none", "events": [], "suggested_scale": 1.0}
    try:
        import httpx
        r = httpx.get(GAMMA_URL, params={
            "closed": "false", "order": "volume24hr",
            "ascending": "false", "limit": 100,
        }, timeout=15)
        if r.status_code != 200:
            print(f"  [event_risk] gamma api {r.status_code} — failing open")
            return default
        events = _parse_markets(r.json(), datetime.now(timezone.utc))
    except Exception as e:
        print(f"  [event_risk] failed open ({e})")
        return default

    if not events:
        return default

    # Uncertain outcomes (prob near 50%) resolving soon are the dangerous ones
    imminent_uncertain = [
        e for e in events
        if e["hours_away"] <= 24
        and (e["prob_yes"] is None or 0.25 <= e["prob_yes"] <= 0.75)
    ]
    if imminent_uncertain:
        return {"event_risk": "high", "events": events, "suggested_scale": 0.5}
    return {"event_risk": "elevated", "events": events, "suggested_scale": 1.0}


def describe(risk: dict) -> str:
    if risk["event_risk"] == "none":
        return ""
    lines = [f"MACRO EVENT RISK: {risk['event_risk'].upper()}"
             + (" — position sizes halved by code" if risk["suggested_scale"] < 1 else "")]
    for e in risk["events"][:3]:
        prob = f", market-implied prob {e['prob_yes']:.0%}" if e["prob_yes"] is not None else ""
        lines.append(f"  • {e['question']} (resolves in {e['hours_away']:.0f}h{prob})")
    return "\n".join(lines)
