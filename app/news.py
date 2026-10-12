"""
Free crypto news per tracked symbol, via Google News' public RSS search --
no API key, no paid tier (CryptoPanic and CryptoCompare's news APIs both
turned out to require a paid key when we checked). Catches exactly the
"coin jumps on a big announcement" case this was built for -- e.g. Hedera's
IBM Cloud deal news turned up cleanly in testing.

Per-coin coverage is inherently noisier/sparser for mid-caps than for BTC/
ETH; a generic "latest crypto headlines" feed almost never mentions HBAR/
XLM/ADA, so each symbol gets its own targeted search query instead.
"""
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

log = logging.getLogger("prophet.news")

# base ticker -> (full name, extra search context). Falls back to just the
# ticker for anything not listed, which still works but is a noisier query.
COIN_NAMES = {
    "HBAR": "Hedera", "XLM": "Stellar", "ADA": "Cardano", "BTC": "Bitcoin",
    "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP Ripple", "DOGE": "Dogecoin",
    "LTC": "Litecoin", "LINK": "Chainlink", "AVAX": "Avalanche", "DOT": "Polkadot",
    "MATIC": "Polygon", "ATOM": "Cosmos", "QNT": "Quant",
}

# Headline patterns that tend to mean "this actually moved the market," not
# routine price-watch filler. A bare percentage figure is NOT part of this --
# routine "X% drop explained" posts mention one constantly; the unusual-move
# verbs are the real signal ("surges"/"plunges" vs. just reporting a number).
_MAJOR_RE = re.compile(
    r"\b(surge[sd]?|soar[sd]?|jump(?:s|ed)?|spike[sd]?|rally|rallies|rallied|"
    r"plunge[sd]?|crash(?:es|ed)?|tumble[sd]?|plummet(?:s|ed)?|rocket(?:s|ed)?)\b", re.I,
)
_MAJOR_KEYWORDS_RE = re.compile(
    r"\b(partnership|deal|acqui(?:re|sition)|merger|ETF|SEC|lawsuit|settlement|"
    r"hack(?:ed)?|exploit|breach|mainnet|listing|delist(?:ed|ing)?|regulat\w*|"
    r"approv\w*|launch(?:es|ed)?|integrat\w*|outage|halt(?:ed|s)?)\b", re.I,
)
# Currency-converter listing spam ("Convert 100 ARS to HBAR") -- zero
# information content, just exchange-rate boilerplate that clutters the feed.
_SPAM_RE = re.compile(r"^convert\s+[\d.,]+\s+\S+.*\bto\b", re.I)


def _strip_source_suffix(title: str, source: str) -> str:
    suffix = f" - {source}"
    return title[: -len(suffix)] if title.endswith(suffix) else title


def _is_major(title: str) -> bool:
    # Err toward over-flagging: missing real news is worse than over-highlighting
    # a routine one, since routine headlines are still shown either way.
    return bool(_MAJOR_RE.search(title)) or bool(_MAJOR_KEYWORDS_RE.search(title))


def _search(query: str) -> list[dict]:
    """One Google News RSS search -> raw parsed items (unfiltered, undeduped)."""
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ProphetNewsBot/1.0)"})
    body = urllib.request.urlopen(req, timeout=15).read()
    root = ET.fromstring(body)

    out = []
    for item in root.iter("item"):
        title_raw = (item.findtext("title") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        source_url = source_el.get("url", "") if source_el is not None else ""
        title = _strip_source_suffix(title_raw, source)
        if not title:
            continue

        pub_raw = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub_raw).astimezone(timezone.utc) if pub_raw else None
        except Exception:
            published = None

        out.append({
            "title": title, "link": item.findtext("link") or "",
            "source": source, "source_url": source_url,
            "published": published.isoformat() if published else None,
            "published_ts": published.timestamp() if published else 0,
            "major": _is_major(title),
        })
    return out


# Separate, narrow risk queries -- not one compound "A OR B OR C" query.
# Google News RSS doesn't honor boolean OR as a real filter; a long
# keyword-stuffed query just gets relevance-ranked as a bag of words, which
# diluted results enough that a live $52M lawsuit against Hedera never
# surfaced even over 80 results. The same two-word query run alone
# ("Hedera lawsuit") found it as the #1 result. One request per category
# costs more HTTP calls but is the only way that actually works.
RISK_QUERY_SUFFIXES = ["lawsuit", "SEC charges", "hack OR exploit", "regulation"]


def fetch_news(symbol: str, limit: int = 12) -> list[dict]:
    """symbol like 'HBAR/USD'. Returns newest-first list of
    {title, link, source, source_url, published, major} dicts.

    Runs a general "latest about this coin" query plus several narrow,
    separate legal/regulatory/security queries. A general query alone misses
    whole categories of real news: routine price-watch posts dominate its
    ranking and crowd out the comparatively rare but far more consequential
    legal/security stories (see RISK_QUERY_SUFFIXES comment above).
    """
    base = symbol.split("/")[0]
    name = COIN_NAMES.get(base, base)
    # Always include the ticker, not just the name, even in risk queries --
    # a name-only query ("Stellar hack") is one word away from matching
    # totally unrelated stories for any coin whose name is also a common
    # word or brand (caught "Stellar" matching an ASOS retailer hack story
    # in testing, which the ticker disambiguates).
    queries = [f"{name} {base} crypto"] + [f"{name} {base} {suffix}" for suffix in RISK_QUERY_SUFFIXES]

    seen_titles = set()
    items = []
    for q in queries:
        try:
            results = _search(q)
        except Exception:
            log.exception(f"News query failed for {symbol!r}: {q!r}")
            continue
        for it in results:
            dedup_key = re.sub(r"[^a-z0-9]+", "", it["title"].lower())
            if dedup_key in seen_titles or _SPAM_RE.match(it["title"]):
                continue  # dedupe near-identical stories across queries/aggregators + drop converter spam
            seen_titles.add(dedup_key)
            items.append(it)

    items.sort(key=lambda x: x["published_ts"], reverse=True)

    # A pure recency cutoff can still bury a major story under a flood of
    # same-day routine price-watch posts from the general query. Guarantee
    # recent majors always make the list, instead of only ever getting in by
    # winning a date-sort tiebreak against noise.
    top = items[:limit]
    extra_majors = [it for it in items[limit:] if it["major"]][:6]
    if extra_majors:
        top = sorted(top + extra_majors, key=lambda x: x["published_ts"], reverse=True)
    return top
