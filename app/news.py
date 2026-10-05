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


def fetch_news(symbol: str, limit: int = 12) -> list[dict]:
    """symbol like 'HBAR/USD'. Returns newest-first list of
    {title, link, source, source_url, published, major} dicts."""
    base = symbol.split("/")[0]
    query = f"{COIN_NAMES.get(base, base)} {base} crypto"
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ProphetNewsBot/1.0)"})
    body = urllib.request.urlopen(req, timeout=15).read()
    root = ET.fromstring(body)

    items = []
    seen_titles = set()
    for item in root.iter("item"):
        title_raw = (item.findtext("title") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        source_url = source_el.get("url", "") if source_el is not None else ""
        title = _strip_source_suffix(title_raw, source)
        # Normalize away punctuation/whitespace differences for the dedupe
        # key only (keeps the real title intact for display) -- aggregators
        # often run the same story with a comma or an "and" vs "&" different.
        dedup_key = re.sub(r"[^a-z0-9]+", "", title.lower())
        if not title or dedup_key in seen_titles or _SPAM_RE.match(title):
            continue  # dedupe near-identical stories across aggregators + drop converter spam
        seen_titles.add(dedup_key)

        pub_raw = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub_raw).astimezone(timezone.utc) if pub_raw else None
        except Exception:
            published = None

        items.append({
            "title": title, "link": item.findtext("link") or "",
            "source": source, "source_url": source_url,
            "published": published.isoformat() if published else None,
            "published_ts": published.timestamp() if published else 0,
            "major": _is_major(title),
        })

    items.sort(key=lambda x: x["published_ts"], reverse=True)
    return items[:limit]
