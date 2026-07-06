"""
Short Volume Collector — FINRA Reg SHO daily short volume files.

FINRA publishes, every trading day, the short volume for every equity —
free, at a static URL, and almost no retail bot reads it:

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

Pipe-delimited: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Signal: short_volume_ratio (SVR) = ShortVolume / TotalVolume.
  ~0.40 is typical baseline for liquid names (market-maker hedging).
  Sustained SVR > 0.55-0.60 = genuine directional short pressure.
  High SVR + price holding = squeeze fuel building.
The interesting number is the DELTA vs the symbol's own trailing baseline,
which the aggregator computes from our stored history.

Fails open: any fetch/parse error yields no signal, never a blocked pipeline.
"""
from datetime import datetime, timedelta
from typing import Optional

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"


def _parse_finra_text(text: str, wanted: set[str]) -> dict[str, dict]:
    """Parse a FINRA Reg SHO daily file for the symbols we care about."""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5 or parts[0] == "Date":
            continue
        sym = parts[1].upper()
        if sym not in wanted:
            continue
        try:
            short_vol = float(parts[2])
            total_vol = float(parts[4])
        except (ValueError, IndexError):
            continue
        if total_vol <= 0:
            continue
        out[sym] = {
            "short_volume": int(short_vol),
            "total_volume": int(total_vol),
            "short_volume_ratio": round(short_vol / total_vol, 4),
        }
    return out


def collect_short_volume(symbols: list[str],
                         lookback_days: int = 4) -> dict[str, dict]:
    """
    Fetch the most recent available FINRA daily file (walks back over
    weekends/holidays) and return {symbol: metrics}. Empty dict on failure.
    """
    try:
        import httpx
    except ImportError:
        return {}

    wanted = {s.upper() for s in symbols}
    day = datetime.utcnow().date()

    for _ in range(lookback_days):
        # FINRA publishes after the close; today's file may not exist yet
        if day.weekday() < 5:
            url = FINRA_URL.format(ymd=day.strftime("%Y%m%d"))
            try:
                r = httpx.get(url, timeout=15, follow_redirects=True)
                if r.status_code == 200 and "|" in r.text[:200]:
                    parsed = _parse_finra_text(r.text, wanted)
                    if parsed:
                        for m in parsed.values():
                            m["file_date"] = day.isoformat()
                        return parsed
            except Exception as e:
                print(f"  [short_volume] fetch {day} failed: {e}")
        day -= timedelta(days=1)

    print("  [short_volume] no FINRA file found in lookback window — failing open")
    return {}


def describe(symbol: str, m: dict, baseline: Optional[dict] = None) -> str:
    svr = m.get("short_volume_ratio")
    if svr is None:
        return ""
    note = f"{symbol}: short volume ratio {svr:.0%}"
    if baseline and baseline.get("short_volume_ratio"):
        delta = svr - baseline["short_volume_ratio"]
        if abs(delta) >= 0.08:
            direction = "up" if delta > 0 else "down"
            note += (f" ({direction} {abs(delta):.0%} vs its recent baseline — "
                     f"{'short pressure building' if delta > 0 else 'short pressure easing'})")
    elif svr >= 0.60:
        note += " (elevated — heavy shorting)"
    return note
