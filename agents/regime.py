"""
Regime Gate — market-level filters that run BEFORE any trade decisions.

Human intraday traders treat these as table stakes: don't fight the tape,
size down when volatility is elevated, sit out chop. Prophet previously had
no concept of market regime — it evaluated setups identically on calm
uptrend days and on panic days.

Two assessments, both from SPY daily bars (no new data source needed):

  TREND:  SPY close vs its 20-day SMA, and 5-day slope of that SMA
          → 'uptrend' | 'downtrend' | 'choppy'
  VOL:    SPY ATR(14) as % of price vs its own 60-day history
          → 'calm' | 'normal' | 'elevated' | 'extreme'

Output: a risk multiplier applied to position sizing, plus hard gates:
  extreme vol            → no new trades (risk_scale 0)
  elevated vol           → half size
  downtrend              → longs half size (shorts unaffected)
  choppy + elevated      → no new trades
"""
import statistics
from typing import Optional


def assess_regime(data_provider, benchmark: str = "SPY") -> dict:
    """
    Returns:
      {
        "trend": "uptrend"|"downtrend"|"choppy"|"unknown",
        "vol_regime": "calm"|"normal"|"elevated"|"extreme"|"unknown",
        "risk_scale": 0.0-1.0,      # multiplier for position sizing
        "long_scale": 0.0-1.0,      # extra multiplier for longs only
        "trade_allowed": bool,
        "reasons": [str, ...],
        "spy_close": float|None,
        "spy_sma20": float|None,
        "atr_pct": float|None,
      }
    Fails OPEN (normal trading) if benchmark data is unavailable — a data
    outage shouldn't halt the bot, but it is logged loudly.
    """
    result = {
        "trend": "unknown", "vol_regime": "unknown",
        "risk_scale": 1.0, "long_scale": 1.0,
        "trade_allowed": True, "reasons": [],
        "spy_close": None, "spy_sma20": None, "atr_pct": None,
    }

    try:
        bars = data_provider.fetch_bars(benchmark, days=90)
    except Exception as e:
        result["reasons"].append(f"regime data unavailable ({e}) — failing open")
        return result

    if not bars or len(bars) < 30:
        result["reasons"].append("insufficient benchmark history — failing open")
        return result

    closes = [b["close"] for b in bars if b.get("close")]
    if len(closes) < 30:
        result["reasons"].append("insufficient close data — failing open")
        return result

    # ── Trend ────────────────────────────────────────────────────────────────
    sma20_now  = statistics.mean(closes[-20:])
    sma20_prev = statistics.mean(closes[-25:-5])   # SMA20 as of 5 days ago
    close_now  = closes[-1]
    result["spy_close"] = round(close_now, 2)
    result["spy_sma20"] = round(sma20_now, 2)

    above = close_now > sma20_now
    slope_up = sma20_now > sma20_prev

    if above and slope_up:
        result["trend"] = "uptrend"
    elif (not above) and (not slope_up):
        result["trend"] = "downtrend"
    else:
        result["trend"] = "choppy"

    # ── Volatility ───────────────────────────────────────────────────────────
    # ATR% now vs distribution of daily true-range% over the last 60 sessions
    tr_pcts = []
    for i in range(1, len(bars)):
        b, prev = bars[i], bars[i - 1]
        if not all(k in b and b[k] for k in ("high", "low", "close")):
            continue
        tr = max(
            b["high"] - b["low"],
            abs(b["high"] - prev["close"]),
            abs(b["low"] - prev["close"]),
        )
        tr_pcts.append(tr / b["close"] * 100)

    if len(tr_pcts) >= 20:
        atr_now = statistics.mean(tr_pcts[-14:])
        hist    = tr_pcts[-60:]
        med     = statistics.median(hist)
        result["atr_pct"] = round(atr_now, 3)

        if atr_now >= med * 2.5:
            result["vol_regime"] = "extreme"
        elif atr_now >= med * 1.5:
            result["vol_regime"] = "elevated"
        elif atr_now <= med * 0.75:
            result["vol_regime"] = "calm"
        else:
            result["vol_regime"] = "normal"

    # ── Gates & sizing ───────────────────────────────────────────────────────
    if result["vol_regime"] == "extreme":
        result["trade_allowed"] = False
        result["risk_scale"] = 0.0
        result["reasons"].append(
            f"EXTREME volatility (ATR {result['atr_pct']}% ≥ 2.5x median) — no new trades")
    elif result["vol_regime"] == "elevated":
        result["risk_scale"] = 0.5
        result["reasons"].append(
            f"Elevated volatility (ATR {result['atr_pct']}%) — half size")

    if result["trend"] == "downtrend":
        result["long_scale"] = 0.5
        result["reasons"].append(
            f"SPY downtrend ({result['spy_close']} < SMA20 {result['spy_sma20']}, "
            "falling) — longs half size")
        if result["vol_regime"] in ("elevated", "extreme"):
            result["trade_allowed"] = False
            result["risk_scale"] = 0.0
            result["reasons"].append("Downtrend + elevated vol — no new trades")
    elif result["trend"] == "choppy":
        result["reasons"].append("SPY choppy — no trend edge, be selective")
        if result["vol_regime"] == "elevated":
            result["trade_allowed"] = False
            result["risk_scale"] = 0.0
            result["reasons"].append("Chop + elevated vol — no new trades")

    if not result["reasons"]:
        result["reasons"].append(
            f"{result['trend']} / {result['vol_regime']} vol — full size")

    return result


def format_regime_for_prompt(regime: dict) -> str:
    """Human-readable regime block for the strategy LLM prompt."""
    lines = [
        "MARKET REGIME (computed, authoritative):",
        f"  Trend: {regime['trend']} | Volatility: {regime['vol_regime']}",
        f"  Risk scale in effect: {regime['risk_scale']:.0%}"
        + (f" (longs further scaled to {regime['long_scale']:.0%})"
           if regime.get("long_scale", 1.0) < 1.0 else ""),
    ]
    for r in regime.get("reasons", []):
        lines.append(f"  • {r}")
    return "\n".join(lines)
