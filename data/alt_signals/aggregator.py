"""
Alt-Signal Aggregator.

Orchestrates the collectors each morning, persists every snapshot to the
alt_signals table, computes each symbol's own trailing baseline from stored
history, and produces a text block for the strategy LLM's briefing.

DESIGN PRINCIPLE — measure before trusting:
  v1 signals are CONTEXT ONLY. They inform the LLM's setup selection but do
  not size trades (except the macro event-risk scale, which is a defensive
  gate like the regime filter, not an alpha claim). After 3-4 weeks of daily
  snapshots, scripts/eval_signals.py joins this table against forward
  returns; only signals that demonstrably predict get promoted into sizing.
"""
from datetime import datetime, timedelta
from statistics import mean
from typing import Optional

BASELINE_DAYS = 20

# ── Measurement universe ─────────────────────────────────────────────────────
#
# Signals are COLLECTED on a wide universe but BRIEFED on the trading
# watchlist only. Those are different jobs and were previously conflated.
#
# Why wider: eval_signals measures a cross-sectional IC — symbols ranked
# against each other within a date. The noise on that per-date estimate is
# ~1/sqrt(n-1) in the number of names, so the standard error of the mean IC
# falls with BOTH more dates and more names. The first real evaluation
# (2026-08-26, 37 dates x 10 names) had se ~0.055, meaning it could only
# resolve |IC| >= 0.11 — far above the 0.02-0.05 that real alt-signals carry.
# Going from 10 names to ~40 cuts the standard error roughly in half, worth
# about a 4x speedup in calendar time to a usable answer.
#
# Why NOT wider in the briefing: the LLM prompt is a scarce resource. On
# 2026-08-26 the research call was truncated mid-JSON because the output
# exceeded its token budget. Pouring 40 symbols of options-flow prose into
# the prompt would make that worse and buy nothing — the LLM can only trade
# the watchlist anyway.
#
# Collection is free: FINRA short volume arrives in one file covering every
# symbol, and options flow is one yfinance call per name that fails open
# independently.
DEFAULT_SIGNAL_UNIVERSE = [
    # megacap tech / the current trading watchlist
    "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "JPM", "SPY", "QQQ",
    # semis + hardware
    "AMD", "AVGO", "MU", "INTC", "QCOM", "TSM",
    # software / internet
    "CRM", "ORCL", "ADBE", "NFLX", "UBER", "PLTR",
    # financials
    "BAC", "GS", "WFC", "V", "MA", "C",
    # healthcare / staples
    "UNH", "JNJ", "PFE", "LLY", "WMT", "COST", "PG", "KO",
    # industrials / energy
    "BA", "CAT", "GE", "XOM", "CVX",
    # broad ETFs (index-level readings, and a sanity check on the collectors)
    "IWM", "DIA", "XLF", "XLE", "XLK",
]

MAX_UNIVERSE = 80        # guard against a runaway env var


def get_signal_universe(watchlist: list[str] = None) -> list[str]:
    """
    Symbols to COLLECT signals on. Always a superset of the watchlist.

    Override with SIGNAL_UNIVERSE (comma-separated). Set it to the watchlist
    to restore the old narrow behaviour. Order is preserved and duplicates
    are dropped so the trading names come first.
    """
    import os
    raw = os.getenv("SIGNAL_UNIVERSE", "")
    configured = ([s.strip().upper() for s in raw.split(",") if s.strip()]
                  if raw.strip() else list(DEFAULT_SIGNAL_UNIVERSE))
    combined = [s.upper() for s in (watchlist or [])] + configured
    seen, out = set(), []
    for sym in combined:
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    if len(out) > MAX_UNIVERSE:
        print(f"  [alt] SIGNAL_UNIVERSE has {len(out)} symbols — "
              f"truncating to {MAX_UNIVERSE}")
        out = out[:MAX_UNIVERSE]
    return out


def _store(db, signal_date, symbol: str, source: str, metrics: dict):
    from db.models import AltSignal
    db.add(AltSignal(signal_date=signal_date, symbol=symbol.upper(),
                     source=source, metrics=metrics))


def _baseline(db, symbol: str, source: str, keys: list[str]) -> Optional[dict]:
    """Average of the last BASELINE_DAYS stored snapshots for given keys."""
    from db.models import AltSignal
    since = datetime.utcnow() - timedelta(days=BASELINE_DAYS + 10)
    rows = (db.query(AltSignal)
            .filter(AltSignal.symbol == symbol.upper(),
                    AltSignal.source == source,
                    AltSignal.signal_date >= since)
            .order_by(AltSignal.signal_date.desc())
            .limit(BASELINE_DAYS)
            .all())
    if len(rows) < 5:            # not enough history for a meaningful baseline
        return None
    out = {}
    for k in keys:
        vals = [r.metrics.get(k) for r in rows if r.metrics.get(k) is not None]
        if vals:
            out[k] = mean(vals)
    return out or None


def collect_all(db, symbols: list[str], universe: list[str] = None) -> dict:
    """
    Run every collector, store snapshots, return:
      {
        "text": <briefing block for the LLM — WATCHLIST ONLY>,
        "event_scale": 1.0 | 0.5,       # enforced by scheduler in code
        "per_symbol": {sym: {...}},     # watchlist only
        "collected": int,               # symbols actually stored
      }

    `symbols`  — the trading watchlist. Drives the LLM briefing.
    `universe` — the wider set to collect and STORE for later measurement.
                 Defaults to get_signal_universe(symbols). Everything here is
                 written to alt_signals; only `symbols` reaches the prompt.

    Never raises — each collector fails open independently.
    """
    from data.alt_signals import options_flow, short_volume, event_risk

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    lines = []
    per_symbol: dict = {}
    event_scale = 1.0

    if universe is None:
        universe = get_signal_universe(symbols)
    briefed = {s.upper() for s in symbols}      # only these reach the prompt
    collected = 0

    # ── Macro event risk (market-wide) ──
    try:
        risk = event_risk.collect_event_risk()
        event_scale = risk.get("suggested_scale", 1.0)
        _store(db, today, "_MACRO", "event_risk",
               {"event_risk": risk["event_risk"],
                "suggested_scale": event_scale,
                "events": risk["events"]})
        desc = event_risk.describe(risk)
        if desc:
            lines.append(desc)
    except Exception as e:
        print(f"  [alt] event_risk failed open: {e}")

    # ── Short volume (one fetch covers all symbols) ──
    sv_lines = []
    try:
        # One FINRA file covers every symbol, so the wide universe is free here.
        sv = short_volume.collect_short_volume(universe)
        for sym, m in sv.items():
            _store(db, today, sym, "short_volume", m)
            collected += 1
            if sym.upper() not in briefed:
                continue                     # stored for measurement, not briefed
            per_symbol.setdefault(sym, {})["short_volume"] = m
            base = _baseline(db, sym, "short_volume", ["short_volume_ratio"])
            desc = short_volume.describe(sym, m, base)
            if desc:
                sv_lines.append("  " + desc)
    except Exception as e:
        print(f"  [alt] short_volume failed open: {e}")
    if sv_lines:
        lines.append("SHORT VOLUME (FINRA daily):")
        lines.extend(sv_lines)

    # ── Options flow (per symbol) ──
    of_lines = []
    for sym in universe:
        try:
            m = options_flow.collect_options_flow(sym)
            if m is None:
                continue
            _store(db, today, sym, "options_flow", m)
            collected += 1
            if sym.upper() not in briefed:
                continue                     # stored for measurement, not briefed
            per_symbol.setdefault(sym, {})["options_flow"] = m
            base = _baseline(db, sym, "options_flow",
                             ["atm_iv", "total_opt_volume"])
            desc = options_flow.describe(sym, m, base)
            if desc:
                of_lines.append("  " + desc)
        except Exception as e:
            print(f"  [alt] options_flow {sym} failed open: {e}")
    if of_lines:
        lines.append("OPTIONS POSITIONING:")
        lines.extend(of_lines)

    try:
        db.commit()
    except Exception as e:
        print(f"  [alt] snapshot commit failed: {e}")
        db.rollback()

    text = ""
    if lines:
        text = ("ALTERNATIVE DATA SIGNALS (differentiated inputs — factor "
                "these into setup selection):\n" + "\n".join(lines))

    print(f"  [alt] stored {collected} snapshot(s) across {len(universe)} "
          f"symbol(s); briefed {len(briefed)}")

    return {"text": text, "event_scale": event_scale,
            "per_symbol": per_symbol, "collected": collected}
