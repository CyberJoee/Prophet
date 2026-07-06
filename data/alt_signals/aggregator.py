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


def collect_all(db, symbols: list[str]) -> dict:
    """
    Run every collector, store snapshots, return:
      {
        "text": <briefing block for the LLM>,
        "event_scale": 1.0 | 0.5,       # enforced by scheduler in code
        "per_symbol": {sym: {...}},
      }
    Never raises — each collector fails open independently.
    """
    from data.alt_signals import options_flow, short_volume, event_risk

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    lines = []
    per_symbol: dict = {}
    event_scale = 1.0

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
        sv = short_volume.collect_short_volume(symbols)
        for sym, m in sv.items():
            _store(db, today, sym, "short_volume", m)
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
    for sym in symbols:
        try:
            m = options_flow.collect_options_flow(sym)
            if m is None:
                continue
            _store(db, today, sym, "options_flow", m)
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

    return {"text": text, "event_scale": event_scale, "per_symbol": per_symbol}
