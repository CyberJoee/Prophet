"""
Audit exit integrity — READ ONLY. Writes nothing, places no orders.

    python scripts/audit_exits.py
    python scripts/audit_exits.py --no-bars     # skip the Alpaca bar fetch

WHY THIS EXISTS
---------------
Investigating an odd trade row on 2026-08-28 turned up three defects that
compound into one problem: Prophet can believe it is flat while still holding
a position at the broker.

  A. PARTIAL FILLS ARE TREATED AS COMPLETE.
     execution/order_tracker.py promotes a trade to OPEN on
     `filled_qty > 0 and fill_price`, not on status == "filled". The
     unfilled remainder keeps working at Alpaca and is never tracked again,
     because the trade is no longer PENDING_FILL for the tracker to re-check.

  B. THE EOD CLOSE IS FIRE-AND-FORGET.
     execution/position_monitor.end_of_day() calls close_position() and then
     immediately marks the trade CLOSED at the LIVE QUOTE, without confirming
     the close filled or what it filled at. This is the v1 phantom-fill bug
     reappearing on the exit side: v2 fixed entries (PENDING_FILL -> confirm
     -> OPEN) and left exits assuming success.

  C. END OF DAY RUNS AT 16:15 ET — AFTER THE 16:00 CLOSE.
     cancel_all_orders() first removes the bracket legs, then a close order
     is submitted into a shut market. Between 16:15 and the next open the
     position is both unprotected and, per B, unknown to the system.

C is what makes A and B dangerous rather than untidy.

Observed instance: Monday 2026-08-24 planned 20 SPY, 15 filled, DB recorded
15 and closed at the 763.66 quote for -19.35. A qty-20 SPY position with the
same 762.37 entry reappeared at the next deploy and was finally reconciled at
-68.40. The real loss was 3.5x the recorded one, and the position was carried
overnight with no stop.

WHAT THIS SCRIPT CHECKS
-----------------------
  1. Live reconciliation — DB open trades vs actual Alpaca positions, both
     directions. This is the "is it happening right now" check.
  2. Resurrection pattern — a sync-imported trade appearing shortly after the
     system marked the same symbol closed. Detectable from the DB alone, and
     the direct fingerprint of the bug above.
  3. Exit price fidelity — recorded eod_close prices against the actual
     session close from 5-min bars.
  4. Fabricated stops — sync-imported trades carry an INVENTED planned_stop
     (a 2%-of-price guess), so their R is computed against a made-up
     denominator. The expectancy gate now suspends setups on R, so anything
     it scores needs a real stop.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSION_END_HHMM = (16, 0)


def _ctx(trade) -> dict:
    c = trade.entry_context
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            return {}
    return c if isinstance(c, dict) else {}


def _is_synced(trade) -> bool:
    return _ctx(trade).get("source") == "alpaca_sync"


def check_live_reconciliation(db):
    from db.models import Trade, TradeStatus
    print("\n1. LIVE RECONCILIATION — does the DB match the broker right now?")
    print("-" * 78)

    open_rows = (db.query(Trade)
                 .filter(Trade.status.in_([TradeStatus.OPEN,
                                           TradeStatus.PENDING_FILL]))
                 .all())
    db_syms = {t.symbol: t for t in open_rows}

    try:
        from execution.broker import get_execution_client
        broker = get_execution_client()
        positions = {p["symbol"] if isinstance(p, dict) else p.symbol: p
                     for p in broker.get_all_positions()}
    except Exception as e:
        print(f"   could not reach the broker ({e}) — skipping")
        return

    print(f"   DB open/pending : {sorted(db_syms) or 'none'}")
    print(f"   Broker positions: {sorted(positions) or 'none'}")

    ghost = sorted(set(positions) - set(db_syms))
    phantom = sorted(set(db_syms) - set(positions))

    if ghost:
        print(f"\n   *** UNTRACKED AT BROKER: {ghost}")
        print("       Positions the system does not know it holds. These have "
              "no stop protection from Prophet's side.")
    if phantom:
        print(f"\n   *** OPEN IN DB, ABSENT AT BROKER: {phantom}")
        print("       The system thinks it holds these and does not.")
    if not ghost and not phantom:
        print("\n   OK — DB and broker agree.")


def check_resurrections(db, hours: int = 30):
    """A sync import shortly after we marked the same symbol closed."""
    from db.models import Trade, TradeStatus
    print(f"\n2. RESURRECTION PATTERN — closed, then re-imported within {hours}h")
    print("-" * 78)

    rows = (db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED)
            .order_by(Trade.entry_time.asc())
            .all())
    by_symbol = defaultdict(list)
    for t in rows:
        by_symbol[t.symbol].append(t)

    hits = []
    for sym, ts in by_symbol.items():
        synced = [t for t in ts if _is_synced(t) and t.entry_time]
        for s in synced:
            for prior in ts:
                if prior.id == s.id or not prior.exit_time:
                    continue
                gap = (s.entry_time - prior.exit_time).total_seconds() / 3600.0
                if 0 <= gap <= hours and not _is_synced(prior):
                    hits.append((sym, prior, s, gap))

    if not hits:
        print("   None found.")
        return hits

    print(f"   {len(hits)} occurrence(s). Each is a position the system "
          f"believed closed and the broker still held:\n")
    total_hidden = 0.0
    for sym, prior, s, gap in hits:
        recorded = prior.pnl or 0.0
        actual = s.pnl or 0.0
        total_hidden += actual
        print(f"   {sym}")
        print(f"     recorded close : {prior.exit_reason:<12} qty={prior.quantity:<6.0f} "
              f"exit=${prior.exit_price or 0:<10.2f} pnl=${recorded:+.2f}")
        print(f"     re-imported    : {s.exit_reason:<12} qty={s.quantity:<6.0f} "
              f"exit=${s.exit_price or 0:<10.2f} pnl=${actual:+.2f}   "
              f"({gap:.1f}h later)")
        if abs(prior.quantity - s.quantity) > 0.001:
            print(f"     >>> QUANTITY MISMATCH {prior.quantity:.0f} vs {s.quantity:.0f}"
                  " — consistent with a partial fill whose remainder kept working")
    print(f"\n   P&L recorded only on the re-import (invisible at the time): "
          f"${total_hidden:+,.2f}")
    return hits


def check_exit_price_fidelity(db, use_bars: bool):
    from db.models import Trade, TradeStatus
    print("\n3. EXIT PRICE FIDELITY — recorded eod_close vs the real session close")
    print("-" * 78)

    rows = (db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED,
                    Trade.exit_reason == "eod_close")
            .order_by(Trade.exit_time.desc())
            .limit(40)
            .all())
    if not rows:
        print("   No eod_close trades found.")
        return
    print(f"   {len(rows)} recent eod_close trade(s). Recorded exit price is the "
          "LIVE QUOTE at 16:15,")
    print("   not a confirmed fill — see the module docstring.")

    if not use_bars:
        print("   (--no-bars: skipping the price comparison)")
        return
    if not os.getenv("ALPACA_API_KEY"):
        print("   ALPACA_API_KEY not set — skipping the price comparison.")
        return

    try:
        from backtesting.engine_v2 import load_alpaca_5min
        syms = sorted({t.symbol for t in rows})
        start = min(t.exit_time for t in rows) - timedelta(days=3)
        end = max(t.exit_time for t in rows) + timedelta(days=1)
        bars = load_alpaca_5min(syms, start, end)
    except Exception as e:
        print(f"   could not load bars ({e}) — skipping")
        return

    closes = {}
    for sym, rs in bars.items():
        byday = defaultdict(list)
        for b in rs:
            if (b["timestamp"].hour, b["timestamp"].minute) <= SESSION_END_HHMM:
                byday[b["timestamp"].date()].append(b)
        for d, bs in byday.items():
            closes[(sym, d)] = bs[-1]["close"]

    print(f"\n   {'symbol':<8}{'date':<12}{'recorded':>10}{'actual':>10}"
          f"{'diff':>9}{'diff %':>9}")
    worst = []
    for t in rows:
        actual = closes.get((t.symbol, t.exit_time.date()))
        if actual is None or not t.exit_price:
            continue
        diff = t.exit_price - actual
        pct = diff / actual * 100
        worst.append((abs(pct), t.symbol, t.exit_time.date(), t.exit_price, actual, diff, pct))
    for a, sym, d, rec, act, diff, pct in sorted(worst, reverse=True)[:15]:
        flag = "  <-- check" if a > 0.25 else ""
        print(f"   {sym:<8}{str(d):<12}{rec:>10.2f}{act:>10.2f}"
              f"{diff:>+9.2f}{pct:>+8.2f}%{flag}")
    if worst:
        avg = sum(w[0] for w in worst) / len(worst)
        print(f"\n   mean absolute deviation: {avg:.3f}%  over {len(worst)} trade(s)")
        print("   Small deviations are normal (quote vs last trade). Large ones "
              "mean the close did not happen where the DB says it did.")


def check_fabricated_stops(db):
    from db.models import Trade, TradeStatus
    from agents.expectancy_gate import trade_r_multiple
    from db.operations import get_stats_cutoff
    print("\n4. FABRICATED STOPS FEEDING THE EXPECTANCY GATE")
    print("-" * 78)

    lookback = int(os.getenv("EXPECTANCY_LOOKBACK_DAYS", "60"))
    cutoff = max(get_stats_cutoff(), datetime.utcnow() - timedelta(days=lookback))

    rows = (db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED,
                    Trade.entry_time >= cutoff)
            .all())
    bad = [t for t in rows if _is_synced(t) and trade_r_multiple(t) is not None]

    print(f"   {len(rows)} closed trade(s) inside the gate's window "
          f"(since {cutoff.date()}).")
    if not bad:
        print("   None of them carry a sync-invented stop. The gate's R values "
              "are computed from real planned stops.")
        return

    print(f"\n   *** {len(bad)} trade(s) are scored on an INVENTED stop.")
    print("       _sync_open_positions sets planned_stop from a 2%-of-price "
          "guess, so their R is fiction.\n")
    per_setup = defaultdict(int)
    for t in bad:
        setup = t.setup_type.value if hasattr(t.setup_type, "value") else str(t.setup_type)
        per_setup[setup] += 1
        print(f"   {t.symbol:<6} {setup:<13} entry=${t.entry_price:<9.2f} "
              f"stop=${t.planned_stop:<9.2f} pnl=${t.pnl or 0:+9.2f} "
              f"R={trade_r_multiple(t):+.2f}")
    print(f"\n   affected setup buckets: {dict(per_setup)}")
    print("   Any suspension decision on those buckets rests partly on "
          "fabricated risk.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bars", action="store_true",
                    help="skip the Alpaca bar fetch in check 3")
    args = ap.parse_args()

    from db.connection import SessionLocal
    db = SessionLocal()
    try:
        print("EXIT INTEGRITY AUDIT — read only, nothing is written")
        print("=" * 78)
        check_live_reconciliation(db)
        check_resurrections(db)
        check_exit_price_fidelity(db, use_bars=not args.no_bars)
        check_fabricated_stops(db)
        print("\n" + "=" * 78)
        print("Done. Nothing was modified.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
