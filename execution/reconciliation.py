"""
Broker reconciliation — does the database agree with reality?

WHY THIS EXISTS
---------------
On 2026-08-28 an audit found an MSFT position sitting at Alpaca that the
database did not know about, and eighteen historical cases of the same shape.
The system had believed itself flat while holding an unhedged position, in
one case overnight with its bracket legs already cancelled.

Every individual bug behind that has been fixed. This module exists because
the next one has not been found yet. It is the standing check that the two
sources of truth still agree, and it is deliberately dumb: no repair logic,
no cleverness, just a loud comparison.

WHY IT DOES NOT AUTO-CORRECT
----------------------------
A disagreement between the database and the broker can mean several very
different things — a fill the tracker missed, a manual trade, a close that
did not go through, a broker outage returning an empty position list. Closing
or opening anything automatically on that basis risks turning a bookkeeping
error into a real trade. This reports; a human decides.

The one thing it must never do is stay quiet. A silent reconciliation failure
is how six weeks of untracked positions went unnoticed.
"""


def reconcile(db, broker, context: str = "") -> dict:
    """
    Compare DB open/pending trades against live broker positions.

    Returns:
      {
        "ok": bool,                  # True when both sides agree
        "untracked": [str, ...],     # held at broker, absent from the DB
        "phantom":   [str, ...],     # open in the DB, absent at the broker
        "db_symbols": [str, ...],
        "broker_symbols": [str, ...],
        "error": str | None,         # set when the broker could not be reached
      }

    Never raises. A reconciliation check that can crash the morning pipeline
    would be worse than the problem it detects.
    """
    from db.models import Trade, TradeStatus

    out = {"ok": True, "untracked": [], "phantom": [],
           "db_symbols": [], "broker_symbols": [], "error": None}

    try:
        rows = (db.query(Trade)
                  .filter(Trade.status.in_([TradeStatus.OPEN,
                                            TradeStatus.PENDING_FILL]))
                  .all())
        db_symbols = {t.symbol.upper() for t in rows}
    except Exception as e:
        out["error"] = f"could not read open trades: {e}"
        out["ok"] = False
        print(f"  [ALERT] [reconcile] {out['error']}")
        return out

    try:
        positions = broker.get_all_positions() or []
        broker_symbols = set()
        for p in positions:
            sym = p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", None)
            if sym:
                broker_symbols.add(str(sym).upper())
    except Exception as e:
        # Cannot conclude anything. Say so rather than reporting a clean bill
        # of health — an unreachable broker must not look like agreement.
        out["error"] = f"could not read broker positions: {e}"
        out["ok"] = False
        print(f"  [ALERT] [reconcile] {out['error']} — cannot verify positions")
        return out

    out["db_symbols"] = sorted(db_symbols)
    out["broker_symbols"] = sorted(broker_symbols)
    out["untracked"] = sorted(broker_symbols - db_symbols)
    out["phantom"] = sorted(db_symbols - broker_symbols)
    out["ok"] = not out["untracked"] and not out["phantom"]

    tag = f" ({context})" if context else ""
    if out["ok"]:
        print(f"  [reconcile]{tag} DB and broker agree "
              f"({len(db_symbols)} open position(s))")
        return out

    print("  " + "!" * 60)
    if out["untracked"]:
        print(f"  [ALERT] [reconcile]{tag} UNTRACKED AT BROKER: "
              f"{', '.join(out['untracked'])}")
        print("  [ALERT] Positions held that the system does not know about. "
              "These carry no stop protection from Prophet.")
    if out["phantom"]:
        print(f"  [ALERT] [reconcile]{tag} OPEN IN DB, ABSENT AT BROKER: "
              f"{', '.join(out['phantom'])}")
        print("  [ALERT] The system believes it holds these and does not.")
    print("  [ALERT] Not auto-corrected on purpose — run "
          "scripts/audit_exits.py and resolve by hand.")
    print("  " + "!" * 60)
    return out


def log_reconciliation(db, result: dict, context: str = ""):
    """Persist a mismatch as a decision row so it is visible after the fact."""
    if result.get("ok"):
        return
    try:
        from db.operations import log_decision
        log_decision(
            db, agent="reconcile", decision_type="broker_mismatch",
            reasoning=(f"{context}: untracked={result.get('untracked')} "
                       f"phantom={result.get('phantom')} "
                       f"error={result.get('error')}"),
            output=result,
        )
    except Exception as e:
        print(f"  [reconcile] could not log mismatch: {e}")
