"""
Preview what the expectancy gate would do — READ ONLY.

Run this before trusting the gate in a live session. It executes the exact
same assess_setups() the morning pipeline calls, against the real database,
and prints the verdict per setup. It writes nothing and places no orders.

    python scripts/check_expectancy.py

If a verdict looks wrong, the gate can be switched off without a deploy by
setting EXPECTANCY_GATE=off in the Railway service variables.

You can also try alternative settings without changing anything permanent:

    EXPECTANCY_Z=2.0 python scripts/check_expectancy.py
    EXPECTANCY_MIN_TRADES=30 python scripts/check_expectancy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from db.connection import SessionLocal
    from db.operations import get_stats_cutoff
    from agents.expectancy_gate import (
        assess_setups, ALLOWED, REDUCED, SUSPENDED, is_enabled,
    )

    db = SessionLocal()
    try:
        print("EXPECTANCY GATE PREVIEW (read-only — nothing is written)\n")
        print(f"  EXPECTANCY_GATE          = {os.getenv('EXPECTANCY_GATE', 'on')}")
        print(f"  EXPECTANCY_WINDOW        = {os.getenv('EXPECTANCY_WINDOW', '30')}")
        print(f"  EXPECTANCY_MIN_TRADES    = {os.getenv('EXPECTANCY_MIN_TRADES', '20')}")
        print(f"  EXPECTANCY_LOOKBACK_DAYS = {os.getenv('EXPECTANCY_LOOKBACK_DAYS', '60')}")
        print(f"  EXPECTANCY_Z             = {os.getenv('EXPECTANCY_Z', '1.0')}")
        print(f"  STATS_SINCE cutoff       = {get_stats_cutoff().date()}")
        print()

        if not is_enabled():
            print("  Gate is OFF — it would take no action at all.\n")

        gate = assess_setups(db)

        if gate.get("errors"):
            for e in gate["errors"]:
                print(f"  ERROR: {e}")
            print()

        setups = gate.get("setups") or {}
        if not setups:
            print("  No verdicts produced.")
            return 0

        hdr = (f"  {'setup':14s} {'verdict':10s} {'size':>6s} {'n':>4s} "
               f"{'unscored':>9s} {'rebuilt':>8s} {'exp':>8s} {'upper':>8s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for key in sorted(setups):
            e = setups[key]
            exp   = f"{e['expectancy_r']:+.2f}R" if e["expectancy_r"] is not None else "   -"
            upper = f"{e['upper_bound']:+.2f}R" if e["upper_bound"] is not None else "   -"
            print(f"  {key:14s} {e['state']:10s} {e['scale']:>5.0%} {e['n']:>4d} "
                  f"{e['n_unscored']:>9d} {e.get('n_reconstructed', 0):>8d} "
                  f"{exp:>8s} {upper:>8s}")

        print()
        for key in sorted(setups):
            print(f"  {key}: {setups[key]['reason']}")

        print()
        if gate.get("suspended"):
            print(f"  WOULD SUSPEND: {', '.join(gate['suspended'])}")
            print("  Picks on these setups will be dropped before sizing.")
        if gate.get("reduced"):
            print(f"  WOULD REDUCE:  {', '.join(gate['reduced'])}")
        if not gate.get("suspended") and not gate.get("reduced"):
            print("  No restrictions — the gate would let everything through today.")

        # A gate that can never act is worth knowing about.
        total_scored = sum(e["n"] for e in setups.values())
        total_unscored = sum(e["n_unscored"] for e in setups.values())
        if total_scored == 0:
            print()
            print("  NOTE: zero scorable trades. The gate cannot act at all until "
                  "closed trades carry a usable planned_stop.")
            if total_unscored:
                print(f"        {total_unscored} closed trade(s) were excluded for "
                      "having no recoverable risk.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
