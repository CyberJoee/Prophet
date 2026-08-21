"""
One-time cleanup for the alpaca_sync duplicate-import bug.

execution/alpaca_sync.py deduped closed trades against the EXIT (sell)
order id, while natively-tracked trades store the ENTRY (buy) order id in
Trade.alpaca_order_id. The two never matched, so every real closed trade
was re-imported as a duplicate row tagged source="alpaca_sync". That
inflated the dashboard's trade count and P&L (observed: dashboard
+$4,543 vs Alpaca account equity +$1,710).

The sync itself is fixed; this removes the rows it already created.

Groups CLOSED trades by (symbol, entry_price, exit_price, quantity) and,
within each group of duplicates, keeps ONE row -- preferring the natively
tracked record (entry_context.source != "alpaca_sync") over the imported
one, and falling back to the oldest row. Never deletes a row that has no
duplicate.

Dry run by default. Nothing is deleted unless you pass --apply:

    python scripts/dedupe_trades.py            # report only
    python scripts/dedupe_trades.py --apply    # actually delete
"""
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_synced(trade) -> bool:
    ctx = trade.entry_context or {}
    return isinstance(ctx, dict) and ctx.get("source") == "alpaca_sync"


def find_duplicate_groups(db):
    """Return [(key, [rows...])] for every CLOSED-trade group with >1 row."""
    from db.models import Trade, TradeStatus

    rows = (db.query(Trade)
              .filter(Trade.status == TradeStatus.CLOSED)
              .order_by(Trade.created_at.asc())
              .all())

    groups = defaultdict(list)
    for t in rows:
        key = (t.symbol,
               round(t.entry_price, 4) if t.entry_price is not None else None,
               round(t.exit_price, 4) if t.exit_price is not None else None,
               round(t.quantity, 6) if t.quantity is not None else None)
        groups[key].append(t)

    return [(k, v) for k, v in groups.items() if len(v) > 1]


def choose_keeper(rows):
    """Prefer a natively-tracked row; otherwise the oldest. Rows are
    already ordered oldest-first by the caller."""
    for t in rows:
        if not _is_synced(t):
            return t
    return rows[0]


def main(apply: bool = False):
    from db.connection import SessionLocal
    from db.models import Trade, TradeStatus

    db = SessionLocal()
    try:
        total_before = (db.query(Trade)
                          .filter(Trade.status == TradeStatus.CLOSED).count())
        dup_groups = find_duplicate_groups(db)

        if not dup_groups:
            print(f"No duplicate CLOSED trades found ({total_before} rows). Nothing to do.")
            return 0

        to_delete = []
        phantom_pnl = 0.0

        print(f"{len(dup_groups)} duplicate group(s) among {total_before} CLOSED trades:\n")
        for key, rows in sorted(dup_groups, key=lambda g: g[0][0]):
            sym, entry, exit_, qty = key
            keeper = choose_keeper(rows)
            drops = [t for t in rows if t.id != keeper.id]
            to_delete.extend(drops)
            phantom_pnl += sum((t.pnl or 0.0) for t in drops)

            print(f"  {sym:<6} entry={entry} exit={exit_} qty={qty}  "
                  f"({len(rows)} rows -> keeping 1)")
            print(f"     KEEP   {keeper.id}  source={(keeper.entry_context or {}).get('source', 'native')}  "
                  f"pnl={keeper.pnl}")
            for t in drops:
                print(f"     DELETE {t.id}  source={(t.entry_context or {}).get('source', 'native')}  "
                      f"pnl={t.pnl}")

        print(f"\n{len(to_delete)} row(s) to delete, removing ${phantom_pnl:,.2f} of phantom P&L.")
        print(f"CLOSED trades: {total_before} -> {total_before - len(to_delete)}")

        if not apply:
            print("\nDRY RUN -- nothing deleted. Re-run with --apply to commit.")
            return 0

        # Journals reference trades; drop dependent rows first.
        from db.models import TradeJournal
        ids = [t.id for t in to_delete]
        journals = (db.query(TradeJournal)
                      .filter(TradeJournal.trade_id.in_(ids)).all())
        for j in journals:
            db.delete(j)
        for t in to_delete:
            db.delete(t)
        db.commit()
        print(f"\nDeleted {len(to_delete)} duplicate trade(s) "
              f"and {len(journals)} orphaned journal entr(y/ies).")

        from db.operations import refresh_strategy_stats
        refresh_strategy_stats(db)
        print("Strategy stats refreshed.")
        return 0

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
