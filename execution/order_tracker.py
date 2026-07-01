"""
Order Tracker
Fixes the phantom-trade bug: a trade is only OPEN once the broker
confirms a fill. Until then it sits in PENDING_FILL.

Lifecycle:
  strategy agent places order  →  trade recorded as PENDING_FILL
  confirm_fills() (every monitor cycle)
      order filled            →  promote to OPEN with real fill price/qty/time
      order cancelled/expired →  mark trade CANCELLED (never was a position)
      order partially filled  →  promote to OPEN with the filled qty
      still working           →  leave PENDING_FILL
  end of day
      still unfilled          →  cancel order + mark trade CANCELLED
"""
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Alpaca statuses that mean the order will never fill
_DEAD_STATUSES = {
    "canceled", "cancelled", "expired", "rejected", "stopped",
    "suspended", "done_for_day", "replaced",
}


def _parse_time(ts) -> datetime:
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def confirm_fills(db, broker) -> dict:
    """
    Reconcile every PENDING_FILL trade against actual broker order status.
    Returns counts of what happened.
    """
    from db.operations import (
        get_pending_trades, confirm_trade_fill, cancel_pending_trade, log_decision
    )

    result = {"confirmed": 0, "cancelled": 0, "still_pending": 0}

    pending = get_pending_trades(db)
    if not pending:
        return result

    for trade in pending:
        if not trade.alpaca_order_id:
            # No order id to check — this should not happen; cancel defensively
            cancel_pending_trade(db, trade.id, reason="no_order_id")
            result["cancelled"] += 1
            continue

        order = broker.get_order(trade.alpaca_order_id)
        if order is None:
            # Broker lookup failed — leave pending, try again next cycle
            result["still_pending"] += 1
            continue

        status = str(order.get("status", "")).lower()
        filled_qty = float(order.get("filled_qty") or 0)
        fill_price = order.get("filled_avg_price")

        if status == "filled" or (filled_qty > 0 and fill_price):
            confirmed = confirm_trade_fill(
                db, trade.id,
                fill_price=float(fill_price),
                fill_time=_parse_time(order.get("filled_at")),
                filled_qty=filled_qty if filled_qty > 0 else None,
            )
            print(f"  [tracker] FILL CONFIRMED {trade.symbol} "
                  f"x{confirmed.quantity:.0f} @ ${confirmed.entry_price:.2f}")
            log_decision(
                db, agent="tracker", decision_type="fill_confirmed",
                symbol=trade.symbol, trade_id=trade.id,
                reasoning=f"Order {trade.alpaca_order_id} filled at ${fill_price}",
                output={"fill_price": fill_price, "filled_qty": filled_qty},
            )
            result["confirmed"] += 1

        elif status in _DEAD_STATUSES:
            cancel_pending_trade(db, trade.id, reason=f"order_{status}")
            print(f"  [tracker] order dead ({status}) — cancelled phantom trade {trade.symbol}")
            log_decision(
                db, agent="tracker", decision_type="order_dead",
                symbol=trade.symbol, trade_id=trade.id,
                reasoning=f"Order {trade.alpaca_order_id} ended {status} without filling",
            )
            result["cancelled"] += 1

        else:
            # new / accepted / pending_new / partially_filled with no avg price yet
            result["still_pending"] += 1

    return result


def cancel_stale_pending(db, broker) -> int:
    """
    EOD cleanup: any order still unfilled gets cancelled at the broker
    and its trade record marked CANCELLED. Prevents phantom trades from
    surviving overnight.
    """
    from db.operations import get_pending_trades, cancel_pending_trade, log_decision

    count = 0
    for trade in get_pending_trades(db):
        if trade.alpaca_order_id:
            broker.cancel_order(trade.alpaca_order_id)
        cancel_pending_trade(db, trade.id, reason="eod_unfilled")
        log_decision(
            db, agent="tracker", decision_type="eod_cancel",
            symbol=trade.symbol, trade_id=trade.id,
            reasoning="Order never filled by end of day — cancelled",
        )
        print(f"  [tracker] EOD cancelled unfilled order for {trade.symbol}")
        count += 1
    return count
