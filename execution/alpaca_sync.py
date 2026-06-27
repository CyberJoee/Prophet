"""
Alpaca Sync
Pulls real positions and closed orders from Alpaca and writes them
into the local DB so the position monitor, journal, and performance
stats all have accurate data to work with.

Runs on scheduler startup and after each morning pipeline.
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def sync_alpaca_to_db(db) -> dict:
    """
    Main entry point. Syncs open positions and recent closed orders
    from Alpaca into the local trades table.

    Returns dict with counts of what was synced.
    """
    result = {"open_synced": 0, "closed_synced": 0, "errors": []}

    try:
        from alpaca.trading.client import TradingClient
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or api_key.startswith("your_"):
            print("  [sync] No Alpaca credentials — skipping sync")
            return result

        client = TradingClient(api_key, secret_key, paper=True)

        # Sync open positions
        open_count = _sync_open_positions(client, db)
        result["open_synced"] = open_count

        # Sync closed orders from last 30 days
        closed_count = _sync_closed_orders(client, db)
        result["closed_synced"] = closed_count

        print(f"  [sync] ✓ Synced {open_count} open positions, {closed_count} closed orders from Alpaca")

    except Exception as e:
        msg = f"Alpaca sync failed: {e}"
        print(f"  [sync] ✗ {msg}")
        result["errors"].append(msg)

    return result


def _sync_open_positions(client, db) -> int:
    """Pull open positions from Alpaca and create/update DB records."""
    from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
    from db.operations import open_trade

    try:
        positions = client.get_all_positions()
    except Exception as e:
        print(f"  [sync] get_all_positions failed: {e}")
        return 0

    count = 0
    for pos in positions:
        symbol = pos.symbol

        # Check if we already have an open trade for this symbol
        existing = (db.query(Trade)
                    .filter(Trade.symbol == symbol,
                            Trade.status == TradeStatus.OPEN)
                    .first())
        if existing:
            continue  # already tracked

        try:
            qty        = abs(float(pos.qty))
            avg_entry  = float(pos.avg_entry_price)
            side       = "buy" if float(pos.qty) > 0 else "sell"
            market_val = float(pos.market_value)
            unrealized = float(pos.unrealized_pl)

            # Estimate stop and target from current price
            current    = float(pos.current_price)
            atr_est    = current * 0.02  # rough 2% ATR estimate
            stop       = round(current - 2 * atr_est, 2) if side == "buy" else round(current + 2 * atr_est, 2)
            target     = round(current + 4 * atr_est, 2) if side == "buy" else round(current - 4 * atr_est, 2)

            # Try to infer setup type from agent_decisions for this symbol
            inferred_setup = SetupType.MOMENTUM  # default to momentum not custom
            try:
                from db.models import AgentDecision
                decision = (db.query(AgentDecision)
                            .filter(AgentDecision.symbol == symbol,
                                    AgentDecision.decision_type == "enter")
                            .order_by(AgentDecision.created_at.desc())
                            .first())
                if decision and decision.inputs and isinstance(decision.inputs, dict):
                    st = decision.inputs.get("setup_type")
                    if st and st != "custom":
                        valid = ["momentum","orb","vwap_bounce","reversal","options_play","earnings"]
                        if st in valid:
                            inferred_setup = SetupType(st)
            except Exception:
                pass

            trade = Trade(
                symbol=symbol,
                asset_type=AssetType.STOCK,
                setup_type=inferred_setup,
                side=OrderSide(side),
                status=TradeStatus.OPEN,
                entry_price=avg_entry,
                entry_time=datetime.utcnow(),
                quantity=qty,
                planned_stop=stop,
                planned_target=target,
                entry_context={
                    "source":        "alpaca_sync",
                    "market_value":  market_val,
                    "unrealized_pl": unrealized,
                    "synced_at":     datetime.utcnow().isoformat(),
                },
            )
            db.add(trade)
            db.commit()
            count += 1
            print(f"  [sync] Imported open position: {symbol} {side} x{qty} @ ${avg_entry:.2f}")

        except Exception as e:
            db.rollback()
            print(f"  [sync] Failed to import {symbol}: {e}")

    return count


def _sync_closed_orders(client, db) -> int:
    """Pull filled+closed orders from Alpaca and create closed trade records."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType

    try:
        since = datetime.utcnow() - timedelta(days=30)
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=since,
            limit=100,
        )
        orders = client.get_orders(req)
    except Exception as e:
        print(f"  [sync] get_orders failed: {e}")
        return 0

    count = 0
    # Group orders by symbol to pair buys with sells
    buy_orders  = {}
    sell_orders = {}

    for order in orders:
        # Only process fully filled orders
        if str(order.status) not in ("filled", "OrderStatus.FILLED"):
            continue
        sym  = order.symbol
        side = str(order.side).lower().replace("orderside.", "")
        try:
            fill_price = float(order.filled_avg_price or 0)
            qty        = float(order.filled_qty or 0)
            filled_at  = order.filled_at or order.updated_at
        except Exception:
            continue

        if side == "buy":
            buy_orders.setdefault(sym, []).append(
                {"price": fill_price, "qty": qty, "time": filled_at, "id": str(order.id)}
            )
        elif side == "sell":
            sell_orders.setdefault(sym, []).append(
                {"price": fill_price, "qty": qty, "time": filled_at, "id": str(order.id)}
            )

    # For each sell, find the matching buy and create a closed trade
    for sym, sells in sell_orders.items():
        buys = buy_orders.get(sym, [])
        for sell in sells:
            # Check if already recorded by order ID
            existing = (db.query(Trade)
                        .filter(Trade.alpaca_order_id == sell["id"])
                        .first())
            if existing:
                continue

            # Also check if there's an open trade we should close instead
            open_trade_rec = (db.query(Trade)
                              .filter(Trade.symbol == sym,
                                      Trade.status == TradeStatus.OPEN)
                              .first())
            if open_trade_rec and entry_price:
                # Close the existing open trade record
                from db.operations import close_trade
                try:
                    close_trade(db, open_trade_rec.id,
                                exit_price=sell["price"],
                                exit_reason="alpaca_sync")
                    count += 1
                    print(f"  [sync] Closed existing open trade: {sym} PnL=${sell['price']-open_trade_rec.entry_price:.2f}/share")
                    continue
                except Exception:
                    pass

            # Find best matching buy (closest in time before the sell)
            entry_price = None
            entry_time  = None
            entry_id    = None
            for buy in sorted(buys, key=lambda b: b["time"] or datetime.utcnow()):
                buy_time  = buy["time"]
                sell_time = sell["time"]
                if buy_time and sell_time and buy_time <= sell_time:
                    entry_price = buy["price"]
                    entry_time  = buy_time
                    entry_id    = buy["id"]

            if not entry_price:
                entry_price = sell["price"] * 0.99  # fallback estimate
                entry_time  = sell["time"]

            try:
                qty  = sell["qty"]
                pnl  = (sell["price"] - entry_price) * qty
                pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price else 0

                trade = Trade(
                    symbol=sym,
                    asset_type=AssetType.STOCK,
                    setup_type=SetupType.CUSTOM,
                    side=OrderSide.BUY,
                    status=TradeStatus.CLOSED,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    quantity=qty,
                    exit_price=sell["price"],
                    exit_time=sell["time"],
                    exit_reason="alpaca_sync",
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 4),
                    alpaca_order_id=sell["id"],
                    entry_context={
                        "source":      "alpaca_sync",
                        "buy_order_id": entry_id,
                        "synced_at":   datetime.utcnow().isoformat(),
                    },
                )
                db.add(trade)
                db.commit()
                count += 1
                print(f"  [sync] Imported closed trade: {sym} buy @ ${entry_price:.2f} → sell @ ${sell['price']:.2f} PnL=${pnl:.2f}")

            except Exception as e:
                db.rollback()
                print(f"  [sync] Failed to import closed trade {sym}: {e}")

    return count


def sync_mock(db) -> dict:
    """
    Mock sync for testing — creates realistic fake trades in the DB.
    Used when Alpaca credentials aren't available.
    """
    from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
    from data.market_data import MockDataProvider

    dp = MockDataProvider()
    count_open = 0
    count_closed = 0

    # Create 2 open positions
    for sym, setup in [("NVDA", SetupType.MOMENTUM), ("SPY", SetupType.VWAP_BOUNCE)]:
        existing = db.query(Trade).filter(Trade.symbol == sym, Trade.status == TradeStatus.OPEN).first()
        if existing:
            continue
        bar = dp.fetch_latest_bar(sym)
        close = bar["close"]
        trade = Trade(
            symbol=sym, asset_type=AssetType.STOCK, setup_type=setup,
            side=OrderSide.BUY, status=TradeStatus.OPEN,
            entry_price=round(close * 0.98, 2), quantity=50,
            entry_time=datetime.utcnow() - timedelta(hours=3),
            planned_stop=round(close * 0.93, 2),
            planned_target=round(close * 1.06, 2),
            entry_context={"source": "mock_sync"},
        )
        db.add(trade)
        db.commit()
        count_open += 1

    # Create 3 closed trades with real PnL
    closed_data = [
        ("AAPL", 0.97, 1.04, "target_hit", SetupType.MOMENTUM),
        ("TSLA", 0.98, 0.94, "stop_hit",   SetupType.ORB),
        ("MSFT", 0.97, 1.03, "target_hit", SetupType.VWAP_BOUNCE),
    ]
    for sym, entry_mult, exit_mult, reason, setup in closed_data:
        existing = db.query(Trade).filter(
            Trade.symbol == sym, Trade.status == TradeStatus.CLOSED,
            Trade.exit_reason == reason,
        ).first()
        if existing:
            continue
        bar = dp.fetch_latest_bar(sym)
        close = bar["close"]
        entry = round(close * entry_mult, 2)
        exit_ = round(close * exit_mult, 2)
        qty   = 30
        pnl   = round((exit_ - entry) * qty, 2)
        trade = Trade(
            symbol=sym, asset_type=AssetType.STOCK, setup_type=setup,
            side=OrderSide.BUY, status=TradeStatus.CLOSED,
            entry_price=entry, quantity=qty,
            entry_time=datetime.utcnow() - timedelta(hours=5),
            exit_price=exit_, exit_time=datetime.utcnow() - timedelta(hours=1),
            exit_reason=reason, pnl=pnl,
            pnl_pct=round((pnl / (entry * qty)) * 100, 4),
            entry_context={"source": "mock_sync"},
        )
        db.add(trade)
        db.commit()
        count_closed += 1

    # Refresh stats
    from db.operations import refresh_strategy_stats
    refresh_strategy_stats(db)

    print(f"  [sync] Mock: {count_open} open + {count_closed} closed trades inserted")
    return {"open_synced": count_open, "closed_synced": count_closed, "errors": []}
