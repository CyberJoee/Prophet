"""
Position Monitor v2 — reconciliation model.

The old design had TWO owners of exits: Alpaca's server-side bracket legs
AND this monitor calling close_position() on 5-minute polls. That caused
"insufficient qty available" errors (shares held by bracket legs), DB drift
when a leg fired between polls, and gap-through-stop risk.

New design — exactly one owner of exits per mode:
  Alpaca (supports_bracket_orders=True):
      Bracket legs ARE the exit. The monitor only RECONCILES:
        1. confirm pending fills (order_tracker)
        2. detect fired bracket legs → close trade in DB at the leg's
           actual fill price
        3. detect vanished positions → reconcile-close at live price
        4. track true running MAE/MFE on open trades
  Mock (supports_bracket_orders=False):
      No server-side legs exist, so the monitor keeps making price-based
      exit decisions like before (test path only).

Runs every 5 minutes during market hours.
"""
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class PositionMonitor:

    def __init__(self, data_provider=None, execution_client=None, db_session=None):
        if data_provider is None:
            from data.market_data import get_provider
            data_provider = get_provider()
        if execution_client is None:
            from execution.broker import get_execution_client
            execution_client = get_execution_client()
        self.data    = data_provider
        self.broker  = execution_client
        self.db      = db_session
        self._snapshot_counter = 0

    # ─── Main entry point ─────────────────────────────────────────────────────

    def check_positions(self) -> list[dict]:
        """Called every 5 minutes. Returns list of actions taken."""
        if self.db is None:
            return []

        from db.operations import get_open_trades
        from execution.order_tracker import confirm_fills

        # 1. Promote confirmed fills, kill dead orders (phantom-trade fix)
        fill_result = confirm_fills(self.db, self.broker)
        if fill_result["confirmed"] or fill_result["cancelled"]:
            print(f"  [monitor] fills: {fill_result}")

        open_trades = get_open_trades(self.db)
        if not open_trades:
            print(f"  [monitor] {datetime.utcnow().strftime('%H:%M')} — no open positions")
            self._maybe_snapshot()
            return []

        # 2. Batch live prices — ONE request for all symbols
        from data.live_price import get_live_prices
        symbols = list({t.symbol for t in open_trades})
        prices  = get_live_prices(symbols, data_provider=self.data)

        actions = []
        for trade in open_trades:
            if self.broker.supports_bracket_orders:
                action = self._reconcile_trade(trade, prices.get(trade.symbol))
            else:
                action = self._price_check_trade(trade, prices.get(trade.symbol))
            if action:
                actions.append(action)

        self._maybe_snapshot()
        return actions

    # ─── Reconciliation path (Alpaca — brackets own the exits) ────────────────

    def _reconcile_trade(self, trade, current_price: Optional[float]) -> Optional[dict]:
        from db.operations import close_trade, log_decision, update_trade_excursions

        try:
            # A. Did a bracket leg fire since last check?
            leg_fill = self._find_filled_exit_leg(trade)
            if leg_fill:
                exit_price, exit_reason, leg_type = leg_fill
                closed = close_trade(self.db, trade.id,
                                     exit_price=exit_price, exit_reason=exit_reason)
                print(f"  [monitor] BRACKET {leg_type.upper()} FIRED — {trade.symbol} "
                      f"@ ${exit_price:.2f} | PnL: ${closed.pnl:+.2f} ({closed.pnl_pct:+.2f}%)")
                log_decision(
                    self.db, agent="monitor", decision_type=exit_reason,
                    symbol=trade.symbol, trade_id=trade.id,
                    reasoning=f"Alpaca bracket {leg_type} leg filled at ${exit_price:.2f}",
                    output={"exit_price": exit_price, "pnl": closed.pnl},
                )
                return {"symbol": trade.symbol, "exit_reason": exit_reason,
                        "exit_price": exit_price, "pnl": closed.pnl}

            # B. Position gone from Alpaca but no leg fill found?
            #    (manual close, liquidation, sync gap) → reconcile at live price
            position = self.broker.get_position(trade.symbol)
            if position is None and current_price is not None:
                closed = close_trade(self.db, trade.id,
                                     exit_price=current_price, exit_reason="reconciled")
                print(f"  [monitor] RECONCILED {trade.symbol} — position gone from "
                      f"broker, closed @ ${current_price:.2f} | PnL: ${closed.pnl:+.2f}")
                log_decision(
                    self.db, agent="monitor", decision_type="reconciled",
                    symbol=trade.symbol, trade_id=trade.id,
                    reasoning="Position no longer exists at broker; no bracket fill found",
                    output={"exit_price": current_price, "pnl": closed.pnl},
                )
                return {"symbol": trade.symbol, "exit_reason": "reconciled",
                        "exit_price": current_price, "pnl": closed.pnl}

            # C. Still open — track true excursions and log status
            if current_price is not None and trade.entry_price:
                is_long = trade.side.value == "buy"
                pnl = (current_price - trade.entry_price) * trade.quantity * (1 if is_long else -1)
                pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
                update_trade_excursions(self.db, trade.id, pnl)
                print(f"  [monitor] {trade.symbol}: ${current_price:.2f} | "
                      f"unrealized {pnl:+.2f} ({pnl_pct:+.2f}%)")
            return None

        except Exception as e:
            print(f"  [monitor] error reconciling {trade.symbol}: {e}")
            return None

    def _find_filled_exit_leg(self, trade) -> Optional[tuple]:
        """
        Check the parent order's bracket legs for a fill.
        Returns (fill_price, exit_reason, leg_type) or None.
        """
        if not trade.alpaca_order_id:
            return None
        order = self.broker.get_order(trade.alpaca_order_id)
        if not order:
            return None

        for leg in order.get("legs", []):
            if str(leg.get("status", "")).lower() != "filled":
                continue
            fill_price = leg.get("filled_avg_price")
            if not fill_price:
                continue
            leg_type = str(leg.get("order_type", "")).lower()
            if "stop" in leg_type:
                return (float(fill_price), "stop_hit", "stop")
            else:  # limit leg = take profit
                return (float(fill_price), "target_hit", "target")
        return None

    # ─── Price-check path (mock broker only — no server-side legs) ────────────

    def _price_check_trade(self, trade, current_price: Optional[float]) -> Optional[dict]:
        from db.operations import close_trade, log_decision, update_trade_excursions

        try:
            if current_price is None:
                return None
            is_long = trade.side.value == "buy"
            stop, target = trade.planned_stop, trade.planned_target

            exit_reason = None
            if stop is not None and (
                (is_long and current_price <= stop) or
                (not is_long and current_price >= stop)
            ):
                exit_reason = "stop_hit"
            elif target is not None and (
                (is_long and current_price >= target) or
                (not is_long and current_price <= target)
            ):
                exit_reason = "target_hit"

            if exit_reason is None:
                if trade.entry_price:
                    pnl = (current_price - trade.entry_price) * trade.quantity * (1 if is_long else -1)
                    update_trade_excursions(self.db, trade.id, pnl)
                    print(f"  [monitor] {trade.symbol}: ${current_price:.2f} | "
                          f"unrealized {pnl:+.2f}")
                return None

            self.broker.close_position(trade.symbol)
            closed = close_trade(self.db, trade.id,
                                 exit_price=current_price, exit_reason=exit_reason)
            print(f"  [monitor] CLOSED {trade.symbol} — {exit_reason} @ "
                  f"${current_price:.2f} | PnL: ${closed.pnl:+.2f}")
            log_decision(
                self.db, agent="monitor", decision_type=exit_reason,
                symbol=trade.symbol, trade_id=trade.id,
                reasoning=f"{exit_reason} at ${current_price:.2f} (mock price-check path)",
                output={"exit_price": current_price, "pnl": closed.pnl},
            )
            return {"symbol": trade.symbol, "exit_reason": exit_reason,
                    "exit_price": current_price, "pnl": closed.pnl}

        except Exception as e:
            print(f"  [monitor] error checking {trade.symbol}: {e}")
            return None

    # ─── Snapshots ────────────────────────────────────────────────────────────

    def _maybe_snapshot(self):
        self._snapshot_counter += 1
        if self._snapshot_counter >= 6:   # every ~30 min
            self._save_snapshot()
            self._snapshot_counter = 0

    def _save_snapshot(self):
        try:
            from db.operations import save_portfolio_snapshot
            from db.models import PortfolioSnapshot
            acct      = self.broker.get_account()
            positions = self.broker.get_all_positions()
            pos_list  = [p for p in positions if p is not None]

            last = (self.db.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.desc())
                    .first())
            start_equity = 100_000.0
            daily_pnl    = acct["equity"] - (last.equity if last else start_equity)
            total_pnl    = acct["equity"] - start_equity

            save_portfolio_snapshot(
                self.db, cash=acct["cash"], equity=acct["equity"],
                open_positions=pos_list, daily_pnl=daily_pnl, total_pnl=total_pnl,
            )
            print(f"  [monitor] snapshot — equity=${acct['equity']:,.2f} "
                  f"total_pnl={total_pnl:+.2f}")
        except Exception as e:
            print(f"  [monitor] snapshot failed: {e}")

    # ─── End of day ───────────────────────────────────────────────────────────

    def end_of_day(self):
        """
        4:15 PM ET routine — ORDER MATTERS here:
          1. Cancel ALL open orders first. Closing a position while bracket
             legs still hold the shares fails with 'insufficient qty
             available'. Cancelling releases the shares.
          2. Confirm any last-second fills / cancel stale pending trades.
          3. Close remaining positions at the broker.
          4. Close remaining OPEN trades in DB at live prices.
          5. Snapshot + refresh stats.
        """
        from db.operations import get_open_trades, close_trade, log_decision
        from execution.order_tracker import confirm_fills, cancel_stale_pending
        from data.live_price import get_live_prices
        import time

        print("  [monitor] End of day — cancelling all open orders first")
        cancelled = self.broker.cancel_all_orders()
        print(f"  [monitor] cancelled {cancelled} open orders")
        time.sleep(2)  # let cancellations settle before touching positions

        # Reconcile last-second bracket fills before force-closing
        confirm_fills(self.db, self.broker)
        cancel_stale_pending(self.db, self.broker)

        open_trades = get_open_trades(self.db)
        if not open_trades:
            print("  [monitor] EOD — nothing left open")
            self._save_snapshot()
            self._refresh_stats()
            return

        symbols = list({t.symbol for t in open_trades})
        prices  = get_live_prices(symbols, data_provider=self.data)

        for trade in open_trades:
            try:
                # Check one last time whether a bracket leg fired
                if self.broker.supports_bracket_orders:
                    leg_fill = self._find_filled_exit_leg(trade)
                    if leg_fill:
                        exit_price, exit_reason, _ = leg_fill
                        closed = close_trade(self.db, trade.id,
                                             exit_price=exit_price, exit_reason=exit_reason)
                        print(f"  [monitor] EOD found bracket fill {trade.symbol} "
                              f"@ ${exit_price:.2f} | PnL: ${closed.pnl:+.2f}")
                        continue

                exit_price = prices.get(trade.symbol) or trade.entry_price
                self.broker.close_position(trade.symbol)
                closed = close_trade(self.db, trade.id,
                                     exit_price=exit_price, exit_reason="eod_close")
                print(f"  [monitor] EOD closed {trade.symbol} @ ${exit_price:.2f} | "
                      f"PnL: ${closed.pnl:+.2f}")
                log_decision(
                    self.db, agent="monitor", decision_type="eod_close",
                    symbol=trade.symbol, trade_id=trade.id,
                    reasoning="End of day — closing all positions before market close",
                    output={"exit_price": exit_price, "pnl": closed.pnl},
                )
            except Exception as e:
                print(f"  [monitor] EOD close failed for {trade.symbol}: {e}")

        self._save_snapshot()
        self._refresh_stats()

    def _refresh_stats(self):
        try:
            from db.operations import refresh_strategy_stats
            refresh_strategy_stats(self.db)
            print("  [monitor] Strategy stats refreshed")
        except Exception as e:
            print(f"  [monitor] Stats refresh failed: {e}")
