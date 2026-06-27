"""
Position Monitor
Runs every 5 minutes during market hours.
- Checks all open trades against current prices
- Closes positions that hit stop loss or take profit
- Logs all actions to agent_decisions table
- Saves portfolio snapshot every 30 minutes
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

    def check_positions(self) -> list[dict]:
        """
        Main entry point — called every 5 minutes.
        Returns list of actions taken.
        """
        if self.db is None:
            return []

        from db.operations import get_open_trades, close_trade, log_decision, save_portfolio_snapshot

        open_trades = get_open_trades(self.db)
        if not open_trades:
            print(f"  [monitor] {datetime.utcnow().strftime('%H:%M')} — no open positions")
            return []

        actions = []
        for trade in open_trades:
            action = self._check_trade(trade)
            if action:
                actions.append(action)

        # Save portfolio snapshot every 6 checks (~30 min)
        self._snapshot_counter += 1
        if self._snapshot_counter >= 6:
            self._save_snapshot()
            self._snapshot_counter = 0

        return actions

    def _get_live_price(self, symbol: str) -> Optional[float]:
        """
        Get the current intraday price.
        Priority: Alpaca latest quote → Alpaca latest bar → daily bar fallback.
        """
        # Try Alpaca live quote first (most accurate during market hours)
        try:
            import os
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed
            api_key    = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if api_key and not api_key.startswith("your_"):
                client = StockHistoricalDataClient(api_key, secret_key)
                req    = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
                quotes = client.get_stock_latest_quote(req)
                if symbol in quotes:
                    q = quotes[symbol]
                    # Use mid-price of bid/ask
                    bid = float(q.bid_price or 0)
                    ask = float(q.ask_price or 0)
                    if bid > 0 and ask > 0:
                        return round((bid + ask) / 2, 2)
                    elif ask > 0:
                        return round(ask, 2)
        except Exception:
            pass

        # Fallback: latest 1-minute bar from Alpaca
        try:
            import os
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest
            from alpaca.data.enums import DataFeed
            from datetime import datetime, timedelta
            api_key    = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if api_key and not api_key.startswith("your_"):
                client = StockHistoricalDataClient(api_key, secret_key)
                req    = StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
                bars   = client.get_stock_latest_bar(req)
                if symbol in bars:
                    return round(float(bars[symbol].close), 2)
        except Exception:
            pass

        # Final fallback: daily bar (mock mode or no credentials)
        bar = self.data.fetch_latest_bar(symbol)
        if bar and bar.get("close"):
            return bar["close"]
        return None

    def _check_trade(self, trade) -> Optional[dict]:
        """Check a single trade against current live price. Returns action dict if closed."""
        from db.operations import close_trade, log_decision

        try:
            # Get current LIVE price (intraday, not daily close)
            current_price = self._get_live_price(trade.symbol)
            if current_price is None:
                return None
            side          = trade.side.value  # "buy" or "sell"
            stop          = trade.planned_stop
            target        = trade.planned_target

            exit_reason = None
            is_long = side == "buy"

            # Check stop loss
            if stop is not None:
                if (is_long and current_price <= stop) or \
                   (not is_long and current_price >= stop):
                    exit_reason = "stop_hit"

            # Check take profit (only if stop wasn't hit)
            if exit_reason is None and target is not None:
                if (is_long and current_price >= target) or \
                   (not is_long and current_price <= target):
                    exit_reason = "target_hit"

            if exit_reason is None:
                # Position still open — log current status
                if trade.entry_price:
                    pnl = (current_price - trade.entry_price) * trade.quantity * (1 if is_long else -1)
                    pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
                    print(f"  [monitor] {trade.symbol}: ${current_price:.2f} | "
                          f"unrealized {'+'if pnl>=0 else ''}{pnl:.2f} ({pnl_pct:+.2f}%)")
                return None

            # Close the position
            close_order = self.broker.close_position(trade.symbol)

            # Calculate excursions
            max_adverse   = None
            max_favorable = None
            if trade.entry_price:
                pnl = (current_price - trade.entry_price) * trade.quantity * (1 if is_long else -1)
                max_adverse   = min(0, pnl)
                max_favorable = max(0, pnl)

            closed = close_trade(
                self.db, trade.id,
                exit_price=current_price,
                exit_reason=exit_reason,
                max_adverse=max_adverse,
                max_favorable=max_favorable,
            )

            pnl_str = f"{'+'if closed.pnl>=0 else ''}{closed.pnl:.2f}"
            print(f"  [monitor] CLOSED {trade.symbol} — {exit_reason} @ ${current_price:.2f} | "
                  f"PnL: ${pnl_str} ({closed.pnl_pct:+.2f}%)")

            log_decision(
                self.db,
                agent="monitor",
                decision_type=exit_reason,
                symbol=trade.symbol,
                trade_id=trade.id,
                reasoning=f"{exit_reason} triggered at ${current_price:.2f}. "
                           f"Entry was ${trade.entry_price:.2f}. PnL: ${closed.pnl:.2f}",
                inputs={"current_price": current_price, "stop": stop, "target": target},
                output={"exit_price": current_price, "pnl": closed.pnl, "pnl_pct": closed.pnl_pct},
            )

            return {
                "symbol":      trade.symbol,
                "exit_reason": exit_reason,
                "exit_price":  current_price,
                "pnl":         closed.pnl,
            }

        except Exception as e:
            print(f"  [monitor] error checking {trade.symbol}: {e}")
            return None

    def _save_snapshot(self):
        """Save current portfolio state to DB."""
        try:
            from db.operations import save_portfolio_snapshot
            acct      = self.broker.get_account()
            positions = self.broker.get_all_positions()
            pos_list  = [p for p in positions if p is not None]

            # Calculate daily and total PnL from DB
            from db.models import PortfolioSnapshot
            last = (self.db.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.desc())
                    .first())

            start_equity = 100_000.0
            daily_pnl    = acct["equity"] - (last.equity if last else start_equity)
            total_pnl    = acct["equity"] - start_equity

            save_portfolio_snapshot(
                self.db,
                cash=acct["cash"],
                equity=acct["equity"],
                open_positions=pos_list,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
            )
            print(f"  [monitor] snapshot saved — equity=${acct['equity']:,.2f} "
                  f"total_pnl={'+'if total_pnl>=0 else ''}{total_pnl:.2f}")
        except Exception as e:
            print(f"  [monitor] snapshot failed: {e}")

    def end_of_day(self):
        """
        4:15 PM ET routine.
        Closes all remaining open positions (no overnight holds in paper mode).
        Saves final portfolio snapshot.
        """
        from db.operations import get_open_trades, close_trade, log_decision

        print("  [monitor] End of day — closing all open positions")
        open_trades = get_open_trades(self.db)

        for trade in open_trades:
            try:
                bar = self.data.fetch_latest_bar(trade.symbol)
                exit_price = bar["close"] if bar and bar.get("close") else trade.entry_price
                is_long = trade.side.value == "buy"

                self.broker.close_position(trade.symbol)
                closed = close_trade(
                    self.db, trade.id,
                    exit_price=exit_price,
                    exit_reason="eod_close",
                )
                print(f"  [monitor] EOD closed {trade.symbol} @ ${exit_price:.2f} | "
                      f"PnL: ${closed.pnl:.2f}")
                log_decision(
                    self.db, agent="monitor", decision_type="eod_close",
                    symbol=trade.symbol, trade_id=trade.id,
                    reasoning="End of day — closing all positions before market close",
                    output={"exit_price": exit_price, "pnl": closed.pnl},
                )
            except Exception as e:
                print(f"  [monitor] EOD close failed for {trade.symbol}: {e}")

        # Final snapshot
        self._save_snapshot()

        # Refresh strategy stats
        try:
            from db.operations import refresh_strategy_stats
            refresh_strategy_stats(self.db)
            print("  [monitor] Strategy stats refreshed")
        except Exception as e:
            print(f"  [monitor] Stats refresh failed: {e}")
