"""
Alpaca paper trading execution client.
Handles order placement, position monitoring, and account status.
Works in two modes:
  - live: real Alpaca paper trading API (requires credentials)
  - mock: simulated fills for testing (no credentials needed)
"""
import os
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─── Abstract base ─────────────────────────────────────────────────────────────

class ExecutionClient:

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        raise NotImplementedError

    def place_limit_order(self, symbol: str, qty: float, side: str,
                          limit_price: float, stop_loss: float = None,
                          take_profit: float = None) -> dict:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def close_position(self, symbol: str) -> dict:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[dict]:
        raise NotImplementedError

    def get_all_positions(self) -> list[dict]:
        raise NotImplementedError

    def get_account(self) -> dict:
        raise NotImplementedError

    def is_market_open(self) -> bool:
        raise NotImplementedError


# ─── Mock execution client ────────────────────────────────────────────────────

class MockExecutionClient(ExecutionClient):
    """
    Simulated paper trading. Fills are instant at limit price (optimistic but fine for testing).
    Maintains a small in-memory portfolio so position queries work correctly.
    """

    def __init__(self, starting_cash: float = 100_000.0):
        self._cash = starting_cash
        self._positions: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}
        # Use realistic mock prices from data provider
        from data.market_data import MockDataProvider
        self._dp = MockDataProvider()

    def _get_price(self, symbol: str) -> float:
        bar = self._dp.fetch_latest_bar(symbol)
        return bar["close"] if bar else 100.0

    def _fill_order(self, order_id: str, symbol: str, qty: float,
                    side: str, fill_price: float) -> dict:
        order = {
            "id":          order_id,
            "symbol":      symbol,
            "qty":         qty,
            "side":        side,
            "fill_price":  fill_price,
            "status":      "filled",
            "filled_at":   datetime.utcnow().isoformat(),
        }
        self._orders[order_id] = order

        # Update positions
        cost = qty * fill_price
        if side == "buy":
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["qty"] + qty
                avg_entry = (pos["avg_entry"] * pos["qty"] + fill_price * qty) / total_qty
                pos["qty"] = total_qty
                pos["avg_entry"] = avg_entry
            else:
                self._positions[symbol] = {"symbol": symbol, "qty": qty, "avg_entry": fill_price}
            self._cash -= cost
        elif side == "sell":
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos["qty"] -= qty
                if pos["qty"] <= 0:
                    del self._positions[symbol]
            self._cash += cost

        return order

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        price = self._get_price(symbol)
        # Simulate slight slippage
        slip = 0.001
        fill_price = price * (1 + slip if side == "buy" else 1 - slip)
        return self._fill_order(str(uuid.uuid4()), symbol, qty, side, round(fill_price, 2))

    def place_limit_order(self, symbol: str, qty: float, side: str,
                          limit_price: float, stop_loss: float = None,
                          take_profit: float = None) -> dict:
        # Mock: fill at limit price (assumes market is within spread)
        return self._fill_order(str(uuid.uuid4()), symbol, qty, side, limit_price)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return True
        return False

    def close_position(self, symbol: str) -> dict:
        pos = self._positions.get(symbol)
        if not pos:
            return {"error": f"No position in {symbol}"}
        price = self._get_price(symbol)
        return self._fill_order(str(uuid.uuid4()), symbol, pos["qty"], "sell", price)

    def get_position(self, symbol: str) -> Optional[dict]:
        pos = self._positions.get(symbol)
        if not pos:
            return None
        price = self._get_price(symbol)
        unrealized_pnl = (price - pos["avg_entry"]) * pos["qty"]
        return {
            "symbol":        symbol,
            "qty":           pos["qty"],
            "avg_entry":     pos["avg_entry"],
            "current_price": price,
            "market_value":  round(price * pos["qty"], 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pct": round((unrealized_pnl / (pos["avg_entry"] * pos["qty"])) * 100, 2),
        }

    def get_all_positions(self) -> list[dict]:
        return [self.get_position(sym) for sym in self._positions]

    def get_account(self) -> dict:
        equity = self._cash + sum(
            self._get_price(sym) * pos["qty"]
            for sym, pos in self._positions.items()
        )
        return {
            "cash":             round(self._cash, 2),
            "equity":           round(equity, 2),
            "buying_power":     round(self._cash, 2),
            "portfolio_value":  round(equity, 2),
        }

    def is_market_open(self) -> bool:
        now = datetime.utcnow()
        # Market hours: Mon-Fri, 9:30 AM - 4:00 PM ET (approx UTC-4 in summer)
        if now.weekday() >= 5:
            return False
        market_open_utc = now.replace(hour=13, minute=30, second=0)
        market_close_utc = now.replace(hour=20, minute=0, second=0)
        return market_open_utc <= now <= market_close_utc


# ─── Alpaca live execution client ─────────────────────────────────────────────

class AlpacaExecutionClient(ExecutionClient):
    """
    Real Alpaca paper trading.
    Requires ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in .env
    """

    def __init__(self):
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper      = os.getenv("ALPACA_BASE_URL", "").find("paper") >= 0

        if not api_key or api_key.startswith("your_"):
            raise ValueError("Set Alpaca credentials in .env")

        self.client = TradingClient(api_key, secret_key, paper=True)
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._StopLossRequest = StopLossRequest
        self._TakeProfitRequest = TakeProfitRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce

    def _side(self, side: str):
        return self._OrderSide.BUY if side.lower() == "buy" else self._OrderSide.SELL

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        try:
            req = self._MarketOrderRequest(
                symbol=symbol, qty=int(qty), side=self._side(side),
                time_in_force=self._TimeInForce.DAY
            )
            order = self.client.submit_order(req)
            return {"id": str(order.id), "symbol": symbol, "qty": qty, "side": side,
                    "status": str(order.status), "fill_price": None}
        except Exception as e:
            print(f"  [broker] place_market_order({symbol}) failed: {e}")
            return {"id": None, "symbol": symbol, "qty": qty, "side": side, "status": "failed"}

    def place_limit_order(self, symbol: str, qty: float, side: str,
                          limit_price: float, stop_loss: float = None,
                          take_profit: float = None) -> dict:
        try:
            kwargs = dict(
                symbol=symbol, qty=int(qty), side=self._side(side),
                time_in_force=self._TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
            if stop_loss:
                kwargs["stop_loss"] = self._StopLossRequest(stop_price=round(stop_loss, 2))
            if take_profit:
                kwargs["take_profit"] = self._TakeProfitRequest(limit_price=round(take_profit, 2))
            order = self.client.submit_order(self._LimitOrderRequest(**kwargs))
            return {"id": str(order.id), "symbol": symbol, "qty": qty, "side": side,
                    "limit_price": limit_price, "status": str(order.status), "fill_price": limit_price}
        except Exception as e:
            print(f"  [broker] place_limit_order({symbol}) failed: {e}")
            return {"id": None, "symbol": symbol, "qty": qty, "side": side, "status": "failed"}

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def close_position(self, symbol: str) -> dict:
        result = self.client.close_position(symbol)
        return {"symbol": symbol, "status": "closed"}

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "symbol":          symbol,
                "qty":             float(pos.qty),
                "avg_entry":       float(pos.avg_entry_price),
                "current_price":   float(pos.current_price),
                "market_value":    float(pos.market_value),
                "unrealized_pnl":  float(pos.unrealized_pl),
                "unrealized_pct":  float(pos.unrealized_plpc) * 100,
            }
        except Exception:
            return None

    def get_all_positions(self) -> list[dict]:
        positions = self.client.get_all_positions()
        return [self.get_position(p.symbol) for p in positions]

    def get_account(self) -> dict:
        acct = self.client.get_account()
        return {
            "cash":            float(acct.cash),
            "equity":          float(acct.equity),
            "buying_power":    float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def is_market_open(self) -> bool:
        clock = self.client.get_clock()
        return clock.is_open


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_execution_client() -> ExecutionClient:
    mode = os.getenv("EXECUTION_CLIENT", "mock").lower()
    if mode == "alpaca":
        return AlpacaExecutionClient()
    return MockExecutionClient()
