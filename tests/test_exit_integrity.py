"""
Tests for exit integrity — the fixes for the 2026-08-28 audit findings.

The audit found an MSFT position held at Alpaca that the database did not know
about, plus eighteen historical cases of the same shape. Three defects
compounded:

  A. order_tracker promoted a trade to OPEN on `filled_qty > 0`, so a partial
     fill looked complete and the working remainder became invisible.
  B. position_monitor.end_of_day() marked trades CLOSED at the live quote
     without confirming the close filled.
  C. End of day ran at 16:15 ET, after the 16:00 bell, so the close was
     submitted into a shut market — with the bracket legs already cancelled.

These are live order-handling paths. A regression here does not produce a
wrong number on a dashboard, it produces an unhedged position nobody knows
about, so every branch is pinned down.

No network, no Postgres, no broker. Run directly:

    python tests/test_exit_integrity.py
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import UUID
import pgvector.sqlalchemy as pgv


@compiles(UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "CHAR(36)"


@compiles(pgv.Vector, "sqlite")
def _vec_sqlite(type_, compiler, **kw):
    return "BLOB"


# data.market_data pulls in pandas/pandas-ta, which the suite deliberately
# does not depend on. Broker and monitor only ever ask it for a bar, so a stub
# module keeps the whole market-data stack out of CI.
import types as _types
_md = _types.ModuleType("data.market_data")


class _MockDataProvider:
    def fetch_latest_bar(self, symbol):
        return {"close": 100.0, "high": 101.0, "low": 99.0,
                "open": 100.0, "volume": 1_000_000, "atr_14": 1.5}

    def fetch_bars(self, symbol, days=90):
        return []


_md.MockDataProvider = _MockDataProvider
_md.get_provider = lambda: _MockDataProvider()
sys.modules["data.market_data"] = _md

from db.connection import engine, Base, SessionLocal
from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
from db.operations import update_trade_quantity
from execution.order_tracker import confirm_fills
from execution.reconciliation import reconcile
from execution.position_monitor import PositionMonitor

Base.metadata.create_all(bind=engine)
db = SessionLocal()

failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def reset():
    db.query(Trade).delete()
    db.commit()


def make_trade(symbol="SPY", qty=20.0, status=TradeStatus.PENDING_FILL,
               order_id="order-1", side=OrderSide.SELL):
    t = Trade(symbol=symbol, asset_type=AssetType.STOCK,
              setup_type=SetupType.MOMENTUM, side=side, status=status,
              entry_price=762.37, entry_time=datetime.utcnow(), quantity=qty,
              planned_stop=765.66, planned_target=755.70,
              alpaca_order_id=order_id, entry_context={"reasoning": "test"})
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


class _StubProvider:
    """Stands in for the market-data provider so tests need no pandas."""
    def fetch_latest_bar(self, symbol):
        return {"close": 763.0, "atr_14": 6.6}

    def get_latest_price(self, symbol):
        return 763.0


class FakeBroker:
    """Minimal broker double. Records what it was asked to do."""
    supports_bracket_orders = False

    def __init__(self, orders=None, positions=None, close_orders=None):
        self.orders = orders or {}
        self.positions = positions or {}
        self.close_orders = close_orders or {}
        self.closed_calls = []
        self.get_order_calls = 0

    def get_order(self, oid):
        self.get_order_calls += 1
        o = self.orders.get(oid)
        return o() if callable(o) else o

    def cancel_order(self, oid):
        return True

    def cancel_all_orders(self):
        return 0

    def get_all_positions(self):
        return list(self.positions.values())

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def get_account(self):
        return {"cash": 100000.0, "equity": 100000.0, "buying_power": 100000.0}

    def close_position(self, symbol):
        self.closed_calls.append(symbol)
        return self.close_orders.get(symbol, {"id": None, "symbol": symbol,
                                              "status": "failed",
                                              "filled_qty": 0.0,
                                              "filled_avg_price": None})


# ── A. Partial fills ────────────────────────────────────────────────────────
print("\n=== A. A partial fill is NOT a fill ===")
reset()
t = make_trade(qty=20.0)
broker = FakeBroker(orders={"order-1": {
    "id": "order-1", "status": "partially_filled",
    "filled_qty": 15.0, "filled_avg_price": 762.37, "filled_at": None}})
res = confirm_fills(db, broker)
db.refresh(t)
check("stays PENDING_FILL", t.status == TradeStatus.PENDING_FILL, str(t.status))
check("counted as still pending", res["still_pending"] == 1, str(res))
check("not counted as confirmed", res["confirmed"] == 0, str(res))
check("quantity tracks what actually filled (15, not 20)", t.quantity == 15.0,
      str(t.quantity))

print("\n--- and once it completes, it promotes ---")
broker.orders["order-1"] = {"id": "order-1", "status": "filled",
                            "filled_qty": 20.0, "filled_avg_price": 762.40,
                            "filled_at": None}
res = confirm_fills(db, broker)
db.refresh(t)
check("promoted to OPEN", t.status == TradeStatus.OPEN, str(t.status))
check("final quantity is the full fill", t.quantity == 20.0, str(t.quantity))
check("entry price is the broker's fill", t.entry_price == 762.40, str(t.entry_price))

print("\n--- a dead order is still cancelled, not left hanging ---")
reset()
t = make_trade(order_id="order-dead")
broker = FakeBroker(orders={"order-dead": {
    "id": "order-dead", "status": "canceled", "filled_qty": 0.0,
    "filled_avg_price": None}})
res = confirm_fills(db, broker)
db.refresh(t)
check("cancelled", t.status == TradeStatus.CANCELLED, str(t.status))

# ── B. EOD close must be confirmed ──────────────────────────────────────────
print("\n=== B. EOD close is not recorded until the fill is confirmed ===")


def eod_with(close_order, orders=None, positions=None):
    reset()
    tr = make_trade(status=TradeStatus.OPEN)
    b = FakeBroker(orders=orders or {}, positions=positions or {},
                   close_orders={"SPY": close_order})
    # A stub provider keeps pandas (and the whole market-data stack) out of
    # the test; end_of_day only needs it to look up quotes it no longer uses
    # for the exit price.
    m = PositionMonitor(data_provider=_StubProvider(),
                        db_session=db, execution_client=b)
    m._close_attempts, m._close_delay = 1, 0     # no sleeping in tests
    m.end_of_day()
    db.refresh(tr)
    return tr, b


print("\n--- the close fills: recorded at the FILL price ---")
tr, b = eod_with({"id": "c1", "symbol": "SPY", "status": "filled",
                  "filled_qty": 20.0, "filled_avg_price": 763.11})
check("marked CLOSED", tr.status == TradeStatus.CLOSED, str(tr.status))
check("exit price is the fill, not a quote", tr.exit_price == 763.11,
      str(tr.exit_price))
check("exit reason eod_close", tr.exit_reason == "eod_close", str(tr.exit_reason))

print("\n--- the close does NOT fill: trade stays OPEN ---")
tr, b = eod_with({"id": "c2", "symbol": "SPY", "status": "accepted",
                  "filled_qty": 0.0, "filled_avg_price": None},
                 orders={"c2": {"id": "c2", "status": "accepted",
                                "filled_qty": 0.0, "filled_avg_price": None}})
check("NOT marked closed", tr.status == TradeStatus.OPEN, str(tr.status))
check("no exit price invented", tr.exit_price is None, str(tr.exit_price))
check("close was actually attempted", b.closed_calls == ["SPY"], str(b.closed_calls))
check("it polled for the fill", b.get_order_calls > 0, str(b.get_order_calls))

print("\n--- the close is rejected: trade stays OPEN, polling stops early ---")
tr, b = eod_with({"id": "c3", "symbol": "SPY", "status": "accepted",
                  "filled_qty": 0.0, "filled_avg_price": None},
                 orders={"c3": {"id": "c3", "status": "rejected",
                                "filled_qty": 0.0, "filled_avg_price": None}})
check("NOT marked closed", tr.status == TradeStatus.OPEN, str(tr.status))
check("stopped polling on a terminal status", b.get_order_calls == 1,
      str(b.get_order_calls))

print("\n--- close_position itself errors: trade stays OPEN ---")
tr, b = eod_with({"id": None, "symbol": "SPY", "status": "failed",
                  "filled_qty": 0.0, "filled_avg_price": None,
                  "error": "market closed"})
check("NOT marked closed", tr.status == TradeStatus.OPEN, str(tr.status))

# ── C. Reconciliation ───────────────────────────────────────────────────────
print("\n=== C. Reconciliation reports disagreement loudly ===")
reset()
make_trade(symbol="NVDA", status=TradeStatus.OPEN, order_id="o-nvda")
r = reconcile(db, FakeBroker(positions={"NVDA": {"symbol": "NVDA", "qty": 10}}))
check("agreement -> ok", r["ok"] is True, str(r))
check("nothing flagged", not r["untracked"] and not r["phantom"], str(r))

r = reconcile(db, FakeBroker(positions={"NVDA": {"symbol": "NVDA", "qty": 10},
                                        "MSFT": {"symbol": "MSFT", "qty": 30}}))
check("untracked position detected", r["untracked"] == ["MSFT"], str(r["untracked"]))
check("not ok", r["ok"] is False)

r = reconcile(db, FakeBroker(positions={}))
check("phantom position detected", r["phantom"] == ["NVDA"], str(r["phantom"]))
check("not ok", r["ok"] is False)


class ExplodingBroker(FakeBroker):
    def get_all_positions(self):
        raise RuntimeError("broker unreachable")


r = reconcile(db, ExplodingBroker())
check("an unreachable broker is NOT reported as agreement", r["ok"] is False, str(r))
check("the error is recorded", "unreachable" in (r["error"] or ""), str(r["error"]))
check("nothing invented while blind", r["untracked"] == [] and r["phantom"] == [])

print("\n--- reconcile never raises, whatever it is handed ---")
try:
    reconcile(db, object())
    check("survives a nonsense broker", True)
except Exception as e:
    check("survives a nonsense broker", False, f"{type(e).__name__}: {e}")

# ── D. Schedule ─────────────────────────────────────────────────────────────
print("\n=== D. End of day runs while the market is open ===")
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "scheduler.py"), encoding="utf-8").read()
check("EOD cron is 15:50, not 16:15",
      "hour=15, minute=50" in src and "hour=16, minute=15" not in src)
check("startup reconciliation is wired in", "reconcile(_db" in src)

# ── E. close_position returns something confirmable ─────────────────────────
print("\n=== E. Both brokers return an order, not a success sentinel ===")
from execution.broker import MockExecutionClient
mock = MockExecutionClient()
mock.place_market_order("AAPL", 10, "buy")
out = mock.close_position("AAPL")
check("has an id", out.get("id"), str(out))
check("has a fill price", out.get("filled_avg_price"), str(out))
check("has a filled qty", out.get("filled_qty"), str(out))
missing = mock.close_position("NOSUCH")
check("no position -> failed, not a fake success",
      missing.get("status") == "failed", str(missing))

src_b = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "execution", "broker.py"), encoding="utf-8").read()
check("Alpaca client no longer returns the {'status': 'closed'} sentinel",
      'return {"symbol": symbol, "status": "closed"}' not in src_b)

db.close()
print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
