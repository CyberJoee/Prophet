"""
Regression test for the alpaca_sync duplicate-import bug.

Runs the REAL execution.alpaca_sync._sync_closed_orders against a real
SQLAlchemy session (in-memory SQLite) with a fake Alpaca client supplying
order fills. Needs no Postgres and no alpaca-py -- run it directly:

    python tests/test_alpaca_sync_dedup.py

The bug: order_tracker/strategy_agent store the ENTRY (buy) order id in
Trade.alpaca_order_id, but the sync deduped against the EXIT (sell) order
id. The two never matched, so every natively-tracked closed trade was
re-imported as a duplicate row on every sync run, inflating the dashboard
trade count and P&L.

All five checks below FAIL on the pre-fix code and pass after it.
"""
import os, sys, uuid, types
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"

# --- make the postgres-only column types renderable on sqlite -----------
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import UUID
import pgvector.sqlalchemy as pgv

@compiles(UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "CHAR(36)"

@compiles(pgv.Vector, "sqlite")
def _vec_sqlite(type_, compiler, **kw):
    return "BLOB"

# stub out the alpaca sdk (not installed locally, only used for request enums)
alpaca = types.ModuleType("alpaca")
trading = types.ModuleType("alpaca.trading")
reqs = types.ModuleType("alpaca.trading.requests")
enums = types.ModuleType("alpaca.trading.enums")
client_mod = types.ModuleType("alpaca.trading.client")
class GetOrdersRequest:
    def __init__(self, **kw): pass
class QueryOrderStatus:
    CLOSED = "closed"
class TradingClient: pass
reqs.GetOrdersRequest = GetOrdersRequest
enums.QueryOrderStatus = QueryOrderStatus
client_mod.TradingClient = TradingClient
for name, mod in [("alpaca", alpaca), ("alpaca.trading", trading),
                  ("alpaca.trading.requests", reqs),
                  ("alpaca.trading.enums", enums),
                  ("alpaca.trading.client", client_mod)]:
    sys.modules[name] = mod

from db.connection import engine, Base, SessionLocal
import db.models as M
from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
import execution.alpaca_sync as sync

Base.metadata.create_all(bind=engine)
db = SessionLocal()

BUY_ID  = "buy-order-aaa"
SELL_ID = "sell-order-zzz"
T0 = datetime(2026, 8, 20, 13, 45)

class Order:
    def __init__(self, oid, sym, side, price, qty, when):
        self.id, self.symbol, self.side = oid, sym, side
        self.filled_avg_price, self.filled_qty = price, qty
        self.filled_at = self.updated_at = when
        self.status = "filled"

class FakeClient:
    def __init__(self, orders): self._orders = orders
    def get_orders(self, req): return self._orders

ORDERS = [
    Order(BUY_ID,  "AAPL", "buy",  100.0, 10, T0),
    Order(SELL_ID, "AAPL", "sell", 110.0, 10, T0 + timedelta(hours=2)),
]

def count_closed():
    return db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).count()

failures = []
def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{detail}]" if detail and not cond else ""))
    if not cond: failures.append(label)

print("\n=== Case 1: trade already recorded natively (entry/buy order id) ===")
# This is exactly what order_tracker/strategy_agent writes: alpaca_order_id = BUY id
db.add(Trade(symbol="AAPL", asset_type=AssetType.STOCK, setup_type=SetupType.MOMENTUM,
             side=OrderSide.BUY, status=TradeStatus.CLOSED,
             entry_price=100.0, entry_time=T0, quantity=10,
             exit_price=110.0, exit_time=T0 + timedelta(hours=2),
             exit_reason="target_hit", pnl=100.0, pnl_pct=10.0,
             alpaca_order_id=BUY_ID))
db.commit()
before = count_closed()
n = sync._sync_closed_orders(FakeClient(ORDERS), db)
after = count_closed()
check("no duplicate row inserted", after == before, f"{before} -> {after}")
check("reported 0 synced", n == 0, f"returned {n}")

print("\n=== Case 2: repeated syncs are idempotent ===")
for i in range(3):
    sync._sync_closed_orders(FakeClient(ORDERS), db)
check("still one CLOSED row after 3 more syncs", count_closed() == 1, f"{count_closed()} rows")

print("\n=== Case 3: genuinely new trade IS imported ===")
db.query(Trade).delete(); db.commit()
n = sync._sync_closed_orders(FakeClient(ORDERS), db)
check("new trade imported", count_closed() == 1, f"{count_closed()} rows")
check("reported 1 synced", n == 1, f"returned {n}")
t = db.query(Trade).first()
check("pnl correct ((110-100)*10)", t.pnl == 100.0, f"pnl={t.pnl}")
# and immediately re-syncing must not duplicate (this run stored the SELL id)
sync._sync_closed_orders(FakeClient(ORDERS), db)
check("re-sync of self-imported row does not duplicate", count_closed() == 1, f"{count_closed()} rows")

print("\n=== Case 4: OPEN record gets closed, not duplicated (old UnboundLocalError path) ===")
db.query(Trade).delete(); db.commit()
db.add(Trade(symbol="AAPL", asset_type=AssetType.STOCK, setup_type=SetupType.MOMENTUM,
             side=OrderSide.BUY, status=TradeStatus.OPEN,
             entry_price=100.0, entry_time=T0, quantity=10,
             alpaca_order_id="some-other-id"))
db.commit()
try:
    n = sync._sync_closed_orders(FakeClient(ORDERS), db)
    crashed = False
except Exception as e:
    crashed = True
    print(f"   raised {type(e).__name__}: {e}")
check("no UnboundLocalError", not crashed)
if not crashed:
    check("exactly one trade row total", db.query(Trade).count() == 1, f"{db.query(Trade).count()} rows")
    t = db.query(Trade).first()
    check("open trade was closed", t.status == TradeStatus.CLOSED, f"status={t.status}")

print("\n=== Case 5: two sells cannot pair to the same single buy ===")
db.query(Trade).delete(); db.commit()
two_sells = [
    Order("b1", "MSFT", "buy",  50.0, 5, T0),
    Order("s1", "MSFT", "sell", 55.0, 5, T0 + timedelta(hours=1)),
    Order("s2", "MSFT", "sell", 57.0, 5, T0 + timedelta(hours=2)),
]
n = sync._sync_closed_orders(FakeClient(two_sells), db)
rows = db.query(Trade).all()
entries = [r.entry_context.get("buy_order_id") for r in rows]
check("one buy reused at most once", entries.count("b1") <= 1, f"buy ids={entries}")

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
