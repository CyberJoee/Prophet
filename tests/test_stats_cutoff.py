"""
Regression test for the STATS_SINCE cutoff.

/api/performance summed P&L over ALL closed trades with no date filter,
including the 67 corrupted pre-v2 trades that StrategyStats excludes -- so
the dashboard reported ~+$4,543 while /api/portfolio, reading real Alpaca
equity, reported ~+$1,710.

Both call sites now share db.operations.get_stats_cutoff().
Run directly: python tests/test_stats_cutoff.py
"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import UUID
import pgvector.sqlalchemy as pgv
@compiles(UUID, "sqlite")
def _u(t,c,**k): return "CHAR(36)"
@compiles(pgv.Vector, "sqlite")
def _v(t,c,**k): return "BLOB"

from db.connection import engine, Base, SessionLocal
from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
from db.operations import get_stats_cutoff
Base.metadata.create_all(bind=engine)
db = SessionLocal()

def mk(when, pnl):
    return Trade(symbol="AAPL", asset_type=AssetType.STOCK, setup_type=SetupType.MOMENTUM,
        side=OrderSide.BUY, status=TradeStatus.CLOSED, entry_price=100.0,
        entry_time=when, quantity=1, exit_price=100+pnl,
        exit_time=when, exit_reason="x", pnl=pnl, pnl_pct=1.0)

# 3 corrupted pre-cutoff trades (+3000 total), 2 real post-cutoff (-100 total)
db.add_all([mk(datetime(2026,6,1),1000.0), mk(datetime(2026,6,15),1000.0),
            mk(datetime(2026,7,1),1000.0),
            mk(datetime(2026,7,2),-50.0),  mk(datetime(2026,8,1),-50.0)])
db.commit()

fails=[]
def check(l,c,d=""):
    print(("  PASS  " if c else "  FAIL  ")+l+(f"  [{d}]" if d and not c else ""))
    if not c: fails.append(l)

cutoff = get_stats_cutoff()
check("default cutoff is 2026-07-02", cutoff == datetime(2026,7,2), str(cutoff))

counted = db.query(Trade).filter(Trade.status==TradeStatus.CLOSED,
                                 Trade.entry_time >= cutoff).all()
excluded = db.query(Trade).filter(Trade.status==TradeStatus.CLOSED,
                                  Trade.entry_time < cutoff).count()
check("counts only post-cutoff trades", len(counted)==2, f"{len(counted)}")
check("sums to -100 not +2900", sum(t.pnl for t in counted)==-100.0,
      f"{sum(t.pnl for t in counted)}")
check("reports 3 excluded", excluded==3, f"{excluded}")
check("cutoff date is inclusive", any(t.entry_time==datetime(2026,7,2) for t in counted))

os.environ["STATS_SINCE"] = "2026-08-01"
check("env override respected", get_stats_cutoff()==datetime(2026,8,1), str(get_stats_cutoff()))
os.environ["STATS_SINCE"] = "garbage"
check("invalid value falls back to epoch", get_stats_cutoff()==datetime(1970,1,1), str(get_stats_cutoff()))
del os.environ["STATS_SINCE"]

print("\n"+("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
