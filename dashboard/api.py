"""
Prophet Dashboard API
FastAPI backend that serves the React frontend and exposes
all trading data from the Postgres database.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
from typing import Optional

load_dotenv()

app = FastAPI(title="Prophet Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    from db.connection import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    """Current portfolio state — cash, equity, open positions, P&L."""
    from db.connection import SessionLocal
    from db.models import PortfolioSnapshot, Trade, TradeStatus
    from execution.broker import get_execution_client

    db = SessionLocal()
    try:
        # Latest snapshot from DB
        snap = (db.query(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .first())

        # Live account from broker
        try:
            broker = get_execution_client()
            acct   = broker.get_account()
            positions = broker.get_all_positions()
        except Exception:
            acct      = {"cash": 0, "equity": 0, "buying_power": 0}
            positions = []

        # Open trades from DB
        open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()

        starting_equity = 100_000.0
        total_pnl       = acct["equity"] - starting_equity
        daily_pnl       = (acct["equity"] - snap.equity) if snap else 0

        return {
            "cash":           round(acct["cash"], 2),
            "equity":         round(acct["equity"], 2),
            "buying_power":   round(acct.get("buying_power", acct["cash"]), 2),
            "total_pnl":      round(total_pnl, 2),
            "total_pnl_pct":  round((total_pnl / starting_equity) * 100, 4),
            "daily_pnl":      round(daily_pnl, 2),
            "open_positions": len(open_trades),
            "last_updated":   datetime.utcnow().isoformat(),
        }
    finally:
        db.close()


@app.get("/api/portfolio/history")
def get_portfolio_history(days: int = 30):
    """Equity curve data for charting."""
    from db.connection import SessionLocal
    from db.models import PortfolioSnapshot

    db = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        snaps = (db.query(PortfolioSnapshot)
                 .filter(PortfolioSnapshot.timestamp >= since)
                 .order_by(PortfolioSnapshot.timestamp.asc())
                 .all())
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "equity":    round(s.equity, 2),
                "cash":      round(s.cash, 2),
                "daily_pnl": round(s.daily_pnl or 0, 2),
                "total_pnl": round(s.total_pnl or 0, 2),
            }
            for s in snaps
        ]
    finally:
        db.close()


# ─── Positions ─────────────────────────────────────────────────────────────────

@app.get("/api/positions")
def get_positions():
    """All currently open positions."""
    from db.connection import SessionLocal
    from db.models import Trade, TradeStatus
    from data.market_data import get_provider

    db = SessionLocal()
    try:
        trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
        try:
            dp = get_provider()
        except Exception:
            dp = None

        result = []
        for t in trades:
            current_price = None
            unrealized_pnl = None
            unrealized_pct = None
            if dp:
                try:
                    bar = dp.fetch_latest_bar(t.symbol)
                    if bar and bar.get("close"):
                        current_price  = bar["close"]
                        mult           = 1 if t.side.value == "buy" else -1
                        unrealized_pnl = (current_price - t.entry_price) * t.quantity * mult
                        unrealized_pct = (unrealized_pnl / (t.entry_price * t.quantity)) * 100
                except Exception:
                    pass

            result.append({
                "id":             str(t.id),
                "symbol":         t.symbol,
                "side":           t.side.value,
                "setup_type":     t.setup_type.value if hasattr(t.setup_type, 'value') else str(t.setup_type),
                "quantity":       t.quantity,
                "entry_price":    t.entry_price,
                "current_price":  current_price,
                "planned_stop":   t.planned_stop,
                "planned_target": t.planned_target,
                "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
                "unrealized_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
                "entry_time":     t.entry_time.isoformat() if t.entry_time else None,
            })
        return result
    finally:
        db.close()


# ─── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/trades")
def get_trades(limit: int = 50, setup_type: Optional[str] = None):
    """Closed trade history."""
    from db.connection import SessionLocal
    from db.models import Trade, TradeStatus

    db = SessionLocal()
    try:
        q = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED)
        if setup_type:
            q = q.filter(Trade.setup_type == setup_type)
        trades = q.order_by(Trade.exit_time.desc()).limit(limit).all()

        return [
            {
                "id":           str(t.id),
                "symbol":       t.symbol,
                "side":         t.side.value,
                "setup_type":   t.setup_type.value if hasattr(t.setup_type, 'value') else str(t.setup_type),
                "quantity":     t.quantity,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "pnl":          round(t.pnl, 2) if t.pnl is not None else None,
                "pnl_pct":      round(t.pnl_pct, 2) if t.pnl_pct is not None else None,
                "exit_reason":  t.exit_reason,
                "entry_time":   t.entry_time.isoformat() if t.entry_time else None,
                "exit_time":    t.exit_time.isoformat() if t.exit_time else None,
            }
            for t in trades
        ]
    finally:
        db.close()


# ─── Performance ───────────────────────────────────────────────────────────────

@app.get("/api/performance")
def get_performance():
    """Strategy performance stats — win rate, expectancy, profit factor per setup."""
    from db.connection import SessionLocal
    from db.models import StrategyStats, Trade, TradeStatus

    db = SessionLocal()
    try:
        stats = db.query(StrategyStats).filter(StrategyStats.total_trades > 0).all()
        total_trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).count()
        total_pnl    = sum(
            t.pnl for t in db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).all()
            if t.pnl is not None
        )
        return {
            "summary": {
                "total_trades": total_trades,
                "total_pnl":    round(total_pnl, 2),
            },
            "by_setup": [
                {
                    "setup_type":    s.setup_type.value if hasattr(s.setup_type, 'value') else str(s.setup_type),
                    "total_trades":  s.total_trades,
                    "winning_trades": s.winning_trades,
                    "win_rate":      round(s.win_rate * 100, 1),
                    "avg_win":       round(s.avg_win, 2),
                    "avg_loss":      round(s.avg_loss, 2),
                    "expectancy":    round(s.expectancy, 2),
                    "profit_factor": round(s.profit_factor, 2),
                    "total_pnl":     round(s.total_pnl, 2),
                    "largest_win":   round(s.largest_win, 2),
                    "largest_loss":  round(s.largest_loss, 2),
                }
                for s in sorted(stats, key=lambda x: x.expectancy or 0, reverse=True)
            ]
        }
    finally:
        db.close()


# ─── Agent decisions / reasoning log ──────────────────────────────────────────

@app.get("/api/decisions")
def get_decisions(limit: int = 50, agent: Optional[str] = None):
    """Agent reasoning log — why each decision was made."""
    from db.connection import SessionLocal
    from db.models import AgentDecision

    db = SessionLocal()
    try:
        q = db.query(AgentDecision)
        if agent:
            q = q.filter(AgentDecision.agent_name == agent)
        decisions = q.order_by(AgentDecision.created_at.desc()).limit(limit).all()
        return [
            {
                "id":            str(d.id),
                "agent":         d.agent_name,
                "decision_type": d.decision_type,
                "symbol":        d.symbol,
                "reasoning":     d.reasoning,
                "output":        d.output,
                "created_at":    d.created_at.isoformat() if d.created_at else None,
            }
            for d in decisions
        ]
    finally:
        db.close()


# ─── Journal ───────────────────────────────────────────────────────────────────

@app.get("/api/journal")
def get_journal(limit: int = 20):
    """Recent trade journal entries with analysis."""
    from db.connection import SessionLocal
    from db.models import TradeJournal, Trade

    db = SessionLocal()
    try:
        entries = (db.query(TradeJournal, Trade)
                   .join(Trade, TradeJournal.trade_id == Trade.id)
                   .order_by(TradeJournal.created_at.desc())
                   .limit(limit)
                   .all())
        return [
            {
                "id":                   str(j.id),
                "symbol":               t.symbol,
                "pnl":                  round(t.pnl, 2) if t.pnl is not None else None,
                "pnl_pct":              round(t.pnl_pct, 2) if t.pnl_pct is not None else None,
                "what_happened":        j.what_happened,
                "what_went_right":      j.what_went_right,
                "what_went_wrong":      j.what_went_wrong,
                "lessons":              j.lessons,
                "entry_quality_score":  j.entry_quality_score,
                "exit_quality_score":   j.exit_quality_score,
                "plan_adherence_score": j.plan_adherence_score,
                "created_at":           j.created_at.isoformat() if j.created_at else None,
            }
            for j, t in entries
        ]
    finally:
        db.close()


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    from db.connection import test_connection
    try:
        test_connection()
        return {"status": "ok", "db": "connected", "time": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "error", "db": str(e)}


# ─── Serve static frontend ────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""):
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not found")
