"""
Database operations layer.
All agents write through here — no raw SQL in agent code.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.models import (
    Trade, TradeJournal, MarketSnapshot, NewsItem,
    StrategyStats, AgentDecision, PortfolioSnapshot, WatchlistItem,
    TradeStatus, SetupType
)
import uuid


# ─── Watchlist ─────────────────────────────────────────────────────────────────

def get_watchlist(db: Session) -> list[str]:
    rows = db.query(WatchlistItem).filter(WatchlistItem.active == True).all()
    return [r.symbol for r in rows]

def add_to_watchlist(db: Session, symbol: str, notes: str = None) -> WatchlistItem:
    item = WatchlistItem(symbol=symbol.upper(), notes=notes)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def seed_default_watchlist(db: Session):
    """Populate watchlist with a default set of liquid stocks."""
    defaults = [
        ("NVDA", "AI/GPU leader, high beta"),
        ("AAPL", "Large cap, consistent"),
        ("TSLA", "High volatility, momentum plays"),
        ("MSFT", "AI tailwind, steady"),
        ("AMZN", "E-commerce + cloud"),
        ("META", "Strong FCF, momentum"),
        ("GOOGL", "Ad revenue + cloud"),
        ("JPM",  "Financials, macro sensitive"),
        ("SPY",  "S&P 500 ETF hedge/reference"),
        ("QQQ",  "Nasdaq ETF hedge/reference"),
    ]
    for sym, note in defaults:
        exists = db.query(WatchlistItem).filter(WatchlistItem.symbol == sym).first()
        if not exists:
            db.add(WatchlistItem(symbol=sym, notes=note))
    db.commit()


# ─── Market Snapshots ──────────────────────────────────────────────────────────

def _json_safe(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if hasattr(v, 'isoformat'):
            result[k] = v.isoformat()
        elif v is None or isinstance(v, (int, float, str, bool)):
            result[k] = v
        else:
            result[k] = str(v)
    return result

def save_snapshot(db: Session, bar: dict) -> MarketSnapshot:
    snap = MarketSnapshot(
        symbol=bar["symbol"],
        timestamp=bar["timestamp"],
        open=bar.get("open"),
        high=bar.get("high"),
        low=bar.get("low"),
        close=bar.get("close"),
        volume=bar.get("volume"),
        vwap=bar.get("vwap"),
        rsi_14=bar.get("rsi_14"),
        macd=bar.get("macd"),
        macd_signal=bar.get("macd_signal"),
        bb_upper=bar.get("bb_upper"),
        bb_lower=bar.get("bb_lower"),
        atr_14=bar.get("atr_14"),
        raw_data=_json_safe(bar),
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)
    return snap

def get_latest_snapshot(db: Session, symbol: str) -> Optional[MarketSnapshot]:
    return (
        db.query(MarketSnapshot)
        .filter(MarketSnapshot.symbol == symbol)
        .order_by(MarketSnapshot.timestamp.desc())
        .first()
    )


# ─── Trades ────────────────────────────────────────────────────────────────────

def open_trade(db: Session, trade_data: dict) -> Trade:
    trade = Trade(
        symbol=trade_data["symbol"],
        asset_type=trade_data["asset_type"],
        setup_type=trade_data.get("setup_type", "custom"),
        side=trade_data["side"],
        entry_price=trade_data["entry_price"],
        entry_time=trade_data.get("entry_time", datetime.utcnow()),
        quantity=trade_data["quantity"],
        planned_stop=trade_data.get("planned_stop"),
        planned_target=trade_data.get("planned_target"),
        entry_context=trade_data.get("entry_context"),
        option_expiry=trade_data.get("option_expiry"),
        option_strike=trade_data.get("option_strike"),
        option_type=trade_data.get("option_type"),
        iv_at_entry=trade_data.get("iv_at_entry"),
        delta_at_entry=trade_data.get("delta_at_entry"),
        alpaca_order_id=trade_data.get("alpaca_order_id"),
        # NEW: default to PENDING_FILL — a trade only becomes OPEN once the
        # broker confirms a fill (see execution/order_tracker.py).
        status=trade_data.get("status", TradeStatus.PENDING_FILL),
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def confirm_trade_fill(db: Session, trade_id, fill_price: float,
                       fill_time: datetime = None, filled_qty: float = None) -> Trade:
    """Promote a PENDING_FILL trade to OPEN with the broker's actual fill data."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise ValueError(f"Trade {trade_id} not found")
    trade.status = TradeStatus.OPEN
    trade.entry_price = fill_price
    trade.entry_time = fill_time or datetime.utcnow()
    if filled_qty is not None and filled_qty > 0:
        trade.quantity = filled_qty
    db.commit()
    db.refresh(trade)
    return trade


def cancel_pending_trade(db: Session, trade_id, reason: str = "order_not_filled") -> Trade:
    """Mark a PENDING_FILL trade as CANCELLED — the order never became a position."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise ValueError(f"Trade {trade_id} not found")
    trade.status = TradeStatus.CANCELLED
    trade.exit_reason = reason
    trade.exit_time = datetime.utcnow()
    db.commit()
    db.refresh(trade)
    return trade


def get_pending_trades(db: Session) -> list[Trade]:
    return db.query(Trade).filter(Trade.status == TradeStatus.PENDING_FILL).all()


def update_trade_excursions(db: Session, trade_id, current_pnl: float) -> None:
    """
    Track true max adverse/favorable excursion while a trade is open.
    Called by the position monitor on every check.
    """
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        return
    mae = trade.max_adverse_excursion or 0.0
    mfe = trade.max_favorable_excursion or 0.0
    trade.max_adverse_excursion = min(mae, current_pnl)
    trade.max_favorable_excursion = max(mfe, current_pnl)
    db.commit()


def close_trade(db: Session, trade_id, exit_price: float,
                exit_reason: str = "manual",
                max_adverse: float = None, max_favorable: float = None) -> Trade:
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise ValueError(f"Trade {trade_id} not found")
    trade.exit_price = exit_price
    trade.exit_time = datetime.utcnow()
    trade.exit_reason = exit_reason
    trade.status = TradeStatus.CLOSED
    # Only overwrite excursions if explicitly provided — the monitor now
    # tracks true running MAE/MFE while the trade is open.
    if max_adverse is not None:
        trade.max_adverse_excursion = max_adverse
    if max_favorable is not None:
        trade.max_favorable_excursion = max_favorable
    # P&L calculation
    multiplier = 1 if trade.side.value == "buy" else -1
    trade.pnl = (exit_price - trade.entry_price) * trade.quantity * multiplier
    trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
    db.commit()
    db.refresh(trade)
    return trade

def get_open_trades(db: Session) -> list[Trade]:
    return db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()

def get_trade_history(db: Session, limit: int = 100, setup_type: str = None) -> list[Trade]:
    q = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED)
    if setup_type:
        q = q.filter(Trade.setup_type == setup_type)
    return q.order_by(Trade.entry_time.desc()).limit(limit).all()


# ─── Agent Decisions ───────────────────────────────────────────────────────────

def log_decision(db: Session, agent: str, decision_type: str,
                 reasoning: str = None, inputs: dict = None,
                 output: dict = None, symbol: str = None,
                 trade_id=None) -> AgentDecision:
    d = AgentDecision(
        agent_name=agent,
        decision_type=decision_type,
        symbol=symbol,
        reasoning=reasoning,
        inputs=inputs,
        output=output,
        trade_id=trade_id,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


# ─── Strategy Stats ────────────────────────────────────────────────────────────

def refresh_strategy_stats(db: Session):
    """Recompute win rate, expectancy, profit factor for each setup type."""
    for setup in SetupType:
        trades = (
            db.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED, Trade.setup_type == setup)
            .all()
        )
        if not trades:
            continue
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        n = len(trades)
        win_rate = len(wins) / n if n else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        stats = db.query(StrategyStats).filter(StrategyStats.setup_type == setup).first()
        if not stats:
            stats = StrategyStats(setup_type=setup)
            db.add(stats)

        stats.total_trades = n
        stats.winning_trades = len(wins)
        stats.win_rate = win_rate
        stats.avg_win = avg_win
        stats.avg_loss = avg_loss
        stats.expectancy = (avg_win * win_rate) - (avg_loss * (1 - win_rate))
        stats.profit_factor = gross_profit / gross_loss if gross_loss else 0
        stats.total_pnl = sum(t.pnl for t in trades if t.pnl)
        stats.largest_win = max((t.pnl for t in wins), default=0)
        stats.largest_loss = min((t.pnl for t in losses), default=0)
        stats.last_updated = datetime.utcnow()

    db.commit()


# ─── Portfolio ─────────────────────────────────────────────────────────────────

def save_portfolio_snapshot(db: Session, cash: float, equity: float,
                             open_positions: list, daily_pnl: float,
                             total_pnl: float) -> PortfolioSnapshot:
    snap = PortfolioSnapshot(
        timestamp=datetime.utcnow(),
        cash=cash,
        equity=equity,
        open_positions=open_positions,
        daily_pnl=daily_pnl,
        total_pnl=total_pnl,
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)
    return snap
