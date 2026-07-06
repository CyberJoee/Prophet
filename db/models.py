from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    Text, JSON, ForeignKey, Enum as SAEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from db.connection import Base
import uuid
import enum

# ─── Enums ────────────────────────────────────────────────────────────────────

class AssetType(str, enum.Enum):
    STOCK = "stock"
    OPTION = "option"

class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class TradeStatus(str, enum.Enum):
    PENDING_FILL = "pending_fill"   # order accepted by broker, not yet filled
    OPEN = "open"                   # fill confirmed — this is a real position
    CLOSED = "closed"
    CANCELLED = "cancelled"         # order expired/cancelled without filling

class SetupType(str, enum.Enum):
    ORB = "orb"               # Opening Range Breakout
    VWAP_BOUNCE = "vwap_bounce"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    OPTIONS_PLAY = "options_play"
    EARNINGS = "earnings"
    CUSTOM = "custom"

# ─── Watchlist ─────────────────────────────────────────────────────────────────

class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, unique=True)
    active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ─── Market Snapshots ──────────────────────────────────────────────────────────

class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float, nullable=True)
    rsi_14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    atr_14 = Column(Float, nullable=True)
    raw_data = Column(JSON, nullable=True)    # full bar/indicator dump
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_snapshots_symbol_ts", "symbol", "timestamp"),
    )

# ─── News ──────────────────────────────────────────────────────────────────────

class NewsItem(Base):
    __tablename__ = "news_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=True)   # null = general market news
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    sentiment_label = Column(String(20), nullable=True)  # bullish / bearish / neutral
    url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ─── Trades ────────────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    asset_type = Column(SAEnum(AssetType), nullable=False)
    setup_type = Column(SAEnum(SetupType), nullable=False, default=SetupType.CUSTOM)
    side = Column(SAEnum(OrderSide), nullable=False)
    status = Column(SAEnum(TradeStatus), nullable=False, default=TradeStatus.OPEN)

    # Entry
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    quantity = Column(Float, nullable=False)
    planned_stop = Column(Float, nullable=True)
    planned_target = Column(Float, nullable=True)

    # Exit
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(100), nullable=True)  # stop_hit, target_hit, manual, time_exit

    # P&L
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    max_adverse_excursion = Column(Float, nullable=True)   # worst drawdown while open
    max_favorable_excursion = Column(Float, nullable=True) # best gain while open

    # Context at entry (snapshot of why the agent entered)
    entry_context = Column(JSON, nullable=True)

    # Options-specific fields
    option_expiry = Column(DateTime, nullable=True)
    option_strike = Column(Float, nullable=True)
    option_type = Column(String(4), nullable=True)   # call / put
    iv_at_entry = Column(Float, nullable=True)
    delta_at_entry = Column(Float, nullable=True)
    theta_at_entry = Column(Float, nullable=True)

    # Alpaca order tracking
    alpaca_order_id = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_status", "status"),
        Index("ix_trades_setup", "setup_type"),
    )

# ─── Trade Journal (post-trade analysis) ──────────────────────────────────────

class TradeJournal(Base):
    __tablename__ = "trade_journals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=False, unique=True)

    # Free-form analysis written by the journal agent
    what_happened = Column(Text, nullable=True)
    what_went_right = Column(Text, nullable=True)
    what_went_wrong = Column(Text, nullable=True)
    lessons = Column(Text, nullable=True)
    market_conditions = Column(Text, nullable=True)

    # Structured scoring (1-10 each)
    entry_quality_score = Column(Integer, nullable=True)
    exit_quality_score = Column(Integer, nullable=True)
    plan_adherence_score = Column(Integer, nullable=True)

    # Vector embedding for semantic search
    # Embed: symbol + setup_type + entry_context + outcome
    embedding = Column(Vector(1536), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

# ─── Strategy Performance ─────────────────────────────────────────────────────

class StrategyStats(Base):
    __tablename__ = "strategy_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    setup_type = Column(SAEnum(SetupType), nullable=False, unique=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    expectancy = Column(Float, default=0.0)        # avg_win*win_rate - avg_loss*(1-win_rate)
    profit_factor = Column(Float, default=0.0)     # gross_profit / gross_loss
    total_pnl = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    avg_hold_minutes = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

# ─── Agent Decisions (reasoning log) ─────────────────────────────────────────

class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name = Column(String(50), nullable=False)   # research / strategy / risk / journal
    decision_type = Column(String(50), nullable=False) # scan / score / enter / skip / exit
    symbol = Column(String(20), nullable=True)
    reasoning = Column(Text, nullable=True)           # full Claude reasoning
    inputs = Column(JSON, nullable=True)              # what data the agent received
    output = Column(JSON, nullable=True)              # structured decision output
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_decisions_agent", "agent_name"),
        Index("ix_decisions_ts", "created_at"),
    )

# ─── Portfolio Snapshots ──────────────────────────────────────────────────────

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)          # cash + open positions value
    open_positions = Column(JSON, nullable=True)    # [{symbol, qty, value, pnl}, ...]
    daily_pnl = Column(Float, nullable=True)
    total_pnl = Column(Float, nullable=True)        # vs starting capital
    created_at = Column(DateTime, default=datetime.utcnow)

# ─── Alternative Data Signals ─────────────────────────────────────────────────

class AltSignal(Base):
    """
    Daily snapshot of differentiated data signals per symbol per source.
    Collected every morning BEFORE the trading decision; evaluated later
    against forward returns (scripts/eval_signals.py) to determine which
    signals actually predict before they're allowed to influence sizing.
    """
    __tablename__ = "alt_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_date = Column(DateTime, nullable=False)   # trading date (00:00 UTC)
    symbol = Column(String(20), nullable=False)      # or "_MACRO" for market-wide
    source = Column(String(30), nullable=False)      # options_flow | short_volume | event_risk
    metrics = Column(JSON, nullable=False)            # source-specific numbers
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_altsignals_lookup", "symbol", "source", "signal_date"),
    )
