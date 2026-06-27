"""
Prophet Trading Agent — full test suite.
Run with: python3 -m pytest tests/test_suite.py -v
Or directly: python3 tests/test_suite.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db():
    from db.connection import engine, Base, SessionLocal
    from db.models import (
        Trade, TradeJournal, MarketSnapshot, NewsItem,
        StrategyStats, AgentDecision, PortfolioSnapshot, WatchlistItem
    )
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="session")
def dp():
    from data.market_data import MockDataProvider
    return MockDataProvider()

@pytest.fixture(scope="session")
def broker():
    from execution.broker import MockExecutionClient
    return MockExecutionClient(starting_cash=100_000.0)


# ─── Data provider ────────────────────────────────────────────────────────────

class TestDataProvider:

    def test_daily_bars_count(self, dp):
        bars = dp.fetch_bars("NVDA", days=60, interval="1d")
        assert len(bars) == 60

    def test_all_indicators_present(self, dp):
        bars = dp.fetch_bars("NVDA", days=60, interval="1d")
        last = bars[-1]
        for key in ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14", "vwap"]:
            assert last[key] is not None, f"Missing indicator: {key}"

    def test_intraday_bars(self, dp):
        bars = dp.fetch_intraday("SPY", days_back=3, interval="5m")
        assert len(bars) > 50

    def test_determinism(self, dp):
        a = dp.fetch_latest_bar("AAPL")
        b = dp.fetch_latest_bar("AAPL")
        assert a["close"] == b["close"]

    def test_watchlist_scan(self, dp):
        results = dp.scan_watchlist(["NVDA", "AAPL", "TSLA"])
        assert len(results) == 3
        for r in results:
            assert r["close"] is not None
            assert r["sector"] is not None

    def test_news(self, dp):
        news = dp.fetch_news("NVDA")
        assert len(news) > 0
        assert news[0]["sentiment_score"] is not None

    def test_fundamentals(self, dp):
        fund = dp.fetch_fundamentals("MSFT")
        assert fund["sector"] == "Technology"
        assert fund["beta"] > 0


# ─── Database ─────────────────────────────────────────────────────────────────

class TestDatabase:

    def test_connection(self):
        from db.connection import test_connection
        assert test_connection() is True

    def test_watchlist_seed(self, db):
        from db.operations import seed_default_watchlist, get_watchlist
        seed_default_watchlist(db)
        wl = get_watchlist(db)
        assert len(wl) == 10
        assert "NVDA" in wl

    def test_open_trade(self, db):
        from db.operations import open_trade, get_open_trades
        from db.models import AssetType, OrderSide
        trade = open_trade(db, {
            "symbol": "NVDA", "asset_type": AssetType.STOCK,
            "setup_type": "momentum", "side": OrderSide.BUY,
            "entry_price": 120.50, "quantity": 100,
            "planned_stop": 115.00, "planned_target": 135.00,
            "entry_context": {"rsi": 55.2, "reason": "test"},
        })
        assert trade.id is not None
        assert trade.status.value == "open"

    def test_close_trade_pnl(self, db):
        from db.operations import open_trade, close_trade, get_open_trades
        from db.models import AssetType, OrderSide
        trade = open_trade(db, {
            "symbol": "AAPL", "asset_type": AssetType.STOCK,
            "setup_type": "vwap_bounce", "side": OrderSide.BUY,
            "entry_price": 200.00, "quantity": 50,
            "planned_stop": 195.00, "planned_target": 215.00,
        })
        closed = close_trade(db, trade.id, exit_price=210.00, exit_reason="target_hit")
        assert closed.pnl == pytest.approx(500.00, rel=0.01)
        assert closed.pnl_pct == pytest.approx(5.0, rel=0.01)

    def test_strategy_stats(self, db):
        from db.operations import refresh_strategy_stats
        from db.models import StrategyStats, SetupType
        refresh_strategy_stats(db)
        stats = db.query(StrategyStats).first()
        assert stats is not None
        assert stats.total_trades > 0

    def test_snapshot_save_and_retrieve(self, db, dp):
        from db.operations import save_snapshot, get_latest_snapshot
        bar = dp.fetch_latest_bar("NVDA")
        snap = save_snapshot(db, bar)
        assert snap.rsi_14 is not None
        latest = get_latest_snapshot(db, "NVDA")
        assert latest is not None
        assert latest.raw_data["symbol"] == "NVDA"

    def test_agent_decision_log(self, db):
        from db.operations import log_decision
        d = log_decision(db, agent="test", decision_type="test_scan",
                         symbol="NVDA", reasoning="unit test",
                         inputs={"test": True}, output={"result": "ok"})
        assert d.id is not None


# ─── Execution ────────────────────────────────────────────────────────────────

class TestExecution:

    def test_initial_account(self, broker):
        acct = broker.get_account()
        assert acct["cash"] == pytest.approx(100_000.0, rel=0.01)

    def test_limit_order_fill(self, broker):
        order = broker.place_limit_order("NVDA", qty=10, side="buy", limit_price=120.00)
        assert order["status"] == "filled"
        assert order["fill_price"] == 120.00

    def test_position_created(self, broker):
        pos = broker.get_position("NVDA")
        assert pos is not None
        assert pos["qty"] >= 10

    def test_close_position(self, broker):
        broker.place_limit_order("TSLA", qty=5, side="buy", limit_price=270.00)
        broker.close_position("TSLA")
        assert broker.get_position("TSLA") is None

    def test_cash_non_negative(self, broker):
        acct = broker.get_account()
        assert acct["cash"] >= 0

    def test_market_hours_returns_bool(self, broker):
        result = broker.is_market_open()
        assert isinstance(result, bool)


# ─── Agents ───────────────────────────────────────────────────────────────────

class TestAgents:

    def test_research_agent_mock(self, dp, db):
        from agents.research_agent import ResearchAgent
        agent = ResearchAgent(data_provider=dp, db_session=db)
        briefing = agent.run_mock(["NVDA", "AAPL", "TSLA", "MSFT"])
        assert briefing["market_mood"] in ["bullish", "bearish", "neutral", "mixed"]
        assert len(briefing["opportunities"]) > 0
        assert len(briefing["symbols_scanned"]) == 4

    def test_research_briefing_structure(self, dp):
        from agents.research_agent import ResearchAgent
        agent = ResearchAgent(data_provider=dp)
        b = agent.run_mock(["NVDA", "AAPL"])
        for opp in b["opportunities"]:
            assert "symbol" in opp
            assert "confidence" in opp
            assert "setup_type" in opp
            assert "thesis" in opp

    def test_strategy_agent_mock(self, dp, db, broker):
        from agents.research_agent import ResearchAgent
        from agents.strategy_agent import StrategyAgent

        # Fresh broker for this test
        from execution.broker import MockExecutionClient
        fresh_broker = MockExecutionClient(100_000.0)

        research = ResearchAgent(data_provider=dp, db_session=db)
        briefing = research.run_mock(["NVDA", "AAPL"])

        strategy = StrategyAgent(data_provider=dp, db_session=db, execution_client=fresh_broker)
        decision = strategy.run_mock(briefing)

        assert "trades" in decision
        assert "executed" in decision
        assert "portfolio_risk_used" in decision

    def test_position_sizing_within_limits(self, dp, db):
        from agents.research_agent import ResearchAgent
        from agents.strategy_agent import StrategyAgent
        from execution.broker import MockExecutionClient

        fresh_broker = MockExecutionClient(100_000.0)
        research = ResearchAgent(data_provider=dp)
        briefing = research.run_mock(["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"])
        strategy = StrategyAgent(data_provider=dp, execution_client=fresh_broker)
        decision = strategy.run_mock(briefing)

        acct = fresh_broker.get_account()
        assert acct["cash"] >= 0, "Cash went negative"

        for p in fresh_broker.get_all_positions():
            pct = p["market_value"] / acct["equity"] * 100
            assert pct <= 16, f"{p['symbol']} position {pct:.1f}% exceeds 15% cap"

        for t in decision["trades"]:
            risk_pct = t["dollar_risk"] / acct["equity"] * 100
            assert risk_pct <= 2.5, f"{t['symbol']} risk {risk_pct:.2f}% exceeds 2% rule"

    def test_decisions_logged_to_db(self, dp, db):
        from db.models import AgentDecision
        decisions = db.query(AgentDecision).filter(
            AgentDecision.agent_name.in_(["research", "strategy"])
        ).all()
        assert len(decisions) > 0


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.exit(result.returncode)


# ─── Scheduler & Position Monitor ─────────────────────────────────────────────

class TestPositionMonitor:

    def test_target_hit_closes_trade(self):
        from db.connection import SessionLocal
        from db.models import AssetType, OrderSide
        from db.operations import open_trade
        from data.market_data import MockDataProvider
        from execution.broker import MockExecutionClient
        from execution.position_monitor import PositionMonitor

        dp = MockDataProvider()
        broker = MockExecutionClient(100_000)
        db = SessionLocal()

        # Use AMZN — not used by other tests
        bar = dp.fetch_latest_bar('AMZN')
        close = bar['close']
        trade = open_trade(db, {
            'symbol': 'AMZN', 'asset_type': AssetType.STOCK,
            'setup_type': 'momentum', 'side': OrderSide.BUY,
            'entry_price': close * 0.95,
            'quantity': 10,
            'planned_stop': close * 0.90,
            'planned_target': close * 0.99,  # below current = triggers
        })
        monitor = PositionMonitor(data_provider=dp, execution_client=broker, db_session=db)
        actions = monitor.check_positions()
        amzn_actions = [a for a in actions if a['symbol'] == 'AMZN']
        assert len(amzn_actions) >= 1
        assert amzn_actions[0]['exit_reason'] == 'target_hit'
        assert amzn_actions[0]['pnl'] > 0
        db.close()

    def test_stop_hit_closes_trade(self):
        from db.connection import SessionLocal
        from db.models import AssetType, OrderSide
        from db.operations import open_trade
        from data.market_data import MockDataProvider
        from execution.broker import MockExecutionClient
        from execution.position_monitor import PositionMonitor

        dp = MockDataProvider()
        broker = MockExecutionClient(100_000)
        db = SessionLocal()

        bar = dp.fetch_latest_bar('AAPL')
        close = bar['close']
        trade = open_trade(db, {
            'symbol': 'AAPL', 'asset_type': AssetType.STOCK,
            'setup_type': 'momentum', 'side': OrderSide.BUY,
            'entry_price': close * 1.10,   # entered above current = in loss
            'quantity': 10,
            'planned_stop': close * 1.05,  # stop above current = triggers
            'planned_target': close * 1.25,
        })
        monitor = PositionMonitor(data_provider=dp, execution_client=broker, db_session=db)
        actions = monitor.check_positions()
        assert any(a['exit_reason'] == 'stop_hit' for a in actions)
        db.close()

    def test_eod_closes_all(self):
        from db.connection import SessionLocal
        from db.models import AssetType, OrderSide
        from db.operations import open_trade, get_open_trades
        from data.market_data import MockDataProvider
        from execution.broker import MockExecutionClient
        from execution.position_monitor import PositionMonitor

        dp = MockDataProvider()
        broker = MockExecutionClient(100_000)
        db = SessionLocal()

        for sym in ['MSFT', 'TSLA']:
            bar = dp.fetch_latest_bar(sym)
            open_trade(db, {
                'symbol': sym, 'asset_type': AssetType.STOCK,
                'setup_type': 'momentum', 'side': OrderSide.BUY,
                'entry_price': bar['close'], 'quantity': 5,
                'planned_stop': bar['close'] * 0.90,
                'planned_target': bar['close'] * 1.20,
            })

        monitor = PositionMonitor(data_provider=dp, execution_client=broker, db_session=db)
        monitor.end_of_day()
        assert len(get_open_trades(db)) == 0
        db.close()

    def test_scheduler_jobs_importable(self):
        from scripts.scheduler import morning_pipeline, monitor_positions, end_of_day
        assert callable(morning_pipeline)
        assert callable(monitor_positions)
        assert callable(end_of_day)


# ─── Phase 3: Journal + Memory ────────────────────────────────────────────────

class TestJournalAgent:

    def _setup_closed_trades(self, db, dp):
        from db.models import AssetType, OrderSide
        from db.operations import open_trade, close_trade
        trades = []
        for sym, setup in [('NVDA','momentum'), ('AAPL','vwap_bounce')]:
            bar = dp.fetch_latest_bar(sym)
            close = bar['close']
            t = open_trade(db, {
                'symbol': sym, 'asset_type': AssetType.STOCK,
                'setup_type': setup, 'side': OrderSide.BUY,
                'entry_price': close * 0.97, 'quantity': 10,
                'planned_stop': close * 0.92,
                'planned_target': close * 1.06,
                'entry_context': {'rsi': 55.0, 'reason': 'test'},
            })
            close_trade(db, t.id, exit_price=close, exit_reason='eod_close')
            trades.append(t)
        return trades

    def test_journal_mock_creates_entries(self):
        from db.connection import SessionLocal
        from db.models import TradeJournal
        from data.market_data import MockDataProvider
        from agents.journal_agent import JournalAgent

        db = SessionLocal()
        dp = MockDataProvider()
        self._setup_closed_trades(db, dp)

        journal = JournalAgent(db_session=db)
        results = journal.run_mock()
        assert len(results) >= 2

        entries = db.query(TradeJournal).all()
        assert len(entries) >= 2
        for e in entries:
            assert e.what_happened is not None
            assert e.entry_quality_score is not None
            assert e.embedding is not None
            assert len(e.embedding) == 1536
        db.close()

    def test_journal_skips_duplicates(self):
        from db.connection import SessionLocal
        from db.models import TradeJournal
        from data.market_data import MockDataProvider
        from agents.journal_agent import JournalAgent

        db = SessionLocal()
        dp = MockDataProvider()
        journal = JournalAgent(db_session=db)

        before = db.query(TradeJournal).count()
        journal.run_mock()   # run again — should skip already-journaled trades
        after = db.query(TradeJournal).count()
        assert after == before   # no new entries
        db.close()

    def test_strategy_stats_updated(self):
        from db.connection import SessionLocal
        from db.models import StrategyStats
        from agents.journal_agent import JournalAgent

        db = SessionLocal()
        journal = JournalAgent(db_session=db)
        journal.run_mock()

        stats = db.query(StrategyStats).filter(StrategyStats.total_trades > 0).all()
        assert len(stats) > 0
        for s in stats:
            assert s.win_rate >= 0
            assert s.total_trades > 0
        db.close()

    def test_memory_retrieval(self):
        from db.connection import SessionLocal
        from agents.memory import find_similar_trades, get_strategy_performance_summary

        db = SessionLocal()
        similar = find_similar_trades(db, symbol='NVDA', setup_type='momentum', limit=3)
        assert isinstance(similar, list)

        summary = get_strategy_performance_summary(db)
        assert isinstance(summary, str)
        assert len(summary) > 0
        db.close()
