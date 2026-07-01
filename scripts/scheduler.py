"""
Prophet Scheduler
The main long-running process that replaces run_once.py in production.
Runs on Railway 24/7 and fires jobs on market days automatically.

Schedule (all times US Eastern):
  09:45 AM  — Morning research scan + strategy (main pipeline)
  Every 5m  — Position monitor (10:00 AM - 3:55 PM)
  04:15 PM  — End of day close + stats refresh
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytz
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

load_dotenv()

ET = pytz.timezone("America/New_York")
scheduler = BlockingScheduler(timezone=ET)

# ─── Shared state ─────────────────────────────────────────────────────────────

def _get_components():
    """Initialise all components. Called fresh each job run."""
    from data.market_data import get_provider
    from execution.broker import get_execution_client
    from db.connection import SessionLocal
    from db.operations import seed_default_watchlist

    dp     = get_provider()
    broker = get_execution_client()
    db     = SessionLocal()
    seed_default_watchlist(db)
    return dp, broker, db


# ─── Jobs ─────────────────────────────────────────────────────────────────────

def morning_pipeline():
    """
    9:45 AM ET — Research scan + strategy decisions.
    Gives 15 minutes after open for price discovery before entering.
    """
    print(f"\n{'='*60}")
    print(f"MORNING PIPELINE — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    dp, broker, db = _get_components()

    try:
        from agents.research_agent import ResearchAgent
        from agents.strategy_agent import StrategyAgent
        from agents.llm_client import is_llm_available
        from db.operations import get_watchlist

        watchlist = get_watchlist(db)
        print(f"Watchlist: {', '.join(watchlist)}\n")

        # Research
        print("Running research scan...")
        research = ResearchAgent(data_provider=dp, db_session=db)
        briefing = None
        if is_llm_available():
            try:
                briefing = research.run()
                print(f"  [LIVE] mood={briefing['market_mood']} "
                      f"opps={[o['symbol'] for o in briefing['opportunities']]}")
            except Exception as e:
                print(f"  [LIVE] research failed: {e} — using mock")
        if briefing is None:
            briefing = research.run_mock()
            print(f"  [MOCK] mood={briefing['market_mood']}")

        # Strategy
        print("Running strategy evaluation...")
        strategy = StrategyAgent(data_provider=dp, db_session=db, execution_client=broker)
        decision = None
        if is_llm_available():
            try:
                decision = strategy.run(briefing)
                print(f"  [LIVE] {len(decision['trades'])} trades planned, "
                      f"{len(decision['executed'])} executed")
            except Exception as e:
                print(f"  [LIVE] strategy failed: {e} — using mock")
        if decision is None:
            decision = strategy.run_mock(briefing)
            print(f"  [MOCK] {len(decision['trades'])} trades")

        for t in decision.get("trades", []):
            try:
                print(f"  → {t['symbol']} {t['side']} x{t['quantity']:.0f} "
                      f"@${t['entry_price']:.2f} stop=${t['stop_loss']:.2f} "
                      f"target=${t['take_profit']:.2f}")
            except Exception:
                pass

        # Sync Alpaca positions into DB after placing orders, then run an
        # immediate fill-confirmation pass so instant fills go OPEN right away
        try:
            import time; time.sleep(3)
            from execution.alpaca_sync import sync_alpaca_to_db
            sync_alpaca_to_db(db)
            from execution.order_tracker import confirm_fills
            fills = confirm_fills(db, broker)
            print(f"  [tracker] post-pipeline: {fills}")
        except Exception as e:
            print(f"  [sync] Post-pipeline sync failed: {e}")

    except Exception as e:
        print(f"  [ERROR] Morning pipeline failed: {e}")
    finally:
        db.close()


def monitor_positions():
    """Every 5 min, 10:00 AM - 3:55 PM ET — Check stops and targets."""
    dp, broker, db = _get_components()
    try:
        from execution.position_monitor import PositionMonitor
        monitor = PositionMonitor(data_provider=dp, execution_client=broker, db_session=db)
        monitor.check_positions()
    except Exception as e:
        print(f"  [monitor] error: {e}")
    finally:
        db.close()


def end_of_day():
    """4:15 PM ET — Close all positions, refresh stats."""
    print(f"\n{'='*60}")
    print(f"END OF DAY — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    dp, broker, db = _get_components()
    try:
        from execution.position_monitor import PositionMonitor
        monitor = PositionMonitor(data_provider=dp, execution_client=broker, db_session=db)
        monitor.end_of_day()
        print("End of day complete.")

        # Run journal agent after positions are closed
        print("Running journal agent...")
        try:
            from agents.journal_agent import JournalAgent
            from agents.llm_client import is_llm_available
            journal = JournalAgent(db_session=db)
            if is_llm_available():
                results = journal.run()
            else:
                results = journal.run_mock()
            print(f"  [journal] {len(results)} trade(s) journaled")
        except Exception as e:
            print(f"  [journal] error: {e}")

    except Exception as e:
        print(f"  [EOD] error: {e}")
    finally:
        db.close()


# ─── Schedule ─────────────────────────────────────────────────────────────────

# 9:45 AM ET — morning pipeline, Mon-Fri only
scheduler.add_job(
    morning_pipeline,
    CronTrigger(day_of_week="mon-fri", hour=9, minute=45, timezone=ET),
    id="morning_pipeline",
    name="Morning Research + Strategy",
    misfire_grace_time=300,   # allow 5 min late start
)

# Every 5 minutes, 10:00 AM - 3:55 PM ET, Mon-Fri
scheduler.add_job(
    monitor_positions,
    CronTrigger(day_of_week="mon-fri", hour="10-15", minute="*/5", timezone=ET),
    id="position_monitor",
    name="Position Monitor",
    misfire_grace_time=60,
)

# 4:15 PM ET — end of day, Mon-Fri
scheduler.add_job(
    end_of_day,
    CronTrigger(day_of_week="mon-fri", hour=16, minute=15, timezone=ET),
    id="end_of_day",
    name="End of Day Close",
    misfire_grace_time=300,
)


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROPHET SCHEDULER STARTING")
    print("=" * 60)
    print(f"Current time: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print()
    print("Scheduled jobs:")
    for job in scheduler.get_jobs():
        print(f"  • {job.name}")
    print()

    # Run init_db on startup to ensure tables exist
    try:
        from db.connection import engine, Base, test_connection
        from db.models import *
        test_connection()
        Base.metadata.create_all(bind=engine)
        from db.connection import SessionLocal
        from db.operations import seed_default_watchlist
        db = SessionLocal()
        seed_default_watchlist(db)
        db.close()
        print("✓ Database ready")
    except Exception as e:
        print(f"✗ Database init failed: {e}")
        sys.exit(1)

    # Sync Alpaca positions on startup
    print("Syncing Alpaca positions with DB...")
    try:
        from execution.alpaca_sync import sync_alpaca_to_db
        from db.connection import SessionLocal as _SL
        _db = _SL()
        sync_alpaca_to_db(_db)
        _db.close()
    except Exception as e:
        print(f"  [sync] Startup sync failed: {e}")

    print("\nScheduler running. Waiting for market hours...\n")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped.")
