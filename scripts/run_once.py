"""
Single pipeline run — useful for testing and manual execution.
Runs: research scan -> strategy decision -> prints results.
Uses DATA_PROVIDER and EXECUTION_CLIENT from .env (defaults to mock).
Falls back to mock mode automatically if live providers fail.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.llm_client import is_llm_available
from db.connection import SessionLocal
from db.operations import seed_default_watchlist, get_watchlist

print("=" * 60)
print("PROPHET — Manual Pipeline Run")
print("=" * 60)

# ── Initialise providers with fallback ──────────────────────
def get_provider_safe():
    mode = os.getenv("DATA_PROVIDER", "mock").lower()
    if mode == "alpaca":
        try:
            from data.market_data import AlpacaDataProvider
            p = AlpacaDataProvider()
            print("  [data] Using Alpaca data provider")
            return p
        except Exception as e:
            print(f"  [data] Alpaca init failed ({e}) — falling back to mock")
    from data.market_data import MockDataProvider
    print("  [data] Using mock data provider")
    return MockDataProvider()

def get_broker_safe():
    mode = os.getenv("EXECUTION_CLIENT", "mock").lower()
    if mode == "alpaca":
        try:
            from execution.broker import AlpacaExecutionClient
            b = AlpacaExecutionClient()
            print("  [broker] Using Alpaca paper trading")
            return b
        except Exception as e:
            print(f"  [broker] Alpaca init failed ({e}) — falling back to mock")
    from execution.broker import MockExecutionClient
    print("  [broker] Using mock broker")
    return MockExecutionClient()

dp     = get_provider_safe()
broker = get_broker_safe()
db     = SessionLocal()

seed_default_watchlist(db)
watchlist = get_watchlist(db)
print(f"\nWatchlist ({len(watchlist)} symbols): {', '.join(watchlist)}\n")

# ── Research scan ────────────────────────────────────────────
print("Running research scan...")
from agents.research_agent import ResearchAgent
research = ResearchAgent(data_provider=dp, db_session=db)

briefing = None
if is_llm_available():
    try:
        briefing = research.run()
        print("  [LIVE] Groq research scan complete")
    except Exception as e:
        print(f"  [LIVE] Groq research scan failed ({e}) — falling back to mock")

if briefing is None:
    briefing = research.run_mock()
    print("  [MOCK] Using mock briefing")

print(f"  Market mood: {briefing['market_mood']} — {briefing.get('market_mood_reason','')}")
print(f"  Opportunities: {[o['symbol'] for o in briefing['opportunities']]}")
print(f"  Avoid: {briefing.get('avoid', [])}")
print()

# ── Strategy decision ────────────────────────────────────────
print("Running strategy evaluation...")
from agents.strategy_agent import StrategyAgent
strategy = StrategyAgent(data_provider=dp, db_session=db, execution_client=broker)

decision = None
if is_llm_available():
    try:
        decision = strategy.run(briefing)
        print("  [LIVE] Groq strategy evaluation complete")
    except Exception as e:
        print(f"  [LIVE] Groq strategy failed ({e}) — falling back to mock")

if decision is None:
    decision = strategy.run_mock(briefing)
    print("  [MOCK] Using mock strategy")

print(f"  Trades planned:  {len(decision['trades'])}")
print(f"  Trades executed: {len(decision['executed'])}")
for t in decision['trades']:
    try:
        print(f"    {t['symbol']:5s} {t['side']:4s} x{t['quantity']:4.0f} @${t['entry_price']:.2f} "
              f"stop=${t['stop_loss']:.2f} target=${t['take_profit']:.2f} R/R={t['risk_reward']:.1f}")
    except Exception:
        print(f"    {t.get('symbol','?')} — partial data")
print()

# ── Portfolio summary ────────────────────────────────────────
try:
    acct = broker.get_account()
    positions = broker.get_all_positions()
    print("Portfolio:")
    print(f"  Cash:   ${acct['cash']:>12,.2f}")
    print(f"  Equity: ${acct['equity']:>12,.2f}")
    if positions:
        print("  Open positions:")
        for p in positions:
            if p:
                print(f"    {p['symbol']:5s} x{p['qty']:4.0f}  ${p['market_value']:>10,.2f}  "
                      f"unrealized: ${p['unrealized_pnl']:>8,.2f}")
    else:
        print("  No open positions")
except Exception as e:
    print(f"  Portfolio summary unavailable: {e}")

db.close()
print()
print("Done. Check agent_decisions table for full reasoning log.")
