"""
Strategy Agent
Receives the research briefing and decides specific trade entries.
- Retrieves similar historical trades from memory (pgvector)
- Calls Claude to produce concrete trade plans
- Applies risk rules before passing to execution
"""
import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


STRATEGY_SYSTEM_PROMPT = """You are a disciplined quantitative trader executing a paper trading strategy.
You receive a market research briefing and must decide specific trade entries.

Rules you MUST follow:
- Only trade setups with confidence >= 0.60
- Never risk more than 2% of portfolio per trade (position size = (portfolio * 0.02) / stop_distance)
- Always specify a stop loss (mandatory) and a profit target (minimum 1.5:1 R/R)
- Stop loss: 0.4x to 0.7x ATR from entry. ATR is the DAILY range — half of that is your intraday stop.
- Profit target: 0.8x to 1.5x ATR from entry. This represents 80-150% of the typical daily move — achievable intraday.
- These are INTRADAY positions that close at 4:15 PM ET — size targets for realistic same-day moves.
- A stock moving 1x ATR intraday is normal. 4x ATR in one day is extremely rare. Be realistic.
- Prefer limit orders over market orders
- For momentum setups: enter near VWAP or on pullbacks, not extended moves
- For options: target 2+ months to expiration, avoid weeklies

Respond ONLY with valid JSON. No preamble, no markdown fences.
Structure:
{
  "trades": [
    {
      "symbol": "NVDA",
      "asset_type": "stock",
      "side": "buy",
      "setup_type": "momentum",
      "quantity": 50,
      "entry_price": 121.00,
      "entry_type": "limit",
      "stop_loss": 119.75,
      "take_profit": 122.50,
      "risk_reward": 2.14,
      "dollar_risk": 175.00,
      "reasoning": "<2-3 sentences>",
      "entry_conditions": "<what needs to happen before entry is valid>"
    }
  ],
  "skip_reason": "<if no trades, why>",
  "portfolio_risk_used": 0.035
}
CRITICAL: ATR is the full daily range. stop_loss = entry - 0.5*ATR, take_profit = entry + 1.0*ATR. Example for $120 stock with ATR=$2.50: stop=$118.75, target=$122.50. NEVER set targets 5%+ away on intraday trades.
Return at most 3 trades. Return empty trades array if conditions aren't right.
"""


def _retrieve_similar_trades(db, embedding_text: str, limit: int = 5) -> list[dict]:
    """
    Retrieve similar historical trades using pgvector cosine similarity.
    Falls back to recent trades if pgvector embedding not yet populated.
    """
    if db is None:
        return []
    try:
        from db.operations import get_trade_history
        trades = get_trade_history(db, limit=limit)
        return [
            {
                "symbol":     t.symbol,
                "setup_type": t.setup_type.value if hasattr(t.setup_type, 'value') else str(t.setup_type),
                "pnl":        t.pnl,
                "pnl_pct":    t.pnl_pct,
                "exit_reason": t.exit_reason,
                "entry_context": t.entry_context,
            }
            for t in trades
        ]
    except Exception:
        return []


def _build_strategy_prompt(briefing: dict, account: dict, similar_trades: list[dict], stats_summary: str = "") -> str:
    parts = []

    parts.append(f"PORTFOLIO STATUS:")
    parts.append(f"  Cash: ${account.get('cash', 0):,.2f}")
    parts.append(f"  Equity: ${account.get('equity', 0):,.2f}")
    parts.append(f"  Buying power: ${account.get('buying_power', 0):,.2f}")
    parts.append("")

    parts.append(f"MARKET BRIEFING:")
    parts.append(f"  Market mood: {briefing.get('market_mood')} — {briefing.get('market_mood_reason', '')}")
    parts.append(f"  Symbols to avoid: {briefing.get('avoid', [])}")
    parts.append(f"  Avoid reason: {briefing.get('avoid_reason', '')}")
    parts.append("")

    parts.append("OPPORTUNITIES TO EVALUATE:")
    for opp in briefing.get("opportunities", []):
        parts.append(
            f"  {opp['symbol']}: {opp['direction'].upper()} | "
            f"setup={opp['setup_type']} | confidence={opp['confidence']:.2f} | "
            f"key_level={opp.get('key_level', 'N/A')}"
        )
        parts.append(f"    Thesis: {opp['thesis']}")
        parts.append(f"    Risk: {opp['risk_note']}")
    parts.append("")

    if similar_trades:
        parts.append("SIMILAR HISTORICAL TRADES (memory):")
        for t in similar_trades[:3]:
            outcome = f"+${t['pnl']:.2f}" if t.get('pnl') and t['pnl'] > 0 else f"-${abs(t.get('pnl', 0)):.2f}"
            parts.append(
                f"  {t['symbol']} {t['setup_type']} → {outcome} ({t.get('exit_reason', 'N/A')})"
            )
            if t.get('lessons'):
                parts.append(f"    Lesson: {t['lessons']}")
        parts.append("")

    if stats_summary:
        parts.append(stats_summary)
        parts.append("")

    parts.append("Decide which trades to enter. Apply the rules strictly.")
    return "\n".join(parts)


class StrategyAgent:

    def __init__(self, data_provider=None, db_session=None, execution_client=None):
        if data_provider is None:
            from data.market_data import get_provider
            data_provider = get_provider()
        if execution_client is None:
            from execution.broker import get_execution_client
            execution_client = get_execution_client()
        self.data = data_provider
        self.db = db_session
        self.broker = execution_client

    def _call_llm(self, prompt: str) -> dict:
        from agents.llm_client import call_llm
        return call_llm(STRATEGY_SYSTEM_PROMPT, prompt)

    def run(self, briefing: dict) -> dict:
        """
        Evaluate a research briefing and produce trade plans.
        Places orders via broker client.
        Returns the strategy decision dict.
        """
        # 1. Get account state
        account = self.broker.get_account()

        # 2. Retrieve memory + strategy stats from pgvector
        similar = []
        stats_summary = ""
        if self.db:
            try:
                from agents.memory import find_similar_trades, get_strategy_performance_summary
                for opp in briefing.get("opportunities", [])[:2]:
                    similar += find_similar_trades(
                        self.db,
                        symbol=opp["symbol"],
                        setup_type=opp.get("setup_type", "momentum"),
                        direction=opp.get("direction", "long"),
                        limit=3,
                    )
                stats_summary = get_strategy_performance_summary(self.db)
            except Exception as e:
                print(f"  [strategy] memory lookup failed: {e}")

        # 3. Build prompt and call LLM
        prompt = _build_strategy_prompt(briefing, account, similar, stats_summary)
        decision = self._call_llm(prompt)
        decision["decided_at"] = datetime.utcnow().isoformat()

        # 4. Execute approved trades
        executed = []
        for trade_plan in decision.get("trades", []):
            try:
                result = self._execute_trade(trade_plan, account)
                if result:
                    executed.append(result)
            except Exception as e:
                print(f"  [strategy] execution error for {trade_plan.get('symbol')}: {e}")

        decision["executed"] = executed

        # 5. Log decision
        if self.db:
            from db.operations import log_decision
            log_decision(
                self.db, agent="strategy", decision_type="evaluate",
                reasoning=decision.get("skip_reason") or f"Evaluated {len(decision.get('trades',[]))} setups",
                inputs={"briefing_mood": briefing.get("market_mood")},
                output=decision,
            )

        return decision

    def _execute_trade(self, plan: dict, account: dict) -> Optional[dict]:
        """Place a single order and record it in DB."""
        symbol = plan["symbol"]
        side = plan["side"]
        qty = plan["quantity"]
        entry_price = plan["entry_price"]
        stop = plan.get("stop_loss")
        target = plan.get("take_profit")

        # Refresh account to get latest buying power after prior fills
        account = self.broker.get_account()

        # Cap position value at 15% of equity (prevents over-concentration)
        equity = account.get("equity", account.get("cash", 100_000))
        max_position_value = equity * 0.15
        max_qty_by_value = int(max_position_value / entry_price)
        qty = min(qty, max_qty_by_value)
        if qty <= 0:
            print(f"  [strategy] skipping {symbol} — position size zero after cap")
            return None

        cost = qty * entry_price
        if cost > account.get("buying_power", 0) * 0.95:
            print(f"  [strategy] skipping {symbol} — insufficient buying power (${cost:.0f} vs ${account.get('buying_power',0):.0f})")
            return None

        # Place order
        order = self.broker.place_limit_order(
            symbol=symbol, qty=qty, side=side,
            limit_price=entry_price, stop_loss=stop, take_profit=target,
        )

        # Persist to DB — Alpaca returns "accepted" not "filled" for limit orders
        if self.db and order.get("status") in ("filled", "accepted", "pending_new", "new"):
            from db.operations import open_trade, log_decision
            from db.models import AssetType, OrderSide
            trade = open_trade(self.db, {
                "symbol":        symbol,
                "asset_type":    AssetType(plan.get("asset_type", "stock")),
                "setup_type":    plan.get("setup_type", "custom"),
                "side":          OrderSide(side),
                "entry_price":   order.get("fill_price", entry_price),
                "quantity":      qty,
                "planned_stop":  stop,
                "planned_target": target,
                "entry_context": {
                    "reasoning":          plan.get("reasoning"),
                    "entry_conditions":   plan.get("entry_conditions"),
                    "risk_reward":        plan.get("risk_reward"),
                    "dollar_risk":        plan.get("dollar_risk"),
                    "alpaca_order_id":    order.get("id"),
                },
                "alpaca_order_id": order.get("id"),
            })
            log_decision(
                self.db, agent="strategy", decision_type="enter",
                symbol=symbol, trade_id=trade.id,
                reasoning=plan.get("reasoning"),
                inputs=plan, output=order,
            )
            order["trade_db_id"] = str(trade.id)

        return order

    def run_mock(self, briefing: dict) -> dict:
        """
        Return a mock strategy decision (no Claude API needed).
        Always uses MockDataProvider for bar data — safe fallback path.
        """
        account = self.broker.get_account()
        equity = account.get("equity", 100_000)

        # Always use mock data for bar lookups in mock mode
        from data.market_data import MockDataProvider
        mock_dp = MockDataProvider()

        # Build a plausible trade from the first opportunity
        opps = briefing.get("opportunities", [])
        trades = []
        for opp in opps[:2]:
            sym = opp["symbol"]
            bar = mock_dp.fetch_latest_bar(sym)
            if not bar:
                continue
            close = bar["close"]
            atr = bar.get("atr_14", close * 0.01)
            # Intraday realistic: stop=0.5x ATR, target=1.0x ATR (2:1 R/R)
            # ATR = typical daily move, so 0.5x/1.0x = realistic intraday levels
            stop = round(close - 0.5 * atr, 2)
            target = round(close + 1.0 * atr, 2)
            stop_dist = close - stop
            qty = max(1, int((equity * 0.02) / stop_dist)) if stop_dist > 0 else 10
            trades.append({
                "symbol":      sym,
                "asset_type":  "stock",
                "side":        "buy" if opp["direction"] == "long" else "sell",
                "setup_type":  opp["setup_type"],
                "quantity":    qty,
                "entry_price": round(close, 2),
                "entry_type":  "limit",
                "stop_loss":   stop,
                "take_profit": target,
                "risk_reward": round((target - close) / stop_dist, 2) if stop_dist > 0 else 2.0,
                "dollar_risk": round(qty * stop_dist, 2),
                "reasoning":   opp["thesis"],
                "entry_conditions": f"Enter on a pullback to VWAP or on break of ${close:.2f}",
            })

        mock_decision = {
            "trades": trades,
            "skip_reason": None if trades else "No high-confidence setups found",
            "portfolio_risk_used": round(sum(t["dollar_risk"] for t in trades) / equity, 4),
            "decided_at": datetime.utcnow().isoformat(),
            "_mock": True,
        }

        # Actually execute via mock broker
        executed = []
        for plan in trades:
            result = self._execute_trade(plan, account)
            if result:
                executed.append(result)
        mock_decision["executed"] = executed

        if self.db:
            from db.operations import log_decision
            log_decision(self.db, agent="strategy", decision_type="evaluate",
                         reasoning=f"Mock: {len(trades)} trades planned",
                         inputs={"mood": briefing.get("market_mood")},
                         output=mock_decision)

        return mock_decision
