"""
Strategy Agent v2 — division of labor.

Old design: the LLM output quantity, entry price, stop, target, dollar_risk,
and risk_reward. LLMs are unreliable at arithmetic, and their entry prices
were based on the 9:45 AM research snapshot — stale by execution time.

New design:
  LLM decides (judgment):   which setups to take, direction, conviction, why
  Code decides (arithmetic): entry from LIVE quote, stop/target from ATR,
                             quantity from the 2% risk rule, all caps
  Pydantic validates:        the LLM's JSON never flows raw into an order

Trades are recorded as PENDING_FILL — they only become OPEN positions when
the order tracker confirms a broker fill (phantom-trade fix).
"""
import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


STRATEGY_SYSTEM_PROMPT = """You are a disciplined quantitative trader evaluating intraday setups.
You receive a market research briefing plus your own historical performance data.

Your ONLY job is to decide WHICH setups (if any) are worth taking and WHY.
Do NOT calculate position sizes, prices, stops, or targets — the execution
engine computes all numbers from live quotes. Focus entirely on judgment:
- Is this setup high quality right now?
- Does the historical performance of this setup type support taking it?
- Is the thesis specific and falsifiable, or vague hope?

Be selective. Skipping is a valid decision — most days do not offer 3 good
trades. A pick with conviction below 0.60 will be discarded, so do not pad.

Respond ONLY with valid JSON. No preamble, no markdown fences.
Structure:
{
  "picks": [
    {
      "symbol": "NVDA",
      "direction": "long",
      "setup_type": "momentum",
      "conviction": 0.72,
      "reasoning": "<2-3 sentences: why this setup is worth risk right now>",
      "entry_conditions": "<what would invalidate this before entry>"
    }
  ],
  "skip_reason": "<if no picks, why — otherwise null>"
}
Return at most 3 picks. Only pick symbols that appear in the briefing's
opportunities and are not on the avoid list.
"""


def _build_strategy_prompt(briefing: dict, account: dict,
                           similar_trades: list[dict], stats_summary: str = "",
                           regime: dict = None) -> str:
    parts = []

    if regime is not None:
        from agents.regime import format_regime_for_prompt
        parts.append(format_regime_for_prompt(regime))
        parts.append("")

    parts.append("PORTFOLIO STATUS:")
    parts.append(f"  Cash: ${account.get('cash', 0):,.2f}")
    parts.append(f"  Equity: ${account.get('equity', 0):,.2f}")
    parts.append("")

    parts.append("MARKET BRIEFING:")
    parts.append(f"  Market mood: {briefing.get('market_mood')} — {briefing.get('market_mood_reason', '')}")
    parts.append(f"  Symbols to avoid: {briefing.get('avoid', [])}")
    parts.append(f"  Avoid reason: {briefing.get('avoid_reason', '')}")
    parts.append("")

    parts.append("OPPORTUNITIES TO EVALUATE:")
    for opp in briefing.get("opportunities", []):
        parts.append(
            f"  {opp['symbol']}: {opp['direction'].upper()} | "
            f"setup={opp['setup_type']} | research_confidence={opp['confidence']:.2f} | "
            f"key_level={opp.get('key_level', 'N/A')}"
        )
        parts.append(f"    Thesis: {opp['thesis']}")
        parts.append(f"    Risk: {opp['risk_note']}")
    parts.append("")

    if similar_trades:
        parts.append("RELEVANT HISTORICAL TRADES (same symbol or setup):")
        for t in similar_trades[:5]:
            pnl = t.get("pnl") or 0
            parts.append(
                f"  {t['symbol']} {t['setup_type']} → {pnl:+.2f} ({t.get('exit_reason', 'N/A')})"
            )
            if t.get("lessons"):
                parts.append(f"    Lesson: {t['lessons']}")
        parts.append("")

    if stats_summary:
        parts.append(stats_summary)
        parts.append("")

    parts.append("Decide which setups (if any) deserve risk today. Be selective.")
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

    # ─── Main pipeline ─────────────────────────────────────────────────────────

    def run(self, briefing: dict, regime: dict = None) -> dict:
        """
        1. LLM picks setups (judgment only) — regime context included in prompt
        2. Pydantic + business rules validate the picks
        3. Code sizes each trade from live quote + ATR, scaled by regime
        4. Orders placed, trades recorded as PENDING_FILL
        """
        from execution.sizing import (
            validate_llm_decision, build_trade_plan, MAX_SESSION_RISK_PCT
        )
        from data.live_price import get_live_prices

        if regime is not None and not regime.get("trade_allowed", True):
            result = {"trades": [], "executed": [],
                      "skip_reason": "regime gate: " + "; ".join(regime.get("reasons", [])),
                      "decided_at": datetime.utcnow().isoformat()}
            self._log(briefing, result)
            return result

        account = self.broker.get_account()
        equity  = account.get("equity", account.get("cash", 100_000))

        # Memory + stats context for the LLM
        similar, stats_summary = self._gather_context(briefing)

        # 1-2. LLM call + validation
        prompt = _build_strategy_prompt(briefing, account, similar, stats_summary,
                                        regime=regime)
        raw = self._call_llm(prompt)
        try:
            decision = validate_llm_decision(raw, briefing)
        except Exception as e:
            print(f"  [strategy] LLM output failed validation: {e}")
            return {"trades": [], "executed": [],
                    "skip_reason": f"invalid LLM output: {e}",
                    "decided_at": datetime.utcnow().isoformat()}

        if not decision.picks:
            result = {"trades": [], "executed": [],
                      "skip_reason": decision.skip_reason or "no picks survived validation",
                      "decided_at": datetime.utcnow().isoformat()}
            self._log(briefing, result)
            return result

        # 3. Deterministic sizing from LIVE quotes, scaled by regime
        symbols = [p.symbol for p in decision.picks]
        prices  = get_live_prices(symbols, data_provider=self.data)
        atrs    = self._get_atrs(symbols)

        base_scale = regime.get("risk_scale", 1.0) if regime else 1.0
        long_scale = regime.get("long_scale", 1.0) if regime else 1.0

        risk_budget = equity * MAX_SESSION_RISK_PCT * base_scale
        plans = []
        for pick in decision.picks:
            scale = base_scale * (long_scale if pick.direction == "long" else 1.0)
            plan = build_trade_plan(
                pick,
                live_price=prices.get(pick.symbol),
                atr=atrs.get(pick.symbol),
                equity=equity,
                risk_budget_left=risk_budget,
                risk_scale=scale,
            )
            if plan:
                plans.append(plan)
                risk_budget -= plan.dollar_risk

        # 4. Execute
        executed = []
        for plan in plans:
            try:
                result = self._execute_plan(plan, account)
                if result:
                    executed.append(result)
            except Exception as e:
                print(f"  [strategy] execution error for {plan.symbol}: {e}")

        result = {
            "trades":   [p.model_dump() for p in plans],
            "executed": executed,
            "skip_reason": None if plans else "no plans survived sizing",
            "portfolio_risk_used": round(sum(p.dollar_risk for p in plans) / equity, 4),
            "decided_at": datetime.utcnow().isoformat(),
        }
        self._log(briefing, result)
        return result

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _gather_context(self, briefing: dict) -> tuple[list[dict], str]:
        similar, stats_summary = [], ""
        if self.db:
            try:
                from agents.memory import find_similar_trades, get_strategy_performance_summary
                seen_ids = set()
                for opp in briefing.get("opportunities", [])[:3]:
                    for t in find_similar_trades(
                        self.db, symbol=opp["symbol"],
                        setup_type=opp.get("setup_type", "momentum"),
                        direction=opp.get("direction", "long"), limit=3,
                    ):
                        key = (t.get("symbol"), t.get("pnl"), t.get("exit_reason"))
                        if key not in seen_ids:
                            seen_ids.add(key)
                            similar.append(t)
                stats_summary = get_strategy_performance_summary(self.db)
            except Exception as e:
                print(f"  [strategy] memory lookup failed: {e}")
        return similar, stats_summary

    def _get_atrs(self, symbols: list[str]) -> dict[str, float]:
        atrs = {}
        for sym in symbols:
            try:
                bar = self.data.fetch_latest_bar(sym)
                if bar and bar.get("atr_14"):
                    atrs[sym] = float(bar["atr_14"])
            except Exception as e:
                print(f"  [strategy] ATR fetch failed for {sym}: {e}")
        return atrs

    def _execute_plan(self, plan, account: dict) -> Optional[dict]:
        """Place the order and record the trade as PENDING_FILL."""
        account = self.broker.get_account()  # refresh after prior fills

        cost = plan.quantity * plan.entry_price
        if plan.side == "buy" and cost > account.get("buying_power", 0) * 0.95:
            print(f"  [strategy] skipping {plan.symbol} — insufficient buying power "
                  f"(${cost:.0f} vs ${account.get('buying_power', 0):.0f})")
            return None

        order = self.broker.place_limit_order(
            symbol=plan.symbol, qty=plan.quantity, side=plan.side,
            limit_price=plan.entry_price,
            stop_loss=plan.stop_loss, take_profit=plan.take_profit,
        )

        if order.get("status") == "failed" or not order.get("id"):
            print(f"  [strategy] order rejected for {plan.symbol}")
            return None

        if self.db:
            from db.operations import open_trade, log_decision
            from db.models import AssetType, OrderSide, TradeStatus

            # Mock broker fills instantly → OPEN; Alpaca → PENDING_FILL until
            # the order tracker confirms the fill.
            instant_fill = (order.get("status") == "filled"
                            and order.get("fill_price") is not None)

            trade = open_trade(self.db, {
                "symbol":        plan.symbol,
                "asset_type":    AssetType(plan.asset_type),
                "setup_type":    plan.setup_type,
                "side":          OrderSide(plan.side),
                "entry_price":   order.get("fill_price") or plan.entry_price,
                "quantity":      plan.quantity,
                "planned_stop":  plan.stop_loss,
                "planned_target": plan.take_profit,
                "status":        TradeStatus.OPEN if instant_fill else TradeStatus.PENDING_FILL,
                "entry_context": {
                    "reasoning":        plan.reasoning,
                    "entry_conditions": plan.entry_conditions,
                    "risk_reward":      plan.risk_reward,
                    "dollar_risk":      plan.dollar_risk,
                    "alpaca_order_id":  order.get("id"),
                    "sized_by":         "deterministic_engine_v2",
                },
                "alpaca_order_id": order.get("id"),
            })
            log_decision(
                self.db, agent="strategy", decision_type="enter",
                symbol=plan.symbol, trade_id=trade.id,
                reasoning=plan.reasoning,
                inputs=plan.model_dump(), output=order,
            )
            order["trade_db_id"] = str(trade.id)
            order["trade_status"] = "open" if instant_fill else "pending_fill"

        return order

    def _log(self, briefing: dict, result: dict):
        if not self.db:
            return
        from db.operations import log_decision
        log_decision(
            self.db, agent="strategy", decision_type="evaluate",
            reasoning=result.get("skip_reason") or
                      f"Planned {len(result.get('trades', []))} trades",
            inputs={"briefing_mood": briefing.get("market_mood")},
            output=result,
        )

    # ─── Mock path (no LLM needed) ────────────────────────────────────────────

    def run_mock(self, briefing: dict) -> dict:
        """Mock decision path — uses the same deterministic sizing engine."""
        from execution.sizing import LLMPick, build_trade_plan, MAX_SESSION_RISK_PCT
        from data.market_data import MockDataProvider

        account = self.broker.get_account()
        equity  = account.get("equity", 100_000)
        mock_dp = MockDataProvider()

        risk_budget = equity * MAX_SESSION_RISK_PCT
        plans = []
        for opp in briefing.get("opportunities", [])[:2]:
            bar = mock_dp.fetch_latest_bar(opp["symbol"])
            if not bar:
                continue
            pick = LLMPick(
                symbol=opp["symbol"],
                direction=opp.get("direction", "long"),
                setup_type=opp.get("setup_type", "momentum"),
                conviction=opp.get("confidence", 0.7),
                reasoning=opp.get("thesis", "mock setup evaluation"),
                entry_conditions="mock",
            )
            plan = build_trade_plan(
                pick, live_price=bar["close"],
                atr=bar.get("atr_14", bar["close"] * 0.01),
                equity=equity, risk_budget_left=risk_budget,
            )
            if plan:
                plans.append(plan)
                risk_budget -= plan.dollar_risk

        executed = []
        for plan in plans:
            result = self._execute_plan(plan, account)
            if result:
                executed.append(result)

        mock_decision = {
            "trades":   [p.model_dump() for p in plans],
            "executed": executed,
            "skip_reason": None if plans else "No high-confidence setups found",
            "portfolio_risk_used": round(sum(p.dollar_risk for p in plans) / equity, 4),
            "decided_at": datetime.utcnow().isoformat(),
            "_mock": True,
        }
        self._log(briefing, mock_decision)
        return mock_decision
