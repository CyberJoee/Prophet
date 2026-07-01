"""
Memory System v2.

The old version called Groq's embeddings endpoint with text-embedding-ada-002
— an OpenAI model that Groq does not serve. The call failed on every single
invocation, so the pgvector path never executed and the fallback silently
returned the N most recent trades of ANY kind, mislabeled as "similar trades."

This version retrieves what the strategy agent actually needs:
  1. Trades on the SAME SYMBOL (most relevant)
  2. Trades with the SAME SETUP TYPE (next most relevant)
Both with journal lessons attached, most recent first.

If you later want true semantic search, wire a real embedding provider
(e.g. sentence-transformers locally, or Voyage/OpenAI API) into
embed_journal_entry() below — the pgvector column and index already exist.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def find_similar_trades(db, symbol: str, setup_type: str,
                        direction: str = "long", limit: int = 5) -> list[dict]:
    """
    Retrieve the most relevant historical trades for a prospective setup:
    same-symbol trades first, then same-setup trades, most recent first.
    """
    from db.models import TradeJournal, Trade, TradeStatus, SetupType

    results = []
    seen_trade_ids = set()

    # 1. Same symbol (any setup) — the strongest relevance signal
    same_symbol = (
        db.query(TradeJournal, Trade)
        .join(Trade, TradeJournal.trade_id == Trade.id)
        .filter(
            Trade.status == TradeStatus.CLOSED,
            Trade.symbol == symbol.upper(),
        )
        .order_by(Trade.exit_time.desc())
        .limit(limit)
        .all()
    )
    for j, t in same_symbol:
        seen_trade_ids.add(t.id)
        results.append(_format_result(j, t))

    # 2. Same setup type (any symbol) — fill remaining slots
    if len(results) < limit:
        try:
            setup_enum = SetupType(setup_type)
        except ValueError:
            setup_enum = None
        if setup_enum is not None:
            same_setup = (
                db.query(TradeJournal, Trade)
                .join(Trade, TradeJournal.trade_id == Trade.id)
                .filter(
                    Trade.status == TradeStatus.CLOSED,
                    Trade.setup_type == setup_enum,
                    Trade.id.notin_(seen_trade_ids) if seen_trade_ids else True,
                )
                .order_by(Trade.exit_time.desc())
                .limit(limit - len(results))
                .all()
            )
            for j, t in same_setup:
                results.append(_format_result(j, t))

    return results


def _format_result(journal, trade) -> dict:
    return {
        "symbol":          trade.symbol,
        "setup_type":      trade.setup_type.value if hasattr(trade.setup_type, 'value') else str(trade.setup_type),
        "side":            trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
        "entry_price":     trade.entry_price,
        "exit_price":      trade.exit_price,
        "pnl":             trade.pnl,
        "pnl_pct":         trade.pnl_pct,
        "exit_reason":     trade.exit_reason,
        "what_happened":   journal.what_happened,
        "lessons":         journal.lessons,
        "entry_quality":   journal.entry_quality_score,
        "exit_quality":    journal.exit_quality_score,
        "plan_adherence":  journal.plan_adherence_score,
    }


def get_strategy_performance_summary(db) -> str:
    """
    Return a formatted string of strategy stats for injection into agent prompts.
    Tells the agent which setups are working and which aren't.
    """
    from db.models import StrategyStats

    stats = db.query(StrategyStats).filter(StrategyStats.total_trades > 0).all()
    if not stats:
        return "No strategy performance data yet — this is early in the learning cycle."

    lines = ["STRATEGY PERFORMANCE (from trade history):"]
    for s in sorted(stats, key=lambda x: x.expectancy or 0, reverse=True):
        setup = s.setup_type.value if hasattr(s.setup_type, 'value') else str(s.setup_type)
        lines.append(
            f"  {setup:15s} | trades={s.total_trades:3d} | "
            f"win_rate={s.win_rate:.0%} | "
            f"expectancy=${s.expectancy:.2f} | "
            f"profit_factor={s.profit_factor:.2f}"
        )
    return "\n".join(lines)
