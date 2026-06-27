"""
Memory System
Semantic search over past trade journals using pgvector.
The strategy agent calls this before entering any new trade
to retrieve similar historical setups and their outcomes.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _embed_query(text: str) -> list[float]:
    """Embed a query string using Groq."""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:2000],
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  [memory] embed failed: {e}")
        return None


def find_similar_trades(db, symbol: str, setup_type: str,
                        direction: str = "long", limit: int = 5) -> list[dict]:
    """
    Find the most similar past trades using pgvector cosine similarity.
    Falls back to recent trades filtered by setup type if embedding fails.

    Returns list of dicts with trade details and journal analysis.
    """
    from db.models import TradeJournal, Trade, TradeStatus
    from sqlalchemy import text

    query_text = f"{symbol} {setup_type} {direction} trade setup entry exit lessons"

    # Try vector similarity search first
    embedding = _embed_query(query_text)
    if embedding is not None:
        try:
            # pgvector cosine similarity — lower distance = more similar
            results = (
                db.query(TradeJournal, Trade)
                .join(Trade, TradeJournal.trade_id == Trade.id)
                .filter(
                    Trade.status == TradeStatus.CLOSED,
                    TradeJournal.embedding.isnot(None),
                )
                .order_by(
                    TradeJournal.embedding.cosine_distance(embedding)
                )
                .limit(limit)
                .all()
            )
            if results:
                return [_format_result(j, t) for j, t in results]
        except Exception as e:
            print(f"  [memory] vector search failed ({e}) — falling back to recent")

    # Fallback: recent trades filtered by setup type
    results = (
        db.query(TradeJournal, Trade)
        .join(Trade, TradeJournal.trade_id == Trade.id)
        .filter(Trade.status == TradeStatus.CLOSED)
        .order_by(Trade.exit_time.desc())
        .limit(limit)
        .all()
    )
    return [_format_result(j, t) for j, t in results]


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
