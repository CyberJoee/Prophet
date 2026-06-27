"""
Journal Agent
Runs after EOD close each market day.
- Writes structured post-trade analysis for every closed trade
- Embeds each journal entry into pgvector for semantic memory
- Updates strategy performance stats
"""
import os
import json
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()


JOURNAL_SYSTEM_PROMPT = """You are a professional trading coach reviewing closed trades.
You receive the details of a completed trade and must write a structured post-trade analysis.

Respond ONLY with valid JSON. No preamble, no markdown fences.
Structure:
{
  "what_happened": "<2-3 sentences describing the trade objectively>",
  "what_went_right": "<what the agent did well, or N/A>",
  "what_went_wrong": "<what went wrong or could be improved, or N/A>",
  "lessons": "<1-2 concrete lessons to apply to future trades>",
  "market_conditions": "<brief description of market context at entry>",
  "entry_quality_score": <1-10>,
  "exit_quality_score": <1-10>,
  "plan_adherence_score": <1-10>
}
Be specific and honest. A losing trade with good process scores higher than a winning trade with bad process.
"""


def _build_journal_prompt(trade, similar_past: list[dict]) -> str:
    lines = []
    lines.append("TRADE TO REVIEW:")
    lines.append(f"  Symbol:     {trade.symbol}")
    lines.append(f"  Setup:      {trade.setup_type.value if hasattr(trade.setup_type, 'value') else trade.setup_type}")
    lines.append(f"  Side:       {trade.side.value if hasattr(trade.side, 'value') else trade.side}")
    lines.append(f"  Entry:      ${trade.entry_price:.2f} x{trade.quantity:.0f} shares")
    lines.append(f"  Exit:       ${trade.exit_price:.2f} ({trade.exit_reason})")
    lines.append(f"  PnL:        ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)")
    lines.append(f"  Entry time: {trade.entry_time}")
    lines.append(f"  Exit time:  {trade.exit_time}")

    if trade.planned_stop:
        lines.append(f"  Stop loss:  ${trade.planned_stop:.2f}")
    if trade.planned_target:
        lines.append(f"  Target:     ${trade.planned_target:.2f}")
    if trade.max_adverse_excursion is not None:
        lines.append(f"  Max adverse excursion: ${trade.max_adverse_excursion:.2f}")
    if trade.max_favorable_excursion is not None:
        lines.append(f"  Max favorable excursion: ${trade.max_favorable_excursion:.2f}")

    if trade.entry_context:
        lines.append(f"  Entry reasoning: {json.dumps(trade.entry_context)}")

    if similar_past:
        lines.append("\nSIMILAR PAST TRADES FOR CONTEXT:")
        for p in similar_past[:3]:
            outcome = f"+${p['pnl']:.2f}" if p.get('pnl', 0) > 0 else f"-${abs(p.get('pnl', 0)):.2f}"
            lines.append(f"  {p['symbol']} {p['setup_type']} → {outcome} ({p.get('exit_reason','?')})")

    lines.append("\nWrite the post-trade journal entry.")
    return "\n".join(lines)


def _build_embedding_text(trade, journal) -> str:
    """Build a rich text description of the trade for embedding."""
    setup = trade.setup_type.value if hasattr(trade.setup_type, 'value') else str(trade.setup_type)
    side  = trade.side.value if hasattr(trade.side, 'value') else str(trade.side)
    outcome = "win" if trade.pnl and trade.pnl > 0 else "loss"
    return (
        f"{trade.symbol} {setup} {side} {outcome} "
        f"entry ${trade.entry_price:.2f} exit ${trade.exit_price:.2f} "
        f"pnl ${trade.pnl:.2f} {trade.pnl_pct:+.2f}% "
        f"exit_reason {trade.exit_reason} "
        f"lessons: {journal.get('lessons', '')} "
        f"market: {journal.get('market_conditions', '')}"
    )


class JournalAgent:

    def __init__(self, db_session=None):
        self.db = db_session
        self._groq = None
        self._embed_client = None

    def _get_groq(self):
        if self._groq is None:
            from groq import Groq
            self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._groq

    def _call_llm(self, trade, similar_past) -> dict:
        prompt = _build_journal_prompt(trade, similar_past)
        client = self._get_groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.3,
            messages=[
                {"role": "system", "content": JOURNAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _embed(self, text: str) -> list[float]:
        """
        Generate a 1536-dim embedding using Groq's embedding model.
        Falls back to a zero vector if embedding fails.
        """
        try:
            client = self._get_groq()
            # Groq supports OpenAI-compatible embeddings
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:2000],  # truncate to avoid token limits
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"  [journal] embedding failed ({e}) — using zero vector")
            return [0.0] * 1536

    def _get_similar_trades(self, trade) -> list[dict]:
        """Get recent closed trades for context. pgvector similarity used once we have embeddings."""
        if self.db is None:
            return []
        try:
            from db.operations import get_trade_history
            recent = get_trade_history(self.db, limit=10)
            setup = trade.setup_type.value if hasattr(trade.setup_type, 'value') else str(trade.setup_type)
            # Filter to same setup type first, then fall back to any recent
            same_setup = [t for t in recent if
                          (t.setup_type.value if hasattr(t.setup_type, 'value') else str(t.setup_type)) == setup
                          and t.id != trade.id]
            return same_setup[:3] if same_setup else [t for t in recent if t.id != trade.id][:3]
        except Exception:
            return []

    def _format_similar(self, trades) -> list[dict]:
        result = []
        for t in trades:
            result.append({
                "symbol":     t.symbol,
                "setup_type": t.setup_type.value if hasattr(t.setup_type, 'value') else str(t.setup_type),
                "pnl":        t.pnl or 0,
                "exit_reason": t.exit_reason,
            })
        return result

    def journal_trade(self, trade) -> dict:
        """Write a journal entry for a single closed trade."""
        from db.models import TradeJournal
        from db.operations import log_decision

        # Skip if already journaled
        existing = self.db.query(TradeJournal).filter(
            TradeJournal.trade_id == trade.id
        ).first()
        if existing:
            print(f"  [journal] {trade.symbol} already journaled — skipping")
            return {}

        # Get similar past trades for context
        similar_raw  = self._get_similar_trades(trade)
        similar_list = self._format_similar(similar_raw)

        # Call LLM for analysis
        try:
            analysis = self._call_llm(trade, similar_list)
        except Exception as e:
            print(f"  [journal] LLM failed for {trade.symbol}: {e} — using template")
            outcome = "profitable" if trade.pnl and trade.pnl > 0 else "losing"
            analysis = {
                "what_happened":       f"{trade.symbol} {outcome} trade closed via {trade.exit_reason}",
                "what_went_right":     "N/A",
                "what_went_wrong":     "N/A",
                "lessons":             "Review setup conditions before next entry",
                "market_conditions":   "Unknown",
                "entry_quality_score": 5,
                "exit_quality_score":  5,
                "plan_adherence_score": 5,
            }

        # Generate embedding
        embed_text = _build_embedding_text(trade, analysis)
        embedding  = self._embed(embed_text)

        # Save journal entry
        journal_entry = TradeJournal(
            trade_id=trade.id,
            what_happened=analysis.get("what_happened"),
            what_went_right=analysis.get("what_went_right"),
            what_went_wrong=analysis.get("what_went_wrong"),
            lessons=analysis.get("lessons"),
            market_conditions=analysis.get("market_conditions"),
            entry_quality_score=analysis.get("entry_quality_score"),
            exit_quality_score=analysis.get("exit_quality_score"),
            plan_adherence_score=analysis.get("plan_adherence_score"),
            embedding=embedding,
        )
        self.db.add(journal_entry)
        self.db.commit()

        pnl_str = f"{'+'if trade.pnl>=0 else ''}{trade.pnl:.2f}"
        print(f"  [journal] {trade.symbol} journaled | PnL=${pnl_str} | "
              f"entry={analysis.get('entry_quality_score')}/10 "
              f"exit={analysis.get('exit_quality_score')}/10")

        log_decision(
            self.db, agent="journal", decision_type="journal",
            symbol=trade.symbol, trade_id=trade.id,
            reasoning=analysis.get("lessons"),
            output=analysis,
        )

        return analysis

    def run(self, target_date: date = None) -> list[dict]:
        """
        Journal all closed trades from target_date (defaults to today).
        Called by the scheduler after EOD close.
        """
        from db.models import Trade, TradeStatus
        if target_date is None:
            target_date = datetime.utcnow().date()

        trades = (
            self.db.query(Trade)
            .filter(
                Trade.status == TradeStatus.CLOSED,
                Trade.exit_time >= datetime.combine(target_date, datetime.min.time()),
            )
            .all()
        )

        if not trades:
            print(f"  [journal] No closed trades on {target_date} to journal")
            return []

        print(f"  [journal] Journaling {len(trades)} trade(s) from {target_date}")
        results = []
        for trade in trades:
            result = self.journal_trade(trade)
            if result:
                results.append(result)

        # Update strategy stats after journaling
        try:
            from db.operations import refresh_strategy_stats
            refresh_strategy_stats(self.db)
            print(f"  [journal] Strategy stats updated")
        except Exception as e:
            print(f"  [journal] Stats update failed: {e}")

        return results

    def run_mock(self, target_date: date = None) -> list[dict]:
        """Run without LLM — generates template journal entries. For testing."""
        from db.models import Trade, TradeStatus
        if target_date is None:
            target_date = datetime.utcnow().date()

        trades = (
            self.db.query(Trade)
            .filter(
                Trade.status == TradeStatus.CLOSED,
                Trade.exit_time >= datetime.combine(target_date, datetime.min.time()),
            )
            .all()
        )

        if not trades:
            print(f"  [journal] No closed trades on {target_date}")
            return []

        results = []
        for trade in trades:
            from db.models import TradeJournal
            existing = self.db.query(TradeJournal).filter(
                TradeJournal.trade_id == trade.id
            ).first()
            if existing:
                continue

            outcome = "profitable" if trade.pnl and trade.pnl > 0 else "losing"
            analysis = {
                "what_happened":        f"{trade.symbol} {outcome} {trade.setup_type.value if hasattr(trade.setup_type,'value') else trade.setup_type} trade. Entered at ${trade.entry_price:.2f}, exited at ${trade.exit_price:.2f} via {trade.exit_reason}.",
                "what_went_right":      "Entry timing was aligned with the planned setup" if outcome == "profitable" else "Stop loss contained the loss as planned",
                "what_went_wrong":      "N/A" if outcome == "profitable" else "Setup did not follow through as expected",
                "lessons":              f"{'Continue trading this setup type' if outcome == 'profitable' else 'Review entry conditions for this setup before next trade'}",
                "market_conditions":    "Mixed market conditions with moderate volatility",
                "entry_quality_score":  7 if outcome == "profitable" else 5,
                "exit_quality_score":   8 if trade.exit_reason == "target_hit" else 6,
                "plan_adherence_score": 8,
            }

            journal_entry = TradeJournal(
                trade_id=trade.id,
                what_happened=analysis["what_happened"],
                what_went_right=analysis["what_went_right"],
                what_went_wrong=analysis["what_went_wrong"],
                lessons=analysis["lessons"],
                market_conditions=analysis["market_conditions"],
                entry_quality_score=analysis["entry_quality_score"],
                exit_quality_score=analysis["exit_quality_score"],
                plan_adherence_score=analysis["plan_adherence_score"],
                embedding=[0.0] * 1536,  # zero vector in mock mode
            )
            self.db.add(journal_entry)
            self.db.commit()

            pnl_str = f"{'+'if trade.pnl>=0 else ''}{trade.pnl:.2f}"
            print(f"  [journal] [MOCK] {trade.symbol} journaled | PnL=${pnl_str}")
            results.append(analysis)

        try:
            from db.operations import refresh_strategy_stats
            refresh_strategy_stats(self.db)
        except Exception:
            pass

        return results
