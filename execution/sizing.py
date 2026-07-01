"""
Deterministic sizing engine + LLM output validation.

The LLM's job is now judgment only: WHICH setups to take and WHY.
All arithmetic — entry price, stop, target, quantity, dollar risk —
is computed here from the live quote and ATR. LLMs are unreliable at
arithmetic and their prices go stale; code is neither.

Risk rules (single source of truth, previously duplicated across prompts):
  - risk per trade:        2% of equity
  - stop distance:         0.5 x ATR(14) from entry
  - target distance:       1.0 x ATR(14) from entry  (2:1 R/R)
  - max position value:    15% of equity
  - max portfolio risk:    6% of equity across all new trades in one session
  - entry sanity:          live quote must exist; qty must be >= 1
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# ─── Risk configuration ───────────────────────────────────────────────────────

RISK_PER_TRADE_PCT   = 0.02
STOP_ATR_MULT        = 0.5
TARGET_ATR_MULT      = 1.0
MAX_POSITION_PCT     = 0.15
MAX_SESSION_RISK_PCT = 0.06
MIN_CONVICTION       = 0.60


# ─── LLM output schema (validated, never trusted raw) ─────────────────────────

class LLMPick(BaseModel):
    """What the LLM is allowed to decide. Nothing numeric about sizing."""
    symbol: str = Field(min_length=1, max_length=10)
    direction: Literal["long", "short"]
    setup_type: Literal["momentum", "orb", "vwap_bounce", "reversal",
                        "options_play", "earnings", "custom"]
    conviction: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10, max_length=1000)
    entry_conditions: str = ""

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class LLMDecision(BaseModel):
    picks: list[LLMPick] = Field(default_factory=list)
    skip_reason: Optional[str] = None

    @field_validator("picks")
    @classmethod
    def _cap_picks(cls, v: list) -> list:
        # LLM was told max 3 — if it pads, truncate rather than fail the
        # whole decision (one bad instinct shouldn't kill valid picks)
        return v[:3]


def validate_llm_decision(raw: dict, briefing: dict) -> LLMDecision:
    """
    Parse + validate the LLM's JSON. Then apply business rules:
      - symbol must appear in the research briefing's opportunities
      - symbol must not be on the avoid list
      - conviction must clear the minimum
    Invalid picks are dropped (with a printed reason), not fatal.
    """
    decision = LLMDecision.model_validate(raw)

    briefed  = {o["symbol"].upper() for o in briefing.get("opportunities", [])}
    avoided  = {s.upper() for s in briefing.get("avoid", []) or []}

    kept = []
    for pick in decision.picks:
        if pick.symbol not in briefed:
            print(f"  [validate] dropped {pick.symbol} — not in research briefing")
            continue
        if pick.symbol in avoided:
            print(f"  [validate] dropped {pick.symbol} — on avoid list")
            continue
        if pick.conviction < MIN_CONVICTION:
            print(f"  [validate] dropped {pick.symbol} — conviction "
                  f"{pick.conviction:.2f} < {MIN_CONVICTION}")
            continue
        kept.append(pick)

    decision.picks = kept
    return decision


# ─── Deterministic trade plan construction ────────────────────────────────────

class TradePlan(BaseModel):
    """Fully computed, ready-to-execute plan. All numbers come from code."""
    symbol: str
    asset_type: str = "stock"
    side: Literal["buy", "sell"]
    setup_type: str
    quantity: int = Field(ge=1)
    entry_price: float = Field(gt=0)
    entry_type: str = "limit"
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_reward: float
    dollar_risk: float
    reasoning: str
    entry_conditions: str = ""


def build_trade_plan(pick: LLMPick, live_price: float, atr: float,
                     equity: float, risk_budget_left: float) -> Optional[TradePlan]:
    """
    Turn a validated LLM pick into an executable plan using the LIVE quote.
    Returns None (with a printed reason) if the trade can't be sized sanely.
    """
    if live_price is None or live_price <= 0:
        print(f"  [sizing] {pick.symbol}: no live price — skipping")
        return None
    if atr is None or atr <= 0:
        print(f"  [sizing] {pick.symbol}: no ATR — skipping")
        return None
    if risk_budget_left <= 0:
        print(f"  [sizing] {pick.symbol}: session risk budget exhausted — skipping")
        return None

    is_long   = pick.direction == "long"
    stop_dist = STOP_ATR_MULT * atr

    if is_long:
        stop   = round(live_price - stop_dist, 2)
        target = round(live_price + TARGET_ATR_MULT * atr, 2)
    else:
        stop   = round(live_price + stop_dist, 2)
        target = round(live_price - TARGET_ATR_MULT * atr, 2)

    # Sanity: stop must be on the correct side and non-degenerate
    if stop_dist < live_price * 0.001:
        print(f"  [sizing] {pick.symbol}: stop distance degenerate "
              f"(${stop_dist:.4f}) — skipping")
        return None
    if is_long and not (stop < live_price < target):
        return None
    if not is_long and not (target < live_price < stop):
        return None

    # Position size: 2% risk, capped by 15% of equity and remaining budget
    risk_dollars = min(equity * RISK_PER_TRADE_PCT, risk_budget_left)
    qty = int(risk_dollars / stop_dist)
    qty = min(qty, int((equity * MAX_POSITION_PCT) / live_price))
    if qty < 1:
        print(f"  [sizing] {pick.symbol}: qty < 1 after caps — skipping")
        return None

    dollar_risk = round(qty * stop_dist, 2)
    reward      = abs(target - live_price)
    rr          = round(reward / stop_dist, 2)

    return TradePlan(
        symbol=pick.symbol,
        side="buy" if is_long else "sell",
        setup_type=pick.setup_type,
        quantity=qty,
        entry_price=round(live_price, 2),
        stop_loss=stop,
        take_profit=target,
        risk_reward=rr,
        dollar_risk=dollar_risk,
        reasoning=pick.reasoning,
        entry_conditions=pick.entry_conditions,
    )
