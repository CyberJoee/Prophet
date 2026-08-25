"""
Deterministic sizing engine + LLM output validation.

The LLM's job is now judgment only: WHICH setups to take and WHY.
All arithmetic — entry price, stop, target, quantity, dollar risk —
is computed here from the live quote and ATR. LLMs are unreliable at
arithmetic and their prices go stale; code is neither.

Risk rules (single source of truth, previously duplicated across prompts):
  - risk per trade:        2% of equity   <-- SEE WARNING BELOW, never binds
  - stop distance:         0.5 x ATR(14) from entry
  - target distance:       1.0 x ATR(14) from entry  (2:1 R/R)
  - max position value:    15% of equity  <-- this is what actually sizes
  - max portfolio risk:    6% of equity across all new trades in one session
  - entry sanity:          live quote must exist; qty must be >= 1

WARNING: THE 2% RISK RULE HAS NEVER BEEN IN EFFECT
--------------------------------------------------
Two independently sensible rules are in conflict, and the notional cap wins
every time:

    risk-derived qty = RISK_PER_TRADE_PCT * equity / stop_distance
    cap-derived qty  = MAX_POSITION_PCT   * equity / price

Setting them equal, the risk rule only binds when

    stop_distance >= price * (RISK_PER_TRADE_PCT / MAX_POSITION_PCT)
                  =  price * 13.3%

Stops here are 0.5 x ATR(14) on 5-minute intraday bars, typically ~0.4-1.5%
of price. So the cap binds on essentially every trade and real risk per
trade is ~0.05-0.2% of equity, not 2%. Verified against live fills:

    SPY  $762.34 -> 20 sh (cap 20.3)    risk taken $66  = 0.064% of equity
    QQQ  $728.05 -> 21 sh (cap 21.2)
    MSFT $496.42 -> 31 sh (cap 31.1)
    NVDA $226.37 -> 68 sh (cap 68.3)

The cap is NOT the bug. Risking 2% with a 0.4% stop implies ~4.6x notional
leverage (620 SPY shares = $472k against $412k buying power) — the order
would be rejected. The cap is the only thing preventing the risk rule from
demanding impossible positions. The 2% figure is simply unreachable given
this stop geometry.

Consequences to keep in mind:
  - Do not "fix" this by raising MAX_POSITION_PCT. Sizing is a multiplier on
    expectancy; scaling up a negative-expectancy system just loses faster.
    Establish edge first (see the geometry sweep), then revisit.
  - The expectancy gate's REDUCED tier is currently inert for the same
    reason: halving risk_dollars moves the risk-derived qty from e.g. 620 to
    310, still far above a cap of 20, so final size is unchanged. The
    SUSPENDED tier still works, because it drops the pick before sizing.
  - At small equity the cap collapses to a handful of shares and integer
    truncation dominates: at $10k, SPY sizes to 1 share ($3.32 risk) and
    QQQ to 2. Below roughly $10k, high-priced symbols round to qty 0 and are
    skipped entirely. See size_report() for a quick check at any equity.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# ─── Risk configuration ───────────────────────────────────────────────────────

# NOTE: RISK_PER_TRADE_PCT is aspirational, not operative — MAX_POSITION_PCT
# binds first on every realistic intraday stop. See the module docstring.
RISK_PER_TRADE_PCT   = 0.02
STOP_ATR_MULT        = 0.5      # geometry under test — see backtesting/geometry_sweep.py
TARGET_ATR_MULT      = 1.0      # only ~7% of live trades ever reach this
MAX_POSITION_PCT     = 0.15     # the rule that actually determines position size
MAX_SESSION_RISK_PCT = 0.06
MIN_CONVICTION       = 0.60


def size_report(equity: float, price: float, atr: float,
                stop_mult: float = None) -> dict:
    """
    What would sizing do at this equity/price/ATR? Pure arithmetic, no order.

    Exists because the interaction between the risk rule and the notional cap
    is not obvious by inspection, and gets worse as equity shrinks — which
    matters if the account is ever run down to ~$10k.
    """
    stop_mult = STOP_ATR_MULT if stop_mult is None else stop_mult
    stop_dist = stop_mult * atr
    risk_qty  = int(equity * RISK_PER_TRADE_PCT / stop_dist) if stop_dist else 0
    cap_qty   = int(equity * MAX_POSITION_PCT / price) if price else 0
    qty       = max(0, min(risk_qty, cap_qty))
    return {
        "equity": equity, "price": price, "stop_distance": round(stop_dist, 4),
        "risk_qty": risk_qty, "cap_qty": cap_qty, "qty": qty,
        "bound_by": ("none — qty 0, trade skipped" if qty < 1
                     else "notional-cap" if cap_qty <= risk_qty else "risk-rule"),
        "dollar_risk": round(qty * stop_dist, 2),
        "risk_pct_of_equity": round(qty * stop_dist / equity, 5) if equity else 0,
        "notional": round(qty * price, 2),
    }


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
                     equity: float, risk_budget_left: float,
                     risk_scale: float = 1.0,
                     stop_mult: float = None,
                     target_mult: float = None,
                     verbose: bool = True) -> Optional[TradePlan]:
    """
    Turn a validated LLM pick into an executable plan using the LIVE quote.
    risk_scale (0-1) comes from the regime gate — scales dollar risk down
    in hostile conditions. 0 means no trade.

    stop_mult / target_mult override the module ATR multiples. They exist so
    the geometry sweep can test alternative stop/target pairs through THIS
    code path rather than a reimplementation — the same discipline that made
    engine_v2 trustworthy. Both default to the live constants, so omitting
    them reproduces production behaviour exactly.

    Returns None (with a printed reason) if the trade can't be sized sanely.
    """
    stop_mult   = STOP_ATR_MULT   if stop_mult   is None else stop_mult
    target_mult = TARGET_ATR_MULT if target_mult is None else target_mult
    if risk_scale <= 0:
        print(f"  [sizing] {pick.symbol}: risk_scale is 0 (regime gate) — skipping")
        return None
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
    stop_dist = stop_mult * atr

    if is_long:
        stop   = round(live_price - stop_dist, 2)
        target = round(live_price + target_mult * atr, 2)
    else:
        stop   = round(live_price + stop_dist, 2)
        target = round(live_price - target_mult * atr, 2)

    # Sanity: stop must be on the correct side and non-degenerate
    if stop_dist < live_price * 0.001:
        print(f"  [sizing] {pick.symbol}: stop distance degenerate "
              f"(${stop_dist:.4f}) — skipping")
        return None
    if is_long and not (stop < live_price < target):
        return None
    if not is_long and not (target < live_price < stop):
        return None

    # Position size: 2% risk scaled by regime, capped by 15% of equity
    # and remaining session budget
    risk_dollars = min(equity * RISK_PER_TRADE_PCT * risk_scale, risk_budget_left)
    risk_qty = int(risk_dollars / stop_dist)
    cap_qty  = int((equity * MAX_POSITION_PCT) / live_price)
    qty = min(risk_qty, cap_qty)
    if qty < 1:
        print(f"  [sizing] {pick.symbol}: qty < 1 after caps "
              f"(risk-qty {risk_qty}, cap-qty {cap_qty}) — skipping")
        return None

    # Name the binding constraint. This is the line whose absence let a dead
    # 2% risk rule sit in the code unnoticed for the life of the project.
    if verbose:
        bound_by = "notional-cap" if cap_qty <= risk_qty else "risk-rule"
        print(f"  [sizing] {pick.symbol}: qty {qty} bound by {bound_by} "
              f"(risk-qty {risk_qty}, cap-qty {cap_qty}); "
              f"risk ${qty * stop_dist:,.2f} = {qty * stop_dist / equity:.3%} of equity")

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
