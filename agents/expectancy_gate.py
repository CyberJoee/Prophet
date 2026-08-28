"""
Expectancy Gate — the first mechanism in Prophet that changes its own
behaviour from outcomes, with no LLM in the loop.

WHY THIS EXISTS
---------------
Until now nothing in the system consumed performance history mechanically.
StrategyStats was computed, formatted into prose, and pasted into a prompt;
whether it changed anything depended entirely on an LLM's mood. Every actual
lever — position size, stops, targets, the regime and earnings gates — was
either a fixed rule or deterministic arithmetic that never looked at whether
a setup had been working.

This gate closes that loop. It reads realised outcomes per setup type and
mechanically suspends or sizes down the ones that are losing money. Code
decides; the LLM is told the verdict but cannot override it, exactly like
regime.py.

THE METRIC: R-MULTIPLE
----------------------
Expectancy is measured in R — profit divided by the dollar risk taken at
entry (|entry - planned_stop| * quantity). Raw dollar P&L is unusable here
because position size varies with equity and with the regime scale, so a
setup could look "worse" purely because it traded during a half-size week.
R normalises that away: +1R means the trade made exactly what it risked.

Trades with no recoverable planned_stop cannot be scored and are excluded
from the sample (they are still counted and reported, so a systematic gap
is visible rather than silent).

THE STATISTICAL RULE
--------------------
A negative average is not evidence. With n=12 and the noise level of
intraday trading, a mean of -0.3R is unremarkable. So a setup is only
SUSPENDED when the upper bound of its expectancy is still below zero:

    upper = mean_R + Z * (stdev_R / sqrt(n))        Z defaults to 1.0

At Z=1.0 that is roughly one standard error — a deliberately modest bar,
chosen because the cost of pausing a setup is low (it resumes automatically)
while the cost of bleeding capital into a dead edge is not. Raise
EXPECTANCY_Z to make suspension harder to trigger.

A setup whose mean is negative but whose upper bound is not is REDUCED to
half size rather than stopped: some evidence, not enough to act on fully.

SELF-LOCKOUT, AND HOW IT IS AVOIDED
-----------------------------------
This is the trap in any gate of this shape, and it is worth stating plainly:
a suspended setup places no trades, so it generates no new data, so a naive
implementation can never un-suspend it. The edge might return and the system
would never find out.

The window is therefore bounded by TIME as well as by count — only trades
from the last EXPECTANCY_LOOKBACK_DAYS are considered. A suspended setup's
losing trades age out of that window; once fewer than EXPECTANCY_MIN_TRADES
remain the setup returns to ALLOWED and gets to prove itself again. The
suspension expires on its own, roughly LOOKBACK_DAYS after the last bad
trade. That aging IS the probation mechanism — there is no separate timer
and no stored state anywhere.

Because the verdict is a pure function of (trade history, today's date),
this module holds no state, needs no migration, and returns the same answer
however many times it is called.

CONFIGURATION
-------------
  EXPECTANCY_GATE            on|off             (default on)
  EXPECTANCY_WINDOW          USABLE trades per setup (default 30)
  EXPECTANCY_MIN_TRADES      min sample to act  (default 20)
  EXPECTANCY_LOOKBACK_DAYS   aging window       (default 60)
  EXPECTANCY_Z               strictness         (default 1.0)
  EXPECTANCY_REDUCED_SCALE   size when reduced  (default 0.5)

WHAT IS NEVER SCORED
--------------------
Trades that alpaca_sync reconstructed from a broker position are excluded
outright (see is_reconstructed). They are not strategy decisions: entry time
is whenever the sync ran, setup type is guessed, and until 2026-08-28 the
stop was invented as a flat 2% of price — which meant R, the number this gate
suspends setups on, had a fabricated denominator.

Fails OPEN (all setups allowed) if anything goes wrong, but says so loudly
in `errors` and in the printed reason — a bug in this file must not silently
halt all trading, but it must never be mistaken for a clean verdict either.
"""
import os
import math
import statistics
from datetime import datetime, timedelta

ALLOWED   = "allowed"
REDUCED   = "reduced"
SUSPENDED = "suspended"


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        print(f"  [expectancy] invalid {name} — using {default}")
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        print(f"  [expectancy] invalid {name} — using {default}")
        return default


def is_enabled() -> bool:
    return os.getenv("EXPECTANCY_GATE", "on").strip().lower() not in ("off", "0", "false", "no")


def is_reconstructed(trade) -> bool:
    """
    True for a trade alpaca_sync built from a broker position rather than one
    the strategy decided.

    These are reconstructions. Their entry time is whenever the sync happened,
    the setup type is guessed from a nearby decision row, and historically the
    stop was invented outright. The audit of 2026-08-28 found eighteen of them
    created when a position the system believed closed was still held.

    Whatever they are, they are not evidence about whether a setup works, so
    the gate does not score them. Excluding at read time rather than deleting
    the rows keeps the audit trail intact — the history is still wrong, it is
    just no longer trusted.
    """
    ctx = getattr(trade, "entry_context", None)
    if isinstance(ctx, str):
        import json
        try:
            ctx = json.loads(ctx)
        except Exception:
            return False
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("reconstructed")) or ctx.get("source") == "alpaca_sync"


def trade_r_multiple(trade) -> float | None:
    """
    Realised R for one trade: pnl / dollar risked at entry.

    Returns None when risk cannot be recovered (no planned_stop, stop equal
    to entry, zero quantity, or no pnl) — the caller excludes those from the
    sample rather than guessing at a denominator.
    """
    if trade.pnl is None or not trade.quantity:
        return None
    if trade.planned_stop is None or trade.entry_price is None:
        return None
    risk_per_share = abs(trade.entry_price - trade.planned_stop)
    if risk_per_share <= 0:
        return None
    dollar_risk = risk_per_share * trade.quantity
    if dollar_risk <= 0:
        return None
    return trade.pnl / dollar_risk


def assess_setups(db, now: datetime = None) -> dict:
    """
    Evaluate every setup type against its recent realised expectancy.

    Returns:
      {
        "enabled": bool,
        "metric": "R",
        "setups": {
           "<setup>": {
              "state": allowed|reduced|suspended,
              "scale": 1.0 | 0.5 | 0.0,
              "n": int,                 # trades scored
              "n_unscored": int,        # closed trades with no recoverable risk
              "expectancy_r": float|None,
              "stdev_r": float|None,
              "stderr": float|None,
              "upper_bound": float|None,
              "reason": str,
           }, ...
        },
        "suspended": [str, ...],
        "reduced": [str, ...],
        "errors": [str, ...],
      }
    """
    now = now or datetime.utcnow()
    out = {
        "enabled": is_enabled(), "metric": "R", "setups": {},
        "suspended": [], "reduced": [], "errors": [],
    }

    if not out["enabled"]:
        out["errors"].append("EXPECTANCY_GATE=off — all setups allowed")
        return out

    window        = _cfg_int("EXPECTANCY_WINDOW", 30)
    min_trades    = _cfg_int("EXPECTANCY_MIN_TRADES", 20)
    lookback_days = _cfg_int("EXPECTANCY_LOOKBACK_DAYS", 60)
    z             = _cfg_float("EXPECTANCY_Z", 1.0)
    reduced_scale = _cfg_float("EXPECTANCY_REDUCED_SCALE", 0.5)

    try:
        from db.models import Trade, TradeStatus, SetupType
        from db.operations import get_stats_cutoff

        # Never let the corrupted pre-v2 era influence a live gate.
        cutoff = max(get_stats_cutoff(), now - timedelta(days=lookback_days))

        for setup in SetupType:
            key = setup.value
            # No SQL LIMIT here. The window is "the last N USABLE trades",
            # not "the last N rows" — applying the limit first let excluded
            # rows eat the window and silently starve the sample. A run of
            # broker-reconstructed imports could push the real trades out and
            # drop a setup below min_trades, quietly switching the gate off
            # for it. The date cutoff already bounds how much this scans.
            rows = (db.query(Trade)
                      .filter(Trade.status == TradeStatus.CLOSED,
                              Trade.setup_type == setup,
                              Trade.entry_time >= cutoff)
                      .order_by(Trade.exit_time.desc())
                      .all())

            rs = []
            unscored = 0
            reconstructed = 0
            for t in rows:
                if len(rs) >= window:
                    break
                if is_reconstructed(t):
                    reconstructed += 1
                    continue
                r = trade_r_multiple(t)
                if r is None:
                    unscored += 1
                else:
                    rs.append(r)

            entry = {
                "state": ALLOWED, "scale": 1.0,
                "n": len(rs), "n_unscored": unscored,
                "n_reconstructed": reconstructed,
                "expectancy_r": None, "stdev_r": None,
                "stderr": None, "upper_bound": None,
                "reason": "",
            }

            if len(rs) < min_trades:
                entry["reason"] = (
                    f"only {len(rs)} scored trade(s) in the last {lookback_days}d "
                    f"(need {min_trades}) — no verdict, trading normally"
                )
                if unscored:
                    entry["reason"] += f"; {unscored} unscorable (no planned_stop)"
                if reconstructed:
                    entry["reason"] += (f"; {reconstructed} excluded as "
                                        "broker-reconstructed")
                out["setups"][key] = entry
                continue

            mean_r = statistics.mean(rs)
            sd_r   = statistics.stdev(rs) if len(rs) > 1 else 0.0
            se     = sd_r / math.sqrt(len(rs)) if len(rs) else 0.0
            upper  = mean_r + z * se

            entry["expectancy_r"] = round(mean_r, 4)
            entry["stdev_r"]      = round(sd_r, 4)
            entry["stderr"]       = round(se, 4)
            entry["upper_bound"]  = round(upper, 4)

            if upper < 0:
                entry["state"] = SUSPENDED
                entry["scale"] = 0.0
                entry["reason"] = (
                    f"expectancy {mean_r:+.2f}R over {len(rs)} trades, "
                    f"upper bound {upper:+.2f}R still < 0 — SUSPENDED "
                    f"(auto-resumes as these age past {lookback_days}d)"
                )
                out["suspended"].append(key)
            elif mean_r < 0:
                entry["state"] = REDUCED
                entry["scale"] = reduced_scale
                entry["reason"] = (
                    f"expectancy {mean_r:+.2f}R over {len(rs)} trades but "
                    f"upper bound {upper:+.2f}R >= 0 — not conclusive, "
                    f"size reduced to {reduced_scale:.0%}"
                )
                out["reduced"].append(key)
            else:
                entry["reason"] = (
                    f"expectancy {mean_r:+.2f}R over {len(rs)} trades — full size"
                )

            out["setups"][key] = entry

    except Exception as e:
        # Fail open, but never quietly: a broken gate must not halt trading,
        # and must not be mistaken for a clean all-clear either.
        msg = f"expectancy gate failed ({e}) — failing OPEN, all setups allowed"
        print(f"  [ALERT] [expectancy] {msg}")
        out["errors"].append(msg)
        out["setups"] = {}
        out["suspended"] = []
        out["reduced"] = []

    return out


def setup_scale(gate: dict, setup_type: str) -> float:
    """Size multiplier for one setup. Unknown setups trade at full size."""
    if not gate:
        return 1.0
    entry = (gate.get("setups") or {}).get(str(setup_type))
    return 1.0 if entry is None else float(entry.get("scale", 1.0))


def is_suspended(gate: dict, setup_type: str) -> bool:
    if not gate:
        return False
    entry = (gate.get("setups") or {}).get(str(setup_type))
    return bool(entry) and entry.get("state") == SUSPENDED


def filter_suspended_picks(picks, gate: dict):
    """
    Split LLM picks into (kept, dropped_labels) by suspension state.

    This is the actual enforcement point for the gate. It lives here rather
    than inline in strategy_agent so it can be tested without a broker, a
    data provider, or an LLM — the whole value of the gate is that it holds
    when the LLM disagrees, and that property needs a test.
    """
    if not gate:
        return list(picks), []
    kept, dropped = [], []
    for pick in picks:
        st = getattr(pick, "setup_type", None)
        st = st.value if hasattr(st, "value") else str(st)
        if is_suspended(gate, st):
            dropped.append(f"{getattr(pick, 'symbol', '?')} ({st})")
        else:
            kept.append(pick)
    return kept, dropped


def format_expectancy_for_prompt(gate: dict) -> str:
    """
    Prompt block for the strategy LLM. Explicitly authoritative — the LLM is
    told these are enforced in code so it does not waste picks on a setup
    that will be discarded anyway.
    """
    if not gate or not gate.get("enabled"):
        return ""
    setups = gate.get("setups") or {}
    if not setups:
        return ""

    lines = ["SETUP EXPECTANCY GATE (computed from realised R, authoritative):"]
    acted = False
    for key, e in sorted(setups.items()):
        if e["state"] == SUSPENDED:
            acted = True
            lines.append(f"  x {key}: SUSPENDED — {e['reason']}. Do not propose this setup.")
        elif e["state"] == REDUCED:
            acted = True
            lines.append(f"  ~ {key}: reduced to {e['scale']:.0%} — {e['reason']}")
    if not acted:
        lines.append("  All setups clear — no expectancy-based restrictions today.")
    for err in gate.get("errors", []):
        lines.append(f"  ! {err}")
    return "\n".join(lines)


def format_expectancy_for_log(gate: dict) -> list[str]:
    """One line per setup for the scheduler's stdout."""
    if not gate.get("enabled"):
        return ["  [expectancy] gate disabled (EXPECTANCY_GATE=off)"]
    if gate.get("errors") and not gate.get("setups"):
        return [f"  [expectancy] {e}" for e in gate["errors"]]

    lines = []
    for key, e in sorted((gate.get("setups") or {}).items()):
        mark = {SUSPENDED: "SUSPENDED", REDUCED: "reduced  ", ALLOWED: "ok       "}[e["state"]]
        exp  = f"{e['expectancy_r']:+.2f}R" if e["expectancy_r"] is not None else "  n/a "
        lines.append(f"  [expectancy] {key:13s} {mark} n={e['n']:3d} exp={exp}")
    return lines
