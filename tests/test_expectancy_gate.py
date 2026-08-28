"""
Tests for agents/expectancy_gate.py — the mechanical learning loop.

Runs the real assess_setups() against in-memory SQLite. No Postgres, no
alpaca-py, no LLM. Run directly:

    python tests/test_expectancy_gate.py

The case that matters most is SELF-LOCKOUT RELEASE. A suspended setup places
no trades, so it generates no new data; a naive gate can never un-suspend it
and would silently kill a setup forever. The time-bounded window is what
prevents that, and test 5 is the thing that proves it.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ.pop("EXPECTANCY_GATE", None)

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import UUID
import pgvector.sqlalchemy as pgv


@compiles(UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "CHAR(36)"


@compiles(pgv.Vector, "sqlite")
def _vec_sqlite(type_, compiler, **kw):
    return "BLOB"


from db.connection import engine, Base, SessionLocal
from db.models import Trade, TradeStatus, AssetType, OrderSide, SetupType
from agents import expectancy_gate as G

Base.metadata.create_all(bind=engine)
db = SessionLocal()

NOW = datetime(2026, 8, 22)

NL = chr(10)
failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def reset():
    db.query(Trade).delete()
    db.commit()


def add_reconstructed(setup, r_values, when=None, source="alpaca_sync"):
    """Sync-imported trades. Historically these carried an INVENTED stop, so
    they looked perfectly scorable while their R was fiction."""
    when = when or (NOW - timedelta(days=5))
    for i, r in enumerate(r_values):
        db.add(Trade(
            symbol="RECON", asset_type=AssetType.STOCK, setup_type=setup,
            side=OrderSide.BUY, status=TradeStatus.CLOSED,
            entry_price=100.0, entry_time=when + timedelta(seconds=i),
            quantity=10.0, planned_stop=99.0,          # invented, but present
            exit_price=100.0 + r, exit_time=when + timedelta(minutes=i + 1),
            exit_reason="reconciled", pnl=r * 10.0, pnl_pct=r,
            entry_context={"source": source, "reconstructed": True},
        ))
    db.commit()


def add_trades(setup, r_values, when=None, with_stop=True):
    """
    Insert closed trades whose realised R is exactly each value in r_values.
    entry=100, stop=99 -> risk/share = 1; qty=10 -> dollar risk = 10.
    So pnl = R * 10.
    """
    when = when or (NOW - timedelta(days=5))
    for i, r in enumerate(r_values):
        pnl = r * 10.0
        db.add(Trade(
            symbol="TEST", asset_type=AssetType.STOCK, setup_type=setup,
            side=OrderSide.BUY, status=TradeStatus.CLOSED,
            entry_price=100.0, entry_time=when + timedelta(seconds=i),
            quantity=10.0,
            planned_stop=99.0 if with_stop else None,
            exit_price=100.0 + r, exit_time=when + timedelta(minutes=i + 1),
            exit_reason="test", pnl=pnl, pnl_pct=r,
        ))
    db.commit()


# ── 1. R-multiple arithmetic ────────────────────────────────────────────────
print("\n=== 1. R-multiple maths ===")
reset()
add_trades(SetupType.MOMENTUM, [-0.5])
t = db.query(Trade).first()
check("R = pnl / (|entry-stop| * qty)", G.trade_r_multiple(t) == -0.5,
      str(G.trade_r_multiple(t)))
t.planned_stop = None
check("no planned_stop -> None (not a guess)", G.trade_r_multiple(t) is None)
t.planned_stop = 100.0
check("stop == entry -> None (no divide by zero)", G.trade_r_multiple(t) is None)

# ── 2. Insufficient data ────────────────────────────────────────────────────
print("\n=== 2. Insufficient sample: no verdict ===")
reset()
add_trades(SetupType.MOMENTUM, [-1.0] * 5)   # awful, but only 5 trades
g = G.assess_setups(db, now=NOW)
m = g["setups"]["momentum"]
check("state allowed", m["state"] == G.ALLOWED, m["state"])
check("full size", m["scale"] == 1.0, str(m["scale"]))
check("no expectancy reported", m["expectancy_r"] is None)
check("reason explains the gap", "need 20" in m["reason"], m["reason"])

# ── 3. Clearly negative -> SUSPENDED ────────────────────────────────────────
print("\n=== 3. Consistently losing setup -> SUSPENDED ===")
reset()
add_trades(SetupType.ORB, [-0.6, -0.4] * 12)   # n=24, mean -0.5, tiny spread
g = G.assess_setups(db, now=NOW)
o = g["setups"]["orb"]
check("state suspended", o["state"] == G.SUSPENDED, o["state"])
check("scale 0", o["scale"] == 0.0, str(o["scale"]))
check("expectancy ~ -0.5R", abs(o["expectancy_r"] + 0.5) < 0.01, str(o["expectancy_r"]))
check("upper bound below zero", o["upper_bound"] < 0, str(o["upper_bound"]))
check("listed in suspended", "orb" in g["suspended"], str(g["suspended"]))
check("is_suspended() agrees", G.is_suspended(g, "orb"))
check("setup_scale() is 0", G.setup_scale(g, "orb") == 0.0)

# ── 4. Negative but noisy -> REDUCED, not suspended ─────────────────────────
print("\n=== 4. Negative mean, wide spread -> REDUCED only ===")
reset()
add_trades(SetupType.REVERSAL, [-3.0, 2.8] * 12)   # mean -0.1, sd ~2.9
g = G.assess_setups(db, now=NOW)
r = g["setups"]["reversal"]
check("mean is negative", r["expectancy_r"] < 0, str(r["expectancy_r"]))
check("state reduced (not suspended)", r["state"] == G.REDUCED, r["state"])
check("half size", r["scale"] == 0.5, str(r["scale"]))
check("upper bound >= 0 - noise respected", r["upper_bound"] >= 0, str(r["upper_bound"]))
check("not in suspended list", "reversal" not in g["suspended"])

# ── 5. SELF-LOCKOUT RELEASE ─────────────────────────────────────────────────
print("\n=== 5. Suspension expires as losses age out (no permanent lockout) ===")
reset()
add_trades(SetupType.ORB, [-0.6, -0.4] * 12, when=datetime(2026, 8, 1))
g_then = G.assess_setups(db, now=datetime(2026, 8, 22))
check("suspended while losses are recent", g_then["setups"]["orb"]["state"] == G.SUSPENDED,
      g_then["setups"]["orb"]["state"])

# Same trades, same DB, no new data — only the clock has moved past the window.
g_later = G.assess_setups(db, now=datetime(2026, 12, 1))
o2 = g_later["setups"]["orb"]
check("allowed again once aged out", o2["state"] == G.ALLOWED, o2["state"])
check("full size restored", o2["scale"] == 1.0, str(o2["scale"]))
check("sample is now empty", o2["n"] == 0, str(o2["n"]))
check("no longer suspended", "orb" not in g_later["suspended"])

# ── 6. Unscorable trades excluded, not guessed at ───────────────────────────
print("\n=== 6. Trades with no planned_stop are excluded and counted ===")
reset()
add_trades(SetupType.VWAP_BOUNCE, [-1.0] * 24, with_stop=False)
g = G.assess_setups(db, now=NOW)
v = g["setups"]["vwap_bounce"]
check("no verdict from unscorable trades", v["state"] == G.ALLOWED, v["state"])
check("scored sample is empty", v["n"] == 0, str(v["n"]))
check("unscored are counted, not hidden", v["n_unscored"] == 24, str(v["n_unscored"]))
check("reason mentions planned_stop", "planned_stop" in v["reason"], v["reason"])

# ── 7. Pre-cutoff (corrupted era) trades never influence the gate ───────────
print("\n=== 7. Pre-STATS_SINCE trades excluded ===")
reset()
add_trades(SetupType.MOMENTUM, [-0.6, -0.4] * 12, when=datetime(2026, 6, 1))
g = G.assess_setups(db, now=NOW)
check("corrupted-era trades ignored", g["setups"]["momentum"]["n"] == 0,
      str(g["setups"]["momentum"]["n"]))
check("no suspension from them", "momentum" not in g["suspended"])

# ── 8. Positive expectancy stays at full size ───────────────────────────────
print("\n=== 8. Profitable setup untouched ===")
reset()
add_trades(SetupType.MOMENTUM, [0.4, 0.6] * 12)
g = G.assess_setups(db, now=NOW)
m = g["setups"]["momentum"]
check("allowed", m["state"] == G.ALLOWED, m["state"])
check("full size", m["scale"] == 1.0, str(m["scale"]))
check("expectancy ~ +0.5R", abs(m["expectancy_r"] - 0.5) < 0.01, str(m["expectancy_r"]))

# ── 9. Kill switch ──────────────────────────────────────────────────────────
print("\n=== 9. EXPECTANCY_GATE=off disables everything ===")
reset()
add_trades(SetupType.ORB, [-0.6, -0.4] * 12)
os.environ["EXPECTANCY_GATE"] = "off"
g = G.assess_setups(db, now=NOW)
check("gate reports disabled", g["enabled"] is False)
check("nothing suspended", g["suspended"] == [], str(g["suspended"]))
check("setup_scale full when off", G.setup_scale(g, "orb") == 1.0)
del os.environ["EXPECTANCY_GATE"]

# ── 10. Fails OPEN and loudly ───────────────────────────────────────────────
print("\n=== 10. Broken DB -> fails open, but records the error ===")


class ExplodingDB:
    def query(self, *a, **k):
        raise RuntimeError("simulated DB outage")


g = G.assess_setups(ExplodingDB(), now=NOW)
check("no setups suspended on failure", g["suspended"] == [], str(g["suspended"]))
check("error recorded, not swallowed", len(g["errors"]) == 1, str(g["errors"]))
check("error names the cause", "simulated DB outage" in g["errors"][0], g["errors"][0])
check("scale defaults to full", G.setup_scale(g, "orb") == 1.0)

# ── 11. Prompt block is honest about enforcement ────────────────────────────
print("\n=== 11. Prompt formatting ===")
reset()
add_trades(SetupType.ORB, [-0.6, -0.4] * 12)
g = G.assess_setups(db, now=NOW)
block = G.format_expectancy_for_prompt(g)
check("names the suspended setup", "orb" in block, block)
check("marked authoritative", "authoritative" in block.lower(), block)
check("tells the LLM not to propose it", "Do not propose" in block, block)
reset()
add_trades(SetupType.MOMENTUM, [0.4, 0.6] * 12)
clear_block = G.format_expectancy_for_prompt(G.assess_setups(db, now=NOW))
check("says so when nothing is restricted", "All setups clear" in clear_block, clear_block)

# ── 12. ENFORCEMENT: suspended picks are dropped, whatever the LLM says ─────
print("\n=== 12. Enforcement against real LLMPick objects ===")
from execution.sizing import LLMPick, LLMDecision

reset()
add_trades(SetupType.ORB, [-0.6, -0.4] * 12)          # orb -> suspended
add_trades(SetupType.MOMENTUM, [0.4, 0.6] * 12)       # momentum -> allowed
gate = G.assess_setups(db, now=NOW)
check("orb suspended, momentum not",
      G.is_suspended(gate, "orb") and not G.is_suspended(gate, "momentum"))


def pick(sym, setup, conviction=0.9):
    return LLMPick(symbol=sym, direction="long", setup_type=setup,
                   conviction=conviction,
                   reasoning="a sufficiently long piece of model reasoning")


# The LLM is maximally confident about the suspended setup — irrelevant.
picks = [pick("AAPL", "orb", 1.0), pick("NVDA", "momentum"), pick("TSLA", "orb", 1.0)]
kept, dropped = G.filter_suspended_picks(picks, gate)
check("both orb picks dropped", len(dropped) == 2, str(dropped))
check("momentum pick kept", len(kept) == 1 and kept[0].symbol == "NVDA",
      str([k.symbol for k in kept]))
check("conviction 1.0 does not override the gate",
      all("orb" in d for d in dropped), str(dropped))
check("dropped labels name symbol and setup", "AAPL (orb)" in dropped, str(dropped))

# Assigning back onto the pydantic model must actually work — strategy_agent
# does exactly this, and a frozen model would break it silently at runtime.
decision = LLMDecision(picks=picks)
decision.picks = kept
check("LLMDecision.picks is assignable", len(decision.picks) == 1, str(decision.picks))

# No gate at all -> nothing dropped.
kept2, dropped2 = G.filter_suspended_picks(picks, None)
check("no gate -> all picks kept", len(kept2) == 3 and dropped2 == [])

# Gate that failed open -> nothing dropped.
broken = G.assess_setups(ExplodingDB(), now=NOW)
kept3, dropped3 = G.filter_suspended_picks(picks, broken)
check("failed-open gate drops nothing", len(kept3) == 3 and dropped3 == [])

# ── 13. Reduced setups are sized down, not dropped ──────────────────────────
print("\n=== 13. REDUCED sizes down rather than blocking ===")
reset()
add_trades(SetupType.REVERSAL, [-3.0, 2.8] * 12)
g = G.assess_setups(db, now=NOW)
rp = [pick("AAPL", "reversal")]
kept4, dropped4 = G.filter_suspended_picks(rp, g)
check("reduced setup is NOT dropped", len(kept4) == 1 and dropped4 == [], str(dropped4))
check("but its scale is 0.5", G.setup_scale(g, "reversal") == 0.5,
      str(G.setup_scale(g, "reversal")))
check("suspended scale 0 would skip in sizing.py",
      G.setup_scale(G.assess_setups(db, now=NOW), "reversal") > 0)

# -- 14. Reconstructed trades are never scored ------------------------------
print(NL + "=== 14. Broker-reconstructed trades are excluded ===")
reset()
add_reconstructed(SetupType.ORB, [-0.6, -0.4] * 12)   # would suspend if scored
g = G.assess_setups(db, now=NOW)
o = g["setups"]["orb"]
check("not scored", o["n"] == 0, str(o["n"]))
check("counted as reconstructed", o["n_reconstructed"] == 24, str(o["n_reconstructed"]))
check("NOT counted as merely unscorable", o["n_unscored"] == 0, str(o["n_unscored"]))
check("no verdict from them", o["state"] == G.ALLOWED, o["state"])
check("not suspended", "orb" not in g["suspended"], str(g["suspended"]))
check("reason explains the exclusion", "reconstructed" in o["reason"], o["reason"])

print(NL + "--- a real losing setup still suspends alongside them ---")
reset()
add_reconstructed(SetupType.MOMENTUM, [0.9] * 20)     # flattering fiction
add_trades(SetupType.MOMENTUM, [-0.6, -0.4] * 12)     # the real record
g = G.assess_setups(db, now=NOW)
m = g["setups"]["momentum"]
check("only real trades scored", m["n"] == 24, str(m["n"]))
check("reconstructed ones set aside", m["n_reconstructed"] == 20,
      str(m["n_reconstructed"]))
check("fiction did not rescue it", m["state"] == G.SUSPENDED, m["state"])
check("expectancy is the real -0.5R", abs(m["expectancy_r"] + 0.5) < 0.01,
      str(m["expectancy_r"]))

print(NL + "--- is_reconstructed handles every context shape ---")


class _Ctx:
    def __init__(self, ctx):
        self.entry_context = ctx


check("dict with source", G.is_reconstructed(_Ctx({"source": "alpaca_sync"})))
check("dict with flag", G.is_reconstructed(_Ctx({"reconstructed": True})))
check("JSON string", G.is_reconstructed(_Ctx('{"source": "alpaca_sync"}')))
check("native trade is not", not G.is_reconstructed(_Ctx({"reasoning": "real"})))
check("None context is not", not G.is_reconstructed(_Ctx(None)))
check("malformed string is not", not G.is_reconstructed(_Ctx("not json")))

# -- 15. Sync no longer invents a stop --------------------------------------
print(NL + "=== 15. alpaca_sync stores no fabricated stop ===")
_sync_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "execution", "alpaca_sync.py"), encoding="utf-8").read()
check("the 2%-of-price guess is gone", "current * 0.02" not in _sync_src)
check("stop is explicitly None", "stop       = None" in _sync_src)
check("imports are marked reconstructed", 'reconstructed' in _sync_src)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
