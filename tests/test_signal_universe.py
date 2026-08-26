"""
Tests for the split between the SIGNAL COLLECTION universe and the LLM
BRIEFING watchlist.

Two separate jobs that used to share one symbol list:

  collect  — as wide as possible. Cross-sectional IC noise falls with the
             number of names, so more symbols per date buys statistical power
             directly. Storage is nearly free.
  brief    — as narrow as possible. The prompt is a scarce resource; the
             research call was truncated mid-JSON on 2026-08-26 by exceeding
             its token budget, and the LLM can only trade the watchlist.

If these ever collapse back into one list, one of the two jobs is being done
badly. That is what these tests pin down.

No network, no Postgres (SQLite). Run directly:

    python tests/test_signal_universe.py
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"
os.environ.pop("SIGNAL_UNIVERSE", None)

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
from db.models import AltSignal
from data.alt_signals import aggregator as A
from data.alt_signals import options_flow, short_volume, event_risk

Base.metadata.create_all(bind=engine)

failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


WATCHLIST = ["NVDA", "AAPL", "SPY"]

# ── 1. get_signal_universe ──────────────────────────────────────────────────
print("\n=== 1. Universe construction ===")
u = A.get_signal_universe(WATCHLIST)
check("universe is a superset of the watchlist",
      set(w.upper() for w in WATCHLIST) <= set(u))
check("universe is meaningfully wider", len(u) >= 30, str(len(u)))
check("watchlist names come first", u[:3] == ["NVDA", "AAPL", "SPY"], str(u[:3]))
check("no duplicates", len(u) == len(set(u)), f"{len(u)} vs {len(set(u))}")

os.environ["SIGNAL_UNIVERSE"] = "AAPL,MSFT,AAPL"
u2 = A.get_signal_universe(["NVDA"])
check("env override respected", set(u2) == {"NVDA", "AAPL", "MSFT"}, str(u2))
check("override dedupes", len(u2) == 3, str(u2))

os.environ["SIGNAL_UNIVERSE"] = ",".join(f"SYM{i}" for i in range(200))
u3 = A.get_signal_universe([])
check("runaway universe is capped", len(u3) == A.MAX_UNIVERSE, str(len(u3)))
del os.environ["SIGNAL_UNIVERSE"]

# Setting SIGNAL_UNIVERSE to the watchlist restores the pre-2026-08-26
# behaviour, which is the documented escape hatch.
os.environ["SIGNAL_UNIVERSE"] = "NVDA,AAPL"
check("narrow override restores old behaviour",
      A.get_signal_universe(["NVDA", "AAPL"]) == ["NVDA", "AAPL"],
      str(A.get_signal_universe(["NVDA", "AAPL"])))
os.environ.pop("SIGNAL_UNIVERSE", None)


# ── 2. collect_all stores wide, briefs narrow ───────────────────────────────
print("\n=== 2. Collection is wide, briefing is narrow ===")

UNIVERSE = ["NVDA", "AAPL", "SPY", "AMD", "COST", "XLE", "BA"]

options_flow.collect_options_flow = lambda sym: {
    "cp_volume_ratio": 1.1, "unusual_contracts": 3,
    "unusual_call_bias": 0.5, "atm_iv": 0.25, "total_opt_volume": 1000,
}
options_flow.describe = lambda sym, m, base: f"{sym}: C/P ratio 1.1"
short_volume.collect_short_volume = lambda syms: {
    s: {"short_volume_ratio": 0.42} for s in syms
}
short_volume.describe = lambda sym, m, base: f"{sym}: short volume ratio 42%"
event_risk.collect_event_risk = lambda: {
    "event_risk": "low", "suggested_scale": 1.0, "events": [],
}
event_risk.describe = lambda risk: ""

db = SessionLocal()
db.query(AltSignal).delete()
db.commit()

res = A.collect_all(db, WATCHLIST, universe=UNIVERSE)

stored = {r.symbol for r in db.query(AltSignal).filter(AltSignal.symbol != "_MACRO").all()}
check("every universe symbol was STORED", stored == set(UNIVERSE),
      str(sorted(stored)))
check("collected count reported", res["collected"] == len(UNIVERSE) * 2,
      str(res["collected"]))

text = res["text"]
for sym in WATCHLIST:
    check(f"{sym} (watchlist) appears in the briefing", sym in text)
for sym in ["AMD", "COST", "XLE", "BA"]:
    check(f"{sym} (collect-only) is NOT in the briefing", sym not in text,
          text[:200])

check("per_symbol covers watchlist only", set(res["per_symbol"]) == set(WATCHLIST),
      str(sorted(res["per_symbol"])))

# The whole point of the narrow briefing is prompt size.
check("briefing stays small despite a wide universe", len(text) < 600,
      f"{len(text)} chars")


# ── 3. Statistical power: why this was done at all ──────────────────────────
print("\n=== 3. Wider universe reduces cross-sectional IC noise ===")
import random
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "eval_signals",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "eval_signals.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def null_se(n_names, n_dates, seed):
    """Standard error of the mean IC under the null, by simulation."""
    random.seed(seed)
    by_date = {}
    for d in range(n_dates):
        by_date[d] = [(random.random(), random.gauss(0, 1))
                      for _ in range(n_names)]
    return E.cross_sectional_ic(by_date)["se"]


se10 = null_se(10, 37, 11)
se40 = null_se(40, 37, 11)
check("40 names gives a smaller standard error than 10", se40 < se10,
      f"se10={se10:.4f} se40={se40:.4f}")
check("the reduction is roughly the expected ~2x", 1.4 < se10 / se40 < 2.8,
      f"ratio {se10 / se40:.2f}")
print(f"         (se over 37 dates: {se10:.4f} at 10 names -> "
      f"{se40:.4f} at 40 names; smallest detectable |IC| at t=2 falls from "
      f"{2*se10:.3f} to {2*se40:.3f})")

db.close()
print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
