"""
Tests for the geometry sweep.

The important one is the NO-EDGE CONTROL: on structureless random-walk data,
no geometry may survive. A sweep that finds edge in noise is worse than no
sweep, because it would justify changing live risk geometry on nothing.

Needs only pydantic (no DB, no network, no LLM). Slow-ish — it runs real
backtests. Run directly:

    python tests/test_geometry_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.engine_v2 import (
    generate_synthetic_5min, BacktestEngineV2, BacktestConfig,
)
from backtesting.geometry_sweep import run_sweep, format_report, _spearman

failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# ── 1. Default config reproduces production geometry ────────────────────────
print("\n=== 1. Default config is production ===")
from execution.sizing import STOP_ATR_MULT, TARGET_ATR_MULT
cfg = BacktestConfig()
check("stop_mult defaults to None (= live constant)", cfg.stop_mult is None)
check("target_mult defaults to None", cfg.target_mult is None)
check("time_exit defaults to None (hold to EOD)", cfg.time_exit_minutes is None)
check("label reports the live constants",
      cfg.label() == f"stop{STOP_ATR_MULT:g}/target{TARGET_ATR_MULT:g}/exit-eod",
      cfg.label())

# ── 2. Geometry actually changes the trades ─────────────────────────────────
print("\n=== 2. Geometry parameters reach the trades ===")
bars = generate_synthetic_5min(["AAPL", "MSFT", "SPY"], sessions=100)

tight = BacktestEngineV2(bars, BacktestConfig(stop_mult=0.25, target_mult=1.0))
wide  = BacktestEngineV2(bars, BacktestConfig(stop_mult=2.0,  target_mult=1.0))
rt, rw = tight.run(), wide.run()
st = rt["overall"]["exits"].get("stop_hit", 0) / max(rt["overall"]["trades"], 1)
sw = rw["overall"]["exits"].get("stop_hit", 0) / max(rw["overall"]["trades"], 1)
check("tighter stops produce more stop-outs", st > sw, f"tight={st:.2f} wide={sw:.2f}")

# ── 3. Time exit is honoured and reported ───────────────────────────────────
print("\n=== 3. Time exit ===")
timed = BacktestEngineV2(bars, BacktestConfig(time_exit_minutes=30)).run()
check("time_exit appears as an exit reason",
      timed["overall"]["exits"].get("time_exit", 0) > 0,
      str(timed["overall"]["exits"]))
no_time = BacktestEngineV2(bars, BacktestConfig()).run()
check("no time exits when unset",
      no_time["overall"]["exits"].get("time_exit", 0) == 0,
      str(no_time["overall"]["exits"]))
check("a 30m exit leaves fewer EOD closes than holding",
      timed["overall"]["exits"].get("eod_close", 0)
      < no_time["overall"]["exits"].get("eod_close", 0))

# ── 4. Spearman helper ──────────────────────────────────────────────────────
print("\n=== 4. Rank correlation helper ===")
check("perfect agreement -> +1", _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0)
check("perfect inversion -> -1", _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0)
check("too few points -> None", _spearman([1, 2], [1, 2]) is None)

# ── 5. Split is chronological, never random ─────────────────────────────────
print("\n=== 5. Fit/validate split ===")
from backtesting.geometry_sweep import _split_sessions
fb, vb, fd, vd = _split_sessions(bars, fit_frac=0.6)
check("fit window precedes validate window", max(fd) < min(vd),
      f"{max(fd)} vs {min(vd)}")
check("no session appears in both", not (set(fd) & set(vd)))
check("split is roughly the requested fraction",
      0.5 < len(fd) / (len(fd) + len(vd)) < 0.7,
      f"{len(fd)}/{len(fd)+len(vd)}")
try:
    _split_sessions(generate_synthetic_5min(["AAPL"], sessions=20))
    check("too-short history is refused", False, "no error raised")
except ValueError:
    check("too-short history is refused", True)

# ── 6. THE CONTROL: no edge may be found in noise ───────────────────────────
print("\n=== 6. NO-EDGE CONTROL (the one that matters) ===")
control_bars = generate_synthetic_5min(["AAPL", "MSFT", "NVDA", "SPY"], sessions=220)
res = run_sweep(control_bars, stops=(0.5, 1.0, 1.5), targets=(0.5, 1.0, 1.5),
                time_exits=(None, 60), min_trades=25)
check("sweep produced eligible cells", len(res["eligible"]) > 0,
      str(len(res["eligible"])))
check("NO geometry survives on structureless data", res["survived"] is False,
      res["verdict"])
check("verdict refuses to recommend a promotion",
      "Do NOT promote" in format_report(res) or not res["survived"])

# ── 7. Report is honest about the rho caveat ────────────────────────────────
print("\n=== 7. Report surfaces the cost-drag caveat ===")
rep = format_report(res)
if res["rank_correlation"] is not None and res["rank_correlation"] >= 0.2:
    check("positive rho carries the cost-drag warning",
          "NOT independent evidence" in rep, rep[-400:])
else:
    check("low rho is called out as noise", "noise" in rep.lower())
check("both windows are shown", "fit:" in rep and "validate:" in rep)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
