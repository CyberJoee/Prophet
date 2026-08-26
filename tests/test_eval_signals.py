"""
Tests for scripts/eval_signals.py statistics.

The script had never been run, and its job is to decide whether a signal is
allowed to influence real position sizing. That makes a wrong answer here
expensive in a way a wrong answer in a dashboard is not — so the maths is
tested against cases with a known correct result, including the case that
matters most: pure noise must come back as noise.

No DB, no network. Run directly:

    python tests/test_eval_signals.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_signals",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "eval_signals.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# ── 1. Rank with tie averaging ──────────────────────────────────────────────
print("\n=== 1. Ranking averages ties ===")
check("distinct values rank in order", E._rank([10, 20, 30]) == [0.0, 1.0, 2.0],
      str(E._rank([10, 20, 30])))
check("a tied pair shares the average rank",
      E._rank([5, 5, 9]) == [0.5, 0.5, 2.0], str(E._rank([5, 5, 9])))
check("all-tied values all share one rank",
      E._rank([7, 7, 7]) == [1.0, 1.0, 1.0], str(E._rank([7, 7, 7])))
# unusual_contracts is a small integer and ties constantly; untied ranking
# would invent an ordering that is not in the data.
check("integer-heavy input does not fabricate an ordering",
      E._rank([2, 2, 2, 5]) == [1.0, 1.0, 1.0, 3.0], str(E._rank([2, 2, 2, 5])))

# ── 2. Spearman ─────────────────────────────────────────────────────────────
print("\n=== 2. Spearman ===")
check("perfect monotone -> +1",
      abs(E.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
check("perfect inverse -> -1",
      abs(E.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9)
check("monotone but non-linear still +1 (rank based)",
      abs(E.spearman([1, 2, 3, 4], [1, 4, 9, 1000]) - 1.0) < 1e-9)
check("too few points -> None", E.spearman([1, 2], [3, 4]) is None)
check("no spread in x -> None (not 0.0)",
      E.spearman([5, 5, 5, 5], [1, 2, 3, 4]) is None)
check("no spread in y -> None", E.spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None)

# ── 3. Cross-sectional IC ───────────────────────────────────────────────────
print("\n=== 3. Cross-sectional IC ===")
# Every date perfectly ranked: IC 1.0, zero variance across dates.
perfect = {d: [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)] for d in range(10)}
r = E.cross_sectional_ic(perfect)
check("perfect signal -> IC 1.0", abs(r["ic"] - 1.0) < 1e-9, str(r["ic"]))
check("counts all dates", r["n_dates"] == 10, str(r["n_dates"]))
check("zero variance -> se 0", r["se"] == 0, str(r["se"]))

thin = {d: [(1, 1.0), (2, 2.0)] for d in range(10)}
r = E.cross_sectional_ic(thin, min_names=4)
check("dates with too few names are skipped", r["n_dates"] == 0, str(r["n_dates"]))
check("no dates -> IC None", r["ic"] is None)

# ── 4. THE ONE THAT MATTERS: noise must read as noise ───────────────────────
print("\n=== 4. Pure noise does not clear |t| = 2 ===")
random.seed(20260826)
false_positives = 0
TRIALS = 40
for _ in range(TRIALS):
    noise = {}
    for d in range(30):                     # 30 dates, 9 symbols — realistic
        noise[d] = [(random.random(), random.gauss(0, 1)) for _ in range(9)]
    res = E.cross_sectional_ic(noise)
    if res["t"] is not None and abs(res["t"]) >= 2.0:
        false_positives += 1
rate = false_positives / TRIALS
check("false-positive rate is near the nominal 5%", rate <= 0.20,
      f"{false_positives}/{TRIALS} = {rate:.0%}")
print(f"         (observed {false_positives}/{TRIALS} = {rate:.0%}; "
      f"~5% expected by construction)")

# ── 5. A real signal IS detected ────────────────────────────────────────────
print("\n=== 5. A genuine signal clears the bar ===")
random.seed(1)
real = {}
for d in range(30):
    day = []
    for _ in range(9):
        x = random.random()
        y = x * 2.0 + random.gauss(0, 0.5)   # signal plus noise
        day.append((x, y))
    real[d] = day
res = E.cross_sectional_ic(real)
check("positive IC recovered", res["ic"] > 0.3, str(res["ic"]))
check("clears |t| = 2", abs(res["t"]) >= 2.0, str(res["t"]))

# ── 6. Pooled vs cross-sectional diverge when beta dominates ────────────────
print("\n=== 6. Pooled IC is inflated by market-wide moves ===")
# Signal has NO cross-sectional information, but on days the signal happens
# to read high across the board, the whole market is up. Pooled correlation
# sees a relationship; cross-sectional correctly sees none.
random.seed(7)
by_date, flat = {}, []
for d in range(30):
    market = random.gauss(0, 1)
    level = market                      # signal level tracks the market that day
    day = []
    for _ in range(9):
        x = level + random.gauss(0, 0.01)   # ~identical across names that day
        y = market + random.gauss(0, 0.05)
        day.append((x, y))
        flat.append((x, y))
    by_date[d] = day

pooled = E.spearman([p[0] for p in flat], [p[1] for p in flat])
cs = E.cross_sectional_ic(by_date)
check("pooled IC looks strong", pooled > 0.8, str(pooled))
check("cross-sectional IC is near zero",
      cs["ic"] is None or abs(cs["ic"]) < 0.4, str(cs["ic"]))
print(f"         (pooled {pooled:+.2f} vs cross-sectional "
      f"{cs['ic']:+.2f} — same data, different question)")

# ── 7. Intraday window definition ───────────────────────────────────────────
print("\n=== 7. Horizon constants match the live system ===")
from datetime import time as dtime
check("entry at 09:45 ET", E.DECISION_TIME == dtime(9, 45), str(E.DECISION_TIME))
check("exit at the session close", E.SESSION_END == dtime(16, 0), str(E.SESSION_END))
check("all five collected signals are evaluated", len(E.SIGNALS) == 5,
      str(len(E.SIGNALS)))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
