"""
Evaluate collected alt-signals against forward returns.

    python scripts/eval_signals.py                 # intraday, the window we trade
    python scripts/eval_signals.py --horizon 1     # close-to-close, 1 day
    python scripts/eval_signals.py --horizon 3

WHAT CHANGED AND WHY (2026-08-26)
---------------------------------
This script had never been run. Auditing it before its first use turned up
three problems that would each have made the output uninterpretable. All are
the same failure the geometry sweep was built to avoid: producing a number
that looks like evidence and is not.

1. IT MEASURED A HORIZON THE BOT NEVER HOLDS.
   Signals are collected at ~09:45 ET. The old code scored them against
   close(d) -> close(d+1), which begins AFTER the bot has already entered and
   exited. Prophet's actual exposure is 09:45 -> session close, same day. A
   signal could predict overnight drift perfectly and be worthless here, or
   vice versa. Intraday is now the default; the daily horizons remain
   available for curiosity, clearly labelled.

2. POOLED IC CONFLATES TWO DIFFERENT QUESTIONS.
   Pooling every (symbol, date) pair mixes "which stock will outperform
   today" with "will the market be up today". On nine megacaps the second
   dominates: they move together, so one market-wide up day contributes nine
   correlated observations. Effective sample size is closer to the number of
   DATES than the number of pairs, and a pooled IC mostly measures beta.

   The headline number is now the CROSS-SECTIONAL IC: rank the symbols
   against each other within each date, correlate with same-date returns,
   then average across dates. That asks the question sizing would actually
   act on — given today's readings, which name is relatively better? Pooled
   IC is still printed, labelled, for comparison.

3. NO SIGNIFICANCE, AND A THRESHOLD THAT GREENLIT NOISE.
   The old guidance was "|IC| > 0.05 with n > 100 is promising". At n = 100
   the standard error of a correlation is about 1/sqrt(n-1) ~ 0.10, so an IC
   of 0.05 is half a standard error from zero — indistinguishable from
   nothing. Every signal now reports the standard error of its mean IC and
   the resulting t-statistic, and the verdict keys off |t|, not |IC|.

Also fixed: the rank function did not average ties. `unusual_contracts` is a
small integer with many repeats, and breaking those ties by array order
invents an ordering that is not in the data.

INTERPRETING THE OUTPUT
-----------------------
  IC        mean cross-sectional rank correlation with forward return
  se        standard error of that mean across dates
  t         IC / se  — this is the column that matters
  n_dates   how many dates contributed (the real sample size)
  spread    top-tercile minus bottom-tercile forward return, pooled

  |t| >= 2.0   worth taking seriously; keep collecting and re-check
  |t| <  2.0   not distinguishable from noise, whatever the IC looks like

Even |t| >= 2 on one run is not a promotion. Five signals are tested here, so
one crossing t=2 by chance is unremarkable. Re-run after more collection and
require the SIGN to be stable before letting anything touch sizing.py.
"""
import os
import sys
import math
import argparse
import statistics
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

DECISION_TIME = dtime(9, 45)     # when signals are collected and trades entered
SESSION_END   = dtime(16, 0)

SIGNALS = (
    ("options_flow", "cp_volume_ratio"),
    ("options_flow", "unusual_contracts"),
    ("options_flow", "unusual_call_bias"),
    ("options_flow", "atm_iv"),
    ("short_volume", "short_volume_ratio"),
)


# ─── Statistics ───────────────────────────────────────────────────────────────

def _rank(vals):
    """Ranks with ties AVERAGED. Integer-valued signals tie constantly."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Rank correlation, or None when undefined (too few points, or no spread)."""
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if not vx or not vy:
        return None          # all readings identical — no information
    return cov / (vx * vy)


def cross_sectional_ic(pairs_by_date, min_names: int = 4) -> dict:
    """
    Mean cross-sectional IC across dates, with its standard error.

    A date contributes one IC, computed by ranking that day's symbols against
    each other. Dates with fewer than min_names usable symbols are skipped —
    a rank correlation over three names is not informative.
    """
    ics = []
    for _, pairs in sorted(pairs_by_date.items()):
        if len(pairs) < min_names:
            continue
        ic = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            ics.append(ic)

    if not ics:
        return {"ic": None, "se": None, "t": None, "n_dates": 0}
    mean = statistics.mean(ics)
    if len(ics) < 2:
        return {"ic": mean, "se": None, "t": None, "n_dates": 1}
    se = statistics.stdev(ics) / math.sqrt(len(ics))
    return {"ic": mean, "se": se,
            "t": (mean / se) if se else None, "n_dates": len(ics)}


# ─── Forward returns ──────────────────────────────────────────────────────────

def load_intraday_returns(symbols, start, end) -> dict:
    """
    {(symbol, date): pct return from the 09:45 bar close to the session close}.

    This is the window Prophet is actually exposed to: it enters at 09:45 and
    is flat by the close. Anything measured outside it is a different
    strategy's question.
    """
    from backtesting.engine_v2 import load_alpaca_5min
    bars = load_alpaca_5min(list(symbols), start, end)

    out = {}
    for sym, rows in bars.items():
        byday = defaultdict(list)
        for b in rows:
            byday[b["timestamp"].date()].append(b)
        for day, bs in byday.items():
            entry = [b for b in bs if b["timestamp"].time() <= DECISION_TIME]
            close = [b for b in bs if b["timestamp"].time() <= SESSION_END]
            if not entry or not close:
                continue
            c0, c1 = entry[-1]["close"], close[-1]["close"]
            if c0:
                out[(sym, day)] = (c1 / c0 - 1) * 100
    return out


def load_daily_forward_returns(symbols, start, horizon) -> dict:
    """{(symbol, date): pct return from close(d) to close(d+horizon)}."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    req = StockBarsRequest(symbol_or_symbols=list(symbols),
                           timeframe=TimeFrame.Day,
                           start=start - timedelta(days=5),
                           end=datetime.utcnow(), feed=DataFeed.IEX)
    bars = client.get_stock_bars(req)

    out = {}
    for sym in symbols:
        rows = sorted(bars.data.get(sym, []), key=lambda b: b.timestamp)
        closes = [(b.timestamp.date(), float(b.close)) for b in rows]
        for i in range(len(closes) - horizon):
            d, c0 = closes[i]
            if c0:
                out[(sym, d)] = (closes[i + horizon][1] / c0 - 1) * 100
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="intraday",
                    help="'intraday' (09:45 -> close, the window we trade) "
                         "or an integer number of trading days close-to-close")
    ap.add_argument("--min-names", type=int, default=4,
                    help="symbols required on a date for it to contribute an IC")
    args = ap.parse_args()

    from db.connection import SessionLocal
    from db.models import AltSignal
    db = SessionLocal()
    try:
        rows = (db.query(AltSignal)
                .filter(AltSignal.symbol != "_MACRO")
                .order_by(AltSignal.signal_date)
                .all())
        if not rows:
            print("No alt_signals collected yet. Let the morning pipeline run "
                  "for a few weeks, then come back.")
            return 0

        symbols = {r.symbol for r in rows}
        start = min(r.signal_date for r in rows)
        end = max(r.signal_date for r in rows) + timedelta(days=2)

        intraday = str(args.horizon).lower() == "intraday"
        label = ("intraday 09:45 -> close (the window Prophet trades)"
                 if intraday else
                 f"{int(args.horizon)}-day close-to-close "
                 f"(NOT the window Prophet trades)")

        print(f"{len(rows):,} snapshots | {len(symbols)} symbols | "
              f"{start.date()} to {max(r.signal_date for r in rows).date()}")
        print(f"horizon: {label}\n")

        if intraday:
            fwd = load_intraday_returns(symbols, start, end)
        else:
            fwd = load_daily_forward_returns(symbols, start, int(args.horizon))
        print()

        if not fwd:
            print("No forward returns could be loaded — check Alpaca "
                  "credentials and that the date range has market data.")
            return 1

        # (source, metric) -> date -> [(signal_value, forward_return)]
        by_signal = defaultdict(lambda: defaultdict(list))
        pooled = defaultdict(list)
        wanted = {(s, m) for s, m in SIGNALS}

        for r in rows:
            day = r.signal_date.date()
            ret = fwd.get((r.symbol, day))
            if ret is None or not isinstance(r.metrics, dict):
                continue
            for metric in (m for s, m in wanted if s == r.source):
                val = r.metrics.get(metric)
                if val is None:
                    continue
                by_signal[(r.source, metric)][day].append((float(val), ret))
                pooled[(r.source, metric)].append((float(val), ret))

        missing = wanted - set(by_signal)
        hdr = (f"{'signal':<34}{'pairs':>7}{'dates':>7}{'IC':>8}{'se':>7}"
               f"{'t':>7}{'pooled':>8}{'spread%':>9}  verdict")
        print(hdr)
        print("-" * len(hdr))

        any_signal = False
        for key in sorted(by_signal):
            source, metric = key
            per_date = by_signal[key]
            flat = pooled[key]
            cs = cross_sectional_ic(per_date, min_names=args.min_names)

            pooled_ic = spearman([p[0] for p in flat], [p[1] for p in flat])
            ranked = sorted(flat, key=lambda p: p[0])
            third = len(ranked) // 3
            if third:
                spread = (sum(p[1] for p in ranked[-third:]) / third
                          - sum(p[1] for p in ranked[:third]) / third)
            else:
                spread = None

            t = cs["t"]
            if cs["n_dates"] < 5:
                verdict = "too few dates"
            elif t is None:
                verdict = "undefined"
            elif abs(t) >= 2.0:
                verdict = "WORTH WATCHING"
                any_signal = True
            else:
                verdict = "noise"

            def f(v, w, p=3):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"

            print(f"{source + '.' + metric:<34}{len(flat):>7}"
                  f"{cs['n_dates']:>7}{f(cs['ic'], 8)}{f(cs['se'], 7)}"
                  f"{f(t, 7, 2)}{f(pooled_ic, 8)}{f(spread, 9)}  {verdict}")

        for source, metric in sorted(missing):
            print(f"{source + '.' + metric:<34}{0:>7}{0:>7}"
                  f"{'-':>8}{'-':>7}{'-':>7}{'-':>8}{'-':>9}  NO DATA")

        print()
        print("t = IC / se. |t| >= 2 is the bar; IC alone means nothing without it.")
        print("Pooled IC mixes stock-picking with market direction and will look")
        print("larger than it deserves — the cross-sectional IC is the honest one.")
        if missing:
            print(f"\n{len(missing)} signal(s) produced NO usable rows. That is a "
                  "collection problem, not a result — check the aggregator.")
        if not any_signal:
            print("\nNothing here clears |t| = 2. No signal should influence "
                  "sizing on this evidence.")
        else:
            print("\nSomething cleared |t| = 2. Five signals are tested, so one "
                  "crossing by chance is expected — re-run after more collection "
                  "and require the sign to hold before promoting anything.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
