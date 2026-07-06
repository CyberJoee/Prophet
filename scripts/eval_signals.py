"""
Evaluate collected alt-signals against forward returns.

Run after 3-4+ weeks of daily collection:

    python scripts/eval_signals.py            # 1-day forward returns
    python scripts/eval_signals.py --horizon 3

For each numeric signal, we ask one question: did high readings lead to
different forward returns than low readings? Reported per signal:

  n            usable (signal, forward-return) pairs
  IC           rank correlation between signal and forward return
  hi_ret       avg forward return, top tercile of signal readings
  lo_ret       avg forward return, bottom tercile
  spread       hi - lo (annualized-ish edge if consistent)

Interpretation:
  |IC| > 0.05 with n > 100 and stable sign  → promising, keep collecting
  |IC| > 0.10 with n > 200                  → candidate for sizing influence
  anything else                             → context at best; not alpha

Do NOT promote a signal into sizing off a small n. Three weeks of 8 symbols
is ~120 observations — treat early reads as direction, not proof.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

SIGNALS = {
    # source, metric key, direction hint for readability
    ("options_flow", "cp_volume_ratio"),
    ("options_flow", "unusual_contracts"),
    ("options_flow", "unusual_call_bias"),
    ("options_flow", "atm_iv"),
    ("short_volume", "short_volume_ratio"),
}


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def load_forward_returns(symbols, start, horizon):
    """{(symbol, date): forward pct return over `horizon` trading days}."""
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
            c1 = closes[i + horizon][1]
            out[(sym, d)] = (c1 / c0 - 1) * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1,
                    help="forward return horizon in trading days")
    args = ap.parse_args()

    from db.connection import SessionLocal
    from db.models import AltSignal
    db = SessionLocal()

    rows = (db.query(AltSignal)
            .filter(AltSignal.symbol != "_MACRO")
            .order_by(AltSignal.signal_date)
            .all())
    if not rows:
        print("No alt_signals collected yet. Let the morning pipeline run "
              "for a few weeks, then come back.")
        return

    symbols = {r.symbol for r in rows}
    start = min(r.signal_date for r in rows)
    print(f"{len(rows)} snapshots | {len(symbols)} symbols | "
          f"since {start.date()} | horizon {args.horizon}d\n")

    fwd = load_forward_returns(symbols, start, args.horizon)

    by_signal = defaultdict(list)   # (source, metric) -> [(value, fwd_ret)]
    for r in rows:
        key_date = r.signal_date.date()
        ret = fwd.get((r.symbol, key_date))
        if ret is None:
            continue
        for source, metric in SIGNALS:
            if r.source == source and r.metrics.get(metric) is not None:
                by_signal[(source, metric)].append(
                    (float(r.metrics[metric]), ret))

    print(f"{'signal':<38}{'n':>6}{'IC':>8}{'hi_ret%':>9}{'lo_ret%':>9}{'spread':>8}")
    print("-" * 78)
    for (source, metric), pairs in sorted(by_signal.items()):
        if len(pairs) < 10:
            print(f"{source+'.'+metric:<38}{len(pairs):>6}   (need more data)")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ic = spearman(xs, ys)
        ranked = sorted(pairs, key=lambda p: p[0])
        third = len(ranked) // 3 or 1
        lo = sum(p[1] for p in ranked[:third]) / third
        hi = sum(p[1] for p in ranked[-third:]) / third
        print(f"{source+'.'+metric:<38}{len(pairs):>6}{ic:>8.3f}"
              f"{hi:>9.3f}{lo:>9.3f}{hi-lo:>8.3f}")

    print("\nRule of thumb: |IC| > 0.05 and n > 100 with a stable sign across")
    print("reruns = worth keeping. Promote to sizing only past |IC| 0.10, n 200.")
    db.close()


if __name__ == "__main__":
    main()
