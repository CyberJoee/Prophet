"""
Run the stop/target/time-exit geometry sweep.

Motivated by the standing finding that only ~7% of trades ever reach target
while ~65% drift to the EOD close — a geometry mismatch rather than bad luck.
This asks whether any other geometry does better, out of sample.

    # the honest run, on real Alpaca 5-min bars
    python scripts/run_geometry_sweep.py --days 365

    # the control: structureless data, where the answer must be "no edge"
    python scripts/run_geometry_sweep.py --synthetic

    # a wider grid (slower; also a worse multiple-comparisons problem)
    python scripts/run_geometry_sweep.py --days 365 --stops 0.5,0.75,1,1.5,2 \\
        --targets 0.5,0.75,1,1.5,2 --time-exits none,30,60,120

READ THE VERDICT, NOT THE TOP ROW. The table is sorted by fit-window
expectancy, and the top row of any grid search looks good by construction.
The verdict line states whether it survived out of sample. If `survived` is
False, nothing here justifies changing execution/sizing.py.

Run the --synthetic control at least once after any change to the engine or
the sweep. If it ever reports survived=True on random-walk data, the tool is
broken and every other result it has produced is void.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SYMBOLS = "NVDA,AAPL,TSLA,MSFT,AMZN,META,GOOGL,JPM,QQQ"


def _parse_floats(text: str) -> tuple:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _parse_time_exits(text: str) -> tuple:
    out = []
    for x in text.split(","):
        x = x.strip().lower()
        if not x:
            continue
        out.append(None if x in ("none", "eod") else int(x))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run the no-edge control instead of real data")
    ap.add_argument("--days", type=int, default=365,
                    help="calendar days of history to load (real data)")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--sessions", type=int, default=220,
                    help="synthetic sessions to generate")
    ap.add_argument("--stops", default="0.5,0.75,1,1.5")
    ap.add_argument("--targets", default="0.5,0.75,1,1.5")
    ap.add_argument("--time-exits", dest="time_exits", default="none,60,120")
    ap.add_argument("--fit-frac", type=float, default=0.6)
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()

    from backtesting.engine_v2 import generate_synthetic_5min, load_alpaca_5min
    from backtesting.geometry_sweep import run_sweep, format_report

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.synthetic:
        print(f"CONTROL RUN — synthetic structureless data, "
              f"{args.sessions} sessions.")
        print("The correct result is survived=False. Anything else means the "
              "sweep is broken.\n")
        bars = generate_synthetic_5min(symbols[:4], sessions=args.sessions)
    else:
        if not os.getenv("ALPACA_API_KEY"):
            print("ALPACA_API_KEY not set — cannot load real bars. "
                  "Use --synthetic for the control run.")
            return 1
        end = datetime.utcnow()
        start = end - timedelta(days=args.days)
        print(f"Loading 5-min bars for {len(symbols)} symbols, "
              f"{start.date()} to {end.date()}...")
        bars = load_alpaca_5min(symbols, start, end)
        got = {s: len(b) for s, b in bars.items()}
        print(f"  bars loaded: {got}\n")
        if not any(got.values()):
            print("No bars returned — check credentials and the date range.")
            return 1

    try:
        res = run_sweep(
            bars,
            stops=_parse_floats(args.stops),
            targets=_parse_floats(args.targets),
            time_exits=_parse_time_exits(args.time_exits),
            fit_frac=args.fit_frac,
            min_trades=args.min_trades,
        )
    except ValueError as e:
        print(f"Sweep could not run: {e}")
        return 1

    print(format_report(res))

    if args.synthetic:
        print()
        if res["survived"]:
            print("!!! CONTROL FAILED — the sweep found an 'edge' in random "
                  "data. Do not trust any sweep result until this is fixed.")
            return 2
        print("Control passed: no geometry survived on structureless data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
