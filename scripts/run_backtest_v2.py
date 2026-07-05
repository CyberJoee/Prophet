"""
Run Backtest v2 — the honest one.

Usage (on Railway or locally with Alpaca creds in .env):

  # Real data, last 12 months, default watchlist:
  python scripts/run_backtest_v2.py

  # Specific symbols and window:
  python scripts/run_backtest_v2.py --symbols NVDA,AAPL,MSFT --months 18

  # Compare with/without the regime gate:
  python scripts/run_backtest_v2.py --regime both

  # Offline synthetic smoke test (no credentials needed):
  python scripts/run_backtest_v2.py --synthetic
"""
import os
import sys
import json
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from backtesting.engine_v2 import (
    BacktestEngineV2, BacktestConfig, load_alpaca_5min, generate_synthetic_5min
)

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "GOOGL", "AMZN"]


def print_report(rep: dict, label: str):
    print(f"\n{'='*64}\nBACKTEST REPORT — {label}\n{'='*64}")
    print(f"Trades: {rep['total_trades']}  |  Final equity: "
          f"${rep['final_equity']:,.2f}  ({rep['total_return_pct']:+.2f}%)")
    print(f"Max drawdown: {rep['max_drawdown_pct']}%  |  "
          f"Total costs paid: ${rep['total_costs']:,.2f}")
    print(f"\n{'setup':<14}{'n':>5}{'win%':>7}{'avgR':>7}{'exp$':>9}"
          f"{'PF':>7}{'stops':>7}{'tgts':>6}{'eod':>6}{'pnl':>11}")
    for setup, s in sorted(rep["by_setup"].items()):
        print(f"{setup:<14}{s['trades']:>5}{s['win_rate']*100:>6.1f}%"
              f"{s['avg_r']:>7.2f}{s['expectancy_$']:>9.2f}"
              f"{s['profit_factor']:>7.2f}{s['stop_hits']:>7}"
              f"{s['target_hits']:>6}{s['eod_closes']:>6}"
              f"{s['total_pnl']:>11,.2f}")
    print("\nMonthly PnL:")
    for m, p in rep["by_month"].items():
        bar = "#" * min(40, int(abs(p) / 50))
        print(f"  {m}  {p:>10,.2f}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--spread-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--regime", choices=["on", "off", "both"], default="both")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic data (offline smoke test)")
    ap.add_argument("--json", action="store_true", help="dump raw JSON too")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print("Loading data...")
    if args.synthetic:
        bars = generate_synthetic_5min(symbols, sessions=120)
        print(f"  synthetic: {len(bars)} symbols x ~120 sessions")
    else:
        end = datetime.utcnow()
        start = end - timedelta(days=args.months * 31)
        bars = load_alpaca_5min(symbols, start, end)

    runs = {"on": [True], "off": [False], "both": [False, True]}[args.regime]
    reports = {}
    for use_regime in runs:
        cfg = BacktestConfig(
            spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
            use_regime_gate=use_regime,
        )
        engine = BacktestEngineV2(bars, cfg)
        rep = engine.run()
        label = f"regime gate {'ON' if use_regime else 'OFF'}"
        reports[label] = rep
        print_report(rep, label)

    if len(reports) == 2:
        (l1, r1), (l2, r2) = reports.items()
        print(f"\n{'='*64}\nREGIME GATE IMPACT\n{'='*64}")
        print(f"{'':<22}{l1:>18}{l2:>18}")
        print(f"{'Return %':<22}{r1['total_return_pct']:>17.2f}%"
              f"{r2['total_return_pct']:>17.2f}%")
        print(f"{'Max drawdown %':<22}{r1['max_drawdown_pct']:>17.2f}%"
              f"{r2['max_drawdown_pct']:>17.2f}%")
        print(f"{'Trades':<22}{r1['total_trades']:>18}{r2['total_trades']:>18}")

    if args.json:
        print(json.dumps(reports, indent=2, default=str))

    print("\nInterpretation guide:")
    print("  expectancy_$ > 0 after costs  → the setup earns; keep it")
    print("  expectancy_$ <= 0             → the LLM should not be offered it")
    print("  PF < 1.2 or win% collapse in the monthly view → regime-fragile")
    print("  This measures the mechanical setups the LLM chooses among.")
    print("  The LLM's selection skill on top can only be judged live.")


if __name__ == "__main__":
    main()
