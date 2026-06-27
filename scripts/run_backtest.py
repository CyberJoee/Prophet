"""
Run a backtest of Prophet's strategy against historical data.
Usage: python scripts/run_backtest.py [--days 90] [--symbols NVDA,AAPL,MSFT]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from backtesting.engine import Backtester
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run Prophet backtest")
    parser.add_argument("--days",    type=int, default=90,
                        help="Number of days to backtest (default: 90)")
    parser.add_argument("--symbols", type=str,
                        default="NVDA,AAPL,MSFT,GOOGL,AMZN,SPY,QQQ,META,JPM,TSLA",
                        help="Comma-separated symbols")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Starting capital (default: 100000)")
    parser.add_argument("--mock",    action="store_true",
                        help="Use mock data (no internet needed)")
    args = parser.parse_args()

    symbols    = [s.strip().upper() for s in args.symbols.split(",")]
    end_date   = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"Starting backtest: {start_date} → {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Data:    {'mock' if args.mock else 'yfinance (live)'}")
    print()

    backtester = Backtester(use_mock=args.mock)

    try:
        result = backtester.run(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            starting_capital=args.capital,
        )
        backtester.print_report(result)

        # Compare to buy-and-hold SPY
        print("\nBuy & Hold comparison (SPY):")
        spy_result = backtester.run(
            symbols=["SPY"],
            start_date=start_date,
            end_date=end_date,
            starting_capital=args.capital,
        )
        # Simple buy-and-hold: just the price change
        if spy_result.trades:
            first_entry = spy_result.trades[0].entry_price
            last_exit   = spy_result.trades[-1].exit_price
            bh_return   = (last_exit - first_entry) / first_entry * 100
            print(f"  SPY buy-and-hold return: {bh_return:+.2f}%")
            print(f"  Prophet return:          {result.total_pnl_pct:+.2f}%")
            diff = result.total_pnl_pct - bh_return
            print(f"  Alpha:                   {diff:+.2f}%")

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
