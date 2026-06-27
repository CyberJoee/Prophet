"""
Backtesting Engine
Replays historical price data through Prophet's research and strategy
logic to measure how the system would have performed.

Uses yfinance for historical data (no Alpaca subscription needed).
Results are stored in backtest_runs and backtest_trades tables.

Run via: python scripts/run_backtest.py
"""
import os
import sys
import json
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol:        str
    setup_type:    str
    side:          str
    entry_date:    date
    exit_date:     date
    entry_price:   float
    exit_price:    float
    quantity:      int
    planned_stop:  float
    planned_target: float
    exit_reason:   str      # target_hit / stop_hit / eod_close
    pnl:           float
    pnl_pct:       float


@dataclass
class BacktestResult:
    run_id:        str
    symbols:       list
    start_date:    date
    end_date:      date
    starting_capital: float
    final_equity:  float
    total_pnl:     float
    total_pnl_pct: float
    total_trades:  int
    winning_trades: int
    losing_trades: int
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    expectancy:    float
    profit_factor: float
    max_drawdown:  float
    max_drawdown_pct: float
    sharpe_ratio:  float
    trades:        list = field(default_factory=list)


# ─── Historical data fetcher (yfinance) ────────────────────────────────────────

def _fetch_historical(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily OHLCV for symbol between start and end using yfinance.
    Returns DataFrame with Date index.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        # Compute ATR
        df["tr"] = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"]  - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = df["tr"].rolling(14).mean()
        df["vwap"]   = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        return df
    except Exception as e:
        print(f"  [backtest] yfinance fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def _fetch_historical_mock(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Generate deterministic mock historical data when yfinance is unavailable.
    Uses the same seeded random walk as MockDataProvider.
    """
    import random, math
    from data.market_data import _MOCK_META, _default_meta

    meta  = _default_meta(symbol)
    seed  = abs(hash(symbol)) % 100_000
    rng   = random.Random(seed)
    price = meta["base"]
    daily_vol = 0.015 * meta["beta"]

    rows  = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # skip weekends
            change = rng.gauss(0, daily_vol)
            price *= (1 + change)
            h = price * (1 + abs(rng.gauss(0, 0.003)))
            l = price * (1 - abs(rng.gauss(0, 0.003)))
            o = price * (1 + rng.gauss(0, 0.001))
            vol = abs(rng.gauss(meta["base"] * 500_000, meta["base"] * 100_000))
            rows.append({"Date": current, "Open": o, "High": h, "Low": l,
                         "Close": price, "Volume": vol})
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Date")
    df["tr"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()
    df["vwap"]   = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


# ─── Signal generation (rule-based, mirrors what Groq does) ───────────────────

def _generate_signal(row: pd.Series, prev_rows: pd.DataFrame) -> Optional[dict]:
    """
    Generate a trade signal for a given day using the same logic
    Prophet's strategy agent applies.

    Returns signal dict or None if no trade.
    """
    if pd.isna(row.get("atr_14")):
        return None

    close = row["Close"]
    atr   = row["atr_14"]
    vwap  = row.get("vwap", close)

    # Compute RSI from previous rows
    rsi = None
    if len(prev_rows) >= 14:
        gains  = prev_rows["Close"].diff().clip(lower=0).tail(14)
        losses = (-prev_rows["Close"].diff()).clip(lower=0).tail(14)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss > 0:
            rs  = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

    if rsi is None:
        return None

    # Setup detection — same logic as research agent
    setup_type = None
    direction  = "long"

    # Momentum: RSI 45-65, price above VWAP, positive trend
    if 45 <= rsi <= 65 and close > vwap:
        setup_type = "momentum"

    # VWAP bounce: RSI 35-50, price just above VWAP (within 0.5%)
    elif 35 <= rsi <= 50 and 0 <= (close - vwap) / vwap <= 0.005:
        setup_type = "vwap_bounce"

    # Reversal: RSI < 35 (oversold)
    elif rsi < 35:
        setup_type = "reversal"

    # Opening range breakout: RSI > 60, high momentum
    elif rsi > 60 and close > vwap * 1.005:
        setup_type = "orb"

    if setup_type is None:
        return None

    # Sizing: 0.5x ATR stop, 1.0x ATR target (matching live system)
    stop   = round(close - 0.5 * atr, 2)
    target = round(close + 1.0 * atr, 2)
    stop_dist = close - stop

    return {
        "setup_type": setup_type,
        "direction":  direction,
        "entry_price": round(close, 2),
        "stop":        stop,
        "target":      target,
        "rsi":         round(rsi, 1),
        "atr":         round(atr, 2),
    }


def _simulate_trade(signal: dict, next_day: pd.Series, capital: float) -> Optional[BacktestTrade]:
    """
    Simulate trade execution and outcome on the next day's data.
    Entry at open, checks if stop or target hit during day, exits at close if neither.
    """
    entry_price = next_day["Open"]
    day_high    = next_day["High"]
    day_low     = next_day["Low"]
    day_close   = next_day["Close"]

    stop   = signal["stop"]
    target = signal["target"]

    # Adjust stop/target to be relative to actual entry (open price)
    atr    = signal["atr"]
    stop   = round(entry_price - 0.5 * atr, 2)
    target = round(entry_price + 1.0 * atr, 2)
    stop_dist = entry_price - stop

    # Position sizing: 2% risk rule, max 15% of capital
    if stop_dist <= 0:
        return None
    qty = max(1, int((capital * 0.02) / stop_dist))
    qty = min(qty, int((capital * 0.15) / entry_price))
    if qty <= 0:
        return None

    # Simulate intraday: assume stop/target can be hit at worst/best price
    exit_price  = day_close
    exit_reason = "eod_close"

    # Check stop hit (day low went below stop)
    if day_low <= stop:
        exit_price  = stop
        exit_reason = "stop_hit"
    # Check target hit (day high exceeded target)
    elif day_high >= target:
        exit_price  = target
        exit_reason = "target_hit"

    pnl     = round((exit_price - entry_price) * qty, 2)
    pnl_pct = round((pnl / (entry_price * qty)) * 100, 4)

    return BacktestTrade(
        symbol=signal.get("symbol", ""),
        setup_type=signal["setup_type"],
        side="buy",
        entry_date=signal.get("entry_date"),
        exit_date=signal.get("exit_date"),
        entry_price=round(entry_price, 2),
        exit_price=round(exit_price, 2),
        quantity=qty,
        planned_stop=stop,
        planned_target=target,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
    )


# ─── Main backtest runner ──────────────────────────────────────────────────────

class Backtester:

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def _fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if self.use_mock:
            return _fetch_historical_mock(symbol, start, end)
        df = _fetch_historical(symbol, start, end)
        if df.empty:
            print(f"  [backtest] yfinance unavailable for {symbol}, using mock data")
            return _fetch_historical_mock(symbol, start, end)
        return df

    def run(
        self,
        symbols: list,
        start_date: date,
        end_date: date,
        starting_capital: float = 100_000.0,
        max_positions: int = 3,
    ) -> BacktestResult:
        """
        Run full backtest over date range.
        On each day, generate signals for all symbols, take top signals,
        simulate outcomes the following day.
        """
        import uuid
        run_id   = str(uuid.uuid4())[:8]
        trades   = []
        equity   = starting_capital
        peak     = starting_capital
        max_dd   = 0.0
        daily_returns = []

        print(f"  [backtest] Running {start_date} → {end_date} | {len(symbols)} symbols | ${starting_capital:,.0f} capital")

        # Fetch all data upfront
        data = {}
        for sym in symbols:
            df = self._fetch(sym, start_date - timedelta(days=30), end_date)
            if not df.empty:
                data[sym] = df
                print(f"  [backtest] {sym}: {len(df)} bars fetched")
            else:
                print(f"  [backtest] {sym}: no data — skipping")

        if not data:
            raise ValueError("No data fetched for any symbol")

        # Get all trading days in range
        trading_days = sorted(set(
            idx.date() if hasattr(idx, 'date') else idx
            for sym_data in data.values()
            for idx in sym_data.index
            if (start_date <= (idx.date() if hasattr(idx, 'date') else idx) <= end_date)
        ))

        prev_equity = equity

        for i, day in enumerate(trading_days):
            if i + 1 >= len(trading_days):
                break  # need next day for simulation

            next_day_date = trading_days[i + 1]
            day_signals   = []

            for sym, df in data.items():
                # Get rows up to and including current day
                df_dates = [d.date() if hasattr(d, 'date') else d for d in df.index]
                df_copy  = df.copy()
                df_copy.index = df_dates

                if day not in df_copy.index:
                    continue
                if next_day_date not in df_copy.index:
                    continue

                row_idx = df_copy.index.get_loc(day)
                current_row = df_copy.iloc[row_idx]
                prev_rows   = df_copy.iloc[max(0, row_idx - 20):row_idx]

                sig = _generate_signal(current_row, prev_rows)
                if sig:
                    sig["symbol"]     = sym
                    sig["entry_date"] = next_day_date
                    sig["exit_date"]  = next_day_date
                    # Score by RSI proximity to ideal ranges
                    rsi = sig["rsi"]
                    sig["score"] = -(abs(rsi - 55))  # closer to 55 = better
                    day_signals.append(sig)

            # Take top signals by score, limit to max_positions
            day_signals.sort(key=lambda s: s["score"], reverse=True)
            day_trades = day_signals[:max_positions]

            for sig in day_trades:
                sym = sig["symbol"]
                df_dates = [d.date() if hasattr(d, 'date') else d for d in data[sym].index]
                df_copy  = data[sym].copy()
                df_copy.index = df_dates

                if next_day_date not in df_copy.index:
                    continue

                next_row = df_copy.loc[next_day_date]
                trade    = _simulate_trade(sig, next_row, equity)
                if trade:
                    equity += trade.pnl
                    trades.append(trade)

            # Track daily return for Sharpe
            daily_return = (equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
            prev_equity = equity

            # Track drawdown
            peak = max(peak, equity)
            dd   = peak - equity
            max_dd = max(max_dd, dd)

        # ── Compute stats ──────────────────────────────────────────────────────
        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n      = len(trades)
        win_rate = len(wins) / n if n else 0
        avg_win  = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        expectancy   = (avg_win * win_rate) - (avg_loss * (1 - win_rate))
        profit_factor = gross_profit / gross_loss if gross_loss else 0
        total_pnl    = equity - starting_capital

        # Sharpe (annualized, assume 252 trading days)
        import statistics
        if len(daily_returns) > 1:
            mean_ret = statistics.mean(daily_returns)
            std_ret  = statistics.stdev(daily_returns)
            sharpe   = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            run_id=run_id,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            starting_capital=starting_capital,
            final_equity=round(equity, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round((total_pnl / starting_capital) * 100, 2),
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            expectancy=round(expectancy, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round((max_dd / starting_capital) * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            trades=trades,
        )

    def print_report(self, result: BacktestResult):
        print()
        print("=" * 60)
        print(f"BACKTEST RESULTS — {result.start_date} to {result.end_date}")
        print("=" * 60)
        print(f"  Symbols:        {', '.join(result.symbols)}")
        print(f"  Starting:       ${result.starting_capital:>12,.2f}")
        print(f"  Final equity:   ${result.final_equity:>12,.2f}")
        print(f"  Total P&L:      ${result.total_pnl:>+12,.2f} ({result.total_pnl_pct:+.2f}%)")
        print(f"  Max drawdown:   ${result.max_drawdown:>12,.2f} ({result.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe ratio:   {result.sharpe_ratio:>12.3f}")
        print()
        print(f"  Total trades:   {result.total_trades}")
        print(f"  Win rate:       {result.win_rate:.1%} ({result.winning_trades}W / {result.losing_trades}L)")
        print(f"  Avg win:        ${result.avg_win:>+12,.2f}")
        print(f"  Avg loss:       ${result.avg_loss:>+12,.2f}")
        print(f"  Expectancy:     ${result.expectancy:>+12,.2f}")
        print(f"  Profit factor:  {result.profit_factor:>12.2f}")
        print()

        # Breakdown by setup type
        setup_trades = {}
        for t in result.trades:
            setup_trades.setdefault(t.setup_type, []).append(t)

        if setup_trades:
            print("  By setup type:")
            for setup, st_trades in sorted(setup_trades.items()):
                st_wins = [t for t in st_trades if t.pnl > 0]
                st_pnl  = sum(t.pnl for t in st_trades)
                st_wr   = len(st_wins) / len(st_trades)
                print(f"    {setup:15s} trades={len(st_trades):3d} win={st_wr:.0%} pnl=${st_pnl:+,.2f}")

        # Exit reason breakdown
        exit_reasons = {}
        for t in result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print()
        print("  Exit reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades * 100 if result.total_trades else 0
            print(f"    {reason:15s} {count:4d} ({pct:.0f}%)")
        print("=" * 60)
