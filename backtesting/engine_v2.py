"""
Backtest Engine v2 — event-driven, intraday, honest.

Why the old backtest couldn't answer anything:
  - It used hand-rolled daily-bar rules that did NOT match the live system
  - Long-only, entered next-day open (live enters intraday same-day)
  - Simulated intraday exits from daily OHLC (unknowable ordering)
  - Zero spread/slippage — fatal with 0.5xATR stops
  - No per-setup attribution

What v2 does:
  - Replays 5-MINUTE bars session by session
  - Decision time is 9:45 ET, matching the live scheduler exactly
  - Signals come from mechanical setup detectors (ORB, VWAP reclaim,
    momentum continuation) — the same setup taxonomy the live LLM chooses
    from. This measures the EDGE OF THE SETUPS the LLM is allowed to take;
    the LLM's discretion layer sits on top and can only be judged live.
  - Sizing/stops/targets come from execution/sizing.py — the ACTUAL live
    code path, not a reimplementation
  - Limit-order fill simulation with DAY expiry (mirrors the live order
    tracker: unfilled by close = cancelled, never a position)
  - Bracket exit simulation on 5-min bars; if stop AND target are inside
    one bar's range, the STOP is assumed hit (conservative)
  - Costs: configurable spread + slippage in bps charged per side
  - Optional regime gate (agents/regime.py) applied day by day
  - Report: per-setup expectancy/win-rate/profit-factor, monthly breakdown,
    max drawdown, cost drag

Data: Alpaca 5-min IEX bars (free tier reaches back years), or a synthetic
generator for offline testing.
"""
import os
import math
import statistics
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from execution.sizing import LLMPick, build_trade_plan, MAX_SESSION_RISK_PCT

DECISION_TIME   = dtime(9, 45)
EOD_TIME        = dtime(15, 55)
OPEN_TIME       = dtime(9, 30)
MAX_CONCURRENT  = 3


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    symbol: str
    setup_type: str
    side: str                 # buy / sell
    quantity: int
    limit_price: float
    stop: float
    target: float
    signal_time: datetime
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    costs: float = 0.0

    @property
    def is_long(self):
        return self.side == "buy"


@dataclass
class BacktestConfig:
    spread_bps: float = 2.0        # half-spread paid per side
    slippage_bps: float = 1.0      # additional slippage per side
    starting_equity: float = 100_000.0
    use_regime_gate: bool = False
    setups: tuple = ("orb", "vwap_bounce", "momentum")


# ─── Setup detectors (mechanical versions of the live setup taxonomy) ─────────

def detect_setups(session_bars: list[dict], prev_day: Optional[dict],
                  enabled: tuple) -> list[dict]:
    """
    Evaluate at the 9:45 decision bar. session_bars = today's 5-min bars up
    to and including 9:45. prev_day = {'open','high','low','close'} of the
    prior session. Returns signal dicts: {setup_type, direction}.
    """
    signals = []
    if len(session_bars) < 3:
        return signals

    decision_bar = session_bars[-1]
    px = decision_bar["close"]

    # Opening range = 9:30-9:45 (first 3 bars)
    or_bars = session_bars[:3]
    or_high = max(b["high"] for b in or_bars)
    or_low  = min(b["low"] for b in or_bars)

    # ORB — price closing beyond the opening range at decision time
    if "orb" in enabled:
        if px > or_high * 1.0005:
            signals.append({"setup_type": "orb", "direction": "long"})
        elif px < or_low * 0.9995:
            signals.append({"setup_type": "orb", "direction": "short"})

    # Session VWAP up to decision time
    cum_pv = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"]
                 for b in session_bars)
    cum_v = sum(b["volume"] for b in session_bars) or 1
    vwap = cum_pv / cum_v

    # VWAP reclaim — was below VWAP, decision bar closes back above (long)
    if "vwap_bounce" in enabled and len(session_bars) >= 3:
        prev_closes_below = all(b["close"] < vwap for b in session_bars[-3:-1])
        prev_closes_above = all(b["close"] > vwap for b in session_bars[-3:-1])
        if prev_closes_below and px > vwap:
            signals.append({"setup_type": "vwap_bounce", "direction": "long"})
        elif prev_closes_above and px < vwap:
            signals.append({"setup_type": "vwap_bounce", "direction": "short"})

    # Momentum continuation — strong prior day, holding above today's open
    if "momentum" in enabled and prev_day:
        prev_ret = (prev_day["close"] - prev_day["open"]) / prev_day["open"]
        today_open = session_bars[0]["open"]
        if prev_ret > 0.015 and px > today_open:
            signals.append({"setup_type": "momentum", "direction": "long"})
        elif prev_ret < -0.015 and px < today_open:
            signals.append({"setup_type": "momentum", "direction": "short"})

    # One signal per symbol per day: first match wins (ORB > VWAP > momentum)
    return signals[:1]


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngineV2:

    def __init__(self, bars_by_symbol: dict[str, list[dict]],
                 config: BacktestConfig = None):
        """
        bars_by_symbol: {symbol: [ {timestamp: datetime, open, high, low,
                                    close, volume}, ... ]}  5-min bars, sorted.
        """
        self.cfg = config or BacktestConfig()
        self.bars = bars_by_symbol
        self.equity = self.cfg.starting_equity
        self.trades: list[SimTrade] = []
        self.daily_equity: list[tuple] = []   # (date, equity)

    # ── Session helpers ──

    def _sessions(self) -> list:
        """All trading dates present in the data, sorted."""
        dates = set()
        for bars in self.bars.values():
            for b in bars:
                dates.add(b["timestamp"].date())
        return sorted(dates)

    def _session_bars(self, symbol: str, day) -> list[dict]:
        return [b for b in self.bars.get(symbol, [])
                if b["timestamp"].date() == day
                and OPEN_TIME <= b["timestamp"].time() <= dtime(16, 0)]

    def _daily_agg(self, symbol: str, day) -> Optional[dict]:
        bars = self._session_bars(symbol, day)
        if not bars:
            return None
        return {"open": bars[0]["open"], "close": bars[-1]["close"],
                "high": max(b["high"] for b in bars),
                "low": min(b["low"] for b in bars)}

    def _atr14(self, symbol: str, day, sessions: list) -> Optional[float]:
        """Daily ATR(14) computed from 5-min data aggregated per session."""
        idx = sessions.index(day)
        if idx < 15:
            return None
        trs = []
        prev_close = None
        for d in sessions[idx - 15: idx]:
            agg = self._daily_agg(symbol, d)
            if not agg:
                continue
            if prev_close is None:
                tr = agg["high"] - agg["low"]
            else:
                tr = max(agg["high"] - agg["low"],
                         abs(agg["high"] - prev_close),
                         abs(agg["low"] - prev_close))
            trs.append(tr)
            prev_close = agg["close"]
        return statistics.mean(trs[-14:]) if len(trs) >= 10 else None

    def _cost(self, price: float, qty: int) -> float:
        """One side of spread + slippage in dollars."""
        bps = (self.cfg.spread_bps + self.cfg.slippage_bps) / 10_000
        return price * qty * bps

    # ── Regime (optional) ──

    def _regime_for_day(self, day, sessions) -> dict:
        if not self.cfg.use_regime_gate or "SPY" not in self.bars:
            return {"trade_allowed": True, "risk_scale": 1.0, "long_scale": 1.0}
        from agents.regime import assess_regime
        idx = sessions.index(day)
        hist_days = sessions[max(0, idx - 90): idx]

        class _DP:
            def __init__(dp_self):
                dp_self.days = hist_days
            def fetch_bars(dp_self, symbol, days=90):
                out = []
                for d in dp_self.days:
                    agg = self._daily_agg("SPY", d)
                    if agg:
                        out.append(agg)
                return out

        return assess_regime(_DP())

    # ── Main loop ──

    def run(self) -> dict:
        sessions = self._sessions()
        symbols = [s for s in self.bars.keys() if s != "SPY"]

        for day in sessions[16:]:   # need ATR warmup
            regime = self._regime_for_day(day, sessions)
            day_trades: list[SimTrade] = []

            if regime["trade_allowed"]:
                risk_budget = self.equity * MAX_SESSION_RISK_PCT * regime["risk_scale"]
                prev_idx = sessions.index(day) - 1

                for symbol in symbols:
                    if len(day_trades) >= MAX_CONCURRENT:
                        break
                    sbars = self._session_bars(symbol, day)
                    decision_bars = [b for b in sbars
                                     if b["timestamp"].time() <= DECISION_TIME]
                    if len(decision_bars) < 3:
                        continue
                    prev_day_agg = self._daily_agg(symbol, sessions[prev_idx])

                    for sig in detect_setups(decision_bars, prev_day_agg,
                                             self.cfg.setups):
                        atr = self._atr14(symbol, day, sessions)
                        px  = decision_bars[-1]["close"]
                        scale = regime["risk_scale"] * (
                            regime.get("long_scale", 1.0)
                            if sig["direction"] == "long" else 1.0)
                        pick = LLMPick(
                            symbol=symbol, direction=sig["direction"],
                            setup_type=sig["setup_type"], conviction=0.7,
                            reasoning="mechanical backtest signal",
                        )
                        plan = build_trade_plan(
                            pick, live_price=px, atr=atr,
                            equity=self.equity,
                            risk_budget_left=risk_budget,
                            risk_scale=scale,
                        )
                        if plan is None:
                            continue
                        risk_budget -= plan.dollar_risk
                        day_trades.append(SimTrade(
                            symbol=symbol, setup_type=plan.setup_type,
                            side=plan.side, quantity=plan.quantity,
                            limit_price=plan.entry_price,
                            stop=plan.stop_loss, target=plan.take_profit,
                            signal_time=decision_bars[-1]["timestamp"],
                        ))

            # Simulate each trade through the rest of the session
            for t in day_trades:
                self._simulate_trade(t, day)
                if t.fill_price is not None:
                    self.equity += t.pnl
                    self.trades.append(t)

            self.daily_equity.append((day, self.equity))

        return self.report()

    def _simulate_trade(self, t: SimTrade, day):
        """Limit fill (DAY expiry) → bracket exits → EOD close. Costs charged."""
        sbars = [b for b in self._session_bars(t.symbol, day)
                 if b["timestamp"] > t.signal_time]

        # 1. Fill simulation: limit fills when price trades through it
        for b in sbars:
            if b["timestamp"].time() > EOD_TIME:
                break
            filled = (b["low"] <= t.limit_price) if t.is_long \
                else (b["high"] >= t.limit_price)
            if filled:
                t.fill_price = t.limit_price
                t.fill_time = b["timestamp"]
                t.costs += self._cost(t.fill_price, t.quantity)
                break

        if t.fill_price is None:
            t.exit_reason = "never_filled"    # mirrors tracker's EOD cancel
            return

        # 2. Bracket simulation bar by bar after fill
        for b in sbars:
            if b["timestamp"] <= t.fill_time:
                continue
            if b["timestamp"].time() > EOD_TIME:
                break
            if t.is_long:
                hit_stop   = b["low"] <= t.stop
                hit_target = b["high"] >= t.target
            else:
                hit_stop   = b["high"] >= t.stop
                hit_target = b["low"] <= t.target
            if hit_stop:                       # conservative: stop wins ties
                self._close(t, t.stop, b["timestamp"], "stop_hit")
                return
            if hit_target:
                self._close(t, t.target, b["timestamp"], "target_hit")
                return

        # 3. EOD close at the last bar before 15:55
        eod_bars = [b for b in sbars if b["timestamp"].time() <= EOD_TIME]
        if eod_bars:
            self._close(t, eod_bars[-1]["close"], eod_bars[-1]["timestamp"],
                        "eod_close")

    def _close(self, t: SimTrade, price: float, when, reason: str):
        t.exit_price = price
        t.exit_time = when
        t.exit_reason = reason
        t.costs += self._cost(price, t.quantity)
        direction = 1 if t.is_long else -1
        gross = (price - t.fill_price) * t.quantity * direction
        t.pnl = gross - t.costs
        risk = abs(t.fill_price - t.stop) * t.quantity
        t.r_multiple = t.pnl / risk if risk else 0.0

    # ── Reporting ──

    def report(self) -> dict:
        filled = [t for t in self.trades if t.fill_price is not None]
        rep = {
            "config": {
                "spread_bps": self.cfg.spread_bps,
                "slippage_bps": self.cfg.slippage_bps,
                "regime_gate": self.cfg.use_regime_gate,
                "setups": list(self.cfg.setups),
            },
            "total_trades": len(filled),
            "final_equity": round(self.equity, 2),
            "total_return_pct": round(
                (self.equity / self.cfg.starting_equity - 1) * 100, 2),
            "total_costs": round(sum(t.costs for t in filled), 2),
            "by_setup": {},
            "by_month": {},
            "max_drawdown_pct": self._max_drawdown(),
        }

        for setup in {t.setup_type for t in filled}:
            ts = [t for t in filled if t.setup_type == setup]
            wins = [t for t in ts if t.pnl > 0]
            losses = [t for t in ts if t.pnl <= 0]
            gp = sum(t.pnl for t in wins)
            gl = abs(sum(t.pnl for t in losses))
            rep["by_setup"][setup] = {
                "trades": len(ts),
                "win_rate": round(len(wins) / len(ts), 3) if ts else 0,
                "avg_r": round(statistics.mean(t.r_multiple for t in ts), 3) if ts else 0,
                "expectancy_$": round(statistics.mean(t.pnl for t in ts), 2) if ts else 0,
                "profit_factor": round(gp / gl, 2) if gl else float("inf"),
                "total_pnl": round(sum(t.pnl for t in ts), 2),
                "stop_hits": sum(1 for t in ts if t.exit_reason == "stop_hit"),
                "target_hits": sum(1 for t in ts if t.exit_reason == "target_hit"),
                "eod_closes": sum(1 for t in ts if t.exit_reason == "eod_close"),
            }

        monthly = defaultdict(float)
        for t in filled:
            monthly[t.exit_time.strftime("%Y-%m")] += t.pnl
        rep["by_month"] = {m: round(p, 2) for m, p in sorted(monthly.items())}
        return rep

    def _max_drawdown(self) -> float:
        peak, max_dd = -math.inf, 0.0
        for _, eq in self.daily_equity:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)
        return round(max_dd * 100, 2)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_alpaca_5min(symbols: list[str], start: datetime,
                     end: datetime) -> dict[str, list[dict]]:
    """Fetch 5-min IEX bars from Alpaca for a symbol list (SPY auto-added
    for the regime gate)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    import pytz

    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    et = pytz.timezone("America/New_York")

    want = list(dict.fromkeys(symbols + ["SPY"]))
    out: dict[str, list[dict]] = {}
    req = StockBarsRequest(
        symbol_or_symbols=want,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end, feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req)
    for sym in want:
        rows = []
        for b in bars.data.get(sym, []):
            ts = b.timestamp.astimezone(et).replace(tzinfo=None)
            rows.append({"timestamp": ts, "open": float(b.open),
                         "high": float(b.high), "low": float(b.low),
                         "close": float(b.close), "volume": float(b.volume)})
        out[sym] = sorted(rows, key=lambda r: r["timestamp"])
        print(f"  loaded {len(rows):,} 5-min bars for {sym}")
    return out


def generate_synthetic_5min(symbols: list[str], sessions: int = 120,
                            seed: int = 42) -> dict[str, list[dict]]:
    """
    Synthetic 5-min bars for offline ENGINE validation (not strategy
    validation). Returns near-random-walk data: zero-mean bar returns with
    mild volatility clustering and slight mean reversion.

    The point: on data with no exploitable structure, a correct engine must
    show ~zero expectancy before costs and negative after. A large positive
    edge on this data = look-ahead bug in the engine.
    """
    import random
    rng = random.Random(seed)
    out = {}
    start_day = datetime(2026, 1, 5)

    for si, sym in enumerate(list(symbols) + ["SPY"]):
        price = 100.0 + si * 80
        rows = []
        day = start_day
        made = 0
        vol = 0.0012
        while made < sessions:
            if day.weekday() < 5:
                # volatility clusters day to day, but returns are zero-mean
                vol = max(0.0006, min(0.004, vol * rng.gauss(1.0, 0.15)))
                t = day.replace(hour=9, minute=30)
                last_ret = 0.0
                while t.time() <= dtime(16, 0):
                    o = price
                    # slight mean reversion (-0.1 autocorr), zero drift
                    ret = rng.gauss(0, vol) - 0.1 * last_ret
                    c = o * (1 + ret)
                    hi = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
                    lo = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
                    rows.append({"timestamp": t, "open": round(o, 4),
                                 "high": round(hi, 4), "low": round(lo, 4),
                                 "close": round(c, 4),
                                 "volume": rng.randint(50_000, 500_000)})
                    price = c
                    last_ret = ret
                    t += timedelta(minutes=5)
                made += 1
            day += timedelta(days=1)
        out[sym] = rows
    return out
