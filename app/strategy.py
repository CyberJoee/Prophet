"""
HA WaveTrend strategy library: data fetch (with cross-exchange history
backfill), Heikin Ashi + WaveTrend + Money Flow indicators, and a backtester.

No CLI, no chart rendering, no browser popups -- this is the pure logic layer
shared by the live dashboard (app/main.py) and anything else that wants it.

Ported from the standalone research script (custom-charts project); kept in
sync in spirit, but this copy owns the web app's runtime behavior.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ───────────────────────── Params ─────────────────────────
@dataclass
class Params:
    use_ha: bool = True
    wt_channel: int = 9
    wt_average: int = 12
    wt_ma: int = 3
    ob: int = 53
    buy_zone: float = 0
    sell_zone: float = 0
    mf_len: int = 60
    fast: int = 9
    slow: int = 21
    use_trend: bool = True
    use_mf: bool = False
    use_vol: bool = False
    exit_mode: str = "red"  # red | cross | ema
    atr_len: int = 14
    atr_mult: float = 2.0
    fee: float = 0.1  # percent per side
    levels: list = field(default_factory=list)


# ───────────────────────── Data ─────────────────────────
# Exchanges to check (roughly in order of how far back their history usually
# goes / how reliable ccxt's public OHLCV endpoint is for them), used when a
# symbol's listing on the primary exchange doesn't cover the requested window.
FALLBACK_EXCHANGES = ["binance", "kraken", "bitfinex", "okx", "bybit", "kucoin", "huobi", "poloniex"]


def fetch_daily(symbol: str, since: str, exchange_id: str = "coinbase") -> pd.DataFrame:
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    now_ms = ex.milliseconds()
    rows = []
    while since_ms < now_ms:
        batch = ex.fetch_ohlcv(symbol, "1d", since=since_ms, limit=300)
        if not batch:
            # No candles in this window -- the symbol may not have been listed
            # yet at `since_ms`. Skip forward a chunk and keep looking instead
            # of treating an empty page as "end of history".
            since_ms += 300 * 86_400_000
            continue
        rows += batch
        since_ms = batch[-1][0] + 86_400_000
        # A short page does NOT mean "end of history" -- the exchange can cap
        # a page well before `now` (e.g. when `since` predates listing). Only
        # stop early once the batch has actually caught up to the present.
        if len(batch) < 300 and since_ms >= now_ms - 86_400_000:
            break
    if not rows:
        raise ValueError(
            f"No OHLCV data returned for {symbol} on {exchange_id} at any point "
            f"since {since}. Check the symbol is correct and was listed on {exchange_id}."
        )
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def _first_candle_ms(ex, symbol: str, since_ms: int, now_ms: int, probe_days: int = 365):
    """Scan forward from since_ms in probe_days-sized steps to find the first
    daily candle timestamp available for `symbol` on `ex`. Returns None if the
    symbol has no candles anywhere in [since_ms, now_ms), or isn't listed."""
    step_ms = probe_days * 86_400_000
    t = since_ms
    while t < now_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, "1d", since=t, limit=1)
        except Exception:
            return None
        if batch:
            return batch[0][0]
        t += step_ms
    return None


def _symbol_variants(symbol: str):
    """USD pairs often don't exist on non-US exchanges; try USDT/USDC as a
    ~1:1 proxy for USD if the exact pair isn't listed."""
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote == "USD":
        variants += [f"{base}/USDT", f"{base}/USDC"]
    return variants


def find_earliest_source(symbol: str, since: str, primary: str = "coinbase", candidates=None):
    """Check `primary` plus a list of fallback exchanges for whichever has the
    earliest available daily history for `symbol` (or a USDT/USDC proxy of
    it). Returns (exchange_id, matched_symbol, first_ts_ms_or_None)."""
    import ccxt
    candidates = candidates if candidates is not None else FALLBACK_EXCHANGES
    since_ms = ccxt.Exchange.parse8601(f"{since}T00:00:00Z")
    now_ms = ccxt.Exchange().milliseconds()

    best = (primary, symbol, None)
    for ex_id in [primary] + [c for c in candidates if c != primary]:
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ex.load_markets()
        except Exception:
            continue
        for sym in _symbol_variants(symbol):
            if sym not in ex.markets:
                continue
            ts = _first_candle_ms(ex, sym, since_ms, now_ms)
            if ts is None:
                continue
            if best[2] is None or ts < best[2]:
                best = (ex_id, sym, ts)
            break  # first listed variant on this exchange is enough
        if best[2] is not None and best[2] <= since_ms:
            break  # already found full coverage back to the requested date
    return best


def fetch_daily_auto(symbol: str, since: str, primary: str = "coinbase", candidates=None,
                      log=None) -> pd.DataFrame:
    """fetch_daily(), but if `primary` doesn't cover `since`, automatically
    check other exchanges for a listing that goes back further and use
    whichever single exchange has the deepest history (no stitching across
    sources, to avoid price discontinuities at the seam). `log`, if given, is
    called with status strings instead of printing."""
    log = log or (lambda s: None)
    df = fetch_daily(symbol, since, primary)
    primary_start = df.index[0]
    since_ts_req = pd.Timestamp(since, tz="utc")

    if primary_start <= since_ts_req:
        return df

    log(f"{symbol} on {primary} only goes back to {primary_start.date()} "
        f"(requested since {since}); checking other exchanges...")
    ex_id, matched_symbol, first_ts = find_earliest_source(symbol, since, primary, candidates)

    if first_ts is None or ex_id == primary:
        log(f"No exchange with earlier {symbol} history found; using {primary} from {primary_start.date()}.")
        return df

    found_start = pd.to_datetime(first_ts, unit="ms", utc=True)
    if found_start >= primary_start:
        log(f"Best alternative ({ex_id}) starts {found_start.date()}, no earlier than "
            f"{primary} ({primary_start.date()}); keeping {primary}.")
        return df

    log(f"Using {ex_id} ({matched_symbol}) instead of {primary} -- history back to {found_start.date()}.")
    return fetch_daily(matched_symbol, since, ex_id)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rules = {"1d": None, "3d": "3D", "1w": "W-MON", "1M": "MS"}
    rule = rules[tf]
    if rule is None:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


# ───────────────────────── Indicators ─────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df.open + df.high + df.low + df.close) / 4
    ho = np.empty(len(df))
    ho[0] = (df.open.iloc[0] + df.close.iloc[0]) / 2
    hc = ha["close"].to_numpy()
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    ha["open"] = ho
    ha["high"] = pd.concat([df.high, ha.open, ha.close], axis=1).max(axis=1)
    ha["low"] = pd.concat([df.low, ha.open, ha.close], axis=1).min(axis=1)
    return ha


def compute(df: pd.DataFrame, a: Params) -> pd.DataFrame:
    src = heikin_ashi(df) if a.use_ha else df
    out = df.copy()
    out[["ha_open", "ha_high", "ha_low", "ha_close"]] = src[["open", "high", "low", "close"]].to_numpy()

    # WaveTrend
    ap = (src.high + src.low + src.close) / 3
    esa = ema(ap, a.wt_channel)
    de = ema((ap - esa).abs(), a.wt_channel)
    ci = ((ap - esa) / (0.015 * de)).replace([np.inf, -np.inf], 0).fillna(0)
    out["wt1"] = ema(ci, a.wt_average)
    out["wt2"] = out.wt1.rolling(a.wt_ma).mean()

    up = (out.wt1 > out.wt2) & (out.wt1.shift() <= out.wt2.shift())
    dn = (out.wt1 < out.wt2) & (out.wt1.shift() >= out.wt2.shift())
    out["green_dot"] = up & (out.wt2 <= a.buy_zone)
    out["red_dot"] = dn & (out.wt2 >= a.sell_zone)
    out["cross_dn"] = dn

    # Money flow
    rng = (src.high - src.low).replace(0, np.nan)
    mf_raw = ((src.close - src.open) / rng * 150).fillna(0)
    out["mf"] = mf_raw.rolling(a.mf_len).mean() - 2.5

    # Trend, volume, ATR (ATR on real prices)
    out["fast"] = ema(src.close, a.fast)
    out["slow"] = ema(src.close, a.slow)
    prev_c = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev_c).abs(), (df.low - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / a.atr_len, adjust=False).mean()

    trend_ok = (src.close > out.slow) if a.use_trend else True
    mf_ok = (out.mf > 0) if a.use_mf else True
    vol_ok = (df.volume > df.volume.rolling(20).mean()) if a.use_vol else True
    out["long_sig"] = out.green_dot & trend_ok & mf_ok & vol_ok

    exits = {"red": out.red_dot, "cross": out.cross_dn,
             "ema": (src.close < out.slow) & (src.close.shift() >= out.slow.shift())}
    out["exit_sig"] = exits[a.exit_mode]
    return out


# ───────────────────────── Backtest ─────────────────────────
def backtest(d: pd.DataFrame, a: Params, capital: float = 10_000.0):
    """Signal on bar close -> fill at next bar's REAL open. Stop checked intrabar on real lows."""
    fee = a.fee / 100
    cash, qty, stop = capital, 0.0, np.nan
    trades, equity = [], []
    entry_px = entry_t = None
    pending_entry = pending_exit = False

    for i, (t, r) in enumerate(d.iterrows()):
        if pending_entry and qty == 0:
            entry_px, entry_t = r.open, t
            qty = cash * (1 - fee) / entry_px
            cash = 0.0
            stop = d.close.iloc[i - 1] - d.atr.iloc[i - 1] * a.atr_mult if a.atr_mult > 0 else np.nan
        elif pending_exit and qty > 0:
            cash = qty * r.open * (1 - fee)
            trades.append((entry_t, t, entry_px, r.open, "signal"))
            qty = 0.0
        pending_entry = pending_exit = False

        if qty > 0 and not np.isnan(stop) and r.low <= stop:
            px = min(r.open, stop)
            cash = qty * px * (1 - fee)
            trades.append((entry_t, t, entry_px, px, "stop"))
            qty = 0.0

        if qty == 0 and r.long_sig and i < len(d) - 1:
            pending_entry = True
        elif qty > 0 and r.exit_sig:
            pending_exit = True

        equity.append(cash + qty * r.close)

    d = d.assign(equity=equity)
    tr = pd.DataFrame(trades, columns=["entry_t", "exit_t", "entry", "exit", "reason"])
    if len(tr):
        tr["pct"] = (tr.exit / tr.entry * (1 - fee) ** 2 - 1) * 100
    # Position still open at the end of the data (no close/stop has fired
    # yet) -- not in `tr` since that only records realized trades. Dashboards
    # and alerting need this to know whether a sell signal would actually
    # close anything right now.
    d.attrs["open_position"] = (
        {"entry_t": entry_t.isoformat(), "entry_px": float(entry_px),
         "stop": None if np.isnan(stop) else float(stop), "qty": float(qty)}
        if qty > 0 else None
    )
    return d, tr


def summarize(d: pd.DataFrame, tr: pd.DataFrame, capital: float = 10_000.0) -> dict:
    eq = d.equity
    dd = (eq / eq.cummax() - 1).min() * 100
    bh = (d.close.iloc[-1] / d.open.iloc[0] - 1) * 100
    last = d.iloc[-1]
    out = {
        "start": d.index[0].date().isoformat(), "end": d.index[-1].date().isoformat(),
        "bars": len(d), "strategy_pct": round((eq.iloc[-1] / capital - 1) * 100, 1),
        "buy_hold_pct": round(bh, 1), "max_dd_pct": round(dd, 1), "trades": len(tr),
        "position": d.attrs.get("open_position"),
        "latest": {
            "date": d.index[-1].date().isoformat(), "close": float(last.close),
            "green_dot": bool(last.green_dot), "red_dot": bool(last.red_dot),
        },
    }
    if len(tr):
        out["win_rate_pct"] = round((tr.pct > 0).mean() * 100)
        out["avg_trade_pct"] = round(tr.pct.mean(), 1)
    return out


def run(symbol: str, tf: str, since: str, a: Params = None, exchange: str = "coinbase",
        capital: float = 10_000.0, log=None):
    """End-to-end: fetch (with auto cross-exchange backfill) -> resample -> compute -> backtest."""
    a = a or Params()
    raw = fetch_daily_auto(symbol, since, exchange, log=log)
    df = resample(raw, tf)
    d, tr = backtest(compute(df, a), a, capital)
    return d, tr, summarize(d, tr, capital)
