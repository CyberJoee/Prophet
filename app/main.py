"""
Prophet -- live HA WaveTrend chart dashboard.

FastAPI app that keeps an in-memory cache of computed charts per (symbol, tf),
refreshed on a background interval, and serves them as a single page. No
external DB required for v1; charts are recomputed from free exchange data
(Coinbase, with automatic fallback to other exchanges for deeper history).

Run locally:
    uvicorn app.main:app --reload
"""
import html
import logging
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.chart import build_html
from app.strategy import Params, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prophet")

# ───────────────────────── Config ─────────────────────────
SYMBOLS = [s.strip() for s in os.environ.get("SYMBOLS", "HBAR/USD").split(",") if s.strip()]
TIMEFRAMES = [t.strip() for t in os.environ.get("TIMEFRAMES", "1d,1w").split(",") if t.strip()]
SINCE = os.environ.get("SINCE", "2021-01-01")
EXCHANGE = os.environ.get("EXCHANGE", "coinbase")
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", 15 * 60))

app = FastAPI(title="Prophet")

# cache[(symbol, tf)] = {"html": str, "summary": dict, "updated_at": iso str, "error": str|None}
_cache: dict[tuple[str, str], dict] = {}
_cache_lock = threading.Lock()


def _refresh_one(symbol: str, tf: str) -> None:
    div_id = f"chart-{symbol.replace('/', '-')}-{tf}"
    try:
        d, tr, summary = run(symbol, tf, SINCE, Params(), EXCHANGE, log=lambda s: log.info(f"[{symbol} {tf}] {s}"))
        chart_html = build_html(d, tr, Params(), symbol, tf, div_id)
        with _cache_lock:
            _cache[(symbol, tf)] = {
                "html": chart_html, "summary": summary, "error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        log.info(f"Refreshed {symbol} {tf}: {summary}")
    except Exception as e:
        log.exception(f"Failed to refresh {symbol} {tf}")
        with _cache_lock:
            prev = _cache.get((symbol, tf), {})
            _cache[(symbol, tf)] = {**prev, "error": str(e),
                                     "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _refresh_all() -> None:
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            _refresh_one(symbol, tf)


def _refresh_loop() -> None:
    while True:
        time.sleep(REFRESH_SECONDS)
        _refresh_all()


@app.on_event("startup")
def startup() -> None:
    _refresh_all()  # populate cache before serving so the first page load isn't empty
    threading.Thread(target=_refresh_loop, daemon=True).start()


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prophet</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<meta http-equiv="refresh" content="{refresh_seconds}">
<style>
  body {{ background:#111318; color:#eee; font-family: -apple-system, Segoe UI, sans-serif; margin:0; padding:24px; }}
  h1 {{ font-weight:600; margin-bottom:4px; }}
  .meta {{ color:#888; font-size:13px; margin-bottom:24px; }}
  .card {{ background:#1a1d24; border-radius:10px; padding:16px; margin-bottom:24px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:16px; margin:8px 0 16px; font-size:14px; }}
  .stat {{ background:#20242e; border-radius:6px; padding:8px 14px; }}
  .stat b {{ color:#8FC7FF; }}
  .err {{ color:#ff6b6b; font-size:13px; }}
  .tf-label {{ color:#aaa; font-size:13px; text-transform:uppercase; letter-spacing:.05em; }}
</style>
</head>
<body>
<h1>Prophet</h1>
<div class="meta">HA WaveTrend live dashboard &middot; auto-refreshes every {refresh_min} min &middot; page rendered {now}</div>
{sections}
</body>
</html>"""

SECTION_TEMPLATE = """<div class="card">
  <span class="tf-label">{symbol} &middot; {tf}</span>
  {stats_html}
  {body}
</div>"""


def _stats_row(summary: dict | None, error: str | None) -> str:
    if error and not summary:
        return f'<div class="err">Error: {html.escape(error)}</div>'
    if not summary:
        return '<div class="err">No data yet</div>'
    items = [
        ("Period", f"{summary['start']} &rarr; {summary['end']}"),
        ("Strategy", f"{summary['strategy_pct']:+.1f}%"),
        ("Buy &amp; hold", f"{summary['buy_hold_pct']:+.1f}%"),
        ("Max DD", f"{summary['max_dd_pct']:.1f}%"),
        ("Trades", str(summary["trades"])),
    ]
    if "win_rate_pct" in summary:
        items.append(("Win rate", f"{summary['win_rate_pct']}%"))
    stats = "".join(f'<div class="stat">{label}: <b>{val}</b></div>' for label, val in items)
    warn = f'<div class="err">Last refresh error (showing stale data): {html.escape(error)}</div>' if error else ""
    return f'<div class="stats">{stats}</div>{warn}'


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    sections = []
    with _cache_lock:
        snapshot = dict(_cache)
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            entry = snapshot.get((symbol, tf))
            body = entry["html"] if entry and entry.get("html") else "<div class='err'>Loading&hellip;</div>"
            stats_html = _stats_row(entry.get("summary") if entry else None, entry.get("error") if entry else None)
            sections.append(SECTION_TEMPLATE.format(symbol=symbol, tf=tf, stats_html=stats_html, body=body))
    return PAGE_TEMPLATE.format(
        refresh_seconds=REFRESH_SECONDS, refresh_min=REFRESH_SECONDS // 60,
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        sections="\n".join(sections),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/summary")
def api_summary() -> dict:
    with _cache_lock:
        snapshot = dict(_cache)
    return {f"{sym}:{tf}": {"summary": v.get("summary"), "error": v.get("error"), "updated_at": v.get("updated_at")}
            for (sym, tf), v in snapshot.items()}
