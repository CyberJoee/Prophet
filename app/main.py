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
import json
import logging
import os
import threading
import time
import urllib.request
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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
# Which timeframes actually fire Discord alerts. Daily dots on a WaveTrend
# this fast flip constantly and would spam the channel; weekly is the one
# worth getting pushed for. Still configurable in case that changes.
ALERT_TIMEFRAMES = {t.strip() for t in os.environ.get("ALERT_TIMEFRAMES", "1w").split(",") if t.strip()}
if DISCORD_WEBHOOK_URL.startswith("https://discordapp.com/"):
    # Legacy domain; discord.com is the current one and the one that's been
    # verified to work reliably with a plain urllib request.
    DISCORD_WEBHOOK_URL = "https://discord.com/" + DISCORD_WEBHOOK_URL[len("https://discordapp.com/"):]

app = FastAPI(title="Prophet")

# cache[(symbol, tf)] = {"html": str, "summary": dict, "updated_at": iso str, "error": str|None}
_cache: dict[tuple[str, str], dict] = {}
_cache_lock = threading.Lock()

# alert_state[(symbol, tf)] = {"date": "2026-09-30", "green": bool, "red": bool}
# Tracks which dot signals we've already alerted on for the current latest
# bar, so a flickering intrabar recompute (the bar isn't closed yet) doesn't
# re-fire the same alert on every refresh, and so a brand new bar's dots are
# alerted exactly once.
_alert_state: dict[tuple[str, str], dict] = {}
# Set True only after the very first population pass completes, so a deploy
# doesn't blast alerts for dots that already existed before this ran.
_alerts_armed = False


def _send_discord_alert(symbol: str, tf: str, kind: str, summary: dict) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    label = TF_LABELS.get(tf, tf)
    action = "🟢 BUY signal (green dot)" if kind == "green" else "🔴 SELL signal (red dot)"
    pos = summary.get("position")
    pos_line = (f"Open position: entered {pos['entry_t'][:10]} @ {pos['entry_px']:.5g}"
                if pos else "No open position")
    content = (f"**{action}**\n"
               f"{symbol} · {label}\n"
               f"Close: {summary['latest']['close']:.5g} on {summary['latest']['date']}\n"
               f"{pos_line}\n"
               f"https://prophetcharts.up.railway.app/")
    try:
        # Discord's API 403s requests with no (or a generic) User-Agent --
        # it's behind bot detection that urllib's default UA trips.
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL, data=json.dumps({"content": content}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "Prophet-Dashboard (https://prophetcharts.up.railway.app, 1.0)"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
        log.info(f"Sent Discord alert: {symbol} {tf} {kind}")
    except Exception:
        log.exception(f"Failed to send Discord alert for {symbol} {tf} {kind}")


def _check_alert(symbol: str, tf: str, summary: dict) -> None:
    if tf not in ALERT_TIMEFRAMES:
        return
    latest = summary["latest"]
    key = (symbol, tf)
    state = _alert_state.get(key) or {}
    if state.get("date") != latest["date"]:
        state = {"date": latest["date"], "green": False, "red": False}

    for kind in ("green", "red"):
        if latest[f"{kind}_dot"] and not state[kind]:
            state[kind] = True
            if _alerts_armed:
                _send_discord_alert(symbol, tf, kind, summary)
    _alert_state[key] = state


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
        _check_alert(symbol, tf, summary)
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
    # Fetch in the background rather than blocking startup: with several
    # symbols/timeframes this can take 30-60s, and Railway (and any other
    # platform health check) starts routing traffic as soon as the process
    # is listening -- blocking here just turns that window into 502s. The
    # dashboard already renders a "Loading..." state for uncached entries.
    def _populate_then_loop():
        global _alerts_armed
        _refresh_all()  # seed _alert_state from whatever's already on the latest bar; no alerts sent yet
        _alerts_armed = True
        log.info("Alerts armed; future new dots will notify Discord" if DISCORD_WEBHOOK_URL
                  else "DISCORD_WEBHOOK_URL not set; alerts disabled")
        _refresh_loop()
    threading.Thread(target=_populate_then_loop, daemon=True).start()


TF_LABELS = {"1d": "Day", "3d": "3-Day", "1w": "Week", "1M": "Month"}


def _slug(symbol: str) -> str:
    return symbol.replace("/", "-")


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
  .meta {{ color:#888; font-size:13px; margin-bottom:20px; }}
  .card {{ background:#1a1d24; border-radius:10px; padding:16px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:16px; margin:8px 0 16px; font-size:14px; }}
  .stat {{ background:#20242e; border-radius:6px; padding:8px 14px; }}
  .stat b {{ color:#8FC7FF; }}
  .err {{ color:#ff6b6b; font-size:13px; }}
  .status {{ border-radius:8px; padding:10px 14px; margin-bottom:14px; font-size:14px; font-weight:600; }}
  .status.open {{ background:#1a3326; color:#7ee0a5; border:1px solid #2d5c40; }}
  .status.flat {{ background:#20242e; color:#999; border:1px solid #2a2e38; }}
  .status.signal {{ background:#3a1e1e; color:#ff8a8a; border:1px solid #5c2d2d; }}

  .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }}
  .tab-btn {{
    background:#1a1d24; color:#aaa; border:1px solid #2a2e38; border-radius:8px;
    padding:9px 18px; font-size:15px; font-weight:600; cursor:pointer;
  }}
  .tab-btn.active {{ background:#2a3350; color:#8FC7FF; border-color:#3b4a7a; }}
  .tab-btn:hover {{ color:#eee; }}

  .tf-tabs {{ display:flex; gap:6px; margin-bottom:14px; }}
  .tf-btn {{
    background:transparent; color:#888; border:1px solid #2a2e38; border-radius:20px;
    padding:5px 16px; font-size:13px; font-weight:600; cursor:pointer;
  }}
  .tf-btn.active {{ background:#20242e; color:#eee; border-color:#444; }}
  .tf-btn:hover {{ color:#eee; }}

  @media (max-width: 480px) {{
    body {{ padding:12px; }}
    h1 {{ font-size:22px; }}
    .card {{ padding:10px; border-radius:8px; }}
    .stat {{ padding:6px 10px; font-size:13px; }}
    .tab-btn {{ padding:8px 14px; font-size:14px; }}
  }}
</style>
</head>
<body>
<h1>Prophet</h1>
<div class="meta">HA WaveTrend live dashboard &middot; auto-refreshes every {refresh_min} min &middot; page rendered {now}</div>
<div class="tabs">{symbol_tabs}</div>
{symbol_panels}
<script>
function showSymbol(sym) {{
  document.querySelectorAll('.symbol-panel').forEach(function(el) {{ el.style.display = 'none'; }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{ el.classList.remove('active'); }});
  var panel = document.getElementById('panel-' + sym);
  var btn = document.getElementById('tab-' + sym);
  if (!panel || !btn) return;
  panel.style.display = 'block';
  btn.classList.add('active');
  try {{ localStorage.setItem('prophet_symbol', sym); }} catch (e) {{}}
  var activeTfBtn = panel.querySelector('.tf-btn.active');
  if (activeTfBtn) resizeChartsIn(sym, activeTfBtn.dataset.tf);
}}
function showTf(sym, tf) {{
  var panel = document.getElementById('panel-' + sym);
  if (!panel) return;
  panel.querySelectorAll('.tf-panel').forEach(function(el) {{ el.style.display = 'none'; }});
  panel.querySelectorAll('.tf-btn').forEach(function(el) {{ el.classList.remove('active'); }});
  var tfPanel = document.getElementById('tfpanel-' + sym + '-' + tf);
  var tfBtn = document.getElementById('tftab-' + sym + '-' + tf);
  if (!tfPanel || !tfBtn) return;
  tfPanel.style.display = 'block';
  tfBtn.classList.add('active');
  try {{ localStorage.setItem('prophet_tf_' + sym, tf); }} catch (e) {{}}
  resizeChartsIn(sym, tf);
}}
function resizeChartsIn(sym, tf) {{
  var container = document.getElementById('tfpanel-' + sym + '-' + tf);
  if (!container) return;
  container.querySelectorAll('.plotly-graph-div').forEach(function(div) {{
    try {{ Plotly.Plots.resize(div); }} catch (e) {{}}
  }});
}}
(function init() {{
  var savedSym = null, savedTf = {{}};
  try {{
    savedSym = localStorage.getItem('prophet_symbol');
    document.querySelectorAll('.symbol-panel').forEach(function(el) {{
      var s = el.dataset.sym;
      var saved = localStorage.getItem('prophet_tf_' + s);
      if (saved) savedTf[s] = saved;
    }});
  }} catch (e) {{}}

  // Every symbol panel needs an active tf-panel, not just the one shown on
  // load -- otherwise switching to a symbol you haven't viewed yet (no saved
  // tf) lands on an empty panel with no sub-tab highlighted.
  document.querySelectorAll('.symbol-panel').forEach(function(el) {{
    var s = el.dataset.sym;
    var wanted = savedTf[s];
    if (!wanted || !document.getElementById('tfpanel-' + s + '-' + wanted)) {{
      var firstTfBtn = el.querySelector('.tf-btn');
      wanted = firstTfBtn ? firstTfBtn.dataset.tf : null;
    }}
    if (wanted) showTf(s, wanted);
  }});

  var sym = savedSym;
  if (!sym || !document.getElementById('panel-' + sym)) {{
    var first = document.querySelector('.symbol-panel');
    sym = first ? first.dataset.sym : null;
  }}
  if (sym) showSymbol(sym);
}})();
</script>
</body>
</html>"""

SYMBOL_PANEL_TEMPLATE = """<div class="symbol-panel card" id="panel-{slug}" data-sym="{slug}" style="display:none;">
  <div class="tf-tabs">{tf_tabs}</div>
  {tf_panels}
</div>"""

TF_PANEL_TEMPLATE = """<div class="tf-panel" id="tfpanel-{slug}-{tf}" style="display:none;">
  {stats_html}
  {body}
</div>"""


def _status_banner(summary: dict) -> str:
    latest = summary["latest"]
    pos = summary.get("position")
    if latest["red_dot"]:
        note = " &mdash; sell signal on the latest bar" + (" (closes the open position)" if pos else " (no open position to close)")
        cls, text = "signal", f"&#128721; Red dot {latest['date']} @ {latest['close']:.5g}{note}"
    elif latest["green_dot"]:
        note = " &mdash; buy signal on the latest bar" + (" (already in position)" if pos else "")
        cls, text = "signal", f"&#128994; Green dot {latest['date']} @ {latest['close']:.5g}{note}"
    elif pos:
        stop_txt = f", stop {pos['stop']:.5g}" if pos["stop"] is not None else ""
        text = f"In position since {pos['entry_t'][:10]} @ {pos['entry_px']:.5g}{stop_txt} &mdash; watching for the next red dot"
        cls = "open"
    else:
        text = "No open position &mdash; watching for the next green dot"
        cls = "flat"
    return f'<div class="status {cls}">{text}</div>'


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
    return f'{_status_banner(summary)}<div class="stats">{stats}</div>{warn}'


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    with _cache_lock:
        snapshot = dict(_cache)

    symbol_tabs, symbol_panels = [], []
    for symbol in SYMBOLS:
        slug = _slug(symbol)
        symbol_tabs.append(
            f'<button class="tab-btn" id="tab-{slug}" onclick="showSymbol(\'{slug}\')">{html.escape(symbol)}</button>'
        )

        tf_tabs, tf_panels = [], []
        for tf in TIMEFRAMES:
            label = TF_LABELS.get(tf, tf)
            tf_tabs.append(
                f'<button class="tf-btn" id="tftab-{slug}-{tf}" data-tf="{tf}" '
                f'onclick="showTf(\'{slug}\',\'{tf}\')">{html.escape(label)}</button>'
            )
            entry = snapshot.get((symbol, tf))
            body = entry["html"] if entry and entry.get("html") else "<div class='err'>Loading&hellip;</div>"
            stats_html = _stats_row(entry.get("summary") if entry else None, entry.get("error") if entry else None)
            tf_panels.append(TF_PANEL_TEMPLATE.format(slug=slug, tf=tf, stats_html=stats_html, body=body))

        symbol_panels.append(SYMBOL_PANEL_TEMPLATE.format(
            slug=slug, tf_tabs="".join(tf_tabs), tf_panels="\n".join(tf_panels),
        ))

    return PAGE_TEMPLATE.format(
        refresh_seconds=REFRESH_SECONDS, refresh_min=REFRESH_SECONDS // 60,
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        symbol_tabs="".join(symbol_tabs), symbol_panels="\n".join(symbol_panels),
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
