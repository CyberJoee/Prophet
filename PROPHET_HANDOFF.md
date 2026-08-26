# Prophet — Project Handoff

**Repo:** `github.com/CyberJoee/Prophet` (branch: `main`, auto-deploys to Railway on push)
**Railway project:** `courteous-rebirth` (id: `69f947d8-57ab-428f-92b5-78d1edefdfa7`)
**Environment:** `production` (id: `e4d500e2-a9dc-4d06-af11-83f8a09c3621`)

Autonomous paper-trading agent. Stack: Python/FastAPI, Postgres + pgvector, SQLAlchemy,
APScheduler, Alpaca (paper trading + market data), Groq (LLM inference), yfinance, FINRA
daily short-volume files, Polymarket public API. Dashboard is a separate service
(`calm-vibrancy`, vanilla JS) at the public domain.

This doc exists so a fresh Claude Code session (or future me) doesn't have to
rediscover the last several weeks of debugging. Read it before touching code.

---

## Railway topology

| Service | ID | Role |
|---|---|---|
| `prophet` | `0700c534-3f9f-4044-afb7-9004d269fadb` | the worker — scheduler, all agents |
| `Postgres` | `5b83d843-228d-4aef-aeeb-16e561c760f1` | DB, pgvector enabled |
| `calm-vibrancy` | `787f5fc0-49f5-4833-b8e3-e8f721662821` | public dashboard, domain `unlimitedprophet.up.railway.app` |

A **Railway MCP connector** is available in Claude with tools for logs, deploy
status, service config, and env vars — verified working. Note `list-variables`
returns secrets in plaintext and may be blocked by the permission classifier;
`get-service-config` returns variable *names* only and is usually enough.

**Working clone + push access (as of 2026-08-21).** The repo is cloned at
`Desktop/claude projects/Prophet/repo` and `git push` to `CyberJoee/Prophet`
works (credentials cached in Windows Credential Manager; there is no `gh` CLI).
Claude should commit and push directly — no more "Add files via upload" through
the web UI. **Pushing to `main` auto-deploys BOTH services**, so push when the
market is closed.

**Both Railway services build from this same repo and branch**, differing only in
start command: `prophet` runs `python scripts/scheduler.py`, `calm-vibrancy` runs
`uvicorn dashboard.api:app`. A change to `dashboard/api.py` redeploys the worker
too, and vice versa.

**Postgres has no query tool via MCP** — SQL still requires the Railway dashboard's
Console tab (`psql -U postgres railway`), and maintenance scripts must be run from
that Console. TCP public proxy was intentionally removed (was getting scanned);
Prophet talks to Postgres over `postgres.railway.internal` only. Claude cannot reach
the DB directly — anything needing SQL has to be a script a human runs in the Console.

**No local dev environment.** None of the runtime deps (sqlalchemy, alpaca-py,
dotenv) are installed on the Windows box. Tests meant to run locally must be
self-contained: see `tests/test_alpaca_sync_dedup.py` for the pattern — in-memory
SQLite, `@compiles` shims for the Postgres-only column types (UUID, Vector), and a
stubbed `alpaca` module.

---

## Schedule (all times ET, APScheduler in `scripts/scheduler.py`)

- **9:45 AM** — morning pipeline: regime gate → alt-signal collection → earnings guard
  → research agent → strategy agent → order placement → fill-confirmation pass
- **Every 5 min during market hours** — position monitor (reconciliation on Alpaca,
  price-check on mock)
- **4:15 PM** — end of day: cancel all open orders → confirm last-second fills →
  close remaining positions → snapshot → refresh strategy stats → journal agent

One full decision cycle per day. No intraday news/event reaction currently exists.

---

## Timeline of major fixes (read this before assuming anything "just works")

### v2 — execution lifecycle rebuild
The original system recorded a trade as OPEN the instant Alpaca *accepted* an
order (not filled). Combined with bracket orders being submitted **without
`order_class=BRACKET`** (a bug — Alpaca silently doesn't attach exit legs
without it), the system had **zero real stop-loss/take-profit protection**
for its entire pre-fix history. 67 early trades were corrupted phantom data:
0% target-hit rate, everything resolved via 5-min polling or EOD close.

Fixed: `TradeStatus.PENDING_FILL` added; `execution/order_tracker.py` confirms
real fills before promoting to OPEN; `execution/position_monitor.py` rewritten
as pure reconciliation against real bracket fills (no competing exit logic);
`execution/sizing.py` takes all arithmetic (entry/stop/target/qty) OUT of the
LLM and into deterministic code, validated with pydantic.

### v3 — gates + honest backtest
Added `agents/regime.py` (SPY trend/vol gate, sizes down or halts on
extreme vol / downtrend+vol), `agents/earnings_guard.py` (blocks entries
within 2 days of earnings via yfinance, fails open), and
`backtesting/engine_v2.py` — a real event-driven intraday backtest that
replays 5-min bars and runs the **actual** `sizing.py` code path (not a
reimplementation). Validated against synthetic no-edge data first (proved
~zero expectancy on structureless noise, confirming no look-ahead bias)
before trusting it on real data.

**Key finding from the honest backtest, since confirmed live:** ORB, VWAP
reclaim, and momentum setups on the megacap watchlist have **~zero gross
edge** before costs — net negative after spread/slippage. Exit distribution:
~65% of trades drift to EOD close, ~27% stop out, **only ~7% hit target**.
This is a geometry mismatch (1.0×ATR target rarely achievable from a 9:45
entry in six hours), not just "bad luck." Live data through mid-August
confirms the same pattern: still ~0 target hits across dozens of real
trades. **The walk-forward geometry sweep (test wider stops/closer
targets/time-exits, fit/validate split) is a planned-but-not-yet-built next
step** — see Open Threads below.

### v4 — alt-data signal layer
Added `data/alt_signals/` — FINRA short-volume ratios, options
positioning (C/P skew, unusual volume, ATM IV vs baseline), Polymarket
macro event-risk gate (halves size on imminent uncertain FOMC/CPI-type
events). All stored in `alt_signals` table, injected into the LLM's
prompt as context only — **still not sized on, and still should not be.**
`scripts/eval_signals.py` computes the IC of each stored signal against
forward returns.

**SUPERSEDED BY v7 (Aug 26).** The eval has since been run and every signal
came back noise, and the |IC| > 0.05 / n > 100 bar quoted here was itself
wrong — at n=100 that is about half a standard error, so it would have
promoted noise into live sizing. Judge on |t| >= 2 against the
cross-sectional IC. See v7 for the results and the pre-registered hypothesis
for the next run.

News/sentiment cleanup rode along in this batch: added an 18-hour
freshness window to Alpaca's news fetch, and replaced a dead
`sentiment_label` (always `None` — Alpaca's feed has no sentiment) with
real article-age labels (`"2h ago"`) in the research prompt.

### Post-v4 bug hunts (the ATR outage)
Five straight trading days (~July 7–14) with **zero trades placed** despite
the pipeline running cleanly end-to-end. Root cause, found by reading
`agent_decisions.reasoning` then the worker logs: `fetch_latest_bar()` was
requesting only 5 calendar days of history — not enough bars to compute
14/20/26-period indicators (ATR, RSI, MACD, Bollinger all came back `None`),
so `sizing.py` skipped every symbol with "no ATR." **Fixed:** bumped to 60
calendar days. This also meant the research LLM had been receiving `None`
for every technical indicator since v1 — likely worse decision quality the
whole time, invisibly.

Also fixed in the same pass: ATM IV baseline poisoning (yfinance returns
~0 IV on thin/pre-open chains; storing zeros produced absurd
"`+252400% vs baseline`" log lines) and noisy 404 spam from asking
yfinance for ETF (SPY/QQQ) earnings dates (ETFs don't have earnings).

### The Groq model deprecation (found Aug 21 — likely broken since mid-June)
Groq deprecated `llama-3.3-70b-versatile` (announced 2026-06-17). Every call
in `agents/llm_client.py` and the separately-hardcoded one in
`agents/journal_agent.py` started 404ing. **The scheduler's fallback design
was the real problem**: on an LLM error it silently substituted a templated
**mock** decision path and kept trading — meaning for roughly two months, no
real LLM reasoning happened, all alt-signal context was computed and
ignored, and journal lessons stopped generating, all without any visible
error in the trade history (mock trades look like normal `custom`-setup
trades on the dashboard).

**Fixed:** `GROQ_MODEL` env var (default `openai/gpt-oss-120b`, Groq's
recommended replacement) makes future model swaps a one-line config change.
More importantly: **the scheduler no longer silently mocks on a genuine LLM
failure** — it logs a loud `[ALERT]` banner, records an `llm_unavailable`
decision, and aborts the day's trading entirely. Mock mode is now reserved
for "no API key configured" only, never "the configured LLM errored."
**This class of bug (silent fallback masking total system failure) is the
single most important lesson from this project — always fail loud, never
fail quiet, when the failure means synthetic data will be recorded as real.**

### v5 — data integrity: sync duplicates + dashboard accounting (Aug 21)
Three separate defects, all found chasing the "+$4,543 dashboard vs +$1,710 real
equity" gap. The gap turned out **not** to be one bug, and not mainly the one this
doc previously blamed.

**1. `alpaca_sync` duplicate imports (the one already known).**
`_sync_closed_orders` deduped against the **exit (sell)** order id, but
natively-tracked trades store the **entry (buy)** order id in
`Trade.alpaca_order_id` (set in `strategy_agent.py` when the bracket is
submitted). Those keys can never match, so every real closed trade was
re-imported once per sync run. Fixed: match the entry leg *before* the dedup
check, and dedup on both legs' order ids plus a symbol/qty/entry/exit content
match as a fallback for legacy rows carrying no order id at all.

**2. An `UnboundLocalError` hiding in the same function.** The OPEN-trade branch
read `entry_price` before assignment. It raised whenever an OPEN record existed
for a symbol with a closed Alpaca order — and the outer `try/except` swallowed it
into a bland `"Alpaca sync failed"`. Same fail-quiet shape as the Groq
deprecation. Also fixed: matched buy fills are now consumed, so two sells of one
symbol can no longer pair to the same buy (a second duplication path).

**3. The dashboard was measuring two different things and calling both
`total_pnl`.** `/api/portfolio` derives it from real Alpaca equity
(`equity - 100_000`). `/api/performance` summed `t.pnl` over **every** closed
trade with no date filter — including the 66 corrupted pre-v2 trades that
`StrategyStats` deliberately excludes. The same response was self-inconsistent:
`summary` counted everything while `by_setup` was cutoff-filtered. **This, not the
duplicates, was the bulk of the gap.** Fixed by extracting
`db.operations.get_stats_cutoff()` as the single source of truth and pointing both
call sites at it; the endpoint now also returns `stats_since` and
`excluded_pre_cutoff_trades` so the number describes its own scope.

**The two problems pushed in opposite directions and partly cancelled**, which is
exactly why the gap looked like one tidy ~$2,833 duplication issue. The 13
duplicate rows were net **−$1,721.89**, so removing them *raised* reported P&L.
A hypothesis that explains a discrepancy's size but not its sign is not yet a
diagnosis.

**Post-cleanup state (2026-08-21):** 153 → 140 closed trades; 74 post-cutoff
(+$2,033.91) + 66 pre-cutoff = 140. Account equity $103,038.01 (+$3,038.01), 0
open positions. The residual ~$1,004 between the two endpoints is **expected and
correct** — roughly the real P&L the account earned before the cutoff, which
equity includes by definition and `/api/performance` now deliberately excludes.
These two numbers should never be expected to match.

Do not read the +$2,033.91 as evidence the agent works: most of those 74 trades
were placed during the Groq outage on templated mock decisions, not LLM reasoning.
If anything it is mild evidence the LLM is not the value-add.

### v6 — the learning loop (Aug 21)
Prompted by the question "is the bot actually learning at all?" — investigated,
and the answer was **no**. Five independent reasons, any one sufficient:

1. **No actuator.** Nothing outside the prompt consumed outcomes. Grep confirmed
   no file in `execution/` or either gate referenced expectancy, win_rate, or
   profit_factor. `StrategyStats`' only non-dashboard consumer was
   `memory.get_strategy_performance_summary()`, which formats it as *prose for a
   prompt*. No parameter anywhere was a function of past P&L.
2. **Lessons carried no information.** "Be more mindful of the overall market
   context." "Continuously monitor and adjust." Not one named a price level, a
   time of day, or a number. Retrieval worked fine — it retrieved vacuum.
3. **Quality scores did not discriminate.** 75 of 101 journal entries scored 8
   or 9 on entry quality; 28 of the 68 *losing* trades scored 8+.
4. **The one metric that looked like learning was an artifact.** Winners
   averaged 7.84 entry quality vs 6.94 for losers — apparently real
   discrimination. It is circular: `_build_journal_prompt` puts `PnL: $X`
   directly in the prompt, so the LLM grades the entry already knowing the
   outcome. That correlation would appear just as strongly in a system with
   zero predictive ability. **Beware any metric computed with hindsight in the
   context window.**
5. **For ~2 months nothing read the lessons anyway** — the Groq outage meant
   journals were written and retrieved into a mock decision path that ignored
   them. The journal agent then broke too: the 2 newest entries (Aug 17) were
   the hardcoded template fallback with fabricated 5/10 scores.

**Built in response — `agents/expectancy_gate.py`**, the first mechanism that
changes behaviour from outcomes with no LLM in the path. Per setup type it
computes realised expectancy and suspends or halves the losers.

- **Metric is R**, not dollars: `pnl / (|entry - planned_stop| * qty)`. Raw P&L
  would penalise a setup for having traded during a half-size regime week.
  Trades with no recoverable stop are excluded **and counted** — a systematic
  gap shows up rather than silently shrinking the sample.
- **A negative mean is not evidence.** SUSPENDED requires `mean + Z*stderr < 0`
  (Z=1.0). Negative-but-noisy gets REDUCED to half size. Tunable via env.
- **Self-lockout is the trap in this design.** A suspended setup places no
  trades, generates no data, and a naive gate could never un-suspend it. The
  window is bounded by TIME as well as count, so losses age out and the setup
  returns to probation on its own. The verdict is a pure function of
  (history, today) — no stored state, no migration, idempotent.
- **Enforced in code**, not the prompt. The LLM is shown the verdict so it does
  not waste picks, but `filter_suspended_picks()` drops them regardless —
  tested against real `LLMPick` objects with conviction 1.0.
- **Fails OPEN**, but logs `[ALERT]` and populates `errors`.

`agents/journal_agent.py` now **fails loud**: factual record kept, every
judgement field NULL, `decision_type="journal_unavailable"`. It no longer
manufactures analysis that looks real.

**Still not done, and deliberately so:** the entry-quality score is still
graded with P&L visible, so it remains hindsight-contaminated and should not
be trusted or used as a signal. Structured/falsifiable lesson fields are also
not built. See Open Threads.

### v7 — the week the answers all came back negative (Aug 24-26)

**Mon Aug 24 — the Groq fix is confirmed.** `[LIVE]` on both research and
strategy for the first time since roughly mid-June. Open thread #2 closed.
One trade (SPY short, reversal setup, 15 sh) closed at EOD for -$19.35.
Everything fired: regime gate, expectancy gate (`momentum reduced`), earnings
guard (blocked NVDA ahead of its 8/26 report), partial fill handled correctly
(planned 20, filled 15, recorded at the real fill).

**The 2% risk rule has never been in effect.** Found while reading that
trade's sizing. `RISK_PER_TRADE_PCT` (2%) and `MAX_POSITION_PCT` (15%) are
each sensible and in direct conflict: the risk rule only binds when
`stop_distance >= price * (0.02/0.15) = price * 13.3%`. Stops here are
0.5x ATR — about 0.4-1.5% of price. **The notional cap has bound on every
trade in the project's history**, verified against seven live fills and all
44 trades of a synthetic backtest. Real risk per trade is ~0.05-0.2%, not 2%.

The cap is not the bug — it is load-bearing. 2% risk on a 0.4% stop implies
~4.6x notional leverage (620 SPY shares = $472k against $412k buying power);
the order would be rejected. Two consequences worth remembering: the
expectancy gate's REDUCED tier is **inert**, because halving a risk-derived
quantity of 620 still lands above a cap of 20 (SUSPENDED still works, since
it drops the pick before sizing); and at $10k equity the cap collapses to
1 share of SPY, where integer truncation dominates everything.

Deliberately **no behaviour change** — sizing multiplies expectancy, and
expectancy is negative. `build_trade_plan` now logs which constraint bound
on every decision, and `size_report()` answers "what would sizing do at $X".

**Geometry sweep: definitive negative.** `backtesting/geometry_sweep.py`,
48 cells over a year of real 5-min bars. **All 48 negative in both the fit
and validate windows** — 96 measurements, not one positive. Best cell
-0.024R fit / -0.035R validate over 231 out-of-sample trades, negative by
more than two standard errors. The ranking is pure cost drag: sort by stop
width and the table falls into perfect blocks, because a fixed dollar cost
divided by a larger R denominator is smaller. Target choice barely registers
(-0.029 to -0.034 across all four targets at stop 1.5) because targets are
almost never reached.

**Entries appear to be worse than random.** Real target-hit rate vs the
random-walk control, holding to EOD: 0.5x ATR 24.2% vs 41.0%; 1.0x ATR
**4.8% vs 18.0%**; 1.5x ATR 0.4% vs 6.6%. Worse at every distance. The
synthetic generator is not a calibrated null so treat the magnitude softly,
but the internal number needs no control: at the live 1.0x target, 4.8% of
trades reach it and 88% drift to the close. **The bracket orders are
decorative** — outcomes are decided by where price happens to be at 4pm.

**Wed Aug 26 — a truncated LLM response cost a trading day.** The pipeline
aborted with `Unterminated string starting at line 39` and the message "fix
GROQ_MODEL / API key". Both were fine. gpt-oss-120b is a reasoning model
whose internal tokens count against `max_tokens`, and at the old 1500 budget
the research JSON was cut off mid-string. Tuesday had run clean, so it is
intermittent — it depends on how much the model reasons. Fixed: budget 3000
(`LLM_MAX_TOKENS`), truncation detected from `finish_reason` AND from the
shape of the parse failure, retried once at double the budget, and split
into `LLMTruncated` (retryable) vs `LLMBadJSON` (more tokens will not help).

**First signal evaluation: noise, and underpowered.** `eval_signals.py` had
never been run. Auditing it before first use found three problems that would
each have made the output uninterpretable — see the Lessons section. After
fixing them, all five signals came back noise on 37 dates x 10 symbols:

    signal                      IC      t(intraday)  t(daily)
    unusual_contracts        -0.097      -1.93        -0.79
    unusual_call_bias        -0.081      -1.17        -1.38
    cp_volume_ratio          -0.028      -0.57        +0.03
    short_volume_ratio       +0.021      +0.38        -0.37
    atm_iv                   -0.012      -0.18        -0.85

**But "nothing cleared" is mostly a statement about sample size.** Observed
standard errors (0.049-0.069) match theory: with ~10 names the null spread
of a cross-sectional IC is ~0.33, so over 37 dates se ~0.055 and the
smallest resolvable |IC| is ~0.11 — far above the 0.02-0.05 that real
alt-signals carry. Detecting IC=0.05 by waiting alone would take ~7 months.

**Response: widen the collection universe.** IC noise falls as
`1/sqrt(n-1)` in the number of NAMES as well as with more dates, so ~45
symbols roughly halves the standard error — about a 4x speedup in calendar
time. Collection and briefing are now separate lists, because they want
opposite things: collect as wide as possible (one FINRA file covers every
symbol; options flow fails open per name), brief as narrow as possible (the
prompt is scarce, as Aug 26 demonstrated, and the LLM can only trade the
watchlist anyway). Trading behaviour unchanged.

**PRE-REGISTERED HYPOTHESIS**, recorded 2026-08-26 before more data arrived
so the next run is a test and not a fishing trip:
`options_flow.unusual_contracts`, **negative**, **intraday** horizon. It was
the largest |t| (-1.93) and held its sign across two non-overlapping
horizons. Re-run after ~3 months; require the sign to hold. With five
signals tested there is a ~23% chance one crosses t=2 by luck alone.

**Where this leaves the project.** Four things are now ruled out with real
evidence: exit geometry (48 configurations), the LLM's contribution, entry
quality, and alt-signals as currently measured. **Prophet has no identified
edge.** The infrastructure, however, is now the opposite of the problem —
it took minutes to rule each of these out properly, where most of this
project's history was fake positives that survived for months.

---

## Architecture map (where things live)

```
agents/
  strategy_agent.py    LLM picks setups (judgment only); sizing.py does all math
  research_agent.py     morning market scan + news; builds the LLM briefing
  regime.py             SPY trend/vol gate — sizes/halts in code, LLM can't override
  expectancy_gate.py    per-setup go/no-go from realised R. THE learning loop:
                         the only thing that changes behaviour from outcomes
                         without an LLM. Stateless — verdict is a pure function
                         of (trade history, today). Enforced in strategy_agent
                         via filter_suspended_picks(), not merely prompted.
  earnings_guard.py     blocks entries near earnings (yfinance, fails open)
  memory.py             same-symbol/same-setup trade retrieval (no embeddings —
                         see note below)
  journal_agent.py      per-trade LLM lesson after close
  llm_client.py         shared Groq call wrapper — GROQ_MODEL env var lives here

execution/
  sizing.py             ALL arithmetic: entry/stop/target/qty from live price
                         + ATR. pydantic-validates LLM output. NOTE the "2%
                         risk rule" is dead code — MAX_POSITION_PCT binds on
                         every realistic intraday stop (see v7). Logs which
                         constraint bound on each decision. size_report()
                         answers "what would sizing do at $X equity".
  broker.py              Alpaca + Mock execution clients (bracket orders fixed
                         in v2 — order_class=BRACKET is mandatory)
  order_tracker.py       confirms real fills before PENDING_FILL -> OPEN
  position_monitor.py    reconciliation (Alpaca) or price-check (mock) —
                         never both own exits at once
  alpaca_sync.py         imports Alpaca's own records into DB. Dedup fixed
                         in v5 — checks BOTH legs' order ids. Still assumes
                         long-only: hardcodes side=BUY and computes
                         pnl=(sell-entry)*qty, so a short would get an
                         inverted P&L.

data/
  market_data.py         Alpaca/yfinance/mock data providers, indicators
                         (60-day lookback minimum — do not reduce this)
  live_price.py           batched, cached live quote fetching
  alt_signals/
    options_flow.py       C/P ratio, unusual volume, ATM IV vs baseline
    short_volume.py        FINRA daily short-volume ratio vs baseline
    event_risk.py           Polymarket macro event gate
    aggregator.py           orchestrates collectors, stores to alt_signals
                             table. COLLECTS on a wide universe
                             (get_signal_universe, ~45 names) but BRIEFS the
                             LLM on the trading watchlist only — those are
                             different jobs with opposite requirements.

backtesting/
  engine_v2.py            the honest one — event-driven, 5-min bars, real
                           sizing.py, spread+slippage, validated against
                           synthetic no-edge data. Bars are indexed by
                           (symbol, date) at construction — do not revert to
                           scanning, it made the sweep 70x slower.
  geometry_sweep.py       stop/target/time-exit grid with a chronological
                           fit/validate split. Ran 2026-08-26: all 48 cells
                           negative in both windows. Read the VERDICT, never
                           the top row — a grid search always has one.
  engine.py               OLD — daily-bar, hand-rolled rules, does NOT match
                           live system. Don't trust its output.

scripts/
  scheduler.py             APScheduler jobs, the morning/EOD pipeline glue
  run_backtest_v2.py        CLI for the honest backtest
  eval_signals.py            IC analysis: do alt-signals predict forward returns
  migrate_pending_status.py  one-time migration (already run)
  migrate_alt_signals.py     one-time migration (already run)
  eval_signals.py            cross-sectional IC of each alt-signal against
                              forward returns. Defaults to the INTRADAY
                              horizon (09:45 -> close) because that is the
                              only window Prophet holds. Judge on |t| >= 2,
                              never on IC alone.
  run_geometry_sweep.py      CLI for the geometry sweep; --synthetic runs the
                              no-edge control, which must always fail to find
                              an edge.
  check_expectancy.py        READ-ONLY preview of what the expectancy gate
                              would do against real data. Run before trusting
                              it live; accepts EXPECTANCY_* overrides inline.
  dedupe_trades.py           one-time cleanup for the alpaca_sync duplicate
                              rows. Dry-run by default, --apply to commit.
                              Already run 2026-08-21 (removed 13 rows).

tests/
  test_suite.py                 the original suite — needs a real Postgres
  test_alpaca_sync_dedup.py     runs the REAL _sync_closed_orders against
                                 in-memory SQLite with a stubbed Alpaca client.
                                 No Postgres, no alpaca-py. All five checks
                                 fail on the pre-fix code (verified).
  test_stats_cutoff.py          STATS_SINCE boundary / override / fallback
  test_expectancy_gate.py       51 checks on the expectancy gate, incl. the
                                 self-lockout release and enforcement against
                                 real pydantic LLMPick objects

db/
  models.py    Trade, TradeJournal, AltSignal, StrategyStats, etc.
  operations.py all DB writes go through here. get_stats_cutoff() is the
                SINGLE SOURCE OF TRUTH for the STATS_SINCE cutoff (default
                2026-07-02), which excludes the corrupted pre-v2 trades.
                Both refresh_strategy_stats() and dashboard/api.py call it —
                do not re-implement the cutoff a third time.

dashboard/
  api.py       FastAPI, deployed as the calm-vibrancy service. NOTE: it serves
                two different numbers both named total_pnl — /api/portfolio is
                all-time real account P&L, /api/performance is realized P&L
                since the cutoff. Correct, but they should be relabelled in the
                UI so the difference is legible.
```

---

## Environment variables of note

- `GROQ_API_KEY` — required
- `GROQ_MODEL` — set to `openai/gpt-oss-120b` on Railway (was hardcoded to a
  now-deprecated model; check `console.groq.com/docs/models` if this 404s
  again — Groq deprecates with only weeks of notice)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` — paper trading
- `DATABASE_URL` — Postgres, internal Railway hostname
- `EARNINGS_GUARD` — `on`/`off`, default on
- `STATS_SINCE` — date cutoff for strategy stats, default `2026-07-02`
- `EXPECTANCY_GATE` — `on`/`off`, default on. Kill switch for the learning
  loop; takes effect on restart, no deploy needed.
- `EXPECTANCY_WINDOW` (30), `EXPECTANCY_MIN_TRADES` (20),
  `EXPECTANCY_LOOKBACK_DAYS` (60), `EXPECTANCY_Z` (1.0),
  `EXPECTANCY_REDUCED_SCALE` (0.5) — gate tuning. Raising Z makes suspension
  harder; LOOKBACK_DAYS also controls how fast a suspension expires.
- `LLM_MAX_TOKENS` — output budget for research/strategy calls, default 3000.
  Raise if `LLMTruncated` appears. `JOURNAL_MAX_TOKENS` (2000) is the same
  knob for the journal agent.
- `SIGNAL_UNIVERSE` — comma-separated symbols to COLLECT alt-signals on,
  default ~45 names. Set it to the watchlist to restore pre-Aug-26 narrow
  collection. Does not affect what is traded or what the LLM sees.

---

## Open threads / next steps, roughly in priority order

**Read this first.** As of 2026-08-26 the honest position is that **Prophet
has no identified edge**. Exit geometry, the LLM's contribution, entry
quality, and the alt-signal layer have each been measured and each came back
negative or unresolvable. The measurement apparatus is now trustworthy — that
is the actual asset. The open question is no longer "which parameter should
change" but "is there anything to find in this domain at all".

Three honest directions, none of them a parameter tweak:

  A. **Wait on the signals.** Collection is widened and free; re-run the eval
     in late November. Cheapest possible option, requires no decisions.
  B. **Change the domain.** Megacap intraday from a 9:45 entry is the most
     arbitraged corner of the market, and the results are what efficiency
     looks like. Different instruments, or a different holding period, is the
     only change large enough to matter.
  C. **Reduce trading while researching.** At ~-0.03R per trade the system
     pays a small fee for data it already has enough of. Not urgent at paper
     scale, but it is not earning anything either.

Do NOT respond to the above by raising risk. Sizing multiplies expectancy,
and expectancy is measurably negative — see v7.



1. ~~Fix the alpaca_sync duplicate bug~~ — **DONE 2026-08-21** (v5 above).
   Shipped, deployed, and the existing duplicate rows cleaned up.
2. ~~Confirm the Groq model fix restores live decisions~~ — **CONFIRMED
   2026-08-24.** `[LIVE]` on both research and strategy. Real LLM reasoning
   resumed after roughly two months of silent mock trading.
3. ~~Signal eval~~ — **RUN 2026-08-26. All five signals: noise.** No signal
   may influence sizing. But the study was underpowered (37 dates x 10 names
   resolves only |IC| >= 0.11), so this is "do not act", not "nothing is
   there". Collection was widened to ~45 names in response; **re-run around
   late November 2026**, when the combination of more dates and more names
   should resolve |IC| ~0.05. Test the pre-registered hypothesis
   (`unusual_contracts`, negative, intraday) rather than re-scanning all five.
4. ~~Geometry sweep~~ — **BUILT AND RUN 2026-08-26. Definitive negative.**
   All 48 stop/target/time-exit combinations negative in both windows over a
   year of real bars. Exits are not the problem; do not revisit this without
   a new reason. The finding that replaced it: entries reach their targets at
   roughly a quarter of the random-walk rate, which points at entry selection
   rather than exit geometry.
5. **Dashboard**: add a view for PENDING_FILL/CANCELLED trades (currently
   invisible — only CLOSED trades render), so "is it broken or just not
   filling" never again requires a psql session to answer.
6. Considered and deliberately **not built**: a web-search "news scout"
   agent (Tavily/Anthropic-search based) — concluded the marginal value
   over the existing Benzinga wire feed didn't justify the added
   complexity for saturated megacap names. EDGAR 8-K filing radar was
   floated as a higher-signal alternative (primary source, keyless) but
   also not yet built — parking here in case it's revisited.
7. ~~Bandit-style expectancy gate~~ — **BUILT 2026-08-21** (v6 above).
   Note the original reasoning for deferring it ("too few trades per setup")
   is handled inside the gate rather than by waiting: it simply produces no
   verdict below `EXPECTANCY_MIN_TRADES`, so it is safe to run while the
   sample is still small and starts acting on its own once it isn't.

8. **De-contaminate the journal's entry-quality score.** It is currently
   graded with the trade's P&L in the prompt, so it measures hindsight, not
   judgement, and its apparent correlation with outcomes is circular. Either
   withhold P&L for that specific question, or replace the score with
   something deterministic (MAE/ATR before target, time-to-stop). Until then
   do not use it as a signal and do not put it on the dashboard as one.

9. **Make journal lessons structured and falsifiable.** Free-text prose
   averages out to "be careful." Constrained fields — an enum of failure
   modes plus measured quantities — could be aggregated across trades
   instead of read one at a time, which is what would make memory retrieval
   worth more than the tokens it costs.

10. **Decide what the 2% risk rule should actually be.** It is currently
    documented-but-dead (v7). Either delete it and describe sizing honestly
    as "15% notional per position", or pick a reachable target. **Do not
    raise risk until something shows positive out-of-sample expectancy.**

11. **Fractional shares, for the $10k goal.** At $10k the notional cap gives
    1 share of SPY and 2 of QQQ; integer truncation dominates and decision
    quality becomes unmeasurable. Fractional sizing is the obvious fix, BUT
    Alpaca is believed not to support bracket/OCO legs on fractional orders —
    which would forfeit the real stop protection v2 was built to establish.
    **Verify that against current Alpaca docs before planning around it.**
    Unverified as of 2026-08-26.

12. **Migrate `journal_agent` onto `call_llm`.** It still makes its own Groq
    call, so it does not inherit the truncation retry or the
    LLMTruncated/LLMBadJSON split added on 2026-08-26. Its budget was raised
    to 2000 as a stopgap.

13. **Dashboard still shows two different numbers named `total_pnl`**
    (`/api/portfolio` = all-time account P&L, `/api/performance` = realised
    since the cutoff). Both correct, easy to confuse. Relabel in the UI.

---

## Hard-won lessons (apply these going forward)

- **Never trust a backtest until it's validated against synthetic
  structureless data first.** The original backtest used different rules
  than the live system and wasn't caught for a while. `engine_v2.py`'s
  synthetic no-edge test (near-zero expectancy on random walk data) is the
  standing bar for any future backtest change.
- **A silent fallback that produces plausible-looking fake data is worse
  than a crash.** Phantom trades (v1), zero-vector embeddings (v1),
  mock-trading-disguised-as-live (Groq deprecation) all share this shape.
  The fix is always the same: fail loud, log it as a distinct decision
  type, and stop rather than substitute.
- **Small samples lie confidently.** 14 trades, 67 trades — not enough to
  conclude much of anything about win rate, journal efficacy, or signal
  predictiveness. Wait for the numbers the eval framework calls for
  before trusting a pattern.
- **The dashboard can be wrong** (duplicate rows) while the DB and Alpaca's
  own account state are the source of truth. Cross-check big numbers
  against `broker.get_account()`'s equity when they look off.
- **Explain the sign, not just the size.** The "$2,833 phantom gap" was blamed
  on duplicate rows for weeks. The duplicates were real — but net *negative*,
  so removing them widened the gap. Two unrelated bugs had been partly
  cancelling. Before accepting a diagnosis, check that it predicts the
  direction of the error, not merely its rough magnitude.
- **Failing loud is necessary but not sufficient — the alarm must point at
  the right thing.** On Aug 26 the pipeline correctly refused to trade on a
  broken LLM call, then told the reader to check GROQ_MODEL and the API key.
  Both were healthy; the model had run out of output tokens. A confident
  wrong diagnosis costs as much time as no alarm, and is harder to doubt.
- **Audit a measurement tool BEFORE its first run, not after.**
  `eval_signals.py` sat unrun for weeks. Reading it first turned up three
  problems — it scored a horizon the bot never holds, pooled cross-sectional
  and time-series correlation so that market beta dominated, and used a
  significance threshold (|IC| > 0.05 at n=100, about half a standard error)
  that would have promoted noise into live sizing. Any of the three would
  have produced a confident, wrong number.
- **"Not significant" and "not there" are different claims.** All five
  signals came back noise, but the study could only resolve |IC| >= 0.11
  against a realistic effect size of 0.02-0.05. Before believing a null,
  compute what the test was actually capable of detecting.
- **Check the sample-size lever you are not pulling.** Statistical power for
  a cross-sectional IC scales with names as well as with dates. Waiting for
  more dates would have taken seven months; widening the universe got the
  same power in about two, and cost nothing.
- **A feedback loop with no actuator is not a feedback loop.** Prophet spent
  months computing performance stats, writing journal lessons, and retrieving
  them into prompts — with nothing downstream that could act on any of it.
  It looked like learning from the outside. Before believing a system adapts,
  find the line of code where an outcome changes a parameter. If there isn't
  one, it doesn't.
- **Distrust any score computed with the answer in the context window.** The
  journal's entry-quality scores correlated with P&L and looked like evidence
  the grader worked. The grader was shown the P&L. Always ask what the model
  could see when it produced the number.
- **Any gate that suspends a behaviour must define how the behaviour resumes.**
  Suspension removes the data that would justify un-suspending. Without an
  explicit escape — here, a time-bounded window — the first bad streak
  becomes permanent.
- **Two numbers with the same name will eventually be compared.** The gap
  survived this long because `total_pnl` meant "all-time account P&L" on one
  endpoint and "sum of every closed trade row" on another. Neither was wrong in
  isolation. When one quantity has two legitimate definitions, name them apart
  and make each report its own scope.
