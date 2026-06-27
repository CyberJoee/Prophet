# Prophet — Autonomous Paper Trading Agent

Multi-agent paper trading system for US equities and options using Claude as the reasoning engine.

## Architecture

```
Data Layer      → Alpaca (live) / Mock (testing)
Agent Layer     → Research → Strategy → Risk → Journal (all Claude-powered)
Memory Layer    → PostgreSQL + pgvector (semantic trade similarity)
Execution       → Alpaca Paper Trading API
Dashboard       → React/Vite (Phase 2)
```

## Stack

- **Python 3.12** — FastAPI backend
- **PostgreSQL 16 + pgvector** — trade journal and vector memory
- **Alpaca Markets** — paper trading (free account at alpaca.markets)
- **Claude API** — multi-agent reasoning core
- **pandas-ta** — technical indicators
- **APScheduler** — market session scheduling (Phase 2)

## Quick Start

### 1. Prerequisites

```bash
# PostgreSQL 16 + pgvector
brew install postgresql@16
pip install pgvector  # or apt-get install postgresql-16-pgvector

# Python deps
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in: ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY
```

Get your free Alpaca paper trading keys at: https://alpaca.markets

### 3. Database setup

```bash
createuser prophet -P            # password: prophet
createdb prophet -O prophet
psql -d prophet -c "CREATE EXTENSION vector;"
python3 scripts/init_db.py       # creates tables + seeds watchlist
```

### 4. Run tests (no API keys needed)

```bash
python3 -m pytest tests/test_suite.py -v
# Expected: 25 passed
```

### 5. Run in mock mode (no API keys needed)

```bash
# All data and execution are simulated
DATA_PROVIDER=mock EXECUTION_CLIENT=mock python3 scripts/run_once.py
```

### 6. Run with real Alpaca paper trading

```bash
# Set DATA_PROVIDER=alpaca EXECUTION_CLIENT=alpaca in .env
python3 scripts/run_once.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Alpaca paper trading key |
| `ALPACA_SECRET_KEY` | — | Alpaca paper trading secret |
| `ALPACA_BASE_URL` | paper-api.alpaca.markets | Use paper URL always |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `DATABASE_URL` | postgresql://prophet:prophet@localhost:5432/prophet | Postgres connection |
| `DATA_PROVIDER` | `mock` | `mock` or `alpaca` |
| `EXECUTION_CLIENT` | `mock` | `mock` or `alpaca` |

## Project Structure

```
prophet/
├── agents/
│   ├── research_agent.py    # Morning scan — calls Claude for market briefing
│   └── strategy_agent.py    # Trade planning — calls Claude for entry decisions
├── data/
│   └── market_data.py       # DataProvider: MockDataProvider + AlpacaDataProvider
├── db/
│   ├── connection.py        # SQLAlchemy engine + session
│   ├── models.py            # All ORM models (trades, journal, stats, decisions)
│   └── operations.py        # CRUD operations for all agents
├── execution/
│   └── broker.py            # ExecutionClient: MockExecutionClient + AlpacaExecutionClient
├── tests/
│   └── test_suite.py        # 25 tests covering all components
├── scripts/
│   ├── init_db.py           # One-time DB setup
│   └── run_once.py          # Single full pipeline run (for testing)
└── .env.example
```

## Phase Roadmap

- [x] **Phase 1** — Data, DB schema, mock providers, research + strategy agents, 25 tests
- [ ] **Phase 2** — APScheduler (9:45am scan, 10am trade, 4:15pm journal), position monitor with trailing stops
- [ ] **Phase 3** — Journal agent (post-trade analysis), pgvector embeddings for trade memory
- [ ] **Phase 4** — Backtesting engine (replay historical data through agents)
- [ ] **Phase 5** — React/Vite dashboard (live P&L, reasoning log, equity curve)

## Switching to Live Alpaca Paper Trading

1. Sign up at https://alpaca.markets (free)
2. Get your paper trading API keys from the dashboard
3. Add to `.env`:
   ```
   ALPACA_API_KEY=PKxxxxx
   ALPACA_SECRET_KEY=xxxxx
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   DATA_PROVIDER=alpaca
   EXECUTION_CLIENT=alpaca
   ```
4. Run: `python3 scripts/run_once.py`

**Note:** Paper trading uses real market data but no real money. Safe to run during market hours.

## Risk Controls (always active)

- Max 2% portfolio risk per trade (position sized by stop distance)
- Max 15% of equity per position (prevents over-concentration)
- Limit orders only (no market orders)
- No trading outside market hours
- All decisions logged to `agent_decisions` table for review
