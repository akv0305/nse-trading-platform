# NSE Trading Platform

A modular algorithmic trading platform for NSE (National Stock Exchange of India)
built with Python 3.11+, designed for multi-strategy live trading, backtesting,
and AI-assisted pre-market analysis.

## Strategies

| # | Strategy | Type | Status |
|---|----------|------|--------|
| 1 | ORB + VWAP Breakout | Intraday Equity | Development |
| 2 | Supply/Demand Zone Trading | Swing / Positional | Planned |
| 3 | Claude LLM Pre-Market Analyst | Decision Support | Phase 4 |

## Architecture

nse-trading-platform/ ├── config/ # Pydantic settings, .env loader ├── core/ │ ├── broker/ # Fyers adapter, paper adapter, base class │ ├── data/ # Models, stock universe, historical data, live candles │ ├── indicators/ # ORB, VWAP, zone detector, sector score │ ├── scanner/ # Stock scanner, sector analyzer │ ├── strategies/ # Strategy base class + implementations │ ├── risk/ # Risk manager, position sizing │ ├── execution/ # Order manager, trade plan builder │ ├── llm/ # Claude LLM analyst (Phase 4) │ └── utils/ # Time, IDs, logging (DBWriter + JSONL) ├── backtest/ # Backtest engine, cost model, performance metrics ├── engine/ # AsyncIO orchestrator, FastAPI control API ├── dashboard/ # Streamlit multi-tab dashboard ├── storage/ # SQLite DB, schema.sql, logs/ ├── scripts/ # Utility scripts (download history, run backtest) ├── tests/ # Pytest test suite └── docs/ # Strategy docs, API docs

## Tech Stack

- **Language**: Python 3.11+
- **Broker API**: Fyers API v3 (WebSocket + REST)
- **Database**: SQLite with WAL mode (concurrent read/write)
- **API Server**: FastAPI + Uvicorn (engine control)
- **Dashboard**: Streamlit + Plotly
- **Config**: Pydantic Settings + python-dotenv
- **LLM**: Anthropic Claude API (Phase 4)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/nse-trading-platform.git
cd nse-trading-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your Fyers credentials and preferences

# 5. Initialize database
python -c "from core.utils.logger import DBWriter; DBWriter()"

# 6. Run backtest (once strategies are implemented)
python scripts/run_backtest.py --strategy orb_vwap --start 2024-01-01 --end 2025-01-01

# 7. Start live engine
python engine/run_engine.py

# 8. Start dashboard (separate terminal)
streamlit run dashboard/app.py --server.port 8501