# NSE Trading Platform — Architecture Reference

## Repository
- Repo: nse-trading-platform (fresh build, not extending Nifty-Trader-Pro)
- Repo structure: config/, core/, backtest/, engine/, dashboard/, storage/, scripts/, tests/, docs/

## Tech Stack
- Python 3.11, FastAPI, Streamlit, SQLite WAL, Pydantic Settings, fyers-apiv3
- Deployment: Hostinger KVM2 VPS Mumbai (₹799/mo)
- IDE: Google Project IDX → GitHub → VPS

## Code Reused from Nifty-Trader-Pro
- FyersAdapter (WebSocket + REST fallback, reconnect watchdog)
- BrokerAdapter base class
- PaperAdapter (simulated fills with slippage)
- Time utilities, ID generators
- DBWriter pattern (rewritten for new schema)

## Strategies
| # | Name | Type | Status |
|---|------|------|--------|
| 1 | ORB + VWAP Breakout | Intraday Equity | To build |
| 2 | Supply/Demand Zones | Swing/Positional | To build |
| 3 | Claude LLM Pre-Market Analyst | Decision support | Phase 4 |
| (shelved) | Pair Trading | Nifty/BankNifty | From old repo, not working |
| (shelved) | Options Butterfly | Positional | From old repo, messed up |
| (active) | Overnight Gap | Intraday | Paper trading in old repo |
| (data collection) | Straddle Sell | Based on conditions | Gathering data |

## Hybrid LLM Approach (Approved)
- Primary decision engine: Deterministic scoring + optional LightGBM
- Claude LLM role: Pre-market analyst + approval gate (not real-time execution)
- LLM runs once daily before market open
- Cost: ~₹50-100/month on Claude Sonnet API
- Config: LLM_ENABLED=false by default, enable in Phase 4

## Risk Architecture
- Per-trade: 0.5% of ₹7.5L = ₹3,750
- Daily loss limit: 2% = ₹15,000 (triggers kill switch)
- Per-strategy daily: ₹6,000
- Max concurrent: 3 positions
- Auto-flatten: 15:22 IST
- No entries after: 14:45 IST
- Paper trading mandatory before live

## Development Phases
1. Scaffold + backtest engine (current)
2. Strategy 1 (ORB+VWAP) + backtest validation
3. Live paper trading + dashboard
4. Claude LLM pre-market analyst
5. Strategy 2 (S/D Zones)
6. Full auto-trading on VPS after 30-day paper profit

## Database
- SQLite WAL mode, 18 tables
- Key tables: trade_plans, signals, orders, trades, positions, risk_events
- All timestamps: epoch milliseconds (IST)

## Conversation Plan
- Current conversation: Scaffold building (batches)
- Strategy 1 conversation: ORB+VWAP implementation + backtest (attach strategy1_orb_vwap.md)
- Strategy 2 conversation: S/D Zones implementation (attach strategy2_sd_zones.md)
- Always attach platform_architecture.md to any new conversation
