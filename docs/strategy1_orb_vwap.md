# Strategy 1: ORB + VWAP Breakout (Intraday Equity)

## Overview
Opening Range Breakout combined with VWAP confirmation for intraday
equity trades on Nifty 50 stocks.

## Core Logic

### Phase 1: Pre-Market Scan (8:50 - 9:15 IST)
- Scan Nifty 50 stocks for sector momentum (previous day performance)
- Identify top N stocks (default 10) with:
  - High relative volume (pre-market auction data if available)
  - Sector showing strength/weakness (sector index % change)
  - Stock near key support/resistance levels
- Claude LLM pre-market analyst (Phase 4) provides market bias

### Phase 2: Opening Range Capture (9:15 - 9:30 IST)
- Capture the Opening Range (OR) for shortlisted stocks
- OR = first 15 minutes (configurable: 15 or 30 min)
- Record OR_HIGH, OR_LOW, OR_MID for each stock
- Calculate OR width as % of price — skip if too wide (>2%) or too narrow (<0.3%)

### Phase 3: Breakout Detection (9:30 onwards)
- Monitor price relative to OR levels and VWAP
- **LONG Entry Conditions (ALL must be true):**
  1. Price closes above OR_HIGH + buffer (0.05%)
  2. Price is above VWAP
  3. VWAP is sloping upward (VWAP now > VWAP 5 candles ago)
  4. Volume on breakout candle > 1.5× average volume of OR period
  5. Sector index for this stock is positive for the day
  6. Time is before ENTRY_CUTOFF (14:45 IST)
  7. Risk manager approves (daily limit, position limit)

- **SHORT Entry Conditions (ALL must be true):**
  1. Price closes below OR_LOW - buffer (0.05%)
  2. Price is below VWAP
  3. VWAP is sloping downward
  4. Volume on breakout candle > 1.5× average
  5. Sector index for this stock is negative for the day
  6. Time is before ENTRY_CUTOFF
  7. Risk manager approves

### Phase 4: Risk Management & Exit
- **Stoploss Modes:**
  - ORB_OPPOSITE (default): SL at opposite end of OR
    - Long entry → SL = OR_LOW
    - Short entry → SL = OR_HIGH
  - ATR_BASED: SL = entry ± 1.5 × ATR(14) on 5-min candles
  - FIXED_PCT: SL = entry ± fixed % (default 0.5%)

- **Position Sizing:**
  - Quantity = RISK_PER_TRADE_INR / |entry_price - stoploss_price|
  - Rounded down to valid lot size
  - Capped by MAX_CONCURRENT_POSITIONS

- **Target:**
  - Default: R:R ratio of 2.0 (i.e., target = entry + 2 × risk)
  - Can be set to VWAP upper/lower band as target

- **Trailing Stop:**
  - Activates after 1.0 R:R achieved
  - Trails at 30% of move from entry
  - Example: Entry 100, SL 98 (risk=2), price reaches 102 (1R)
    → Trail activates, SL moves to 100 + 0.3×2 = 100.60
    → Price reaches 104 → SL = 100 + 0.3×4 = 101.20

- **Time Exit:**
  - Force flatten all positions at FLATTEN_TIME (15:22 IST)
  - No new entries after ENTRY_CUTOFF (14:45 IST)

### Phase 5: VWAP Calculation
- Standard VWAP: cumulative(typical_price × volume) / cumulative(volume)
- Typical price = (H + L + C) / 3
- Reset daily at 9:15
- VWAP Bands: VWAP ± (multiplier × rolling std-dev of typical_price)
- Default band multiplier: 1.5

## Indicators Required
1. **ORB** (core/indicators/orb.py): OR_HIGH, OR_LOW, OR_MID, OR_WIDTH
2. **VWAP** (core/indicators/vwap.py): VWAP line, upper band, lower band, slope
3. **Volume Profile**: average volume during OR, breakout volume ratio
4. **ATR**: 14-period ATR on 5-min candles (for ATR-based SL mode)
5. **Sector Score** (core/indicators/sector_score.py): sector index % change

## Candle Timeframe
- Primary: 5-minute candles for signal generation
- OR period: captured from 1-minute candles aggregated to OR_HIGH/LOW
- VWAP: calculated on every 1-minute or 5-minute candle

## Backtest Parameters
- Universe: Nifty 50
- Period: 2024-01-01 to 2025-01-01
- Capital: ₹7,50,000
- Risk per trade: 0.5% = ₹3,750
- Max concurrent positions: 3
- Include all costs (Fyers brokerage model)
- Skip: first 3 minutes after open, holidays, expiry days (optional)

## Expected Metrics (to validate)
- Win rate: 40-55% (typical for breakout strategies)
- Avg R:R: 1.5-2.5
- Profit factor: >1.5
- Max drawdown: <10%
- Avg trades per day: 1-3

## Config Parameters (from settings.py)
- ORB_PERIOD_MINUTES = 15
- ORB_BREAKOUT_BUFFER_PCT = 0.05
- VWAP_BAND_STD_MULTIPLIER = 1.5
- ORB_STOPLOSS_MODE = "ORB_OPPOSITE"
- ORB_TARGET_RR = 2.0
- ORB_TRAIL_AFTER_RR = 1.0
- ORB_TRAIL_PCT = 0.3
- ENTRY_CUTOFF_IST = "14:45"
- FLATTEN_TIME_IST = "15:22"
