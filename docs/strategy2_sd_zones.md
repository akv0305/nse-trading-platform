# Strategy 2: Supply/Demand Zone Trading (Swing / Positional)

## Overview
Identify institutional supply and demand zones from price structure,
score them for quality, and trade reversals when price revisits fresh zones.
Suitable for swing trades (2-10 days) on Nifty 50 stocks.

## Core Logic

### Phase 1: Zone Detection (Daily scan after market close)
- Scan last 60 days of daily OHLCV data for each Nifty 50 stock
- Identify zones using the "Base-Departure" pattern:
  - **Demand Zone (buying):**
    1. Price drops into a consolidation area (1-4 small-body candles)
    2. Followed by a strong bullish departure (large green candle)
    3. Zone = low of consolidation to high of consolidation
  - **Supply Zone (selling):**
    1. Price rallies into a consolidation area (1-4 small-body candles)
    2. Followed by a strong bearish departure (large red candle)
    3. Zone = low of consolidation to high of consolidation

### Phase 2: Zone Scoring (0 to 5 scale)
Each zone gets a composite score based on:
1. **Departure Strength (0-1):** How fast/strong price left the zone
   - Measured by: departure candle body size / average candle body size
2. **Freshness (0-1):** Has price returned to test the zone?
   - Fresh (never tested) = 1.0
   - Tested once = 0.5
   - Tested twice+ = 0.0 (zone is spent)
3. **Time Decay (0-1):** How old is the zone?
   - < 10 days = 1.0
   - 10-20 days = 0.7
   - 20-40 days = 0.4
   - > 40 days = 0.1
4. **Zone Width (0-1):** Tighter zones are better
   - < 1% of price = 1.0
   - 1-2% = 0.7
   - 2-3% = 0.3
   - > 3% = 0.0 (too wide, skip)
5. **Trend Alignment (0-1):** Is the zone in the direction of the trend?
   - 50-day SMA direction matches zone type = 1.0
   - Counter-trend = 0.3

- **Total Score = sum of all 5 components (max 5.0)**
- **Minimum tradeable score: 2.0 (configurable)**

### Phase 3: Entry Signal
- **LONG (Demand Zone):**
  1. Price drops into a fresh demand zone (touches zone_high)
  2. Zone score ≥ SD_MIN_ZONE_STRENGTH (2.0)
  3. Confirmation: bullish candle pattern at zone (e.g., hammer, engulfing)
     OR price bounces off zone and closes above zone_high
  4. R:R ≥ SD_RISK_REWARD_MIN (2.0) — target = nearest supply zone above
  5. Risk manager approves

- **SHORT (Supply Zone):**
  1. Price rallies into a fresh supply zone (touches zone_low)
  2. Zone score ≥ SD_MIN_ZONE_STRENGTH
  3. Confirmation: bearish pattern or price rejects and closes below zone_low
  4. R:R ≥ SD_RISK_REWARD_MIN — target = nearest demand zone below
  5. Risk manager approves

### Phase 4: Risk Management & Exit
- **Stoploss:** Beyond the far edge of the zone + buffer
  - Long: SL = zone_low - (0.2% of price)
  - Short: SL = zone_high + (0.2% of price)

- **Target:** Nearest opposing zone
  - Long at demand zone → target = nearest supply zone above
  - Short at supply zone → target = nearest demand zone below

- **Position Sizing:** Same as Strategy 1
  - Quantity = RISK_PER_TRADE_INR / |entry - SL|

- **Exit Rules:**
  - Target hit
  - Stoploss hit
  - Zone violation (price closes beyond SL level for 2 consecutive candles)
  - Time-based: exit after 10 trading days if neither target nor SL hit
  - Manual kill switch

### Phase 5: Sector Filter
- Only trade stocks in sectors showing relative strength (for longs)
  or relative weakness (for shorts)
- Sector score based on sector index performance vs Nifty 50

## Indicators Required
1. **Zone Detector** (core/indicators/zone_detector.py): Find S/D zones
2. **Zone Scorer** (core/indicators/zone_scorer.py): Score zones 0-5
3. **SMA** (50-day): For trend alignment scoring
4. **ATR**: For position sizing and zone width context
5. **Sector Score**: For sector filter

## Candle Timeframe
- Zone detection: Daily candles (1D)
- Entry trigger: 15-minute or 1-hour candles for timing
- Exit monitoring: Daily candles

## Backtest Parameters
- Universe: Nifty 50
- Period: 2024-01-01 to 2025-01-01
- Capital: ₹7,50,000
- Risk per trade: 0.5% = ₹3,750
- Max concurrent positions: 3
- Max holding period: 10 trading days
- Include all costs (Fyers delivery brokerage for swing)

## Expected Metrics
- Win rate: 50-65% (higher than breakout due to zone quality filter)
- Avg R:R: 2.0-3.0
- Profit factor: >1.8
- Avg holding: 3-7 days
- Trades per month: 5-15

## Config Parameters (from settings.py)
- SD_LOOKBACK_DAYS = 60
- SD_MIN_ZONE_STRENGTH = 2.0
- SD_ZONE_FRESHNESS_DAYS = 20
- SD_RISK_REWARD_MIN = 2.0
- SD_MAX_ZONE_WIDTH_PCT = 3.0
