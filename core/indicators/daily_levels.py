"""
Daily Support / Resistance Levels
=================================
Computes key daily levels from intraday candle history
for higher-timeframe confluence filtering.

Levels computed:
  - Previous day's high / low
  - 5-day high / low (weekly swing)
  - 20-day high / low (monthly swing)

A pin bar that forms near one of these levels has higher
probability of reversal (confluence with intraday VWAP band).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyLevels:
    """Key support/resistance levels from daily data."""
    prev_day_high: float
    prev_day_low: float
    week_high: float       # 5-day high
    week_low: float        # 5-day low
    month_high: float      # 20-day high
    month_low: float       # 20-day low
    prev_day_close: float


def compute_daily_levels_from_intraday(
    candle_history: list,
    current_day_str: str,
    lookback_short: int = 5,
    lookback_long: int = 20,
) -> Optional[DailyLevels]:
    """
    Compute daily S/R levels by aggregating intraday candles into
    daily bars, then extracting highs/lows from completed days.

    Parameters
    ----------
    candle_history : list[Candle]
        All intraday candles for this symbol (across multiple days),
        sorted by timestamp ascending.
    current_day_str : str
        Today's date as 'YYYY-MM-DD'. Only days BEFORE this are used.
    lookback_short : int
        Days for weekly high/low (default 5).
    lookback_long : int
        Days for monthly high/low (default 20).

    Returns
    -------
    DailyLevels or None
        None if fewer than 1 completed prior day.
    """
    from core.utils.time_utils import epoch_ms_to_ist

    # Group candles by date
    daily_bars: Dict[str, Dict[str, float]] = {}

    for c in candle_history:
        try:
            dt = epoch_ms_to_ist(c.timestamp)
            day = dt.strftime("%Y-%m-%d")
        except Exception:
            continue

        # Only include completed days (before today)
        if day >= current_day_str:
            continue

        if day not in daily_bars:
            daily_bars[day] = {
                "high": c.high,
                "low": c.low,
                "close": c.close,  # will be overwritten by last candle
            }
        else:
            bar = daily_bars[day]
            bar["high"] = max(bar["high"], c.high)
            bar["low"] = min(bar["low"], c.low)
            bar["close"] = c.close  # last candle's close = day close

    if not daily_bars:
        return None

    # Sort by date
    sorted_days = sorted(daily_bars.keys())

    if len(sorted_days) < 1:
        return None

    prev_day = daily_bars[sorted_days[-1]]

    # Short-term (5-day)
    short_days = sorted_days[-lookback_short:]
    week_high = max(daily_bars[d]["high"] for d in short_days)
    week_low = min(daily_bars[d]["low"] for d in short_days)

    # Long-term (20-day)
    long_days = sorted_days[-lookback_long:]
    month_high = max(daily_bars[d]["high"] for d in long_days)
    month_low = min(daily_bars[d]["low"] for d in long_days)

    return DailyLevels(
        prev_day_high=prev_day["high"],
        prev_day_low=prev_day["low"],
        week_high=week_high,
        week_low=week_low,
        month_high=month_high,
        month_low=month_low,
        prev_day_close=prev_day["close"],
    )


def is_near_support(
    price: float,
    levels: DailyLevels,
    buffer_pct: float = 0.5,
) -> bool:
    """
    Check if price is near a daily support level.

    For LONG entries: pin bar low should be near a support level.

    Parameters
    ----------
    price : float
        The pin bar's low price.
    levels : DailyLevels
        Daily support/resistance levels.
    buffer_pct : float
        How close price must be to a level (as % of price). Default 0.5%.

    Returns
    -------
    bool
        True if price is near any support level.
    """
    if price <= 0:
        return False

    buffer = price * (buffer_pct / 100.0)

    support_levels = [
        levels.prev_day_low,
        levels.week_low,
        levels.month_low,
    ]

    for level in support_levels:
        if abs(price - level) <= buffer:
            return True

    return False


def is_near_resistance(
    price: float,
    levels: DailyLevels,
    buffer_pct: float = 0.5,
) -> bool:
    """
    Check if price is near a daily resistance level.

    For SHORT entries: pin bar high should be near a resistance level.

    Parameters
    ----------
    price : float
        The pin bar's high price.
    levels : DailyLevels
        Daily support/resistance levels.
    buffer_pct : float
        How close price must be to a level (as % of price). Default 0.5%.

    Returns
    -------
    bool
        True if price is near any resistance level.
    """
    if price <= 0:
        return False

    buffer = price * (buffer_pct / 100.0)

    resistance_levels = [
        levels.prev_day_high,
        levels.week_high,
        levels.month_high,
    ]

    for level in resistance_levels:
        if abs(price - level) <= buffer:
            return True

    return False
