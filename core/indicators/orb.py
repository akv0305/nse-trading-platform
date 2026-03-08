"""
NSE Trading Platform — Opening Range Breakout (ORB) Indicator

Captures the Opening Range (first N minutes after market open)
and provides breakout detection signals.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data.models import Candle
from core.utils.time_utils import IST, epoch_ms_to_ist


@dataclass(frozen=True, slots=True)
class ORBLevels:
    """Opening Range levels for a symbol on a given day."""
    symbol: str
    date: str                # YYYY-MM-DD
    or_high: float           # Highest price during OR period
    or_low: float            # Lowest price during OR period
    or_mid: float            # Midpoint
    or_width: float          # Absolute width (high - low)
    or_width_pct: float      # Width as % of midpoint
    is_valid: bool           # True if OR was captured cleanly
    captured_at_ms: int      # Epoch ms when OR was finalized


def compute_opening_range(
    candles: list[Candle],
    or_period_minutes: int = 15,
    market_open_time: str = "09:15",
) -> ORBLevels | None:
    """
    Compute Opening Range from a list of intraday candles.

    Parameters
    ----------
    candles : list[Candle]
        Intraday candles (1m or 5m), sorted by timestamp ascending.
        Must include candles from market open through OR period end.
    or_period_minutes : int
        Duration of opening range in minutes (15 or 30).
    market_open_time : str
        Market open time in 'HH:MM' format.

    Returns
    -------
    ORBLevels or None
        None if insufficient data to compute OR.
    """
    if not candles:
        return None

    # ── Parse market open time ────────────────────────────────────────
    open_h, open_m = map(int, market_open_time.split(":"))

    # ── Filter candles that fall within the OR period ─────────────────
    # OR period: [market_open, market_open + or_period_minutes)
    # For 5m candles with 15-min OR: candles opening at 9:15, 9:20, 9:25
    # The candle at 9:25 covers 9:25–9:30, so its OPEN time is still
    # within the 15-min window (9:15 to 9:30 exclusive of candle open).
    or_candles: list[Candle] = []

    for candle in candles:
        candle_dt = epoch_ms_to_ist(candle.timestamp)
        candle_h = candle_dt.hour
        candle_m = candle_dt.minute

        # Convert to minutes-since-midnight for easy comparison
        candle_minutes = candle_h * 60 + candle_m
        open_minutes = open_h * 60 + open_m
        or_end_minutes = open_minutes + or_period_minutes

        # Include candles whose open time is >= market_open and < OR end
        if open_minutes <= candle_minutes < or_end_minutes:
            or_candles.append(candle)

    if not or_candles:
        return None

    # ── Compute OR high/low from selected candles ─────────────────────
    or_high = max(c.high for c in or_candles)
    or_low = min(c.low for c in or_candles)
    or_mid = round((or_high + or_low) / 2.0, 2)
    or_width = round(or_high - or_low, 2)

    # Width as percentage of midpoint
    or_width_pct = round((or_width / or_mid) * 100.0, 4) if or_mid > 0 else 0.0

    # ── Validity check ────────────────────────────────────────────────
    # OR is valid if we have at least the expected number of candles.
    # For 5m candles: 15-min OR needs 3 candles, 30-min OR needs 6.
    # For 1m candles: 15-min OR needs 15 candles, etc.
    # We also check that OR width is reasonable (0.3%–2.0% of price).
    first_candle = or_candles[0]
    candle_dt = epoch_ms_to_ist(first_candle.timestamp)

    # Detect timeframe from candle metadata
    tf = first_candle.timeframe
    if tf == "5m":
        expected_candles = or_period_minutes // 5
    elif tf == "1m":
        expected_candles = or_period_minutes
    elif tf == "15m":
        expected_candles = or_period_minutes // 15
    else:
        # Default: just need at least 1 candle
        expected_candles = 1

    has_enough_candles = len(or_candles) >= expected_candles
    width_reasonable = 0.3 <= or_width_pct <= 2.0

    is_valid = has_enough_candles and or_width > 0

    # Determine the symbol and date from first candle
    symbol = first_candle.symbol
    date_str = candle_dt.strftime("%Y-%m-%d")

    # captured_at_ms is the timestamp of the last OR candle's close
    # (i.e., the end of the OR period)
    captured_at_ms = or_candles[-1].timestamp

    return ORBLevels(
        symbol=symbol,
        date=date_str,
        or_high=round(or_high, 2),
        or_low=round(or_low, 2),
        or_mid=or_mid,
        or_width=or_width,
        or_width_pct=round(or_width_pct, 4),
        is_valid=is_valid,
        captured_at_ms=captured_at_ms,
    )


def detect_breakout(
    current_price: float,
    orb_levels: ORBLevels,
    buffer_pct: float = 0.05,
) -> str:
    """
    Check if current price has broken out of the Opening Range.

    Parameters
    ----------
    current_price : float
    orb_levels : ORBLevels
    buffer_pct : float
        Buffer percentage above/below OR for breakout confirmation.

    Returns
    -------
    str
        'BREAKOUT_LONG' | 'BREAKOUT_SHORT' | 'INSIDE_RANGE'
    """
    if not orb_levels.is_valid:
        return "INSIDE_RANGE"

    # ── Compute breakout thresholds with buffer ───────────────────────
    # Buffer is a percentage of the OR high/low themselves
    long_threshold = orb_levels.or_high * (1.0 + buffer_pct / 100.0)
    short_threshold = orb_levels.or_low * (1.0 - buffer_pct / 100.0)

    if current_price > long_threshold:
        return "BREAKOUT_LONG"
    elif current_price < short_threshold:
        return "BREAKOUT_SHORT"
    else:
        return "INSIDE_RANGE"
