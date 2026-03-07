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
    raise NotImplementedError("ORB computation — implement in Strategy 1 conversation")


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
    raise NotImplementedError("Breakout detection — implement in Strategy 1 conversation")
