"""
NSE Trading Platform — Supply/Demand Zone Detector

Identifies supply and demand zones from daily OHLCV data
using the Base-Departure pattern.

Implementation: Strategy 2 conversation.
"""

from __future__ import annotations

import pandas as pd

from core.data.models import Zone


def detect_zones(
    df: pd.DataFrame,
    lookback_days: int = 60,
    min_departure_ratio: float = 1.5,
    max_base_candles: int = 4,
    max_zone_width_pct: float = 3.0,
) -> list[Zone]:
    """
    Detect supply and demand zones from daily OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data with columns: open, high, low, close, volume.
        Should cover at least lookback_days of history.
    lookback_days : int
        How many days back to scan for zones.
    min_departure_ratio : float
        Minimum ratio of departure candle body to average body for zone validity.
    max_base_candles : int
        Maximum number of consolidation candles in the base.
    max_zone_width_pct : float
        Maximum zone width as % of price. Zones wider than this are discarded.

    Returns
    -------
    list[Zone]
        Detected zones sorted by strength_score descending.
    """
    raise NotImplementedError("Zone detection — implement in Strategy 2 conversation")
