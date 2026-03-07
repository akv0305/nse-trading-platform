"""
NSE Trading Platform — Zone Quality Scorer

Scores supply/demand zones on a 0-5 scale based on:
  1. Departure strength
  2. Freshness (untested)
  3. Time decay
  4. Zone width
  5. Trend alignment

Implementation: Strategy 2 conversation.
"""

from __future__ import annotations

import pandas as pd

from core.data.models import Zone


def score_zone(
    zone: Zone,
    current_price: float,
    daily_df: pd.DataFrame,
    sma_period: int = 50,
    freshness_days: int = 20,
) -> float:
    """
    Compute composite quality score for a zone.

    Parameters
    ----------
    zone : Zone
        The zone to score.
    current_price : float
        Current market price of the stock.
    daily_df : pd.DataFrame
        Recent daily OHLCV data for trend analysis.
    sma_period : int
        SMA period for trend alignment scoring.
    freshness_days : int
        Zones older than this get a time decay penalty.

    Returns
    -------
    float
        Score from 0.0 to 5.0.
    """
    raise NotImplementedError("Zone scoring — implement in Strategy 2 conversation")


def filter_tradeable_zones(
    zones: list[Zone],
    min_score: float = 2.0,
    max_zones: int = 5,
) -> list[Zone]:
    """
    Filter zones to only tradeable ones.

    Parameters
    ----------
    zones : list[Zone]
        All detected zones (already scored).
    min_score : float
        Minimum score to be tradeable.
    max_zones : int
        Maximum number of zones to return.

    Returns
    -------
    list[Zone]
        Filtered and sorted zones.
    """
    raise NotImplementedError("Zone filtering — implement in Strategy 2 conversation")
