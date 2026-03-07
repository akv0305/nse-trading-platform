"""
NSE Trading Platform — VWAP (Volume Weighted Average Price) Indicator

Computes intraday VWAP with configurable standard deviation bands.
Resets daily at market open.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data.models import Candle


@dataclass(frozen=True, slots=True)
class VWAPData:
    """VWAP calculation result at a point in time."""
    vwap: float              # Current VWAP value
    upper_band: float        # VWAP + (multiplier × std_dev)
    lower_band: float        # VWAP - (multiplier × std_dev)
    std_dev: float           # Rolling standard deviation
    slope: float             # VWAP slope (positive = rising, negative = falling)
    cum_volume: int          # Cumulative volume used in calculation


def compute_vwap(
    candles: list[Candle],
    band_multiplier: float = 1.5,
    slope_lookback: int = 5,
) -> VWAPData | None:
    """
    Compute VWAP and bands from intraday candles.

    Parameters
    ----------
    candles : list[Candle]
        Today's intraday candles, sorted by timestamp ascending.
        Must start from market open (9:15 IST).
    band_multiplier : float
        Standard deviation multiplier for bands.
    slope_lookback : int
        Number of candles to look back for slope calculation.

    Returns
    -------
    VWAPData or None
        None if insufficient data.
    """
    raise NotImplementedError("VWAP computation — implement in Strategy 1 conversation")


def compute_vwap_series(
    df: pd.DataFrame,
    band_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Compute VWAP as a pandas Series added to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: high, low, close, volume.
        Should be single-day intraday data.
    band_multiplier : float

    Returns
    -------
    pd.DataFrame
        Original df with added columns: vwap, vwap_upper, vwap_lower.
    """
    raise NotImplementedError("VWAP series — implement in Strategy 1 conversation")
