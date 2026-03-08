"""
NSE Trading Platform — VWAP (Volume Weighted Average Price) Indicator

Computes intraday VWAP with configurable standard deviation bands.
Resets daily at market open.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
    if not candles or len(candles) < 1:
        return None

    # ── Accumulate VWAP components ────────────────────────────────────
    cum_tp_vol: float = 0.0     # Σ(typical_price × volume)
    cum_volume: int = 0         # Σ(volume)
    cum_tp_sq_vol: float = 0.0  # Σ(typical_price² × volume), for std-dev

    # We also track per-candle VWAP values to compute slope
    vwap_history: list[float] = []

    for candle in candles:
        tp = candle.typical_price  # (H + L + C) / 3
        vol = candle.volume

        if vol <= 0:
            # Zero-volume candle — carry forward VWAP, skip accumulation
            vwap_history.append(vwap_history[-1] if vwap_history else tp)
            continue

        cum_tp_vol += tp * vol
        cum_volume += vol
        cum_tp_sq_vol += (tp * tp) * vol

        current_vwap = cum_tp_vol / cum_volume
        vwap_history.append(current_vwap)

    if cum_volume <= 0:
        return None

    # ── Final VWAP ────────────────────────────────────────────────────
    vwap = cum_tp_vol / cum_volume

    # ── Standard Deviation ────────────────────────────────────────────
    # Variance = Σ(tp² × vol) / Σ(vol) - vwap²
    # This is the volume-weighted variance of typical price around VWAP
    variance = (cum_tp_sq_vol / cum_volume) - (vwap * vwap)
    std_dev = math.sqrt(max(variance, 0.0))

    # ── Bands ─────────────────────────────────────────────────────────
    upper_band = vwap + (band_multiplier * std_dev)
    lower_band = vwap - (band_multiplier * std_dev)

    # ── Slope ─────────────────────────────────────────────────────────
    # Slope = (current_vwap - vwap_N_candles_ago) / N
    # Positive slope → VWAP rising, negative → falling
    slope = 0.0
    if len(vwap_history) >= 2:
        lookback = min(slope_lookback, len(vwap_history) - 1)
        if lookback > 0:
            slope = (vwap_history[-1] - vwap_history[-1 - lookback]) / lookback

    return VWAPData(
        vwap=round(vwap, 2),
        upper_band=round(upper_band, 2),
        lower_band=round(lower_band, 2),
        std_dev=round(std_dev, 4),
        slope=round(slope, 4),
        cum_volume=cum_volume,
    )


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
    if df.empty:
        df["vwap"] = pd.Series(dtype="float64")
        df["vwap_upper"] = pd.Series(dtype="float64")
        df["vwap_lower"] = pd.Series(dtype="float64")
        return df

    # Work on a copy so we don't mutate the caller's DataFrame
    df = df.copy()

    # ── Typical price ─────────────────────────────────────────────────
    tp = (df["high"] + df["low"] + df["close"]) / 3.0

    # ── Cumulative sums ───────────────────────────────────────────────
    tp_vol = tp * df["volume"]
    cum_tp_vol = tp_vol.cumsum()
    cum_volume = df["volume"].cumsum().replace(0, np.nan)  # avoid div-by-zero

    # ── VWAP line ─────────────────────────────────────────────────────
    df["vwap"] = (cum_tp_vol / cum_volume).round(2)

    # ── Standard deviation (volume-weighted) ──────────────────────────
    tp_sq_vol = (tp * tp) * df["volume"]
    cum_tp_sq_vol = tp_sq_vol.cumsum()

    variance = (cum_tp_sq_vol / cum_volume) - (df["vwap"] ** 2)
    # Clamp to zero to avoid sqrt of negative due to float precision
    std_dev = np.sqrt(variance.clip(lower=0.0))

    # ── Bands ─────────────────────────────────────────────────────────
    df["vwap_upper"] = (df["vwap"] + band_multiplier * std_dev).round(2)
    df["vwap_lower"] = (df["vwap"] - band_multiplier * std_dev).round(2)

    return df
