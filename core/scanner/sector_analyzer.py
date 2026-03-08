"""
NSE Trading Platform — Sector Rotation Analyzer

Analyzes sector index performance to identify sector rotation
patterns and momentum shifts.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_sector_rotation(
    sector_data: dict[str, pd.DataFrame],
    nifty_data: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
) -> dict[str, dict]:
    """
    Analyze sector rotation and momentum.

    Uses the Relative Rotation Graph (RRG) concept:
      - Relative Strength (RS): sector return vs Nifty over long_period
      - RS-Momentum: rate of change of RS over short_period
      - Rotation phase determined by (RS, RS-Momentum) quadrant

    Quadrants (like RRG):
      - LEADING:   RS > 0 and momentum > 0  (outperforming & accelerating)
      - WEAKENING: RS > 0 and momentum <= 0  (outperforming but slowing)
      - LAGGING:   RS <= 0 and momentum <= 0 (underperforming & decelerating)
      - IMPROVING: RS <= 0 and momentum > 0  (underperforming but accelerating)

    Parameters
    ----------
    sector_data : dict[str, pd.DataFrame]
        Sector index → daily OHLCV.
    nifty_data : pd.DataFrame
        Nifty 50 daily OHLCV.
    short_period : int
        Short-term lookback for momentum.
    long_period : int
        Long-term lookback for trend.

    Returns
    -------
    dict[str, dict]
        Sector → {'momentum': float, 'trend': str, 'relative_strength': float,
                   'rotation_phase': str}
        rotation_phase: 'IMPROVING' | 'LEADING' | 'WEAKENING' | 'LAGGING'
    """
    results: dict[str, dict] = {}

    # ── Compute Nifty returns ─────────────────────────────────────────
    nifty_long_ret = _compute_return(nifty_data, long_period)
    nifty_short_ret = _compute_return(nifty_data, short_period)

    if nifty_long_ret is None or nifty_short_ret is None:
        # Insufficient benchmark data — return neutral for all
        for sector_name in sector_data:
            results[sector_name] = {
                "momentum": 0.0,
                "trend": "NEUTRAL",
                "relative_strength": 0.0,
                "rotation_phase": "LAGGING",
            }
        return results

    for sector_name, df in sector_data.items():
        # ── Relative Strength: long-term outperformance ───────────
        sector_long_ret = _compute_return(df, long_period)
        if sector_long_ret is None:
            results[sector_name] = {
                "momentum": 0.0,
                "trend": "NEUTRAL",
                "relative_strength": 0.0,
                "rotation_phase": "LAGGING",
            }
            continue

        rs = sector_long_ret - nifty_long_ret  # positive = outperforming

        # ── RS-Momentum: short-term change in relative strength ───
        sector_short_ret = _compute_return(df, short_period)
        if sector_short_ret is None:
            momentum = 0.0
        else:
            # Momentum = short-term RS - portion of long-term RS
            # This captures the *change* in relative performance
            short_rs = sector_short_ret - nifty_short_ret
            momentum = short_rs  # positive = RS improving

        # ── Trend label ───────────────────────────────────────────
        if sector_long_ret > 1.0:
            trend = "BULLISH"
        elif sector_long_ret < -1.0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # ── Rotation Phase (RRG quadrants) ────────────────────────
        if rs > 0 and momentum > 0:
            rotation_phase = "LEADING"
        elif rs > 0 and momentum <= 0:
            rotation_phase = "WEAKENING"
        elif rs <= 0 and momentum > 0:
            rotation_phase = "IMPROVING"
        else:
            rotation_phase = "LAGGING"

        results[sector_name] = {
            "momentum": round(momentum, 4),
            "trend": trend,
            "relative_strength": round(rs, 4),
            "rotation_phase": rotation_phase,
        }

    return results


# ── Private Helpers ───────────────────────────────────────────────────────

def _compute_return(df: pd.DataFrame, lookback_days: int) -> float | None:
    """
    Compute percentage return over the last `lookback_days` rows.

    Returns
    -------
    float or None
        Percentage return (e.g. 2.5 means +2.5%), or None if insufficient data.
    """
    if df is None or df.empty:
        return None

    if "close" not in df.columns:
        return None

    closes = df["close"].dropna()

    if len(closes) < 2:
        return None

    if len(closes) < lookback_days + 1:
        lookback_days = len(closes) - 1

    current_close = float(closes.iloc[-1])
    past_close = float(closes.iloc[-1 - lookback_days])

    if past_close == 0:
        return None

    return ((current_close - past_close) / past_close) * 100.0
