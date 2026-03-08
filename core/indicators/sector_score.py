"""
NSE Trading Platform — Sector Strength Score

Computes relative strength/weakness of sectors based on
sector index performance vs Nifty 50.

Used by both ORB+VWAP (sector filter) and S/D Zones (trend alignment).

Implementation: Strategy 1 conversation (shared indicator).
"""

from __future__ import annotations

import pandas as pd

from core.data.universe import SECTOR_MAP


def compute_sector_scores(
    sector_data: dict[str, pd.DataFrame],
    nifty_data: pd.DataFrame,
    lookback_days: int = 5,
) -> dict[str, float]:
    """
    Compute relative strength score for each sector.

    The score measures how much each sector index outperformed or
    underperformed Nifty 50 over the lookback period.

    Score = sector_return% - nifty_return%  (normalized to -1..+1 range)

    Parameters
    ----------
    sector_data : dict[str, pd.DataFrame]
        Sector index name → daily OHLCV DataFrame.
        DataFrame must have columns: close (at minimum).
        Rows sorted by date ascending.
    nifty_data : pd.DataFrame
        Nifty 50 index daily OHLCV.
    lookback_days : int
        Period for relative strength calculation.

    Returns
    -------
    dict[str, float]
        Sector name → score (-1.0 to +1.0).
        Positive = outperforming Nifty, Negative = underperforming.
    """
    scores: dict[str, float] = {}

    # ── Nifty benchmark return over lookback ──────────────────────────
    nifty_return = _compute_return(nifty_data, lookback_days)
    if nifty_return is None:
        # If we can't compute Nifty return, return 0.0 for all sectors
        return {sector: 0.0 for sector in sector_data}

    # ── Compute raw relative strength for each sector ─────────────────
    raw_scores: dict[str, float] = {}
    for sector_name, df in sector_data.items():
        sector_return = _compute_return(df, lookback_days)
        if sector_return is None:
            raw_scores[sector_name] = 0.0
            continue
        # Relative strength = sector return minus benchmark return
        raw_scores[sector_name] = sector_return - nifty_return

    # ── Normalize to -1.0 .. +1.0 ────────────────────────────────────
    if not raw_scores:
        return scores

    max_abs = max(abs(v) for v in raw_scores.values()) if raw_scores else 1.0
    if max_abs == 0:
        max_abs = 1.0  # Avoid division by zero

    for sector_name, raw in raw_scores.items():
        normalized = raw / max_abs
        # Clamp to [-1.0, 1.0] for safety
        normalized = max(-1.0, min(1.0, normalized))
        scores[sector_name] = round(normalized, 4)

    return scores


def get_stock_sector_bias(
    symbol: str,
    sector_scores: dict[str, float],
) -> float:
    """
    Get the sector bias for a specific stock.

    Parameters
    ----------
    symbol : str
        Plain symbol (e.g. 'RELIANCE').
    sector_scores : dict[str, float]
        Sector → score mapping from compute_sector_scores().

    Returns
    -------
    float
        Sector score for this stock's sector. 0.0 if unknown.
    """
    # ── Look up the stock's sector from SECTOR_MAP ────────────────────
    sector = SECTOR_MAP.get(symbol)

    if sector is None:
        return 0.0

    # ── Map sector name to the sector_scores keys ─────────────────────
    # sector_scores keys are sector index names like "NIFTY BANK",
    # "NIFTY IT", etc.  SECTOR_MAP values are like "Banking", "IT", etc.
    # We need to map between the two naming conventions.
    sector_to_index: dict[str, str] = {
        "Banking": "NIFTY BANK",
        "IT": "NIFTY IT",
        "Pharma": "NIFTY PHARMA",
        "Auto": "NIFTY AUTO",
        "FMCG": "NIFTY FMCG",
        "Metals & Mining": "NIFTY METAL",
        "Energy": "NIFTY ENERGY",
        "Infrastructure": "NIFTY INFRA",
        "Financial Services": "NIFTY FIN SERVICE",
        "Insurance": "NIFTY FIN SERVICE",  # Insurance falls under fin services
        "Telecom": "NIFTY INFRA",          # No dedicated telecom index, closest proxy
        "Consumer": "NIFTY FMCG",          # Consumer mapped to FMCG as proxy
        "Cement": "NIFTY INFRA",           # Cement mapped to Infra as proxy
        "Conglomerate": "NIFTY 50",        # Conglomerate (Reliance) — use broad market
        "Healthcare": "NIFTY PHARMA",      # Healthcare mapped to Pharma
    }

    index_name = sector_to_index.get(sector)

    if index_name is None:
        return 0.0

    # ── Direct match in sector_scores ─────────────────────────────────
    if index_name in sector_scores:
        return sector_scores[index_name]

    # ── Try case-insensitive / partial match ──────────────────────────
    index_name_lower = index_name.lower()
    for key, score in sector_scores.items():
        if key.lower() == index_name_lower:
            return score

    return 0.0


# ── Private Helpers ───────────────────────────────────────────────────────

def _compute_return(df: pd.DataFrame, lookback_days: int) -> float | None:
    """
    Compute percentage return over the last `lookback_days` rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'close' column. Rows sorted by date ascending.
    lookback_days : int

    Returns
    -------
    float or None
        Percentage return (e.g. 2.5 for +2.5%), or None if insufficient data.
    """
    if df is None or df.empty:
        return None

    if "close" not in df.columns:
        return None

    closes = df["close"].dropna()

    if len(closes) < lookback_days + 1:
        # Not enough data — use whatever is available if at least 2 rows
        if len(closes) < 2:
            return None
        lookback_days = len(closes) - 1

    current_close = closes.iloc[-1]
    past_close = closes.iloc[-1 - lookback_days]

    if past_close == 0:
        return None

    return ((current_close - past_close) / past_close) * 100.0
