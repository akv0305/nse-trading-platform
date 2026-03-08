"""
NSE Trading Platform — Stock Scanner

Scans the stock universe and shortlists candidates for each strategy.
Runs as part of pre_market_scan() in strategy implementations.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.universe import SECTOR_MAP, from_fyers_symbol
from core.indicators.sector_score import get_stock_sector_bias


def scan_for_orb(
    universe: list[str],
    historical_data: dict[str, pd.DataFrame],
    sector_scores: dict[str, float],
    top_n: int = 10,
) -> list[dict]:
    """
    Scan universe for ORB+VWAP candidates.

    Criteria:
      - High relative volume (last 5 days avg)
      - Stock in strong/weak sector (directional)
      - Adequate daily range (ATR > 1% of price)
      - Not in consolidation (some volatility)

    Parameters
    ----------
    universe : list[str]
        Fyers-format symbols.
    historical_data : dict[str, pd.DataFrame]
        Symbol → recent daily OHLCV.
    sector_scores : dict[str, float]
        Sector strength scores.
    top_n : int
        Max stocks to return.

    Returns
    -------
    list[dict]
        Ranked candidates: [{'symbol': str, 'score': float, 'reason': str}, ...]
    """
    candidates: list[dict] = []

    for symbol in universe:
        df = historical_data.get(symbol)
        if df is None or df.empty:
            continue

        # Need at least 6 rows for 5-day lookback + current
        if len(df) < 6:
            continue

        # ── Extract plain symbol for sector lookup ────────────────────
        try:
            _, plain_symbol, _ = from_fyers_symbol(symbol)
        except (ValueError, IndexError):
            plain_symbol = symbol

        # ── 1. Relative Volume Score (0–1) ────────────────────────────
        # Compare latest day volume to 5-day average
        volumes = df["volume"].values
        if len(volumes) < 2:
            continue

        latest_volume = float(volumes[-1])
        avg_volume_5d = float(np.mean(volumes[-6:-1])) if len(volumes) >= 6 else float(np.mean(volumes[:-1]))

        if avg_volume_5d <= 0:
            continue

        rel_volume = latest_volume / avg_volume_5d
        # Normalize: 1.0 is average, cap at 3.0
        vol_score = min(rel_volume / 3.0, 1.0)

        # ── 2. Average True Range score (0–1) ─────────────────────────
        # ATR as % of close → measures daily volatility
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        if len(closes) < 2:
            continue

        # True Range for last 5 days
        tr_values: list[float] = []
        for i in range(-min(5, len(closes) - 1), 0):
            h = float(highs[i])
            l = float(lows[i])
            prev_c = float(closes[i - 1])
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_values.append(tr)

        if not tr_values:
            continue

        atr = np.mean(tr_values)
        current_price = float(closes[-1])

        if current_price <= 0:
            continue

        atr_pct = (atr / current_price) * 100.0

        # ATR between 1% and 5% is ideal for ORB
        # Below 1% = too tight (skip), above 5% = too volatile
        if atr_pct < 1.0:
            atr_score = 0.0
            skip_reason = "low_volatility"
        elif atr_pct > 5.0:
            atr_score = 0.3  # penalize but don't exclude
            skip_reason = ""
        else:
            # Linear scale: 1% → 0.5, 2.5% → 1.0, 5% → 0.5
            atr_score = 1.0 - abs(atr_pct - 2.5) / 2.5
            atr_score = max(0.0, min(1.0, atr_score))
            skip_reason = ""

        if atr_score == 0.0:
            continue

        # ── 3. Sector Bias Score (0–1) ────────────────────────────────
        # Directional score: we want stocks in sectors that are
        # strongly positive OR strongly negative (both are tradeable).
        sector_bias = get_stock_sector_bias(plain_symbol, sector_scores)
        sector_abs = abs(sector_bias)
        # Directional strength: higher absolute score = better candidate
        sector_score_norm = min(sector_abs, 1.0)

        # ── 4. Price trend score (0–1) ────────────────────────────────
        # Simple: compare last close to 5-day SMA
        sma_5 = float(np.mean(closes[-5:])) if len(closes) >= 5 else float(np.mean(closes))
        if sma_5 > 0:
            trend_strength = abs(current_price - sma_5) / sma_5
            trend_score = min(trend_strength * 20.0, 1.0)  # 5% move = 1.0
        else:
            trend_score = 0.0

        # ── Composite Score (weighted) ────────────────────────────────
        composite = (
            vol_score * 0.30          # 30% weight on volume
            + atr_score * 0.25        # 25% weight on volatility
            + sector_score_norm * 0.25  # 25% weight on sector strength
            + trend_score * 0.20      # 20% weight on trend
        )

        # Build reason string
        direction = "bullish" if sector_bias > 0 else "bearish" if sector_bias < 0 else "neutral"
        reason_parts = [
            f"RelVol={rel_volume:.1f}x",
            f"ATR={atr_pct:.2f}%",
            f"Sector={direction}({sector_bias:+.2f})",
            f"Trend={'up' if current_price > sma_5 else 'down'}",
        ]

        candidates.append({
            "symbol": symbol,
            "score": round(composite, 4),
            "reason": " | ".join(reason_parts),
            "sector_bias": sector_bias,
            "rel_volume": round(rel_volume, 2),
            "atr_pct": round(atr_pct, 2),
            "current_price": round(current_price, 2),
        })

    # ── Sort by composite score descending, take top_n ────────────────
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def scan_for_sd_zones(
    universe: list[str],
    historical_data: dict[str, pd.DataFrame],
    sector_scores: dict[str, float],
    top_n: int = 10,
) -> list[dict]:
    """
    Scan universe for S/D Zone trading candidates.

    Criteria:
      - Has fresh, high-scoring zones near current price
      - Adequate liquidity (avg volume > threshold)
      - Trending (50 SMA slope positive for demand, negative for supply)

    Parameters
    ----------
    universe : list[str]
    historical_data : dict[str, pd.DataFrame]
    sector_scores : dict[str, float]
    top_n : int

    Returns
    -------
    list[dict]
        Ranked candidates.
    """
    raise NotImplementedError("S/D Zone scanner — implement in Strategy 2 conversation")
