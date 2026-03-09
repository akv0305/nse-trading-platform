"""
Wick Rejection (Pin Bar) Detection Indicator
=============================================

Detects hammer and shooting-star patterns at VWAP band extremes
for the VWAP Mean-Reversion (VMR) strategy.

Research-backed filters (Indian market / NSE):
  - Wick ≥ 66% of candle range (2:1 wick-to-body minimum)
  - Body ≤ 25–30% of candle range
  - Nose (opposite wick) ≤ 10% of candle range
  - Volume ≥ 20-period SMA
  - Candle range ≥ 1.0× ATR-14 (noise elimination)
  - Prior micro-trend: 5-candle directional drift required

Functions
---------
- detect_wick_rejection : Analyse a single candle for wick-rejection signal
- compute_formation_stop : 3-candle structure-based stop-loss (legacy)
- compute_atr            : Average True Range over N candles
- has_prior_trend        : Check if last N candles show directional drift
- resample_candles       : Aggregate finer-TF candles into coarser TF
- compute_vwap_bands     : Session VWAP + SD bands
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Enums & Data Classes ──────────────────────────────────────────────────

class RejectionType(str, Enum):
    """Type of wick rejection detected."""
    NONE = "NONE"
    BULLISH_HAMMER = "BULLISH_HAMMER"
    BEARISH_SHOOTING_STAR = "BEARISH_SHOOTING_STAR"


@dataclass
class VWAPBands:
    """VWAP and standard-deviation bands for the current session."""
    vwap: float
    upper_1sd: float
    lower_1sd: float
    upper_1_5sd: float
    lower_1_5sd: float
    upper_2sd: float
    lower_2sd: float
    std_dev: float


@dataclass
class WickRejection:
    """Result of wick-rejection detection on a single candle."""
    rejection_type: RejectionType = RejectionType.NONE
    wick_body_ratio: float = 0.0
    body_position: float = 0.0
    volume_ratio: float = 0.0
    band_touch: str = ""
    vwap_distance_sd: float = 0.0
    score: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    details: Dict = field(default_factory=dict)


# ─── Constants ──────────────────────────────────────────────────────────────

MIN_BODY_SIZE_PCT = 0.02
MIN_WICK_BODY_RATIO = 2.0
BODY_POSITION_THRESHOLD = 0.33
MIN_VOLUME_RATIO = 1.0
TICK_SIZE = 0.05
MAX_NOSE_PCT_DEFAULT = 0.10
MIN_ATR_MULTIPLIER_DEFAULT = 1.0
PRIOR_TREND_LOOKBACK_DEFAULT = 5


# ─── ATR & Trend Helpers ───────────────────────────────────────────────────

def compute_atr(candles: list, period: int = 14) -> float:
    """
    Compute Average True Range over the last `period` candles.

    Parameters
    ----------
    candles : list[Candle]
        Candle history (at least 2 required).
    period : int
        ATR lookback period (default 14).

    Returns
    -------
    float
        ATR value, or 0.0 if insufficient data.
    """
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        pc = candles[i - 1]
        tr = max(
            c.high - c.low,
            abs(c.high - pc.close),
            abs(c.low - pc.close),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    return sum(trs[-period:]) / period


def has_prior_trend(
    recent_candles: list,
    is_bullish: bool,
    lookback: int = PRIOR_TREND_LOOKBACK_DEFAULT,
) -> bool:
    """
    Check if the last `lookback` candles show a directional drift.

    For bullish hammer: need prior downward drift (lower closes).
    For bearish shooting star: need prior upward drift (higher closes).

    Parameters
    ----------
    recent_candles : list[Candle]
        Candles before the current pin bar (most recent last).
    is_bullish : bool
        True if checking for bullish hammer, False for bearish star.
    lookback : int
        Number of candles to check (default 5).

    Returns
    -------
    bool
        True if the required prior trend exists.
    """
    if len(recent_candles) < lookback:
        # Not enough data — be permissive to avoid blocking all signals
        # during early session
        if len(recent_candles) < 3:
            return True
        lookback = len(recent_candles)

    start_close = recent_candles[-lookback].close
    end_close = recent_candles[-1].close
    if start_close == 0:
        return False
    drift_pct = (end_close - start_close) / start_close * 100.0

    if is_bullish:
        return drift_pct < -0.1  # price was falling → hammer is valid
    else:
        return drift_pct > 0.1   # price was rising → shooting star valid


# ─── Public Functions ───────────────────────────────────────────────────────

def detect_wick_rejection(
    candle,
    vwap_bands: VWAPBands,
    recent_candles: List,
    volume_lookback: int = 20,
    min_wick_ratio: float = MIN_WICK_BODY_RATIO,
    band_threshold_sd: float = 1.5,
    min_atr_multiplier: float = MIN_ATR_MULTIPLIER_DEFAULT,
    max_nose_pct: float = MAX_NOSE_PCT_DEFAULT,
    prior_trend_lookback: int = PRIOR_TREND_LOOKBACK_DEFAULT,
) -> WickRejection:
    """
    Detect whether *candle* shows a wick-rejection pattern at a VWAP band.

    Applies research-backed filters:
      1. Standard pin-bar shape (wick ≥ 2× body, body in upper/lower third)
      2. Nose wick ≤ 10% of candle range
      3. Candle range ≥ 1.0× ATR-14 (eliminates noise)
      4. Prior micro-trend in correct direction
      5. Volume ≥ 20-candle SMA
      6. Price touches ≥ 1.5 SD VWAP band

    Parameters
    ----------
    candle : Candle
        The candle to evaluate.
    vwap_bands : VWAPBands
        Current session VWAP + SD bands.
    recent_candles : list[Candle]
        Previous candles (for volume avg, ATR, trend check).
    volume_lookback : int
        Number of candles for volume SMA (default 20).
    min_wick_ratio : float
        Minimum wick-to-body ratio (default 2.0).
    band_threshold_sd : float
        Minimum SD distance for band touch (default 1.5).
    min_atr_multiplier : float
        Candle range must be ≥ this × ATR-14 (default 1.0).
    max_nose_pct : float
        Max nose wick as fraction of candle range (default 0.10).
    prior_trend_lookback : int
        Candles to check for prior directional drift (default 5).

    Returns
    -------
    WickRejection
        Populated result; rejection_type == NONE if no pattern found.
    """
    result = WickRejection()

    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    candle_range = h - l

    if candle_range <= 0:
        result.details["reason"] = "zero_range"
        return result

    body_size = abs(c - o)
    body_top = max(o, c)
    body_bottom = min(o, c)
    upper_wick = h - body_top
    lower_wick = body_bottom - l

    # ── Body position within range (0=bottom, 1=top) ───────────────
    body_mid = (body_top + body_bottom) / 2.0
    body_position = (body_mid - l) / candle_range

    # ── Avoid tiny / doji candles ──────────────────────────────────
    mid_price = (h + l) / 2.0
    if mid_price <= 0:
        result.details["reason"] = "zero_price"
        return result

    body_pct = (body_size / mid_price) * 100.0
    if body_pct < MIN_BODY_SIZE_PCT:
        result.details["reason"] = "doji_body"
        return result

    # ── Filter 1: Minimum candle range vs ATR ──────────────────────
    # Ensures the pin bar has meaningful size, not just random noise
    if recent_candles and len(recent_candles) >= 2:
        atr = compute_atr(recent_candles, period=14)
        if atr > 0 and candle_range < min_atr_multiplier * atr:
            result.details["reason"] = "candle_range_below_atr"
            result.details["candle_range"] = round(candle_range, 2)
            result.details["atr"] = round(atr, 2)
            return result

    # ── Volume ratio ───────────────────────────────────────────────
    avg_vol = _avg_volume(recent_candles, volume_lookback)
    vol_ratio = (candle.volume / avg_vol) if avg_vol > 0 else 0.0

    # ── VWAP distance in SDs ───────────────────────────────────────
    sd = vwap_bands.std_dev if vwap_bands.std_dev > 0 else 1e-9
    price_for_dist = c
    vwap_dist_sd = (price_for_dist - vwap_bands.vwap) / sd

    # ── Check BULLISH HAMMER at lower band ─────────────────────────
    if (l <= vwap_bands.lower_1_5sd and
            body_size > 0 and
            lower_wick / body_size >= min_wick_ratio and
            body_position >= (1.0 - BODY_POSITION_THRESHOLD) and
            vol_ratio >= MIN_VOLUME_RATIO):

        # Filter 2: Nose wick check — upper wick must be small
        nose_pct = upper_wick / candle_range if candle_range > 0 else 1.0
        if nose_pct > max_nose_pct:
            result.details["reason"] = "nose_too_large"
            result.details["nose_pct"] = round(nose_pct, 3)
            return result

        # Filter 3: Prior micro-trend — price should have been falling
        if not has_prior_trend(recent_candles, is_bullish=True,
                               lookback=prior_trend_lookback):
            result.details["reason"] = "no_prior_downtrend"
            return result

        band_touch = "-2.0SD" if l <= vwap_bands.lower_2sd else "-1.5SD"
        # Stop price is computed in strategy, not here anymore
        # (strategy uses pin bar wick extreme + ticks)
        stop = round(l - 2 * TICK_SIZE, 2)  # legacy fallback
        target = _compute_target(c, vwap_bands, direction="LONG")
        score = _compute_rejection_score(
            wick_ratio=lower_wick / body_size,
            vol_ratio=vol_ratio,
            sd_distance=abs(vwap_dist_sd),
            is_green=(c >= o),
        )
        return WickRejection(
            rejection_type=RejectionType.BULLISH_HAMMER,
            wick_body_ratio=round(lower_wick / body_size, 2),
            body_position=round(body_position, 3),
            volume_ratio=round(vol_ratio, 2),
            band_touch=band_touch,
            vwap_distance_sd=round(vwap_dist_sd, 3),
            score=round(score, 3),
            stop_price=stop,
            target_price=target,
            details={
                "body_size": round(body_size, 2),
                "lower_wick": round(lower_wick, 2),
                "upper_wick": round(upper_wick, 2),
                "avg_volume": round(avg_vol, 0),
                "nose_pct": round(upper_wick / candle_range, 3),
            },
        )

    # ── Check BEARISH SHOOTING STAR at upper band ──────────────────
    if (h >= vwap_bands.upper_1_5sd and
            body_size > 0 and
            upper_wick / body_size >= min_wick_ratio and
            body_position <= BODY_POSITION_THRESHOLD and
            vol_ratio >= MIN_VOLUME_RATIO):

        # Filter 2: Nose wick check — lower wick must be small
        nose_pct = lower_wick / candle_range if candle_range > 0 else 1.0
        if nose_pct > max_nose_pct:
            result.details["reason"] = "nose_too_large"
            result.details["nose_pct"] = round(nose_pct, 3)
            return result

        # Filter 3: Prior micro-trend — price should have been rising
        if not has_prior_trend(recent_candles, is_bullish=False,
                               lookback=prior_trend_lookback):
            result.details["reason"] = "no_prior_uptrend"
            return result

        band_touch = "+2.0SD" if h >= vwap_bands.upper_2sd else "+1.5SD"
        stop = round(h + 2 * TICK_SIZE, 2)  # legacy fallback
        target = _compute_target(c, vwap_bands, direction="SHORT")
        score = _compute_rejection_score(
            wick_ratio=upper_wick / body_size,
            vol_ratio=vol_ratio,
            sd_distance=abs(vwap_dist_sd),
            is_green=(c < o),
        )
        return WickRejection(
            rejection_type=RejectionType.BEARISH_SHOOTING_STAR,
            wick_body_ratio=round(upper_wick / body_size, 2),
            body_position=round(body_position, 3),
            volume_ratio=round(vol_ratio, 2),
            band_touch=band_touch,
            vwap_distance_sd=round(vwap_dist_sd, 3),
            score=round(score, 3),
            stop_price=stop,
            target_price=target,
            details={
                "body_size": round(body_size, 2),
                "lower_wick": round(lower_wick, 2),
                "upper_wick": round(upper_wick, 2),
                "avg_volume": round(avg_vol, 0),
                "nose_pct": round(lower_wick / candle_range, 3),
            },
        )

    # ── No pattern ─────────────────────────────────────────────────
    result.body_position = round(body_position, 3)
    result.volume_ratio = round(vol_ratio, 2)
    result.vwap_distance_sd = round(vwap_dist_sd, 3)
    result.details["reason"] = "no_pattern"
    return result


def compute_formation_stop(
    recent_candles: List,
    pin_candle,
    direction: str,
    tick_size: float = TICK_SIZE,
    lookback: int = 2,
) -> float:
    """
    Legacy: Compute structure-based stop-loss using the pin-bar candle
    and the *lookback* candles before it.

    Note: VMR strategy now uses pin-bar wick extreme + ticks instead.
    This function is kept for backward compatibility.
    """
    formation_candles = list(recent_candles[-lookback:]) + [pin_candle]

    if direction.upper() == "LONG":
        lowest = min(c.low for c in formation_candles)
        return round(lowest - 2 * tick_size, 2)
    else:
        highest = max(c.high for c in formation_candles)
        return round(highest + 2 * tick_size, 2)


def resample_candles(
    candles: List,
    source_tf_minutes: int,
    target_tf_minutes: int,
) -> List:
    """
    Aggregate finer-timeframe candles into coarser-timeframe candles.
    """
    if not candles or target_tf_minutes <= source_tf_minutes:
        return list(candles)

    from core.data.models import Candle as CandleClass

    resampled = []
    bucket: List = []
    bucket_start: Optional[int] = None

    for c in candles:
        ts = c.timestamp
        if isinstance(ts, (int, float)):
            from core.utils.time_utils import epoch_ms_to_ist
            ts = epoch_ms_to_ist(ts)

        minute_of_day = ts.hour * 60 + ts.minute
        bucket_id = minute_of_day // target_tf_minutes

        if bucket_start is None:
            bucket_start = bucket_id
            bucket.append(c)
        elif bucket_id == bucket_start:
            bucket.append(c)
        else:
            if bucket:
                resampled.append(_merge_bucket(bucket, f"{target_tf_minutes}m"))
            bucket = [c]
            bucket_start = bucket_id

    if bucket:
        resampled.append(_merge_bucket(bucket, f"{target_tf_minutes}m"))

    return resampled


def compute_vwap_bands(
    candles: List,
    sd_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0),
) -> Optional[VWAPBands]:
    """
    Compute session VWAP and standard-deviation bands from candles.
    Uses typical price = (H + L + C) / 3 weighted by volume.
    """
    if not candles:
        return None

    cum_tp_vol = 0.0
    cum_vol = 0.0
    cum_tp2_vol = 0.0

    for c in candles:
        tp = (c.high + c.low + c.close) / 3.0
        vol = c.volume if c.volume > 0 else 1
        cum_tp_vol += tp * vol
        cum_vol += vol
        cum_tp2_vol += (tp ** 2) * vol

    if cum_vol <= 0:
        return None

    vwap = cum_tp_vol / cum_vol
    variance = max((cum_tp2_vol / cum_vol) - vwap ** 2, 0.0)
    std_dev = variance ** 0.5

    return VWAPBands(
        vwap=round(vwap, 2),
        upper_1sd=round(vwap + 1.0 * std_dev, 2),
        lower_1sd=round(vwap - 1.0 * std_dev, 2),
        upper_1_5sd=round(vwap + 1.5 * std_dev, 2),
        lower_1_5sd=round(vwap - 1.5 * std_dev, 2),
        upper_2sd=round(vwap + 2.0 * std_dev, 2),
        lower_2sd=round(vwap - 2.0 * std_dev, 2),
        std_dev=round(std_dev, 2),
    )


# ─── Private Helpers ────────────────────────────────────────────────────────

def _avg_volume(candles: List, lookback: int) -> float:
    """Average volume of the last `lookback` candles (SMA)."""
    if not candles:
        return 0.0
    recent = candles[-lookback:]
    vols = [c.volume for c in recent if c.volume > 0]
    return sum(vols) / len(vols) if vols else 0.0


def _compute_target(
    entry_price: float,
    bands: VWAPBands,
    direction: str,
) -> float:
    """
    Target = VWAP (mean reversion).
    Ensures at least a minimal target distance.
    """
    if direction.upper() == "LONG":
        target = bands.vwap
        if target - entry_price < bands.std_dev * 0.3:
            target = bands.lower_1sd
        return round(max(target, entry_price + 0.10), 2)
    else:
        target = bands.vwap
        if entry_price - target < bands.std_dev * 0.3:
            target = bands.upper_1sd
        return round(min(target, entry_price - 0.10), 2)


def _compute_rejection_score(
    wick_ratio: float,
    vol_ratio: float,
    sd_distance: float,
    is_green: bool,
) -> float:
    """
    Composite quality score (0-1) for a wick rejection.

    Weights:
      - Wick ratio   : 30%  (capped at ratio=5)
      - Volume ratio  : 25%  (capped at ratio=3)
      - SD distance   : 25%  (capped at 2.5 SD)
      - Candle colour : 20%  (preferred colour = full points)
    """
    wick_score = min(wick_ratio / 5.0, 1.0)
    vol_score = min(vol_ratio / 3.0, 1.0)
    sd_score = min(sd_distance / 2.5, 1.0)
    colour_score = 1.0 if is_green else 0.5

    composite = (
        0.30 * wick_score
        + 0.25 * vol_score
        + 0.25 * sd_score
        + 0.20 * colour_score
    )
    return min(composite, 1.0)


def _merge_bucket(candles: List, timeframe: str):
    """Merge a list of candles into a single aggregated candle."""
    from core.data.models import Candle as CandleClass

    first = candles[0]
    return CandleClass(
        symbol=first.symbol,
        timestamp=first.timestamp,
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=sum(c.volume for c in candles),
        timeframe=timeframe,
    )
