"""
Wick Rejection (Pin Bar) Detection Indicator
=============================================

Detects hammer and shooting-star patterns at VWAP band extremes
for the VWAP Mean-Reversion (VMR) strategy.

Functions
---------
- detect_wick_rejection : Analyse a single candle for wick-rejection signal
- compute_formation_stop : 3-candle structure-based stop-loss
- resample_candles       : Aggregate finer-TF candles into coarser TF
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
    BULLISH_HAMMER = "BULLISH_HAMMER"           # Long lower wick at lower band
    BEARISH_SHOOTING_STAR = "BEARISH_SHOOTING_STAR"  # Long upper wick at upper band


@dataclass
class VWAPBands:
    """VWAP and standard-deviation bands for the current session."""
    vwap: float
    upper_1sd: float        # VWAP + 1.0 × SD
    lower_1sd: float        # VWAP - 1.0 × SD
    upper_1_5sd: float      # VWAP + 1.5 × SD
    lower_1_5sd: float      # VWAP - 1.5 × SD
    upper_2sd: float        # VWAP + 2.0 × SD
    lower_2sd: float        # VWAP - 2.0 × SD
    std_dev: float          # Current rolling SD


@dataclass
class WickRejection:
    """Result of wick-rejection detection on a single candle."""
    rejection_type: RejectionType = RejectionType.NONE
    wick_body_ratio: float = 0.0          # wick length / body size
    body_position: float = 0.0            # 0 = bottom of range, 1 = top
    volume_ratio: float = 0.0             # candle vol / avg vol
    band_touch: str = ""                  # which band was touched
    vwap_distance_sd: float = 0.0         # how many SDs from VWAP
    score: float = 0.0                    # composite quality 0-1
    stop_price: float = 0.0               # structure-based SL
    target_price: float = 0.0             # VWAP or 0.5 SD target
    details: Dict = field(default_factory=dict)


# ─── Constants ──────────────────────────────────────────────────────────────

MIN_BODY_SIZE_PCT = 0.02       # 0.02 % of price – ignore doji-like bodies
MIN_WICK_BODY_RATIO = 2.0      # wick must be ≥ 2× body
BODY_POSITION_THRESHOLD = 0.33  # body in upper/lower third of range
MIN_VOLUME_RATIO = 1.0         # volume ≥ average of lookback candles
TICK_SIZE = 0.05               # NSE tick size for equity


# ─── Public Functions ───────────────────────────────────────────────────────

def detect_wick_rejection(
    candle,                          # Candle dataclass
    vwap_bands: VWAPBands,
    recent_candles: List,            # last N candles (for avg volume)
    volume_lookback: int = 10,
    min_wick_ratio: float = MIN_WICK_BODY_RATIO,
    band_threshold_sd: float = 1.5,
) -> WickRejection:
    """
    Detect whether *candle* shows a wick-rejection pattern at a VWAP band.

    Parameters
    ----------
    candle : Candle
        The candle to evaluate (signal-TF candle).
    vwap_bands : VWAPBands
        Current session VWAP + SD bands (computed on band-TF).
    recent_candles : list[Candle]
        Previous candles on the same TF (for volume average).
    volume_lookback : int
        Number of candles to average for volume comparison.
    min_wick_ratio : float
        Minimum wick-to-body ratio (default 2.0).
    band_threshold_sd : float
        Minimum SD distance for band touch (default 1.5).

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
    body_position = (body_mid - l) / candle_range  # 0..1

    # ── Avoid tiny / doji candles ──────────────────────────────────
    mid_price = (h + l) / 2.0
    if mid_price <= 0:
        result.details["reason"] = "zero_price"
        return result

    body_pct = (body_size / mid_price) * 100.0
    if body_pct < MIN_BODY_SIZE_PCT:
        result.details["reason"] = "doji_body"
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
            body_position >= (1.0 - BODY_POSITION_THRESHOLD) and  # body in upper 1/3
            vol_ratio >= MIN_VOLUME_RATIO):

        band_touch = "-2.0SD" if l <= vwap_bands.lower_2sd else "-1.5SD"
        stop = compute_formation_stop(
            recent_candles, candle, direction="LONG", tick_size=TICK_SIZE
        )
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
            },
        )

    # ── Check BEARISH SHOOTING STAR at upper band ──────────────────
    if (h >= vwap_bands.upper_1_5sd and
            body_size > 0 and
            upper_wick / body_size >= min_wick_ratio and
            body_position <= BODY_POSITION_THRESHOLD and  # body in lower 1/3
            vol_ratio >= MIN_VOLUME_RATIO):

        band_touch = "+2.0SD" if h >= vwap_bands.upper_2sd else "+1.5SD"
        stop = compute_formation_stop(
            recent_candles, candle, direction="SHORT", tick_size=TICK_SIZE
        )
        target = _compute_target(c, vwap_bands, direction="SHORT")
        score = _compute_rejection_score(
            wick_ratio=upper_wick / body_size,
            vol_ratio=vol_ratio,
            sd_distance=abs(vwap_dist_sd),
            is_green=(c < o),  # red candle preferred for bearish
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
    Compute structure-based stop-loss using the pin-bar candle and
    the *lookback* candles before it.

    For LONG:  stop = min(low of pin + 2 prior candles) − 2 ticks
    For SHORT: stop = max(high of pin + 2 prior candles) + 2 ticks

    Parameters
    ----------
    recent_candles : list[Candle]
        Candles preceding the pin bar (most recent last).
    pin_candle : Candle
        The pin-bar candle itself.
    direction : str
        "LONG" or "SHORT".
    tick_size : float
        Minimum price increment (default 0.05 for NSE equity).
    lookback : int
        Number of prior candles to include (default 2).

    Returns
    -------
    float
        The stop-loss price.
    """
    # Gather the candles: last `lookback` from recent + the pin bar
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

    E.g., resample 3-minute candles into 5-minute candles.

    Parameters
    ----------
    candles : list[Candle]
        Input candles sorted by timestamp ascending.
    source_tf_minutes : int
        Source timeframe in minutes (e.g. 3).
    target_tf_minutes : int
        Target timeframe in minutes (e.g. 5).

    Returns
    -------
    list[Candle]
        Resampled candles. The last bucket may be incomplete.
    """
    if not candles or target_tf_minutes <= source_tf_minutes:
        return list(candles)

    from core.data.models import Candle as CandleClass

    resampled = []
    bucket: List = []
    bucket_start: Optional[datetime] = None

    for c in candles:
        ts = c.timestamp
        if isinstance(ts, (int, float)):
            # epoch ms → datetime
            from core.utils.time_utils import epoch_ms_to_ist
            ts = epoch_ms_to_ist(ts)

        # Determine bucket boundary: floor timestamp to target_tf_minutes
        minute_of_day = ts.hour * 60 + ts.minute
        bucket_minute = (minute_of_day // target_tf_minutes) * target_tf_minutes

        if bucket_start is None:
            bucket_start = minute_of_day // target_tf_minutes
            bucket.append(c)
        elif minute_of_day // target_tf_minutes == bucket_start:
            bucket.append(c)
        else:
            # Flush previous bucket
            if bucket:
                resampled.append(_merge_bucket(
                    bucket, f"{target_tf_minutes}m"
                ))
            bucket = [c]
            bucket_start = minute_of_day // target_tf_minutes

    # Flush last bucket
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

    Parameters
    ----------
    candles : list[Candle]
        Session candles so far (sorted by time, ascending).
    sd_multipliers : tuple
        SD multipliers for band computation.

    Returns
    -------
    VWAPBands or None
        None if insufficient data.
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
    # Variance = E[X²] - E[X]²
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
    """Average volume of the last `lookback` candles."""
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
    Target = VWAP (or the 0.5-SD band if entry is very close to VWAP).
    Ensures at least a minimal target distance.
    """
    if direction.upper() == "LONG":
        # Target is VWAP; if very close, use lower_1sd
        target = bands.vwap
        if target - entry_price < bands.std_dev * 0.3:
            target = bands.lower_1sd  # partial reversion
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
      - Wick ratio   : 30 %  (capped contribution at ratio=5)
      - Volume ratio  : 25 %  (capped at ratio=3)
      - SD distance   : 25 %  (capped at 2.5 SD)
      - Candle colour : 20 %  (preferred colour = full points)
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
