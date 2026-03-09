"""
Tests for wick_rejection indicator module.
Updated for v2.0 filters: ATR, nose-wick, prior micro-trend.
"""
import pytest
from dataclasses import dataclass
from typing import Optional


# ── Minimal Candle stub for isolated testing ──────────────────────────────

@dataclass
class _Candle:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "5m"


def _c(o, h, l, c, vol=50000, sym="NSE:TEST-EQ"):
    """Shorthand candle constructor."""
    return _Candle(symbol=sym, timestamp=0, open=o, high=h, low=l, close=c, volume=vol)


# ── Import SUT ────────────────────────────────────────────────────────────

from core.indicators.wick_rejection import (
    detect_wick_rejection,
    compute_formation_stop,
    compute_vwap_bands,
    resample_candles,
    RejectionType,
    VWAPBands,
    WickRejection,
)


# ── Helper: build VWAPBands from params ───────────────────────────────────

def _bands(vwap=1000.0, sd=10.0) -> VWAPBands:
    return VWAPBands(
        vwap=vwap,
        upper_1sd=vwap + sd,
        lower_1sd=vwap - sd,
        upper_1_5sd=vwap + 1.5 * sd,
        lower_1_5sd=vwap - 1.5 * sd,
        upper_2sd=vwap + 2.0 * sd,
        lower_2sd=vwap - 2.0 * sd,
        std_dev=sd,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST: detect_wick_rejection
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectWickRejection:
    """Tests for the wick-rejection (pin bar) detection function."""

    def _recent_downtrend(self, n=12, start=1000.0, vol=50000):
        """
        Generate N candles with a clear downward drift.
        Each candle closes ~1 point lower than the previous.
        Ranges ~6 points each (for ATR ~6, so pin bars with range ≥6 pass).
        """
        candles = []
        for i in range(n):
            price = start - i * 1.0
            candles.append(_c(
                o=price + 0.5,
                h=price + 3,
                l=price - 3,
                c=price,
                vol=vol,
            ))
        return candles

    def _recent_uptrend(self, n=12, start=1000.0, vol=50000):
        """
        Generate N candles with a clear upward drift.
        Each candle closes ~1 point higher than the previous.
        Ranges ~6 points each.
        """
        candles = []
        for i in range(n):
            price = start + i * 1.0
            candles.append(_c(
                o=price - 0.5,
                h=price + 3,
                l=price - 3,
                c=price,
                vol=vol,
            ))
        return candles

    # ── Bullish Hammer Tests ──────────────────────────────────────

    def test_bullish_hammer_at_lower_band(self):
        """Classic hammer: long lower wick, body in upper third, at -1.5SD.
        Nose wick must be ≤ 10% of candle range."""
        bands = _bands(vwap=1000.0, sd=10.0)  # lower_1_5sd = 985
        # Hammer: open=984, high=985.5, low=975, close=985
        # body = 1, lower wick = 984-975 = 9, ratio = 9
        # candle_range = 985.5-975 = 10.5
        # nose (upper wick) = 985.5-985 = 0.5, nose_pct = 0.5/10.5 = 0.048 < 0.10 ✓
        # body_mid = 984.5, body_pos = (984.5-975)/10.5 = 0.905 (upper third) ✓
        candle = _c(o=984, h=985.5, l=975, c=985, vol=80000)
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BULLISH_HAMMER
        assert wr.wick_body_ratio >= 2.0
        assert wr.body_position >= 0.67
        assert wr.volume_ratio >= 1.0
        assert wr.band_touch in ("-1.5SD", "-2.0SD")
        assert wr.score > 0.0
        assert wr.stop_price < candle.low
        assert wr.target_price > candle.close

    def test_bullish_hammer_at_2sd(self):
        """Hammer at -2.0SD should flag band_touch as -2.0SD."""
        bands = _bands(vwap=1000.0, sd=10.0)  # lower_2sd = 980
        # Hammer: open=979, high=980, low=970, close=980
        # body = 1, lower wick = 979-970 = 9, ratio = 9
        # range = 980-970 = 10, nose = 980-980 = 0, nose_pct = 0 ✓
        candle = _c(o=979, h=980, l=970, c=980, vol=70000)
        recent = self._recent_downtrend(n=12, start=990.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BULLISH_HAMMER
        assert wr.band_touch == "-2.0SD"

    def test_no_hammer_if_low_volume(self):
        """Hammer shape but volume below average → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        candle = _c(o=984, h=985, l=975, c=985, vol=30000)  # below avg
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_short_wick(self):
        """Lower wick too short relative to body."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # body = |985-983| = 2, lower wick = 983-982 = 1, ratio = 0.5
        candle = _c(o=983, h=986, l=982, c=985, vol=60000)
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_not_at_band(self):
        """Good hammer shape but price is near VWAP, not at -1.5SD."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # Low = 991, above lower_1_5sd (985) → not at band
        candle = _c(o=997, h=998, l=991, c=998, vol=60000)
        recent = self._recent_downtrend(n=12, start=1002.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_nose_too_large(self):
        """Hammer shape but nose (upper wick) > 10% of range → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # open=984, high=987, low=978, close=985
        # range = 987-978 = 9, nose = 987-985 = 2, nose_pct = 2/9 = 0.222 > 0.10
        candle = _c(o=984, h=987, l=978, c=985, vol=80000)
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_no_prior_downtrend(self):
        """Hammer at band but prior candles trending UP → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        candle = _c(o=984, h=985, l=975, c=985, vol=80000)
        # Uptrend recent candles — wrong direction for bullish hammer
        recent = self._recent_uptrend(n=12, start=980.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    # ── Bearish Shooting Star Tests ───────────────────────────────

    def test_bearish_shooting_star_at_upper_band(self):
        """Classic shooting star: long upper wick, body in lower third, at +1.5SD.
        Nose wick must be ≤ 10% of candle range."""
        bands = _bands(vwap=1000.0, sd=10.0)  # upper_1_5sd = 1015
        # SS: open=1016, high=1025, low=1015, close=1015
        # body = 1, upper wick = 1025-1016 = 9, ratio = 9
        # range = 1025-1015 = 10, nose (lower wick) = 1015-1015 = 0, nose_pct = 0 ✓
        # body_mid = 1015.5, body_pos = (1015.5-1015)/10 = 0.05 (lower third) ✓
        candle = _c(o=1016, h=1025, l=1015, c=1015, vol=80000)
        recent = self._recent_uptrend(n=12, start=1005.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BEARISH_SHOOTING_STAR
        assert wr.wick_body_ratio >= 2.0
        assert wr.body_position <= 0.33
        assert wr.volume_ratio >= 1.0
        assert wr.band_touch in ("+1.5SD", "+2.0SD")
        assert wr.stop_price > candle.high
        assert wr.target_price < candle.close

    def test_bearish_shooting_star_at_2sd(self):
        """Shooting star at +2.0SD should flag band_touch as +2.0SD."""
        bands = _bands(vwap=1000.0, sd=10.0)  # upper_2sd = 1020
        # SS: open=1021, high=1030, low=1020, close=1020
        # body = 1, upper wick = 1030-1021 = 9, ratio = 9
        # range = 10, nose = 0 ✓
        candle = _c(o=1021, h=1030, l=1020, c=1020, vol=70000)
        recent = self._recent_uptrend(n=12, start=1010.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BEARISH_SHOOTING_STAR
        assert wr.band_touch == "+2.0SD"

    def test_no_shooting_star_if_body_too_high(self):
        """Upper wick OK but body is not in lower third."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # body_mid = 1018, range = 1022-1014 = 8
        # body_pos = (1018-1014)/8 = 0.5 → not lower third
        candle = _c(o=1017, h=1022, l=1014, c=1019, vol=70000)
        recent = self._recent_uptrend(n=12, start=1005.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_shooting_star_if_nose_too_large(self):
        """Shooting star shape but nose (lower wick) > 10% of range → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # open=1016, high=1022, low=1013, close=1015
        # range = 1022-1013 = 9, nose = min(1016,1015)-1013 = 2, nose_pct = 2/9 = 0.222
        candle = _c(o=1016, h=1022, l=1013, c=1015, vol=80000)
        recent = self._recent_uptrend(n=12, start=1005.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_shooting_star_if_no_prior_uptrend(self):
        """Shooting star at band but prior candles trending DOWN → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        candle = _c(o=1016, h=1025, l=1015, c=1015, vol=80000)
        # Downtrend recent candles — wrong direction for bearish star
        recent = self._recent_downtrend(n=12, start=1020.0, vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    # ── Edge Cases ────────────────────────────────────────────────

    def test_zero_range_candle(self):
        """Candle with H == L should return NONE."""
        bands = _bands()
        candle = _c(o=985, h=985, l=985, c=985, vol=50000)
        wr = detect_wick_rejection(candle, bands, [])
        assert wr.rejection_type == RejectionType.NONE

    def test_candle_below_atr_rejected(self):
        """A tiny candle whose range is below ATR of recent candles → NONE."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # Recent candles have range ~6 each → ATR ≈ 6
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)
        # Tiny candle: range = 2, well below ATR of 6
        candle = _c(o=984, h=985, l=983, c=985, vol=80000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_score_higher_for_stronger_signal(self):
        """Signal at -2SD with 3× volume should score higher than -1.5SD with 1.1× volume."""
        bands = _bands(vwap=1000.0, sd=10.0)
        recent = self._recent_downtrend(n=12, start=995.0, vol=50000)

        # Strong signal: deeper at -2SD, high volume, clean shape
        # open=979, high=980, low=968, close=980
        # body=1, lower wick=979-968=11, range=12, nose=0
        strong = _c(o=979, h=980, l=968, c=980, vol=150000)
        wr_strong = detect_wick_rejection(strong, bands, recent)

        # Weak signal: at -1.5SD, barely above volume threshold
        # open=984, high=985, low=975, close=985
        # body=1, lower wick=984-975=9, range=10, nose=0
        weak = _c(o=984, h=985, l=975, c=985, vol=55000)
        wr_weak = detect_wick_rejection(weak, bands, recent)

        assert wr_strong.rejection_type == RejectionType.BULLISH_HAMMER
        assert wr_weak.rejection_type == RejectionType.BULLISH_HAMMER
        assert wr_strong.score > wr_weak.score


# ═══════════════════════════════════════════════════════════════════════════
# TEST: compute_formation_stop
# ═══════════════════════════════════════════════════════════════════════════

class TestFormationStop:
    """Tests for structure-based stop-loss computation."""

    def test_long_stop_uses_lowest_low_minus_2_ticks(self):
        prior = [_c(100, 103, 97, 101), _c(99, 102, 96, 100)]
        pin = _c(98, 101, 95, 100, vol=80000)

        stop = compute_formation_stop(prior, pin, direction="LONG")

        assert stop == 94.90

    def test_short_stop_uses_highest_high_plus_2_ticks(self):
        prior = [_c(100, 106, 99, 105), _c(101, 108, 100, 107)]
        pin = _c(107, 110, 104, 105, vol=80000)

        stop = compute_formation_stop(prior, pin, direction="SHORT")

        assert stop == 110.10

    def test_lookback_limits_candles(self):
        """With lookback=1, only last 1 prior candle + pin bar used."""
        prior = [
            _c(100, 105, 90, 102),
            _c(100, 103, 97, 101),
        ]
        pin = _c(98, 101, 95, 100)

        stop = compute_formation_stop(prior, pin, direction="LONG", lookback=1)

        assert stop == 94.90

    def test_empty_recent_candles(self):
        """With no prior candles, only pin bar is used."""
        pin = _c(98, 101, 93, 100)

        stop = compute_formation_stop([], pin, direction="LONG")

        assert stop == 92.90


# ═══════════════════════════════════════════════════════════════════════════
# TEST: compute_vwap_bands
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeVWAPBands:
    """Tests for VWAP band calculation from candle data."""

    def test_single_candle(self):
        candles = [_c(100, 102, 98, 101, vol=1000)]
        bands = compute_vwap_bands(candles)

        assert bands is not None
        assert abs(bands.vwap - 100.33) < 0.1
        assert bands.std_dev == 0.0
        assert bands.upper_1_5sd == bands.vwap

    def test_multiple_candles_bands_widen(self):
        candles = [
            _c(100, 105, 95, 102, vol=1000),
            _c(102, 110, 100, 108, vol=2000),
            _c(108, 112, 104, 106, vol=1500),
        ]
        bands = compute_vwap_bands(candles)

        assert bands is not None
        assert bands.std_dev > 0
        assert bands.upper_1_5sd > bands.upper_1sd > bands.vwap
        assert bands.lower_1_5sd < bands.lower_1sd < bands.vwap

    def test_empty_candles_returns_none(self):
        assert compute_vwap_bands([]) is None

    def test_symmetry(self):
        """Upper and lower bands should be symmetric around VWAP."""
        candles = [
            _c(100, 105, 95, 100, vol=1000),
            _c(100, 103, 97, 100, vol=1000),
        ]
        bands = compute_vwap_bands(candles)
        assert bands is not None

        assert abs(
            (bands.upper_1_5sd - bands.vwap) -
            (bands.vwap - bands.lower_1_5sd)
        ) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# TEST: resample_candles
# ═══════════════════════════════════════════════════════════════════════════

class TestResampleCandles:
    """Tests for candle resampling utility."""

    def test_same_tf_returns_copy(self):
        candles = [_c(100, 102, 98, 101)]
        result = resample_candles(candles, source_tf_minutes=5, target_tf_minutes=5)
        assert len(result) == 1

    def test_empty_returns_empty(self):
        result = resample_candles([], source_tf_minutes=3, target_tf_minutes=5)
        assert result == []

    def test_target_smaller_returns_copy(self):
        candles = [_c(100, 102, 98, 101)]
        result = resample_candles(candles, source_tf_minutes=5, target_tf_minutes=3)
        assert len(result) == 1