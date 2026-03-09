"""
Tests for wick_rejection indicator module.
"""
import pytest
from dataclasses import dataclass
from typing import Optional


# ── Minimal Candle stub for isolated testing ──────────────────────────────

@dataclass
class _Candle:
    symbol: str
    timestamp: int       # epoch ms (unused in indicator logic)
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

    def _recent(self, n=12, base=1000.0, vol=50000):
        """Generate N neutral candles for volume averaging."""
        return [_c(base, base + 2, base - 2, base + 1, vol) for _ in range(n)]

    # ── Bullish Hammer Tests ──────────────────────────────────────

    def test_bullish_hammer_at_lower_band(self):
        """Classic hammer: long lower wick, body in upper third, at -1.5SD."""
        bands = _bands(vwap=1000.0, sd=10.0)  # lower_1_5sd = 985
        # Hammer: open=984, high=986, low=978, close=985
        # body = |985-984| = 1, lower wick = 984-978 = 6, ratio = 6/1 = 6
        # body_mid = 984.5, body_pos = (984.5 - 978) / (986 - 978) = 0.8125 (upper third)
        candle = _c(o=984, h=986, l=978, c=985, vol=80000)
        recent = self._recent(vol=50000)

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
        candle = _c(o=979, h=981, l=974, c=980, vol=70000)
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BULLISH_HAMMER
        assert wr.band_touch == "-2.0SD"

    def test_no_hammer_if_low_volume(self):
        """Hammer shape but volume below average → no signal."""
        bands = _bands(vwap=1000.0, sd=10.0)
        candle = _c(o=984, h=986, l=978, c=985, vol=30000)  # below avg
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_short_wick(self):
        """Lower wick too short relative to body."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # body = |985-983| = 2, lower wick = 983-982 = 1, ratio = 0.5
        candle = _c(o=983, h=986, l=982, c=985, vol=60000)
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    def test_no_hammer_if_not_at_band(self):
        """Good hammer shape but price is near VWAP, not at -1.5SD."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # Low = 996, above lower_1_5sd (985) → not at band
        candle = _c(o=997, h=999, l=991, c=998, vol=60000)
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    # ── Bearish Shooting Star Tests ───────────────────────────────

    def test_bearish_shooting_star_at_upper_band(self):
        """Classic shooting star: long upper wick, body in lower third, at +1.5SD."""
        bands = _bands(vwap=1000.0, sd=10.0)  # upper_1_5sd = 1015
        # SS: open=1016, high=1022, low=1014, close=1015
        # body = 1, upper wick = 1022-1016 = 6, ratio = 6
        # body_mid = 1015.5, body_pos = (1015.5-1014)/(1022-1014) = 0.1875 (lower third)
        candle = _c(o=1016, h=1022, l=1014, c=1015, vol=80000)
        recent = self._recent(vol=50000)

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
        candle = _c(o=1021, h=1027, l=1019, c=1020, vol=70000)
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.BEARISH_SHOOTING_STAR
        assert wr.band_touch == "+2.0SD"

    def test_no_shooting_star_if_body_too_high(self):
        """Upper wick OK but body is not in lower third."""
        bands = _bands(vwap=1000.0, sd=10.0)
        # body_mid = 1018, range = 1022-1014 = 8
        # body_pos = (1018-1014)/8 = 0.5 → not lower third
        candle = _c(o=1017, h=1022, l=1014, c=1019, vol=70000)
        recent = self._recent(vol=50000)

        wr = detect_wick_rejection(candle, bands, recent)

        assert wr.rejection_type == RejectionType.NONE

    # ── Edge Cases ────────────────────────────────────────────────

    def test_zero_range_candle(self):
        """Candle with H == L should return NONE."""
        bands = _bands()
        candle = _c(o=985, h=985, l=985, c=985, vol=50000)
        wr = detect_wick_rejection(candle, bands, [])
        assert wr.rejection_type == RejectionType.NONE

    def test_score_higher_for_stronger_signal(self):
        """Signal at -2SD with 3× volume should score higher than -1.5SD with 1.1× volume."""
        bands = _bands(vwap=1000.0, sd=10.0)
        recent = self._recent(vol=50000)

        # Strong signal
        strong = _c(o=979, h=981, l=970, c=980, vol=150000)
        wr_strong = detect_wick_rejection(strong, bands, recent)

        # Weak signal
        weak = _c(o=984, h=986, l=978, c=985, vol=55000)
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

        # Lowest low = 95 (pin bar), stop = 95 - 2*0.05 = 94.90
        assert stop == 94.90

    def test_short_stop_uses_highest_high_plus_2_ticks(self):
        prior = [_c(100, 106, 99, 105), _c(101, 108, 100, 107)]
        pin = _c(107, 110, 104, 105, vol=80000)

        stop = compute_formation_stop(prior, pin, direction="SHORT")

        # Highest high = 110 (pin bar), stop = 110 + 2*0.05 = 110.10
        assert stop == 110.10

    def test_lookback_limits_candles(self):
        """With lookback=1, only last 1 prior candle + pin bar used."""
        prior = [
            _c(100, 105, 90, 102),   # very low = 90, but outside lookback
            _c(100, 103, 97, 101),   # this one is within lookback=1
        ]
        pin = _c(98, 101, 95, 100)

        stop = compute_formation_stop(prior, pin, direction="LONG", lookback=1)

        # Only prior[-1] (low=97) and pin (low=95); lowest=95; stop=94.90
        assert stop == 94.90

    def test_empty_recent_candles(self):
        """With no prior candles, only pin bar is used."""
        pin = _c(98, 101, 93, 100)

        stop = compute_formation_stop([], pin, direction="LONG")

        # Only pin (low=93); stop = 93 - 0.10 = 92.90
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
        # TP = (102+98+101)/3 = 100.333..., VWAP = same (single candle)
        assert abs(bands.vwap - 100.33) < 0.1
        # SD = 0 for single candle
        assert bands.std_dev == 0.0
        assert bands.upper_1_5sd == bands.vwap  # 0 SD → bands collapse

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
