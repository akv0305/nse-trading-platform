"""
NSE Trading Platform — Step 1 Tests: Indicators

Tests for VWAP, ORB, and Sector Score indicators.
Run: pytest tests/test_step1_indicators.py -v
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import pytest

from core.data.models import Candle
from core.utils.time_utils import IST, ist_to_epoch_ms


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build candles for a given day starting at market open
# ═══════════════════════════════════════════════════════════════════════════

def _make_candle(
    symbol: str,
    dt: datetime.datetime,
    o: float, h: float, l: float, c: float,
    volume: int,
    timeframe: str = "5m",
) -> Candle:
    """Create a Candle from a naive IST datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    ts = ist_to_epoch_ms(dt)
    return Candle(
        symbol=symbol, timestamp=ts,
        open=o, high=h, low=l, close=c,
        volume=volume, timeframe=timeframe,
    )


def _build_5m_candles_for_day(
    symbol: str = "NSE:RELIANCE-EQ",
    date_str: str = "2024-06-03",
    base_price: float = 2500.0,
) -> list[Candle]:
    """
    Build a realistic set of 5m candles for a full trading day (9:15–15:25).
    75 candles total.  Price drifts upward with some noise.
    """
    dt_base = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=IST)
    candles: list[Candle] = []
    price = base_price

    start_min = 9 * 60 + 15   # 9:15
    end_min = 15 * 60 + 25    # 15:25 (last candle open time)

    for minute in range(start_min, end_min + 1, 5):
        h = minute // 60
        m = minute % 60
        dt = dt_base.replace(hour=h, minute=m, second=0, microsecond=0)

        # Simulate price movement
        move = (hash((date_str, minute)) % 200 - 100) / 100.0  # -1.0 to +1.0
        drift = 0.05  # slight upward drift
        price = price + move + drift
        o = round(price, 2)
        h_price = round(price + abs(move) + 0.5, 2)
        l_price = round(price - abs(move) - 0.3, 2)
        c = round(price + move * 0.3, 2)
        vol = 50000 + (hash((date_str, minute, "v")) % 30000)

        candles.append(_make_candle(symbol, dt, o, h_price, l_price, c, vol, "5m"))

    return candles


# ═══════════════════════════════════════════════════════════════════════════
# VWAP Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeVwap:
    """Tests for compute_vwap()."""

    def test_returns_none_for_empty_list(self):
        from core.indicators.vwap import compute_vwap
        assert compute_vwap([]) is None

    def test_returns_none_for_zero_volume_only(self):
        from core.indicators.vwap import compute_vwap
        candle = _make_candle(
            "NSE:TEST-EQ",
            datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST),
            100.0, 102.0, 99.0, 101.0, 0, "5m",
        )
        assert compute_vwap([candle]) is None

    def test_single_candle_vwap_equals_typical_price(self):
        from core.indicators.vwap import compute_vwap
        candle = _make_candle(
            "NSE:TEST-EQ",
            datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST),
            100.0, 105.0, 98.0, 103.0, 10000, "5m",
        )
        result = compute_vwap([candle])
        assert result is not None
        expected_tp = (105.0 + 98.0 + 103.0) / 3.0
        assert result.vwap == round(expected_tp, 2)

    def test_single_candle_std_dev_is_zero(self):
        from core.indicators.vwap import compute_vwap
        candle = _make_candle(
            "NSE:TEST-EQ",
            datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST),
            100.0, 100.0, 100.0, 100.0, 10000, "5m",
        )
        result = compute_vwap([candle])
        assert result is not None
        assert result.std_dev == 0.0
        assert result.upper_band == result.vwap
        assert result.lower_band == result.vwap

    def test_multi_candle_vwap_correct(self):
        """Manually verify VWAP for 3 candles."""
        from core.indicators.vwap import compute_vwap
        base_dt = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        c1 = _make_candle("NSE:TEST-EQ", base_dt, 100, 102, 99, 101, 1000, "5m")
        c2 = _make_candle(
            "NSE:TEST-EQ",
            base_dt + datetime.timedelta(minutes=5),
            101, 104, 100, 103, 2000, "5m",
        )
        c3 = _make_candle(
            "NSE:TEST-EQ",
            base_dt + datetime.timedelta(minutes=10),
            103, 106, 102, 105, 1500, "5m",
        )
        candles = [c1, c2, c3]
        result = compute_vwap(candles)
        assert result is not None

        # Manual calc
        tp1 = (102 + 99 + 101) / 3.0   # 100.6667
        tp2 = (104 + 100 + 103) / 3.0  # 102.3333
        tp3 = (106 + 102 + 105) / 3.0  # 104.3333
        cum_tp_vol = tp1 * 1000 + tp2 * 2000 + tp3 * 1500
        cum_vol = 1000 + 2000 + 1500
        expected_vwap = cum_tp_vol / cum_vol

        assert abs(result.vwap - round(expected_vwap, 2)) < 0.01
        assert result.cum_volume == 4500

    def test_bands_symmetric_around_vwap(self):
        from core.indicators.vwap import compute_vwap
        candles = _build_5m_candles_for_day()[:10]
        result = compute_vwap(candles, band_multiplier=2.0)
        assert result is not None
        band_width_upper = result.upper_band - result.vwap
        band_width_lower = result.vwap - result.lower_band
        assert abs(band_width_upper - band_width_lower) < 0.02

    def test_slope_positive_for_rising_vwap(self):
        """Build candles with strictly increasing prices → slope should be > 0."""
        from core.indicators.vwap import compute_vwap
        base_dt = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = []
        for i in range(10):
            p = 100.0 + i * 2  # monotonically rising
            dt = base_dt + datetime.timedelta(minutes=5 * i)
            candles.append(_make_candle("NSE:TEST-EQ", dt, p, p + 1, p - 0.5, p + 0.5, 10000, "5m"))
        result = compute_vwap(candles, slope_lookback=5)
        assert result is not None
        assert result.slope > 0

    def test_slope_negative_for_falling_vwap(self):
        from core.indicators.vwap import compute_vwap
        base_dt = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = []
        for i in range(10):
            p = 200.0 - i * 2  # monotonically falling
            dt = base_dt + datetime.timedelta(minutes=5 * i)
            candles.append(_make_candle("NSE:TEST-EQ", dt, p, p + 0.5, p - 1, p - 0.5, 10000, "5m"))
        result = compute_vwap(candles, slope_lookback=5)
        assert result is not None
        assert result.slope < 0

    def test_zero_volume_candles_skipped_gracefully(self):
        from core.indicators.vwap import compute_vwap
        base_dt = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        c1 = _make_candle("NSE:TEST-EQ", base_dt, 100, 102, 99, 101, 5000, "5m")
        c2 = _make_candle("NSE:TEST-EQ", base_dt + datetime.timedelta(minutes=5), 101, 103, 100, 102, 0, "5m")
        c3 = _make_candle("NSE:TEST-EQ", base_dt + datetime.timedelta(minutes=10), 102, 105, 101, 104, 8000, "5m")
        result = compute_vwap([c1, c2, c3])
        assert result is not None
        assert result.cum_volume == 5000 + 8000  # zero-vol candle excluded


class TestComputeVwapSeries:
    """Tests for compute_vwap_series()."""

    def test_empty_df_returns_empty_with_columns(self):
        from core.indicators.vwap import compute_vwap_series
        df = pd.DataFrame(columns=["high", "low", "close", "volume"])
        result = compute_vwap_series(df)
        assert "vwap" in result.columns
        assert "vwap_upper" in result.columns
        assert "vwap_lower" in result.columns
        assert result.empty

    def test_single_row(self):
        from core.indicators.vwap import compute_vwap_series
        df = pd.DataFrame([{"high": 105, "low": 98, "close": 103, "volume": 10000}])
        result = compute_vwap_series(df)
        expected_tp = (105 + 98 + 103) / 3.0
        assert abs(result["vwap"].iloc[0] - round(expected_tp, 2)) < 0.01

    def test_vwap_series_matches_candle_vwap(self):
        """Cross-check: series VWAP at last row should match compute_vwap()."""
        from core.indicators.vwap import compute_vwap, compute_vwap_series
        candles = _build_5m_candles_for_day()[:20]

        # Series approach
        rows = [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
        df = pd.DataFrame(rows)
        series_result = compute_vwap_series(df, band_multiplier=1.5)

        # Candle approach
        candle_result = compute_vwap(candles, band_multiplier=1.5)

        assert candle_result is not None
        assert abs(series_result["vwap"].iloc[-1] - candle_result.vwap) < 0.02

    def test_upper_always_above_lower(self):
        from core.indicators.vwap import compute_vwap_series
        candles = _build_5m_candles_for_day()[:30]
        rows = [{"high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
        df = pd.DataFrame(rows)
        result = compute_vwap_series(df)
        assert (result["vwap_upper"] >= result["vwap_lower"]).all()

    def test_does_not_mutate_original_df(self):
        from core.indicators.vwap import compute_vwap_series
        df = pd.DataFrame([{"high": 105, "low": 98, "close": 103, "volume": 10000}])
        original_cols = set(df.columns)
        _ = compute_vwap_series(df)
        assert set(df.columns) == original_cols  # original unchanged


# ═══════════════════════════════════════════════════════════════════════════
# ORB Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeOpeningRange:
    """Tests for compute_opening_range()."""

    def test_returns_none_for_empty_list(self):
        from core.indicators.orb import compute_opening_range
        assert compute_opening_range([]) is None

    def test_returns_none_when_no_candles_in_range(self):
        """All candles are after OR period."""
        from core.indicators.orb import compute_opening_range
        # Candles starting at 10:00 — well past the 9:15-9:30 OR window
        dt = datetime.datetime(2024, 6, 3, 10, 0, tzinfo=IST)
        c = _make_candle("NSE:TEST-EQ", dt, 100, 102, 99, 101, 5000, "5m")
        assert compute_opening_range([c]) is None

    def test_15min_orb_captures_3_candles_of_5m(self):
        """15-min OR from 5m candles: 9:15, 9:20, 9:25 → exactly 3 candles."""
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = [
            _make_candle("NSE:RELIANCE-EQ", base, 2500, 2520, 2490, 2510, 50000, "5m"),
            _make_candle("NSE:RELIANCE-EQ", base + datetime.timedelta(minutes=5), 2510, 2530, 2500, 2525, 60000, "5m"),
            _make_candle("NSE:RELIANCE-EQ", base + datetime.timedelta(minutes=10), 2525, 2540, 2505, 2535, 55000, "5m"),
            # This 9:30 candle should NOT be included in OR
            _make_candle("NSE:RELIANCE-EQ", base + datetime.timedelta(minutes=15), 2535, 2550, 2530, 2545, 45000, "5m"),
        ]
        result = compute_opening_range(candles, or_period_minutes=15)
        assert result is not None
        assert result.or_high == 2540.0  # max high of first 3 candles
        assert result.or_low == 2490.0   # min low of first 3 candles
        assert result.is_valid

    def test_orb_mid_is_average_of_high_low(self):
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = [
            _make_candle("NSE:TEST-EQ", base, 100, 110, 90, 105, 10000, "5m"),
            _make_candle("NSE:TEST-EQ", base + datetime.timedelta(minutes=5), 105, 108, 95, 106, 12000, "5m"),
            _make_candle("NSE:TEST-EQ", base + datetime.timedelta(minutes=10), 106, 112, 92, 110, 11000, "5m"),
        ]
        result = compute_opening_range(candles)
        assert result is not None
        assert result.or_mid == round((result.or_high + result.or_low) / 2.0, 2)

    def test_orb_width_pct(self):
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = [
            _make_candle("NSE:TEST-EQ", base, 100, 110, 90, 105, 10000, "5m"),
            _make_candle("NSE:TEST-EQ", base + datetime.timedelta(minutes=5), 105, 108, 95, 106, 12000, "5m"),
            _make_candle("NSE:TEST-EQ", base + datetime.timedelta(minutes=10), 106, 112, 92, 110, 11000, "5m"),
        ]
        result = compute_opening_range(candles)
        assert result is not None
        expected_pct = round((result.or_width / result.or_mid) * 100.0, 4)
        assert result.or_width_pct == expected_pct

    def test_30min_orb_captures_6_candles_of_5m(self):
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = []
        for i in range(8):  # 0,5,10,15,20,25,30,35 min
            dt = base + datetime.timedelta(minutes=5 * i)
            candles.append(_make_candle("NSE:TEST-EQ", dt, 100 + i, 102 + i, 99 + i, 101 + i, 10000, "5m"))
        result = compute_opening_range(candles, or_period_minutes=30)
        assert result is not None
        assert result.is_valid
        # 30 min / 5m = 6 candles → indices 0-5 (9:15 to 9:40)
        # or_high = max high of first 6 candles
        assert result.or_high == max(c.high for c in candles[:6])
        assert result.or_low == min(c.low for c in candles[:6])

    def test_date_and_symbol_captured(self):
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = [
            _make_candle("NSE:INFY-EQ", base, 1500, 1520, 1490, 1510, 30000, "5m"),
            _make_candle("NSE:INFY-EQ", base + datetime.timedelta(minutes=5), 1510, 1525, 1500, 1520, 35000, "5m"),
            _make_candle("NSE:INFY-EQ", base + datetime.timedelta(minutes=10), 1520, 1530, 1505, 1525, 32000, "5m"),
        ]
        result = compute_opening_range(candles)
        assert result is not None
        assert result.symbol == "NSE:INFY-EQ"
        assert result.date == "2024-06-03"

    def test_invalid_when_too_few_candles(self):
        """Only 1 candle for 15-min OR (need 3) → is_valid=False."""
        from core.indicators.orb import compute_opening_range
        base = datetime.datetime(2024, 6, 3, 9, 15, tzinfo=IST)
        candles = [
            _make_candle("NSE:TEST-EQ", base, 100, 110, 90, 105, 10000, "5m"),
        ]
        result = compute_opening_range(candles, or_period_minutes=15)
        assert result is not None
        assert not result.is_valid


class TestDetectBreakout:
    """Tests for detect_breakout()."""

    def _make_orb(self, or_high=110.0, or_low=90.0, is_valid=True) -> object:
        from core.indicators.orb import ORBLevels
        return ORBLevels(
            symbol="NSE:TEST-EQ",
            date="2024-06-03",
            or_high=or_high,
            or_low=or_low,
            or_mid=(or_high + or_low) / 2.0,
            or_width=or_high - or_low,
            or_width_pct=((or_high - or_low) / ((or_high + or_low) / 2.0)) * 100.0,
            is_valid=is_valid,
            captured_at_ms=1000000,
        )

    def test_inside_range(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(110.0, 90.0)
        assert detect_breakout(100.0, orb) == "INSIDE_RANGE"

    def test_breakout_long(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(110.0, 90.0)
        # With 0.05% buffer: threshold = 110 * 1.0005 = 110.055
        assert detect_breakout(110.10, orb) == "BREAKOUT_LONG"

    def test_breakout_short(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(110.0, 90.0)
        # With 0.05% buffer: threshold = 90 * 0.9995 = 89.955
        assert detect_breakout(89.90, orb) == "BREAKOUT_SHORT"

    def test_at_exact_threshold_is_inside(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(110.0, 90.0)
        # Exactly at OR high (no buffer crossed) → INSIDE
        assert detect_breakout(110.0, orb) == "INSIDE_RANGE"

    def test_just_above_buffer_long(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(1000.0, 980.0)
        # buffer=0.05%, threshold = 1000 * 1.0005 = 1000.50
        assert detect_breakout(1000.51, orb, buffer_pct=0.05) == "BREAKOUT_LONG"
        assert detect_breakout(1000.49, orb, buffer_pct=0.05) == "INSIDE_RANGE"

    def test_invalid_orb_returns_inside(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(110.0, 90.0, is_valid=False)
        assert detect_breakout(200.0, orb) == "INSIDE_RANGE"

    def test_custom_buffer(self):
        from core.indicators.orb import detect_breakout
        orb = self._make_orb(100.0, 80.0)
        # buffer=1.0%, long threshold = 100 * 1.01 = 101.0
        assert detect_breakout(101.5, orb, buffer_pct=1.0) == "BREAKOUT_LONG"
        assert detect_breakout(100.5, orb, buffer_pct=1.0) == "INSIDE_RANGE"


# ═══════════════════════════════════════════════════════════════════════════
# Sector Score Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSectorScores:
    """Tests for compute_sector_scores()."""

    def _make_close_df(self, closes: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"close": closes})

    def test_empty_sector_data(self):
        from core.indicators.sector_score import compute_sector_scores
        nifty = self._make_close_df([100, 102, 104, 106, 108, 110])
        result = compute_sector_scores({}, nifty)
        assert result == {}

    def test_all_sectors_matching_nifty_gives_zero(self):
        """If all sectors match Nifty performance → all scores ≈ 0."""
        from core.indicators.sector_score import compute_sector_scores
        closes = [100, 102, 104, 106, 108, 110]
        nifty = self._make_close_df(closes)
        sectors = {
            "NIFTY BANK": self._make_close_df(closes),
            "NIFTY IT": self._make_close_df(closes),
        }
        result = compute_sector_scores(sectors, nifty, lookback_days=5)
        for score in result.values():
            assert abs(score) < 0.01

    def test_outperformer_gets_positive_score(self):
        from core.indicators.sector_score import compute_sector_scores
        nifty = self._make_close_df([100, 101, 102, 103, 104, 105])  # +5%
        bank = self._make_close_df([100, 103, 106, 109, 112, 115])   # +15%
        it = self._make_close_df([100, 100, 100, 100, 100, 100])     # 0%
        result = compute_sector_scores(
            {"NIFTY BANK": bank, "NIFTY IT": it},
            nifty, lookback_days=5,
        )
        assert result["NIFTY BANK"] > 0
        assert result["NIFTY IT"] < 0

    def test_scores_bounded_minus_one_to_plus_one(self):
        from core.indicators.sector_score import compute_sector_scores
        nifty = self._make_close_df([100, 105])
        sectors = {
            "NIFTY BANK": self._make_close_df([100, 120]),
            "NIFTY IT": self._make_close_df([100, 80]),
            "NIFTY AUTO": self._make_close_df([100, 105]),
        }
        result = compute_sector_scores(sectors, nifty, lookback_days=1)
        for score in result.values():
            assert -1.0 <= score <= 1.0

    def test_insufficient_nifty_data(self):
        from core.indicators.sector_score import compute_sector_scores
        nifty = self._make_close_df([100])  # only 1 row → can't compute return
        sectors = {"NIFTY BANK": self._make_close_df([100, 105])}
        result = compute_sector_scores(sectors, nifty, lookback_days=5)
        assert result["NIFTY BANK"] == 0.0


class TestGetStockSectorBias:
    """Tests for get_stock_sector_bias()."""

    def test_known_banking_stock(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY BANK": 0.75, "NIFTY IT": -0.3}
        assert get_stock_sector_bias("HDFCBANK", scores) == 0.75

    def test_known_it_stock(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY BANK": 0.75, "NIFTY IT": -0.3}
        assert get_stock_sector_bias("INFY", scores) == -0.3

    def test_pharma_stock(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY PHARMA": 0.5}
        assert get_stock_sector_bias("SUNPHARMA", scores) == 0.5

    def test_unknown_symbol_returns_zero(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY BANK": 0.5}
        assert get_stock_sector_bias("UNKNOWN_STOCK", scores) == 0.0

    def test_missing_sector_in_scores_returns_zero(self):
        from core.indicators.sector_score import get_stock_sector_bias
        # HDFCBANK maps to NIFTY BANK, but that's not in scores
        scores = {"NIFTY IT": 0.5}
        assert get_stock_sector_bias("HDFCBANK", scores) == 0.0

    def test_insurance_maps_to_fin_service(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY FIN SERVICE": 0.6}
        # HDFCLIFE is Insurance → should map to NIFTY FIN SERVICE
        assert get_stock_sector_bias("HDFCLIFE", scores) == 0.6

    def test_auto_stock(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY AUTO": -0.4}
        assert get_stock_sector_bias("MARUTI", scores) == -0.4

    def test_metal_stock(self):
        from core.indicators.sector_score import get_stock_sector_bias
        scores = {"NIFTY METAL": 0.9}
        assert get_stock_sector_bias("TATASTEEL", scores) == 0.9

    def test_all_nifty50_stocks_return_a_value(self):
        """Every Nifty 50 stock should map to some sector and not crash."""
        from core.indicators.sector_score import get_stock_sector_bias
        from core.data.universe import NIFTY50
        # Build a complete score map
        scores = {
            "NIFTY BANK": 0.5, "NIFTY IT": -0.3, "NIFTY PHARMA": 0.2,
            "NIFTY AUTO": 0.1, "NIFTY FMCG": -0.1, "NIFTY METAL": 0.4,
            "NIFTY ENERGY": 0.3, "NIFTY INFRA": -0.2, "NIFTY FIN SERVICE": 0.6,
        }
        for symbol in NIFTY50:
            bias = get_stock_sector_bias(symbol, scores)
            assert isinstance(bias, float), f"{symbol} returned {type(bias)}"
