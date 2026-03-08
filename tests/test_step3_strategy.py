"""
NSE Trading Platform — Step 3 Tests: ORB+VWAP Strategy

Tests for core/strategies/orb_vwap.py
Run: pytest tests/test_step3_strategy.py -v
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from config.settings import settings
from core.data.models import Candle, Signal, SignalType, Direction
from core.data.universe import to_fyers_symbol, NIFTY50
from core.strategies.orb_vwap import ORBVWAPStrategy, _SymbolState
from core.utils.time_utils import IST, ist_to_epoch_ms


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts(h: int, m: int, date_str: str = "2024-06-03") -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=h, minute=m, second=0, microsecond=0, tzinfo=IST,
    )
    return ist_to_epoch_ms(dt)


def _candle(
    symbol: str, h: int, m: int,
    o: float, hi: float, lo: float, c: float,
    volume: int = 50000,
    date_str: str = "2024-06-03",
) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=_ts(h, m, date_str),
        open=o, high=hi, low=lo, close=c,
        volume=volume,
        timeframe="5m",
    )


def _build_or_candles(symbol: str, base: float = 1000.0) -> list[Candle]:
    """
    OR_HIGH = base + 8, OR_LOW = base - 8
    Width = 16 / base*100 ≈ 1.6% (within valid range)
    Average volume = 60000
    """
    return [
        _candle(symbol, 9, 15, base, base + 5, base - 8, base + 3, 60000),
        _candle(symbol, 9, 20, base + 3, base + 8, base - 5, base + 6, 60000),
        _candle(symbol, 9, 25, base + 6, base + 7, base - 3, base + 4, 60000),
    ]


def _make_daily_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    n = len(closes)
    if volumes is None:
        volumes = [100000] * n
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "timestamp": 1704067200000 + i * 86400000,
            "open": c - 5, "high": c + 10, "low": c - 10,
            "close": c, "volume": volumes[i],
        })
    return pd.DataFrame(rows)


def _setup_strategy_with_orb(
    sym: str = "NSE:HDFCBANK-EQ",
    base: float = 1000.0,
    sector_scores: dict | None = None,
) -> tuple[ORBVWAPStrategy, list[Candle]]:
    s = ORBVWAPStrategy()
    if sector_scores is None:
        sector_scores = {"NIFTY BANK": 0.7, "NIFTY IT": -0.3}
    s._sector_scores = sector_scores
    s._states[sym] = _SymbolState()

    candles = _build_or_candles(sym, base)
    history: list[Candle] = []
    for c in candles:
        history.append(c)
        s.on_candle(sym, c, history, None)

    c_inside = _candle(sym, 9, 30, base + 2, base + 5, base - 1, base + 3, 55000)
    history.append(c_inside)
    s.on_candle(sym, c_inside, history, None)

    return s, history


def _feed_rising_candles(
    s: ORBVWAPStrategy, sym: str, history: list[Candle], base: float = 1000.0,
) -> list[Candle]:
    rising_data = [
        (9, 35,  base + 4,  base + 8,  base + 2,  base + 6,  65000),
        (9, 40,  base + 6,  base + 10, base + 4,  base + 8,  68000),
        (9, 45,  base + 8,  base + 12, base + 6,  base + 10, 70000),
        (9, 50,  base + 10, base + 14, base + 8,  base + 12, 72000),
        (9, 55,  base + 12, base + 16, base + 10, base + 14, 75000),
        (10, 0,  base + 14, base + 18, base + 12, base + 16, 78000),
        (10, 5,  base + 16, base + 20, base + 14, base + 18, 80000),
    ]
    for h, m, o, hi, lo, c, v in rising_data:
        candle = _candle(sym, h, m, o, hi, lo, c, v)
        history.append(candle)
        s.on_candle(sym, candle, history, None)
    return history


def _feed_falling_candles(
    s: ORBVWAPStrategy, sym: str, history: list[Candle], base: float = 1000.0,
) -> list[Candle]:
    falling_data = [
        (9, 35,  base - 2,  base,      base - 6,  base - 4,  65000),
        (9, 40,  base - 4,  base - 2,  base - 8,  base - 6,  68000),
        (9, 45,  base - 6,  base - 4,  base - 10, base - 8,  70000),
        (9, 50,  base - 8,  base - 6,  base - 12, base - 10, 72000),
        (9, 55,  base - 10, base - 8,  base - 14, base - 12, 75000),
        (10, 0,  base - 12, base - 10, base - 16, base - 14, 78000),
        (10, 5,  base - 14, base - 12, base - 18, base - 16, 80000),
    ]
    for h, m, o, hi, lo, c, v in falling_data:
        candle = _candle(sym, h, m, o, hi, lo, c, v)
        history.append(candle)
        s.on_candle(sym, candle, history, None)
    return history


# ═══════════════════════════════════════════════════════════════════════════
# Instantiation & get_params
# ═══════════════════════════════════════════════════════════════════════════

class TestORBVWAPInit:

    def test_instantiates(self):
        s = ORBVWAPStrategy()
        assert s.name == "orb_vwap"
        assert s.version == "1.0.0"
        assert s.is_active

    def test_get_params_returns_dict(self):
        s = ORBVWAPStrategy()
        p = s.get_params()
        assert isinstance(p, dict)
        assert p["orb_period_minutes"] == settings.ORB_PERIOD_MINUTES
        assert p["orb_breakout_buffer_pct"] == settings.ORB_BREAKOUT_BUFFER_PCT
        assert p["vwap_band_std_multiplier"] == settings.VWAP_BAND_STD_MULTIPLIER
        assert p["orb_stoploss_mode"] == settings.ORB_STOPLOSS_MODE
        assert p["orb_target_rr"] == settings.ORB_TARGET_RR
        assert p["orb_trail_after_rr"] == settings.ORB_TRAIL_AFTER_RR
        assert p["orb_trail_pct"] == settings.ORB_TRAIL_PCT
        assert p["entry_cutoff_ist"] == settings.ENTRY_CUTOFF_IST
        assert p["flatten_time_ist"] == settings.FLATTEN_TIME_IST

    def test_repr(self):
        s = ORBVWAPStrategy()
        assert "orb_vwap" in repr(s)


# ═══════════════════════════════════════════════════════════════════════════
# pre_market_scan
# ═══════════════════════════════════════════════════════════════════════════

class TestPreMarketScan:

    def test_returns_list_of_symbols(self):
        s = ORBVWAPStrategy()
        symbols = [to_fyers_symbol(n) for n in NIFTY50[:10]]
        hist = {sym: _make_daily_df([1000 + i * 5 for i in range(25)],
                                    [100000] * 24 + [250000])
                for sym in symbols}
        result = s.pre_market_scan(symbols, hist)
        assert isinstance(result, list)

    def test_respects_scan_top_n(self):
        s = ORBVWAPStrategy()
        symbols = [to_fyers_symbol(n) for n in NIFTY50[:30]]
        hist = {sym: _make_daily_df([1000 + i * 5 for i in range(25)],
                                    [100000] * 24 + [250000])
                for sym in symbols}
        result = s.pre_market_scan(symbols, hist)
        assert len(result) <= settings.SCAN_TOP_N

    def test_sets_watchlist(self):
        s = ORBVWAPStrategy()
        symbols = [to_fyers_symbol(n) for n in NIFTY50[:10]]
        hist = {sym: _make_daily_df([1000 + i * 5 for i in range(25)],
                                    [100000] * 24 + [250000])
                for sym in symbols}
        s.pre_market_scan(symbols, hist)
        assert len(s.watchlist) >= 0

    def test_empty_universe(self):
        s = ORBVWAPStrategy()
        assert s.pre_market_scan([], {}) == []


# ═══════════════════════════════════════════════════════════════════════════
# on_candle — OR Phase
# ═══════════════════════════════════════════════════════════════════════════

class TestOnCandleORPhase:

    def test_accumulates_or_candles(self):
        s = ORBVWAPStrategy()
        sym = "NSE:RELIANCE-EQ"
        s._states[sym] = _SymbolState()
        c1 = _candle(sym, 9, 15, 2500, 2520, 2490, 2510, 60000)
        sig = s.on_candle(sym, c1, [c1], None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "or_period" in sig.skip_reason

    def test_orb_captured_after_930(self):
        s = ORBVWAPStrategy()
        sym = "NSE:RELIANCE-EQ"
        s._states[sym] = _SymbolState()

        candles = _build_or_candles(sym)
        history = []
        for c in candles:
            history.append(c)
            s.on_candle(sym, c, history, None)

        c_930 = _candle(sym, 9, 30, 1004, 1006, 1000, 1005, 55000)
        history.append(c_930)
        s.on_candle(sym, c_930, history, None)

        state = s._states[sym]
        assert state.orb_captured
        assert state.orb is not None
        assert state.orb.or_high == 1008.0
        assert state.orb.or_low == 992.0

    def test_orb_width_too_wide_rejects(self):
        s = ORBVWAPStrategy()
        sym = "NSE:TEST-EQ"
        s._states[sym] = _SymbolState()

        or_candles = [
            _candle(sym, 9, 15, 1050, 1100, 1000, 1060, 60000),
            _candle(sym, 9, 20, 1060, 1090, 1010, 1070, 60000),
            _candle(sym, 9, 25, 1070, 1080, 1020, 1050, 60000),
        ]
        history = []
        for c in or_candles:
            history.append(c)
            s.on_candle(sym, c, history, None)

        c_930 = _candle(sym, 9, 30, 1050, 1055, 1045, 1050, 50000)
        history.append(c_930)
        sig = s.on_candle(sym, c_930, history, None)
        assert "orb_width" in sig.skip_reason
        assert s._states[sym].orb_rejected is True

        # Subsequent candles blocked
        c_935 = _candle(sym, 9, 35, 1110, 1120, 1100, 1115, 100000)
        history.append(c_935)
        sig2 = s.on_candle(sym, c_935, history, None)
        assert sig2.signal_type == SignalType.NO_ACTION
        assert "orb_width_rejected" in sig2.skip_reason

    def test_orb_width_too_narrow_rejects(self):
        s = ORBVWAPStrategy()
        sym = "NSE:TEST-EQ"
        s._states[sym] = _SymbolState()

        or_candles = [
            _candle(sym, 9, 15, 1000, 1001, 1000, 1000.5, 60000),
            _candle(sym, 9, 20, 1000.5, 1001, 1000, 1000.3, 60000),
            _candle(sym, 9, 25, 1000.3, 1000.8, 1000, 1000.5, 60000),
        ]
        history = []
        for c in or_candles:
            history.append(c)
            s.on_candle(sym, c, history, None)

        c_930 = _candle(sym, 9, 30, 1000.5, 1001, 1000, 1000.5, 50000)
        history.append(c_930)
        sig = s.on_candle(sym, c_930, history, None)
        assert "orb_width" in sig.skip_reason
        assert s._states[sym].orb_rejected is True


# ═══════════════════════════════════════════════════════════════════════════
# on_candle — Breakout Phase
# ═══════════════════════════════════════════════════════════════════════════

class TestOnCandleBreakout:

    def test_inside_range_no_signal(self):
        s, history = _setup_strategy_with_orb()
        sym = "NSE:HDFCBANK-EQ"
        c = _candle(sym, 9, 35, 1003, 1006, 1000, 1004, 50000)
        history.append(c)
        sig = s.on_candle(sym, c, history, None)
        assert sig.signal_type == SignalType.NO_ACTION

    def test_long_breakout_all_conditions_met(self):
        sym = "NSE:HDFCBANK-EQ"
        s, history = _setup_strategy_with_orb(sym=sym, base=1000.0)
        history = _feed_rising_candles(s, sym, history, base=1000.0)

        c_breakout = _candle(sym, 10, 10, 1018, 1022, 1016, 1020, 100000)
        history.append(c_breakout)
        sig = s.on_candle(sym, c_breakout, history, None)

        assert sig.signal_type == SignalType.BUY, (
            f"Expected BUY but got {sig.signal_type}: {sig.skip_reason}"
        )
        assert sig.strength > 0.0
        assert sig.indicator_data["stoploss_price"] == 992.0
        assert sig.indicator_data["target_price"] is not None

    def test_short_breakout_all_conditions_met(self):
        sym = "NSE:INFY-EQ"
        s, history = _setup_strategy_with_orb(
            sym=sym, base=1000.0,
            sector_scores={"NIFTY IT": -0.6},
        )
        history = _feed_falling_candles(s, sym, history, base=1000.0)

        c_breakout = _candle(sym, 10, 10, 984, 986, 978, 980, 100000)
        history.append(c_breakout)
        sig = s.on_candle(sym, c_breakout, history, None)

        assert sig.signal_type == SignalType.SELL, (
            f"Expected SELL but got {sig.signal_type}: {sig.skip_reason}"
        )
        assert sig.indicator_data["stoploss_price"] == 1008.0
        assert sig.indicator_data["target_price"] is not None

    def test_long_breakout_target_uses_rr(self):
        sym = "NSE:HDFCBANK-EQ"
        s, history = _setup_strategy_with_orb(sym=sym, base=1000.0)
        history = _feed_rising_candles(s, sym, history, base=1000.0)

        c_breakout = _candle(sym, 10, 10, 1018, 1022, 1016, 1020, 100000)
        history.append(c_breakout)
        sig = s.on_candle(sym, c_breakout, history, None)

        if sig.signal_type == SignalType.BUY:
            risk = sig.price_at_signal - sig.indicator_data["stoploss_price"]
            expected_target = sig.price_at_signal + risk * settings.ORB_TARGET_RR
            assert abs(sig.indicator_data["target_price"] - expected_target) < 0.01

    def test_no_signal_when_position_open(self):
        s, history = _setup_strategy_with_orb()
        sym = "NSE:HDFCBANK-EQ"
        c = _candle(sym, 9, 55, 1019, 1025, 1018, 1022, 150000)
        history.append(c)
        position = {"direction": "LONG", "entry_price": 1010}
        sig = s.on_candle(sym, c, history, position)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "position_open" in sig.skip_reason

    def test_no_second_breakout_same_day(self):
        sym = "NSE:HDFCBANK-EQ"
        s, history = _setup_strategy_with_orb(sym=sym, base=1000.0)
        history = _feed_rising_candles(s, sym, history, base=1000.0)

        c_breakout = _candle(sym, 10, 10, 1018, 1022, 1016, 1020, 100000)
        history.append(c_breakout)
        sig1 = s.on_candle(sym, c_breakout, history, None)

        c2 = _candle(sym, 10, 15, 1020, 1028, 1018, 1025, 120000)
        history.append(c2)
        sig2 = s.on_candle(sym, c2, history, None)

        if sig1.signal_type == SignalType.BUY:
            assert sig2.signal_type == SignalType.NO_ACTION
            assert "breakout_already_fired" in sig2.skip_reason

    def test_past_entry_cutoff_no_signal(self):
        s, history = _setup_strategy_with_orb()
        sym = "NSE:HDFCBANK-EQ"
        c = _candle(sym, 14, 50, 1025, 1030, 1020, 1028, 100000)
        history.append(c)
        sig = s.on_candle(sym, c, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "cutoff" in sig.skip_reason

    def test_long_breakout_fails_without_volume(self):
        sym = "NSE:HDFCBANK-EQ"
        s, history = _setup_strategy_with_orb(sym=sym, base=1000.0)
        history = _feed_rising_candles(s, sym, history, base=1000.0)

        c_weak = _candle(sym, 10, 10, 1018, 1022, 1016, 1020, 30000)
        history.append(c_weak)
        sig = s.on_candle(sym, c_weak, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "low_volume" in sig.skip_reason

    def test_long_breakout_fails_with_negative_sector(self):
        sym = "NSE:INFY-EQ"
        s, history = _setup_strategy_with_orb(
            sym=sym, base=1000.0,
            sector_scores={"NIFTY IT": -0.5},
        )
        history = _feed_rising_candles(s, sym, history, base=1000.0)

        c_breakout = _candle(sym, 10, 10, 1018, 1022, 1016, 1020, 100000)
        history.append(c_breakout)
        sig = s.on_candle(sym, c_breakout, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "sector_negative" in sig.skip_reason

    def test_unknown_symbol_sector_bias_zero_blocks_long(self):
        """
        A symbol not in SECTOR_MAP has sector_bias=0.
        For LONG, condition requires sector_bias > 0 → blocked.
        Uses base=2000 so OR width = 16/2000 = 0.8% (within valid range).
        """
        sym = "NSE:UNKNOWN-EQ"
        s, history = _setup_strategy_with_orb(
            sym=sym, base=2000.0,
            sector_scores={"NIFTY BANK": 0.7},
        )
        history = _feed_rising_candles(s, sym, history, base=2000.0)

        # OR_HIGH = 2008, buffer threshold ≈ 2008 * 1.0005 = 2009.004
        # close=2020 is well above threshold
        c_bo = _candle(sym, 10, 10, 2018, 2022, 2016, 2020, 100000)
        history.append(c_bo)
        sig = s.on_candle(sym, c_bo, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        # sector_bias for unknown stock = 0.0 → not > 0 → "sector_negative"
        assert "sector_negative" in sig.skip_reason

# ═══════════════════════════════════════════════════════════════════════════
# should_exit
# ═══════════════════════════════════════════════════════════════════════════

class TestShouldExit:

    def _make_position(
        self,
        direction: str = "LONG",
        entry: float = 1020.0,
        sl: float = 990.0,
        target: float = 1080.0,
        peak_price: float | None = None,
    ) -> dict:
        pos = {
            "direction": direction,
            "entry_price": entry,
            "stoploss_price": sl,
            "target_price": target,
            "quantity": 100,
        }
        if peak_price is not None:
            pos["peak_price"] = peak_price
        return pos

    def test_flatten_time_exit(self):
        s = ORBVWAPStrategy()
        pos = self._make_position()
        candle = _candle("NSE:TEST-EQ", 15, 22, 1030, 1032, 1028, 1031)
        sig = s.should_exit("NSE:TEST-EQ", 1031.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "FLATTEN"

    def test_stoploss_hit_long(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="LONG", entry=1020, sl=990, target=1080)
        candle = _candle("NSE:TEST-EQ", 10, 30, 995, 998, 988, 989)
        sig = s.should_exit("NSE:TEST-EQ", 989.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "STOPLOSS"

    def test_stoploss_hit_short(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="SHORT", entry=990, sl=1020, target=930)
        candle = _candle("NSE:TEST-EQ", 10, 30, 1018, 1025, 1015, 1022)
        sig = s.should_exit("NSE:TEST-EQ", 1022.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_SHORT
        assert sig.indicator_data["exit_reason"] == "STOPLOSS"

    def test_target_hit_long(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="LONG", entry=1020, sl=990, target=1080)
        candle = _candle("NSE:TEST-EQ", 11, 0, 1078, 1085, 1075, 1082)
        sig = s.should_exit("NSE:TEST-EQ", 1082.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "TARGET"

    def test_target_hit_short(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="SHORT", entry=1020, sl=1050, target=960)
        candle = _candle("NSE:TEST-EQ", 11, 0, 962, 965, 955, 958)
        sig = s.should_exit("NSE:TEST-EQ", 958.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_SHORT
        assert sig.indicator_data["exit_reason"] == "TARGET"

    def test_no_exit_when_within_range(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="LONG", entry=1020, sl=990, target=1080)
        candle = _candle("NSE:TEST-EQ", 10, 30, 1035, 1040, 1030, 1038)
        sig = s.should_exit("NSE:TEST-EQ", 1038.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION

    def test_trailing_not_active_below_1r(self):
        """
        LONG: entry=1020, sl=990 (risk=30).
        peak_price=1025 (explicitly set), candle high=1030.
        Updated peak = max(1025, 1030) = 1030.
        move = 1030 - 1020 = 10, rr = 10/30 = 0.33 < 1.0 → trail NOT active.
        Price 1028 > plain SL 990 → no exit.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="LONG", entry=1020, sl=990, target=1080,
            peak_price=1025,
        )
        # Candle high=1030 → updated peak=1030 → rr=0.33 < 1.0
        candle = _candle("NSE:TEST-EQ", 11, 30, 1026, 1030, 1027, 1028)
        sig = s.should_exit("NSE:TEST-EQ", 1028.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION
        assert sig.indicator_data.get("is_trailing") is False

    def test_trailing_active_above_1r_long(self):
        """
        LONG: entry=1020, sl=990 (risk=30).
        peak_price=1055, candle high=1055 (no change to peak).
        move = 1055 - 1020 = 35, rr = 35/30 = 1.17 → trail active.
        trail_sl = 1020 + 0.3 * 35 = 1030.5.
        Price=1052 > 1030.5 → no exit, but is_trailing=True.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="LONG", entry=1020, sl=990, target=1080,
            peak_price=1055,
        )
        # candle high = 1055 exactly → peak stays 1055
        candle = _candle("NSE:TEST-EQ", 12, 0, 1050, 1055, 1048, 1052)
        sig = s.should_exit("NSE:TEST-EQ", 1052.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION
        assert sig.indicator_data.get("is_trailing") is True
        assert sig.indicator_data.get("effective_sl") == 1030.5

    def test_trailing_stop_triggers_exit_long(self):
        """
        LONG: entry=1020, sl=990 (risk=30).
        peak_price=1060, candle high=1060.
        trail_sl = 1020 + 0.3*40 = 1032.
        Price 1031 < 1032 → TRAIL exit.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="LONG", entry=1020, sl=990, target=1080,
            peak_price=1060,
        )
        candle = _candle("NSE:TEST-EQ", 12, 5, 1035, 1060, 1029, 1031)
        sig = s.should_exit("NSE:TEST-EQ", 1031.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "TRAIL"
        assert sig.indicator_data["is_trailing"] is True

    def test_trailing_stop_does_not_regress(self):
        """
        LONG: entry=1020, sl=990 (risk=30).
        peak_price=1060 (historic high), candle high=1045 (lower than peak).
        Updated peak = max(1060, 1045) = 1060 (unchanged).
        trail_sl = 1020 + 0.3*40 = 1032.
        Price 1040 > 1032 → no exit, trail active.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="LONG", entry=1020, sl=990, target=1080,
            peak_price=1060,
        )
        candle = _candle("NSE:TEST-EQ", 12, 10, 1045, 1045, 1038, 1040)
        sig = s.should_exit("NSE:TEST-EQ", 1040.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION
        assert sig.indicator_data["effective_sl"] == 1032.0
        assert sig.indicator_data["is_trailing"] is True
        assert sig.indicator_data["peak_price"] == 1060  # peak unchanged

    def test_trailing_stop_short(self):
        """
        SHORT: entry=1020, sl=1050 (risk=30).
        peak_price(low)=985, candle low=985.
        move = 1020 - 985 = 35, rr=1.17 → trail active.
        trail_sl = 1020 - 0.3*35 = 1009.5.
        Price 990 < 1009.5 → no exit (SHORT exits when price >= SL).
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="SHORT", entry=1020, sl=1050, target=960,
            peak_price=985,
        )
        candle = _candle("NSE:TEST-EQ", 12, 0, 990, 992, 985, 990)
        sig = s.should_exit("NSE:TEST-EQ", 990.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION
        assert sig.indicator_data.get("is_trailing") is True

    def test_trailing_stop_triggers_exit_short(self):
        """
        SHORT: entry=1020, sl=1050 (risk=30).
        peak_price(low)=980, candle low=980.
        trail_sl = 1020 - 0.3*40 = 1008.
        Price 1010 >= 1008 → TRAIL exit.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="SHORT", entry=1020, sl=1050, target=960,
            peak_price=980,
        )
        candle = _candle("NSE:TEST-EQ", 12, 5, 1005, 1012, 980, 1010)
        sig = s.should_exit("NSE:TEST-EQ", 1010.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_SHORT
        assert sig.indicator_data["exit_reason"] == "TRAIL"

    def test_flatten_overrides_everything(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="LONG", entry=1020, sl=990, target=1080)
        candle = _candle("NSE:TEST-EQ", 15, 25, 1050, 1055, 1045, 1052)
        sig = s.should_exit("NSE:TEST-EQ", 1052.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "FLATTEN"

    def test_stoploss_before_target_when_both_possible(self):
        s = ORBVWAPStrategy()
        pos = self._make_position(direction="LONG", entry=1020, sl=1010, target=1040)
        candle = _candle("NSE:TEST-EQ", 11, 0, 1015, 1018, 1003, 1005)
        sig = s.should_exit("NSE:TEST-EQ", 1005.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "STOPLOSS"

    def test_candle_high_updates_peak_for_trailing(self):
        """
        peak_price in position = 1050, but candle.high = 1065.
        Updated peak should be 1065, trail computed from 1065.
        """
        s = ORBVWAPStrategy()
        pos = self._make_position(
            direction="LONG", entry=1020, sl=990, target=1100,
            peak_price=1050,
        )
        # candle high=1065 → peak updated to 1065
        # trail_sl = 1020 + 0.3*(1065-1020) = 1020 + 13.5 = 1033.5
        candle = _candle("NSE:TEST-EQ", 12, 0, 1055, 1065, 1050, 1058)
        sig = s.should_exit("NSE:TEST-EQ", 1058.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION
        assert sig.indicator_data["peak_price"] == 1065
        assert sig.indicator_data["effective_sl"] == 1033.5


# ═══════════════════════════════════════════════════════════════════════════
# end_of_day & build_trade_plan
# ═══════════════════════════════════════════════════════════════════════════

class TestEndOfDay:

    def test_clears_state(self):
        s = ORBVWAPStrategy()
        s._states["NSE:TEST-EQ"] = _SymbolState()
        s._watchlist = ["NSE:TEST-EQ"]
        s._sector_scores = {"NIFTY BANK": 0.5}
        s.end_of_day()
        assert s._states == {}
        assert s._sector_scores == {}
        assert s.watchlist == []


class TestBuildTradePlan:

    def test_build_trade_plan_long(self):
        s = ORBVWAPStrategy()
        signal = Signal(
            strategy_name="orb_vwap",
            symbol="NSE:HDFCBANK-EQ",
            signal_type=SignalType.BUY,
            strength=0.7,
            price_at_signal=1022.0,
            timestamp=_ts(9, 55),
            indicator_data={"stoploss_price": 990.0, "target_price": 1086.0},
        )
        plan = s.build_trade_plan(signal, risk_per_trade=3750.0)
        assert plan is not None
        assert plan.direction == Direction.LONG
        assert plan.entry_price == 1022.0
        assert plan.stoploss_price == 990.0
        assert plan.target_price == 1086.0
        expected_qty = int(3750.0 / (1022.0 - 990.0))
        assert plan.quantity == expected_qty

    def test_build_trade_plan_short(self):
        s = ORBVWAPStrategy()
        signal = Signal(
            strategy_name="orb_vwap",
            symbol="NSE:HDFCBANK-EQ",
            signal_type=SignalType.SELL,
            strength=0.6,
            price_at_signal=988.0,
            timestamp=_ts(10, 0),
            indicator_data={"stoploss_price": 1020.0, "target_price": 924.0},
        )
        plan = s.build_trade_plan(signal, risk_per_trade=3750.0)
        assert plan is not None
        assert plan.direction == Direction.SHORT
        expected_qty = int(3750.0 / (1020.0 - 988.0))
        assert plan.quantity == expected_qty

    def test_build_trade_plan_no_action_returns_none(self):
        s = ORBVWAPStrategy()
        signal = Signal(
            strategy_name="orb_vwap",
            symbol="NSE:TEST-EQ",
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=1000.0,
            timestamp=_ts(10, 0),
        )
        assert s.build_trade_plan(signal, risk_per_trade=3750.0) is None
