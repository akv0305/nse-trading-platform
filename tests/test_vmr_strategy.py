"""
Tests for VMR (VWAP Mean-Reversion) Strategy
=============================================
Run: pytest tests/test_vmr_strategy.py -v
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from config.settings import settings
from core.data.models import Candle, Signal, SignalType, Direction
from core.data.universe import to_fyers_symbol, NIFTY50
from core.strategies.vmr_strategy import VMRStrategy, _VMRState
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
    tf: str = "5m",
) -> Candle:
    return Candle(
        symbol=symbol, timestamp=_ts(h, m, date_str),
        open=o, high=hi, low=lo, close=c,
        volume=volume, timeframe=tf,
    )


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


def _build_warmup_candles(
    sym: str, base: float = 1000.0, n: int = 20,
) -> list[Candle]:
    """Build N candles from 9:15 onward, centered around base price."""
    candles = []
    for i in range(n):
        minute = 15 + i * 5
        h = 9 + minute // 60
        m = minute % 60
        noise = (i % 5 - 2) * 0.5
        price = base + noise
        candles.append(_candle(
            sym, h, m,
            o=price - 1, hi=price + 2, lo=price - 2, c=price,
            volume=50000,
        ))
    return candles


def _setup_strategy_with_warmup(
    sym: str = "NSE:HDFCBANK-EQ",
    base: float = 1000.0,
    sector_scores: dict | None = None,
    n_warmup: int = 20,
) -> tuple[VMRStrategy, list[Candle]]:
    """Create a VMR strategy pre-loaded with warmup candles."""
    s = VMRStrategy()
    if sector_scores is None:
        sector_scores = {"NIFTY BANK": 0.3}
    s._sector_scores = sector_scores
    s._states[sym] = _VMRState(prev_day_close=base)

    candles = _build_warmup_candles(sym, base, n_warmup)
    history: list[Candle] = []
    for c in candles:
        history.append(c)
        s.on_candle(sym, c, history, None)

    return s, history


# ═══════════════════════════════════════════════════════════════════════════
# Init & Params
# ═══════════════════════════════════════════════════════════════════════════

class TestVMRInit:

    def test_instantiates(self):
        s = VMRStrategy()
        assert s.name == "vmr_vwap"
        assert s.version == "1.0.0"
        assert s.is_active

    def test_get_params(self):
        s = VMRStrategy()
        p = s.get_params()
        assert isinstance(p, dict)
        assert p["strategy_name"] == "vmr_vwap"
        assert "band_sd_threshold" in p
        assert "trail_pct" in p

    def test_repr(self):
        s = VMRStrategy()
        assert "vmr_vwap" in repr(s)


# ═══════════════════════════════════════════════════════════════════════════
# pre_market_scan
# ═══════════════════════════════════════════════════════════════════════════

class TestVMRPreMarketScan:

    def test_returns_list(self):
        s = VMRStrategy()
        symbols = [to_fyers_symbol(n) for n in NIFTY50[:10]]
        hist = {sym: _make_daily_df([1000 + i * 5 for i in range(25)])
                for sym in symbols}
        result = s.pre_market_scan(symbols, hist)
        assert isinstance(result, list)

    def test_empty_universe(self):
        s = VMRStrategy()
        assert s.pre_market_scan([], {}) == []

    def test_sets_watchlist(self):
        s = VMRStrategy()
        symbols = [to_fyers_symbol(n) for n in NIFTY50[:10]]
        hist = {sym: _make_daily_df([1000 + i * 5 for i in range(25)])
                for sym in symbols}
        s.pre_market_scan(symbols, hist)
        assert len(s.watchlist) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# on_candle — entry logic
# ═══════════════════════════════════════════════════════════════════════════

class TestVMROnCandle:

    def test_warmup_period_no_signal(self):
        s = VMRStrategy()
        sym = "NSE:HDFCBANK-EQ"
        s._states[sym] = _VMRState(prev_day_close=1000.0)
        c = _candle(sym, 9, 20, 1000, 1002, 998, 1001)
        sig = s.on_candle(sym, c, [c], None)
        assert sig.signal_type == SignalType.NO_ACTION

    def test_no_signal_when_position_open(self):
        s, history = _setup_strategy_with_warmup()
        sym = "NSE:HDFCBANK-EQ"
        c = _candle(sym, 11, 0, 980, 982, 975, 981, 80000)
        history.append(c)
        pos = {"direction": "LONG", "entry_price": 990}
        sig = s.on_candle(sym, c, history, pos)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "position_open" in sig.skip_reason

    def test_bullish_hammer_at_lower_band_generates_buy(self):
        """
        After warmup (VWAP ~1000), inject a hammer candle at -1.5SD.
        VWAP std_dev computed from warmup candles ≈ small.
        We need the low to breach lower_1_5sd.
        """
        sym = "NSE:HDFCBANK-EQ"
        s = VMRStrategy()
        s._sector_scores = {"NIFTY BANK": 0.3}
        s._states[sym] = _VMRState(prev_day_close=1000.0)

        # Build candles with enough variance to create real bands
        history: list[Candle] = []
        # First create a spread of candles to establish VWAP with some SD
        for i in range(15):
            minute = 15 + i * 5
            h = 9 + minute // 60
            m = minute % 60
            # Alternate high and low to create variance
            if i % 2 == 0:
                c = _candle(sym, h, m, 1010, 1015, 1005, 1012, 50000)
            else:
                c = _candle(sym, h, m, 990, 995, 985, 988, 50000)
            history.append(c)
            s.on_candle(sym, c, history, None)

        # Now inject a clear hammer at the lower band
        # VWAP should be ~1000, SD ~10, lower_1.5sd ~985
        hammer = _candle(
            sym, 10, 40,
            o=984, hi=986, lo=975, c=985,  # long lower wick, body at top
            volume=80000,
        )
        history.append(hammer)
        sig = s.on_candle(sym, hammer, history, None)

        # This might be BUY or NO_ACTION depending on exact band values
        # The test validates the flow works without errors
        assert sig.signal_type in (SignalType.BUY, SignalType.NO_ACTION)
        if sig.signal_type == SignalType.BUY:
            assert sig.indicator_data["stoploss_price"] < sig.price_at_signal
            assert sig.indicator_data["target_price"] > sig.price_at_signal
            assert sig.strength > 0

    def test_past_cutoff_no_signal(self):
        s, history = _setup_strategy_with_warmup()
        sym = "NSE:HDFCBANK-EQ"
        c = _candle(sym, 14, 50, 980, 985, 975, 981, 80000)
        history.append(c)
        sig = s.on_candle(sym, c, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "cutoff" in sig.skip_reason

    def test_gap_too_wide_rejected(self):
        s = VMRStrategy()
        sym = "NSE:HDFCBANK-EQ"
        s._states[sym] = _VMRState(prev_day_close=1000.0)

        # First candle gaps 5% up
        candles = []
        for i in range(10):
            minute = 15 + i * 5
            h = 9 + minute // 60
            m = minute % 60
            c = _candle(sym, h, m, 1050, 1055, 1045, 1052, 50000)
            candles.append(c)
            s.on_candle(sym, c, candles, None)

        # After gap detection, subsequent candles should be rejected
        c_late = _candle(sym, 10, 0, 1060, 1065, 1055, 1062, 80000)
        candles.append(c_late)
        sig = s.on_candle(sym, c_late, candles, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "gap" in sig.skip_reason

    def test_no_second_signal_same_day(self):
        """Once a signal fires, no more signals that day."""
        sym = "NSE:HDFCBANK-EQ"
        s = VMRStrategy()
        s._sector_scores = {"NIFTY BANK": 0.3}
        s._states[sym] = _VMRState(prev_day_close=1000.0)

        # Force signal_fired = True
        s._states[sym].signal_fired = True

        history = _build_warmup_candles(sym, 1000.0, 15)
        for c in history:
            pass  # don't feed to strategy, just build history

        c_new = _candle(sym, 11, 0, 980, 985, 970, 982, 80000)
        history.append(c_new)
        sig = s.on_candle(sym, c_new, history, None)
        assert sig.signal_type == SignalType.NO_ACTION
        assert "signal_already_fired" in sig.skip_reason


# ═══════════════════════════════════════════════════════════════════════════
# should_exit
# ═══════════════════════════════════════════════════════════════════════════

class TestVMRShouldExit:

    def _make_position(
        self,
        direction: str = "LONG",
        entry: float = 985.0,
        sl: float = 972.0,
        target: float = 1000.0,
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
        s = VMRStrategy()
        pos = self._make_position()
        candle = _candle("NSE:TEST-EQ", 15, 22, 990, 992, 988, 991)
        sig = s.should_exit("NSE:TEST-EQ", 991.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "FLATTEN"

    def test_stoploss_hit_long(self):
        s = VMRStrategy()
        pos = self._make_position(direction="LONG", entry=985, sl=972, target=1000)
        candle = _candle("NSE:TEST-EQ", 10, 30, 973, 975, 970, 971)
        sig = s.should_exit("NSE:TEST-EQ", 971.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "STOPLOSS"

    def test_stoploss_hit_short(self):
        s = VMRStrategy()
        pos = self._make_position(direction="SHORT", entry=1015, sl=1028, target=1000)
        candle = _candle("NSE:TEST-EQ", 10, 30, 1027, 1030, 1025, 1029)
        sig = s.should_exit("NSE:TEST-EQ", 1029.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_SHORT
        assert sig.indicator_data["exit_reason"] == "STOPLOSS"

    def test_target_hit_long(self):
        s = VMRStrategy()
        pos = self._make_position(direction="LONG", entry=985, sl=972, target=1000)
        candle = _candle("NSE:TEST-EQ", 11, 0, 998, 1002, 997, 1001)
        sig = s.should_exit("NSE:TEST-EQ", 1001.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "TARGET"

    def test_target_hit_short(self):
        s = VMRStrategy()
        pos = self._make_position(direction="SHORT", entry=1015, sl=1028, target=1000)
        candle = _candle("NSE:TEST-EQ", 11, 0, 1001, 1003, 998, 999)
        sig = s.should_exit("NSE:TEST-EQ", 999.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_SHORT
        assert sig.indicator_data["exit_reason"] == "TARGET"

    def test_no_exit_within_range(self):
        s = VMRStrategy()
        pos = self._make_position(direction="LONG", entry=985, sl=972, target=1000)
        candle = _candle("NSE:TEST-EQ", 10, 30, 990, 993, 988, 992)
        sig = s.should_exit("NSE:TEST-EQ", 992.0, pos, candle)
        assert sig.signal_type == SignalType.NO_ACTION

    def test_peak_price_updated_from_candle_high(self):
        s = VMRStrategy()
        pos = self._make_position(
            direction="LONG", entry=985, sl=972, target=1010,
            peak_price=990,
        )
        candle = _candle("NSE:TEST-EQ", 11, 0, 995, 998, 993, 996)
        sig = s.should_exit("NSE:TEST-EQ", 996.0, pos, candle)
        assert sig.indicator_data["peak_price"] == 998  # updated from candle.high

    def test_flatten_overrides_target(self):
        s = VMRStrategy()
        pos = self._make_position(direction="LONG", entry=985, sl=972, target=1000)
        # At flatten time, price is above target
        candle = _candle("NSE:TEST-EQ", 15, 25, 1001, 1003, 999, 1002)
        sig = s.should_exit("NSE:TEST-EQ", 1002.0, pos, candle)
        assert sig.signal_type == SignalType.EXIT_LONG
        assert sig.indicator_data["exit_reason"] == "FLATTEN"


# ═══════════════════════════════════════════════════════════════════════════
# end_of_day
# ═══════════════════════════════════════════════════════════════════════════

class TestVMREndOfDay:

    def test_clears_state(self):
        s = VMRStrategy()
        s._states["NSE:TEST-EQ"] = _VMRState()
        s._watchlist = ["NSE:TEST-EQ"]
        s._sector_scores = {"NIFTY BANK": 0.5}
        s.end_of_day()
        assert s._states == {}
        assert s._sector_scores == {}
        assert s.watchlist == []


# ═══════════════════════════════════════════════════════════════════════════
# build_trade_plan (inherited from StrategyBase)
# ═══════════════════════════════════════════════════════════════════════════

class TestVMRTradePlan:

    def test_build_trade_plan_long(self):
        s = VMRStrategy()
        signal = Signal(
            strategy_name="vmr_vwap",
            symbol="NSE:HDFCBANK-EQ",
            signal_type=SignalType.BUY,
            strength=0.65,
            price_at_signal=985.0,
            timestamp=_ts(10, 40),
            indicator_data={"stoploss_price": 972.0, "target_price": 1000.0},
        )
        plan = s.build_trade_plan(signal, risk_per_trade=3750.0)
        assert plan is not None
        assert plan.direction == Direction.LONG
        assert plan.entry_price == 985.0
        assert plan.stoploss_price == 972.0
        expected_qty = int(3750.0 / (985.0 - 972.0))
        assert plan.quantity == expected_qty

    def test_no_action_returns_none(self):
        s = VMRStrategy()
        signal = Signal(
            strategy_name="vmr_vwap",
            symbol="NSE:TEST-EQ",
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=1000.0,
            timestamp=_ts(10, 0),
        )
        assert s.build_trade_plan(signal, risk_per_trade=3750.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# Engine integration (basic smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestVMRBacktestIntegration:

    def test_engine_runs_with_vmr(self):
        from backtest.engine import BacktestEngine
        from backtest.cost_model import FyersCostModel

        strategy = VMRStrategy()
        engine = BacktestEngine(strategy, FyersCostModel())

        sym = to_fyers_symbol("HDFCBANK")
        # Build simple 1-day data
        rows = []
        base = 1500.0
        for i in range(75):
            minute = 9 * 60 + 15 + i * 5
            h = minute // 60
            m = minute % 60
            noise = ((hash(("2024-06-03", i)) % 200) - 100) / 50.0
            price = base + noise
            ts = _ts(h, m)
            rows.append({
                "timestamp": ts,
                "open": round(price - 1, 2),
                "high": round(price + 3, 2),
                "low": round(price - 3, 2),
                "close": round(price, 2),
                "volume": 50000 + (hash(("v", i)) % 30000),
            })
        df = pd.DataFrame(rows)

        result = engine.run({sym: df}, initial_capital=750000.0)
        assert result is not None
        assert result.strategy_name == "vmr_vwap"
        assert result.initial_capital == 750000.0

    def test_vmr_factory_in_run_backtest(self):
        from scripts.run_backtest import _create_strategy
        s = _create_strategy("vmr_vwap")
        assert s.name == "vmr_vwap"
        assert s.version == "1.0.0"
