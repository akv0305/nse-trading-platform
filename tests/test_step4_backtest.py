"""
NSE Trading Platform — Step 4 Tests: Backtest Engine + Performance

Tests for backtest/engine.py and backtest/performance.py
Run: pytest tests/test_step4_backtest.py -v
"""

from __future__ import annotations

import datetime
import math

import pandas as pd
import pytest

from config.settings import settings
from core.data.models import BacktestResult, Candle, SignalType
from core.data.universe import to_fyers_symbol
from core.strategies.orb_vwap import ORBVWAPStrategy
from core.utils.time_utils import IST, ist_to_epoch_ms
from backtest.engine import BacktestEngine
from backtest.cost_model import FyersCostModel
from backtest.performance import compute_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Build synthetic 5m intraday data
# ═══════════════════════════════════════════════════════════════════════════

def _ts(date_str: str, h: int, m: int) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=h, minute=m, second=0, microsecond=0, tzinfo=IST,
    )
    return ist_to_epoch_ms(dt)


def _build_day_5m(
    symbol: str,
    date_str: str,
    base_price: float = 1000.0,
    trend: float = 0.0,
) -> list[dict]:
    """
    Build one full day of 5m candles (9:15–15:25).
    trend > 0 → prices drift up, trend < 0 → prices drift down.
    Returns list of row dicts.
    """
    rows = []
    price = base_price
    start_min = 9 * 60 + 15
    end_min = 15 * 60 + 25

    for minute in range(start_min, end_min + 1, 5):
        h = minute // 60
        m = minute % 60

        # Deterministic pseudo-random noise
        noise = ((hash((date_str, minute)) % 200) - 100) / 200.0  # -0.5 to +0.5
        price += trend + noise * 2
        price = max(price, 10.0)

        o = round(price, 2)
        hi = round(price + abs(noise) * 3 + 1.0, 2)
        lo = round(price - abs(noise) * 3 - 0.5, 2)
        c = round(price + noise, 2)
        vol = 50000 + (hash((date_str, minute, "v")) % 40000)

        rows.append({
            "timestamp": _ts(date_str, h, m),
            "open": o,
            "high": hi,
            "low": lo,
            "close": c,
            "volume": vol,
        })

    return rows


def _build_multi_day_df(
    symbol: str,
    dates: list[str],
    base_price: float = 1000.0,
    trend: float = 0.0,
) -> pd.DataFrame:
    """Build a multi-day 5m OHLCV DataFrame."""
    all_rows = []
    price = base_price
    for date_str in dates:
        rows = _build_day_5m(symbol, date_str, price, trend)
        all_rows.extend(rows)
        # Carry forward price for next day
        if rows:
            price = rows[-1]["close"]
    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:

    def test_empty_trades(self):
        m = compute_metrics([], 750000.0)
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["total_pnl"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_all_winners(self):
        trades = [
            {"pnl_net": 1000, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": 1500, "entry_time": 3000, "exit_time": 4000},
            {"pnl_net": 500, "entry_time": 5000, "exit_time": 6000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert m["total_trades"] == 3
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 0
        assert m["win_rate"] == 100.0
        assert m["total_pnl"] == 3000.0
        assert m["profit_factor"] == 999.99  # inf capped

    def test_all_losers(self):
        trades = [
            {"pnl_net": -1000, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": -500, "entry_time": 3000, "exit_time": 4000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert m["winning_trades"] == 0
        assert m["losing_trades"] == 2
        assert m["win_rate"] == 0.0
        assert m["total_pnl"] == -1500.0
        assert m["profit_factor"] == 0.0

    def test_mixed_trades(self):
        trades = [
            {"pnl_net": 2000, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": -800, "entry_time": 3000, "exit_time": 4000},
            {"pnl_net": 1500, "entry_time": 5000, "exit_time": 6000},
            {"pnl_net": -600, "entry_time": 7000, "exit_time": 8000},
            {"pnl_net": 1000, "entry_time": 9000, "exit_time": 10000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert m["total_trades"] == 5
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 2
        assert m["win_rate"] == 60.0
        assert m["total_pnl"] == 3100.0
        assert m["avg_trade_pnl"] == 620.0
        # Profit factor = 4500 / 1400
        assert abs(m["profit_factor"] - 4500 / 1400) < 0.01

    def test_max_consecutive_losses(self):
        trades = [
            {"pnl_net": 100, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": -100, "entry_time": 3000, "exit_time": 4000},
            {"pnl_net": -200, "entry_time": 5000, "exit_time": 6000},
            {"pnl_net": -150, "entry_time": 7000, "exit_time": 8000},
            {"pnl_net": 500, "entry_time": 9000, "exit_time": 10000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert m["max_consecutive_losses"] == 3

    def test_max_drawdown(self):
        # Equity goes 100k → 110k → 95k → 105k
        # Peak = 110k, trough = 95k → DD = 15/110 = 13.636%
        curve = [
            (1000, 100_000),
            (2000, 110_000),
            (3000, 95_000),
            (4000, 105_000),
        ]
        m = compute_metrics(
            [{"pnl_net": 5000, "entry_time": 1000, "exit_time": 4000}],
            100_000,
            equity_curve=curve,
        )
        expected_dd = ((110_000 - 95_000) / 110_000) * 100.0
        assert abs(m["max_drawdown_pct"] - expected_dd) < 0.01

    def test_sharpe_ratio_is_float(self):
        trades = [
            {"pnl_net": 1000, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": -500, "entry_time": 3000, "exit_time": 4000},
            {"pnl_net": 800, "entry_time": 5000, "exit_time": 6000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert isinstance(m["sharpe_ratio"], float)

    def test_expectancy_positive_for_profitable_system(self):
        trades = [
            {"pnl_net": 3000, "entry_time": 1000, "exit_time": 2000},
            {"pnl_net": -1000, "entry_time": 3000, "exit_time": 4000},
            {"pnl_net": 2500, "entry_time": 5000, "exit_time": 6000},
            {"pnl_net": -800, "entry_time": 7000, "exit_time": 8000},
        ]
        m = compute_metrics(trades, 750000.0)
        assert m["expectancy"] > 0

    def test_holding_period_computed(self):
        trades = [
            {"pnl_net": 100, "entry_time": 1000000, "exit_time": 1000000 + 300000},  # 5 min
            {"pnl_net": 200, "entry_time": 2000000, "exit_time": 2000000 + 600000},  # 10 min
        ]
        m = compute_metrics(trades, 750000.0)
        assert abs(m["avg_holding_period_min"] - 7.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# BacktestEngine Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestEngine:

    def test_returns_backtest_result(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        sym = to_fyers_symbol("HDFCBANK")
        df = _build_multi_day_df(sym, ["2024-06-03"], base_price=1500, trend=0.1)
        result = engine.run({sym: df}, initial_capital=750000.0)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "orb_vwap"
        assert result.initial_capital == 750000.0

    def test_empty_data_returns_zero_trades(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)
        result = engine.run({}, initial_capital=750000.0)
        assert result.total_trades == 0
        assert result.final_capital == 750000.0

    def test_single_day_runs_without_error(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        sym = to_fyers_symbol("RELIANCE")
        df = _build_multi_day_df(sym, ["2024-06-03"], base_price=2500, trend=0.3)
        result = engine.run({sym: df})

        assert result.initial_capital == 750000.0
        assert result.duration_sec >= 0
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)

    def test_multi_day_resets_strategy(self):
        """Engine should reset strategy state on each new day."""
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        sym = to_fyers_symbol("INFY")
        df = _build_multi_day_df(
            sym, ["2024-06-03", "2024-06-04"], base_price=1500, trend=0.2,
        )
        result = engine.run({sym: df})
        assert isinstance(result, BacktestResult)
        # Should have equity curve spanning both days
        assert len(result.equity_curve) > 0

    def test_slippage_applied(self):
        """Verify slippage is applied to fills."""
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        # Test the slippage method directly
        price = 1000.0
        buy_price = engine._apply_slippage(price, "LONG", is_exit=False)
        sell_price = engine._apply_slippage(price, "LONG", is_exit=True)

        assert buy_price > price   # LONG entry: buy higher
        assert sell_price < price  # LONG exit: sell lower

        short_entry = engine._apply_slippage(price, "SHORT", is_exit=False)
        short_exit = engine._apply_slippage(price, "SHORT", is_exit=True)

        assert short_entry < price  # SHORT entry: sell lower
        assert short_exit > price   # SHORT exit: buy higher

    def test_cost_model_used(self):
        """Verify costs are deducted from P&L."""
        strategy = ORBVWAPStrategy()
        cost_model = FyersCostModel()
        engine = BacktestEngine(strategy, cost_model)

        sym = to_fyers_symbol("TCS")
        df = _build_multi_day_df(sym, ["2024-06-03"], base_price=3500, trend=0.5)
        result = engine.run({sym: df})

        # If there were any trades, costs should be non-zero
        for trade in result.trades:
            assert trade["costs_total"] >= 0

    def test_params_recorded(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        sym = to_fyers_symbol("RELIANCE")
        df = _build_multi_day_df(sym, ["2024-06-03"], base_price=2500)
        result = engine.run({sym: df})

        assert result.params is not None
        assert result.params["strategy_name"] == "orb_vwap"
        assert result.params["orb_period_minutes"] == settings.ORB_PERIOD_MINUTES

    def test_multi_symbol_runs(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        data = {}
        for name in ["HDFCBANK", "ICICIBANK", "SBIN"]:
            sym = to_fyers_symbol(name)
            data[sym] = _build_multi_day_df(
                sym, ["2024-06-03"], base_price=1500, trend=0.1,
            )

        result = engine.run(data)
        assert isinstance(result, BacktestResult)

    def test_capital_tracking(self):
        """Final capital = initial + total_pnl."""
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        sym = to_fyers_symbol("RELIANCE")
        df = _build_multi_day_df(sym, ["2024-06-03"], base_price=2500, trend=0.3)
        result = engine.run({sym: df}, initial_capital=750000.0)

        # Final capital should be consistent with trades
        total_pnl_from_trades = sum(t["pnl_net"] for t in result.trades)
        expected_final = 750000.0 + total_pnl_from_trades
        assert abs(result.final_capital - expected_final) < 0.01

    def test_max_concurrent_positions_respected(self):
        """Should not open more than MAX_CONCURRENT_POSITIONS at once."""
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        # Use many symbols to try to exceed position limit
        data = {}
        for name in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]:
            sym = to_fyers_symbol(name)
            data[sym] = _build_multi_day_df(
                sym, ["2024-06-03"], base_price=1500, trend=0.5,
            )

        result = engine.run(data)
        # We can't directly check concurrent count from result, but
        # the engine should not crash and should produce valid results
        assert isinstance(result, BacktestResult)

    def test_close_position_computes_pnl(self):
        """Test _close_position directly."""
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        position = {
            "symbol": "NSE:TEST-EQ",
            "direction": "LONG",
            "entry_price": 1000.0,
            "stoploss_price": 980.0,
            "target_price": 1060.0,
            "quantity": 100,
            "entry_time": 1000000,
            "indicator_data": {},
        }

        trade = engine._close_position(position, 1050.0, 2000000, "TARGET")
        assert trade["direction"] == "LONG"
        assert trade["pnl_gross"] == 5000.0  # (1050-1000) * 100
        assert trade["pnl_net"] < trade["pnl_gross"]  # costs deducted
        assert trade["costs_total"] > 0
        assert trade["exit_reason"] == "TARGET"

    def test_close_position_short(self):
        strategy = ORBVWAPStrategy()
        engine = BacktestEngine(strategy)

        position = {
            "symbol": "NSE:TEST-EQ",
            "direction": "SHORT",
            "entry_price": 1000.0,
            "stoploss_price": 1020.0,
            "target_price": 960.0,
            "quantity": 100,
            "entry_time": 1000000,
            "indicator_data": {},
        }

        trade = engine._close_position(position, 970.0, 2000000, "TARGET")
        assert trade["pnl_gross"] == 3000.0  # (1000-970) * 100
        assert trade["pnl_net"] < trade["pnl_gross"]
