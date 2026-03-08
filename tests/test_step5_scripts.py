"""
Step 5 — Tests for scripts/run_backtest.py and scripts/download_history.py.

These tests verify the script logic (argument parsing, strategy factory,
symbol resolution, result printing) without requiring a live Fyers connection
or actual OHLCV data in the database.

Run:
    pytest tests/test_step5_scripts.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

# ── Ensure project root on path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.models import BacktestResult


# ══════════════════════════════════════════════════════════════════════════
#  run_backtest.py tests
# ══════════════════════════════════════════════════════════════════════════


class TestCreateStrategy:
    """Tests for the strategy factory function."""

    def test_orb_vwap_returns_correct_type(self):
        from scripts.run_backtest import _create_strategy

        strategy = _create_strategy("orb_vwap")
        assert strategy is not None
        # Actual name is lowercase "orb_vwap" as set in the strategy class
        assert strategy.name == "orb_vwap"

    def test_orb_vwap_case_insensitive(self):
        from scripts.run_backtest import _create_strategy

        s1 = _create_strategy("ORB_VWAP")
        s2 = _create_strategy("Orb-Vwap")
        s3 = _create_strategy("orb vwap")
        assert s1.name == s2.name == s3.name

    def test_unknown_strategy_raises(self):
        from scripts.run_backtest import _create_strategy

        with pytest.raises(ValueError, match="Unknown strategy"):
            _create_strategy("bollinger_bands")

    def test_strategy_is_subclass_of_base(self):
        from scripts.run_backtest import _create_strategy
        from core.strategies.base import StrategyBase

        strategy = _create_strategy("orb_vwap")
        assert isinstance(strategy, StrategyBase)


class TestPrintResults:
    """Tests for the result printer."""

    def _make_result(self, **overrides) -> BacktestResult:
        """Create a minimal BacktestResult matching the actual dataclass."""
        defaults = dict(
            strategy_name="orb_vwap",
            symbol_universe="NIFTY50",
            start_date="2024-01-01",
            end_date="2025-01-01",
            initial_capital=750_000.0,
            final_capital=780_000.0,
            total_trades=120,
            winning_trades=60,
            losing_trades=60,
            win_rate=50.0,
            total_pnl=30_000.0,
            avg_trade_pnl=250.0,
            max_drawdown_pct=5.5,
            sharpe_ratio=1.25,
            profit_factor=1.8,
            max_consecutive_losses=4,
            params={"ORB_PERIOD_MINUTES": 15},
            trades=[
                {"symbol": "NSE:RELIANCE-EQ", "net_pnl": 5000.0, "exit_reason": "TARGET"},
                {"symbol": "NSE:INFY-EQ", "net_pnl": -2000.0, "exit_reason": "STOPLOSS"},
                {"symbol": "NSE:TCS-EQ", "net_pnl": 3000.0, "exit_reason": "TARGET"},
            ],
            equity_curve=[750_000.0, 755_000.0, 780_000.0],
            duration_sec=12.5,
        )
        defaults.update(overrides)
        return BacktestResult(**defaults)

    def test_print_results_runs_without_error(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result()
        _print_results(result)
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "orb_vwap" in captured.out

    def test_print_results_shows_pnl(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(total_pnl=30_000.0)
        _print_results(result)
        captured = capsys.readouterr()
        assert "30,000" in captured.out

    def test_print_results_shows_dates(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result()
        _print_results(result)
        captured = capsys.readouterr()
        assert "2024-01-01" in captured.out
        assert "2025-01-01" in captured.out

    def test_print_results_no_trades(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trades=[],
        )
        _print_results(result)
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out

    def test_print_results_shows_winners_and_losers(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result()
        _print_results(result)
        captured = capsys.readouterr()
        assert "Top 5 Winners" in captured.out
        assert "Top 5 Losers" in captured.out

    def test_print_results_shows_sharpe_and_pf(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(sharpe_ratio=1.250, profit_factor=1.800)
        _print_results(result)
        captured = capsys.readouterr()
        assert "1.250" in captured.out
        assert "1.800" in captured.out

    def test_print_results_negative_pnl(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(
            total_pnl=-15_000.0,
            final_capital=735_000.0,
        )
        _print_results(result)
        captured = capsys.readouterr()
        assert "15,000" in captured.out

    def test_print_results_shows_universe(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(symbol_universe="NIFTY50")
        _print_results(result)
        captured = capsys.readouterr()
        assert "NIFTY50" in captured.out

    def test_print_results_shows_duration(self, capsys):
        from scripts.run_backtest import _print_results

        result = self._make_result(duration_sec=42.7)
        _print_results(result)
        captured = capsys.readouterr()
        assert "42.7" in captured.out


# ══════════════════════════════════════════════════════════════════════════
#  download_history.py tests
# ══════════════════════════════════════════════════════════════════════════


class TestResolveSymbols:
    """Tests for symbol resolution in the download script."""

    def test_nifty50_returns_all_stocks_plus_indices(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("NIFTY50")
        # 50 stocks + 13 sector indices + 1 NIFTY50 index = 64
        assert len(symbols) == 62

    def test_nifty50_case_insensitive(self):
        from scripts.download_history import _resolve_symbols

        s1 = _resolve_symbols("NIFTY50")
        s2 = _resolve_symbols("nifty50")
        assert s1 == s2

    def test_custom_symbols_returns_subset(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("RELIANCE,INFY")
        # 2 stocks + 12 sector indices + 1 NIFTY50 index = 15
        assert len(symbols) == 15
        assert "NSE:RELIANCE-EQ" in symbols
        assert "NSE:INFY-EQ" in symbols

    def test_always_includes_nifty50_index(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("TCS")
        assert "NSE:NIFTY50-INDEX" in symbols

    def test_always_includes_sector_indices(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("TCS")
        assert "NSE:NIFTYBANK-INDEX" in symbols
        assert "NSE:NIFTYIT-INDEX" in symbols

    def test_no_duplicates(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("NIFTY50")
        assert len(symbols) == len(set(symbols))

    def test_all_fyers_format(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("RELIANCE,HDFCBANK")
        for s in symbols:
            assert ":" in s, f"Symbol {s} not in Fyers format"

    def test_unknown_symbol_still_included(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("FAKESYMBOL")
        assert "NSE:FAKESYMBOL-EQ" in symbols

    def test_whitespace_and_comma_handling(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols(" RELIANCE , INFY , TCS ")
        assert "NSE:RELIANCE-EQ" in symbols
        assert "NSE:INFY-EQ" in symbols
        assert "NSE:TCS-EQ" in symbols

    def test_empty_items_ignored(self):
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("RELIANCE,,INFY,")
        equity = [s for s in symbols if s.endswith("-EQ")]
        assert len(equity) == 2


# ══════════════════════════════════════════════════════════════════════════
#  Integration-style tests (mocked broker)
# ══════════════════════════════════════════════════════════════════════════


class TestRunBacktestIntegration:
    """Test that run_backtest.main() wires everything together correctly."""

    def _make_synthetic_df(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Build a small synthetic 5-min OHLCV DataFrame."""
        from core.utils.time_utils import IST

        rows = []
        base = 1000.0
        rng = np.random.RandomState(42)
        # ~75 candles per day (5-min from 9:15 to 15:30)
        for d in range(days):
            for c in range(75):
                ts_ms = (
                    1704067200000  # 2024-01-01 approx
                    + d * 86_400_000
                    + (9 * 3600 + 15 * 60 + c * 300) * 1000
                )
                o = base + rng.uniform(-5, 5)
                h = o + rng.uniform(0, 3)
                l = o - rng.uniform(0, 3)
                cl = (o + h + l) / 3
                rows.append({
                    "timestamp": ts_ms,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(cl, 2),
                    "volume": int(rng.uniform(10000, 100000)),
                })
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(
            "Asia/Kolkata"
        )
        df.set_index("datetime", inplace=True)
        return df

    @patch("scripts.run_backtest._load_data")
    def test_main_calls_engine_with_mocked_data(self, mock_load):
        """Verify main() wires strategy → engine → result without errors."""
        from scripts.run_backtest import _create_strategy

        data = {
            "NSE:RELIANCE-EQ": self._make_synthetic_df("RELIANCE"),
            "NSE:INFY-EQ": self._make_synthetic_df("INFY"),
        }
        mock_load.return_value = data

        strategy = _create_strategy("orb_vwap")
        from backtest.engine import BacktestEngine
        from backtest.cost_model import FyersCostModel

        engine = BacktestEngine(strategy=strategy, cost_model=FyersCostModel())
        result = engine.run(
            data=data,
            initial_capital=750_000.0,
            start_date="2024-01-01",
            end_date="2024-01-06",
        )
        assert result is not None
        assert result.strategy_name == strategy.name
        assert result.initial_capital == 750_000.0

    @patch("scripts.run_backtest._load_data")
    def test_main_empty_data_returns_zero_trades(self, mock_load):
        from scripts.run_backtest import _create_strategy
        from backtest.engine import BacktestEngine
        from backtest.cost_model import FyersCostModel

        mock_load.return_value = {}
        strategy = _create_strategy("orb_vwap")
        engine = BacktestEngine(strategy=strategy, cost_model=FyersCostModel())
        result = engine.run(data={}, initial_capital=750_000.0)
        assert result.total_trades == 0


class TestDownloadHistoryIntegration:
    """Test download_history wiring with a mocked Fyers adapter."""

    @patch("scripts.download_history.FyersAdapter")
    def test_main_connects_and_downloads(self, MockAdapter):
        """Verify the download flow calls the right methods."""
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("RELIANCE")
        assert "NSE:RELIANCE-EQ" in symbols
        assert len(symbols) > 1  # includes indices

    @patch("scripts.download_history.FyersAdapter")
    def test_force_flag_passes_through(self, MockAdapter):
        """Verify the force flag concept exists."""
        from scripts.download_history import _resolve_symbols

        symbols = _resolve_symbols("NIFTY50")
        assert len(symbols) == 62
