"""
NSE Trading Platform — Backtest Engine

Event-driven backtester that simulates strategy execution on historical data.
Uses the same StrategyBase interface as live trading for consistency.

Implementation: Strategy 1 conversation (alongside ORB+VWAP).
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from config.settings import settings
from core.data.models import BacktestResult, Candle, SignalType
from core.strategies.base import StrategyBase
from core.utils.time_utils import IST, epoch_ms_to_ist, ist_to_epoch_ms
from backtest.cost_model import FyersCostModel
from backtest.performance import compute_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates:
      - Candle-by-candle strategy execution
      - Order fills with slippage
      - Transaction costs (FyersCostModel)
      - Position tracking and P&L
      - Risk management rules

    Usage
    -----
    >>> engine = BacktestEngine(strategy, cost_model)
    >>> result = engine.run(data, initial_capital=750000)
    >>> print(f"Net P&L: ₹{result.total_pnl:,.2f}")
    """

    def __init__(
        self,
        strategy: StrategyBase,
        cost_model: FyersCostModel | None = None,
    ) -> None:
        self._strategy = strategy
        self._cost_model = cost_model or FyersCostModel()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 750_000.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """
        Run a backtest.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Symbol → OHLCV DataFrame (intraday 5m candles).
            DataFrame must have columns: timestamp, open, high, low, close, volume.
        initial_capital : float
            Starting capital in INR.
        start_date : str, optional
            Filter data from this date. 'YYYY-MM-DD'.
        end_date : str, optional
            Filter data to this date.

        Returns
        -------
        BacktestResult
            Complete backtest results with trade list and equity curve.
        """
        wall_start = time.time()

        capital = initial_capital
        all_trades: list[dict] = []
        equity_curve: list[tuple[int, float]] = []
        open_positions: dict[str, dict] = {}  # symbol → position dict

        # ── 1. Pre-market scan using daily data ───────────────────────
        # Strategy expects Fyers-format symbols and DataFrames
        watchlist = self._strategy.pre_market_scan(
            universe=list(data.keys()),
            historical_data=data,
        )

        if not watchlist:
            watchlist = list(data.keys())

        # ── 2. Build unified candle timeline ──────────────────────────
        # Convert all DataFrames to Candle objects and merge into a
        # single timeline sorted by (timestamp, symbol).
        all_candles: list[Candle] = []
        candle_history: dict[str, list[Candle]] = {sym: [] for sym in watchlist}

        for symbol in watchlist:
            df = data.get(symbol)
            if df is None or df.empty:
                continue

            df = df.copy()

            # Apply date filters
            if start_date and "timestamp" in df.columns:
                start_dt = pd.Timestamp(start_date, tz=IST)
                start_ms = int(start_dt.timestamp() * 1000)
                df = df[df["timestamp"] >= start_ms]
            if end_date and "timestamp" in df.columns:
                end_dt = pd.Timestamp(end_date, tz=IST) + pd.Timedelta(days=1)
                end_ms = int(end_dt.timestamp() * 1000)
                df = df[df["timestamp"] < end_ms]

            for _, row in df.iterrows():
                candle = Candle(
                    symbol=symbol,
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    timeframe="5m",
                )
                all_candles.append(candle)

        # Sort by timestamp, then symbol for deterministic ordering
        all_candles.sort(key=lambda c: (c.timestamp, c.symbol))

        if not all_candles:
            return self._empty_result(initial_capital, start_date, end_date, wall_start)

        # ── 3. Track trading days for daily resets ────────────────────
        current_day: str = ""

        # ── 4. Candle-by-candle simulation ────────────────────────────
        for candle in all_candles:
            symbol = candle.symbol
            ts = candle.timestamp
            candle_dt = epoch_ms_to_ist(ts)
            day_str = candle_dt.strftime("%Y-%m-%d")

            # ── Daily reset (new trading day) ─────────────────────────
            if day_str != current_day:
                # Force-flatten any overnight positions from previous day
                for sym, pos in list(open_positions.items()):
                    trade = self._close_position(pos, pos["last_price"], ts, "FLATTEN")
                    all_trades.append(trade)
                    capital += trade["pnl_net"]
                open_positions.clear()

                # Reset strategy state for new day
                self._strategy.end_of_day()

                # Re-run pre_market_scan for new day
                self._strategy.pre_market_scan(
                    universe=list(data.keys()),
                    historical_data=data,
                )

                current_day = day_str

            # ── Update candle history ─────────────────────────────────
            if symbol not in candle_history:
                candle_history[symbol] = []
            candle_history[symbol].append(candle)

            # ── Check exits first for open positions ──────────────────
            if symbol in open_positions:
                pos = open_positions[symbol]
                # Update peak price for trailing stop
                if pos["direction"] == "LONG":
                    pos["peak_price"] = max(pos.get("peak_price", pos["entry_price"]), candle.high)
                else:
                    pos["peak_price"] = min(pos.get("peak_price", pos["entry_price"]), candle.low)
                pos["last_price"] = candle.close

                exit_signal = self._strategy.should_exit(
                    symbol=symbol,
                    current_price=candle.close,
                    position=pos,
                    candle=candle,
                )

                if exit_signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
                    exit_reason = exit_signal.indicator_data.get("exit_reason", "UNKNOWN")
                    exit_price = self._apply_slippage(
                        candle.close, pos["direction"], is_exit=True,
                    )
                    trade = self._close_position(pos, exit_price, ts, exit_reason)
                    all_trades.append(trade)
                    capital += trade["pnl_net"]
                    del open_positions[symbol]

                    equity_curve.append((ts, capital))
                    continue

            # ── Check entries (only if no position in this symbol) ────
            if symbol not in open_positions:
                current_pos = None
            else:
                current_pos = open_positions[symbol]

            entry_signal = self._strategy.on_candle(
                symbol=symbol,
                candle=candle,
                candle_history=candle_history[symbol],
                current_position=current_pos,
            )

            if entry_signal.signal_type in (SignalType.BUY, SignalType.SELL):
                # ── Position sizing ───────────────────────────────────
                stoploss = entry_signal.indicator_data.get("stoploss_price")
                target = entry_signal.indicator_data.get("target_price")

                if stoploss is None:
                    continue

                entry_price = self._apply_slippage(
                    candle.close,
                    "LONG" if entry_signal.signal_type == SignalType.BUY else "SHORT",
                    is_exit=False,
                )

                risk_per_share = abs(entry_price - stoploss)
                if risk_per_share <= 0:
                    continue

                # Max concurrent positions check
                if len(open_positions) >= settings.MAX_CONCURRENT_POSITIONS:
                    continue

                # Capital check
                risk_budget = min(settings.RISK_PER_TRADE_INR, capital * 0.02)
                quantity = int(risk_budget / risk_per_share)
                if quantity <= 0:
                    continue

                # Check if we have enough capital for the position
                position_value = entry_price * quantity
                if position_value > capital:
                    quantity = int(capital / entry_price)
                    if quantity <= 0:
                        continue

                direction = "LONG" if entry_signal.signal_type == SignalType.BUY else "SHORT"

                open_positions[symbol] = {
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stoploss_price": stoploss,
                    "target_price": target,
                    "quantity": quantity,
                    "entry_time": ts,
                    "peak_price": entry_price,
                    "last_price": candle.close,
                    "indicator_data": entry_signal.indicator_data,
                }

            # ── Record equity at each candle ──────────────────────────
            unrealised = 0.0
            for sym, pos in open_positions.items():
                if pos["direction"] == "LONG":
                    unrealised += (pos["last_price"] - pos["entry_price"]) * pos["quantity"]
                else:
                    unrealised += (pos["entry_price"] - pos["last_price"]) * pos["quantity"]

            equity_curve.append((ts, capital + unrealised))

        # ── 5. Force-flatten any remaining positions ──────────────────
        last_ts = all_candles[-1].timestamp if all_candles else 0
        for sym, pos in list(open_positions.items()):
            trade = self._close_position(pos, pos["last_price"], last_ts, "FLATTEN")
            all_trades.append(trade)
            capital += trade["pnl_net"]
        open_positions.clear()

        if equity_curve:
            equity_curve.append((last_ts, capital))

        # ── 6. Compute performance metrics ────────────────────────────
        metrics = compute_metrics(all_trades, initial_capital, equity_curve)

        wall_end = time.time()

        return BacktestResult(
            strategy_name=self._strategy.name,
            symbol_universe=",".join(watchlist[:10]) if len(watchlist) <= 10 else f"NIFTY50({len(watchlist)})",
            start_date=start_date or "",
            end_date=end_date or "",
            initial_capital=initial_capital,
            final_capital=capital,
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            losing_trades=metrics.get("losing_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            total_pnl=metrics.get("total_pnl", 0.0),
            max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            avg_trade_pnl=metrics.get("avg_trade_pnl", 0.0),
            max_consecutive_losses=metrics.get("max_consecutive_losses", 0),
            params=self._strategy.get_params(),
            trades=all_trades,
            equity_curve=equity_curve,
            duration_sec=round(wall_end - wall_start, 3),
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _close_position(
        self,
        position: dict,
        exit_price: float,
        exit_time: int,
        exit_reason: str,
    ) -> dict:
        """Close a position and compute P&L including costs."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Compute costs
        pnl_result = self._cost_model.compute_net_pnl(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            direction=direction,
            trade_type="INTRADAY",
        )

        return {
            "symbol": position["symbol"],
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": position["entry_time"],
            "exit_price": exit_price,
            "exit_time": exit_time,
            "stoploss_price": position.get("stoploss_price"),
            "target_price": position.get("target_price"),
            "pnl_gross": round(gross_pnl, 2),
            "pnl_net": round(pnl_result["net_pnl"], 2),
            "costs_total": round(pnl_result["costs"], 2),
            "exit_reason": exit_reason,
            "indicator_data": position.get("indicator_data", {}),
        }

    def _apply_slippage(
        self,
        price: float,
        direction: str,
        is_exit: bool,
    ) -> float:
        """
        Apply slippage to simulate real fills.

        Entry: unfavorable direction (LONG buys higher, SHORT sells lower).
        Exit: unfavorable direction (LONG sells lower, SHORT buys higher).
        """
        slippage_pct = settings.SLIPPAGE_PCT / 100.0

        if direction == "LONG":
            if is_exit:
                # Selling: worse fill = lower price
                return round(price * (1.0 - slippage_pct), 2)
            else:
                # Buying: worse fill = higher price
                return round(price * (1.0 + slippage_pct), 2)
        else:  # SHORT
            if is_exit:
                # Buying to cover: worse fill = higher price
                return round(price * (1.0 + slippage_pct), 2)
            else:
                # Selling short: worse fill = lower price
                return round(price * (1.0 - slippage_pct), 2)

    def _empty_result(
        self,
        initial_capital: float,
        start_date: str | None,
        end_date: str | None,
        wall_start: float,
    ) -> BacktestResult:
        """Return an empty result when no data is available."""
        return BacktestResult(
            strategy_name=self._strategy.name,
            symbol_universe="",
            start_date=start_date or "",
            end_date=end_date or "",
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_trades=0,
            total_pnl=0.0,
            params=self._strategy.get_params(),
            duration_sec=round(time.time() - wall_start, 3),
        )
