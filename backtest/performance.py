"""
NSE Trading Platform — Performance Metrics Calculator

Computes standard trading performance metrics from a list of trades.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

import pandas as pd

from core.data.models import BacktestResult


def compute_metrics(
    trades: list[dict],
    initial_capital: float,
    equity_curve: list[tuple[int, float]] | None = None,
) -> dict:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    trades : list[dict]
        Each dict: {'entry_price', 'exit_price', 'quantity', 'direction',
                     'pnl_net', 'entry_time', 'exit_time', ...}
    initial_capital : float
    equity_curve : list[tuple], optional
        (timestamp, equity) pairs for drawdown calculation.

    Returns
    -------
    dict
        Keys: total_trades, winning_trades, losing_trades, win_rate,
        total_pnl, avg_trade_pnl, max_drawdown_pct, sharpe_ratio,
        profit_factor, max_consecutive_losses, avg_holding_period,
        expectancy, calmar_ratio.
    """
    raise NotImplementedError("Performance metrics — implement in Strategy 1 conversation")
