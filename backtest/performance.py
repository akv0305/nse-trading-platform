"""
NSE Trading Platform — Performance Metrics Calculator

Computes standard trading performance metrics from a list of trades.

Implementation: Strategy 1 conversation.
"""

from __future__ import annotations

import math

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
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_losses": 0,
            "avg_holding_period_min": 0.0,
            "expectancy": 0.0,
            "calmar_ratio": 0.0,
        }

    # ── Basic counts ──────────────────────────────────────────────────
    total_trades = len(trades)
    pnls = [t.get("pnl_net", 0.0) for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]
    winning_trades = len(winning)
    losing_trades = len(losing)

    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum(pnls)
    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

    # ── Gross profit / gross loss for profit factor ───────────────────
    gross_profit = sum(winning) if winning else 0.0
    gross_loss = abs(sum(losing)) if losing else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # ── Max consecutive losses ────────────────────────────────────────
    max_consec_losses = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    # ── Average holding period (minutes) ──────────────────────────────
    holding_periods: list[float] = []
    for t in trades:
        entry_t = t.get("entry_time", 0)
        exit_t = t.get("exit_time", 0)
        if entry_t and exit_t and exit_t > entry_t:
            holding_periods.append((exit_t - entry_t) / 60_000.0)  # ms to minutes
    avg_holding_min = (sum(holding_periods) / len(holding_periods)) if holding_periods else 0.0

    # ── Expectancy (avg win × win_rate - avg loss × loss_rate) ────────
    avg_win = (sum(winning) / len(winning)) if winning else 0.0
    avg_loss = (abs(sum(losing)) / len(losing)) if losing else 0.0
    win_rate_dec = winning_trades / total_trades if total_trades > 0 else 0.0
    loss_rate_dec = losing_trades / total_trades if total_trades > 0 else 0.0
    expectancy = avg_win * win_rate_dec - avg_loss * loss_rate_dec

    # ── Max Drawdown from equity curve ────────────────────────────────
    max_drawdown_pct = _compute_max_drawdown(equity_curve, initial_capital)

    # ── Sharpe Ratio (annualized, assuming ~252 trading days) ─────────
    sharpe_ratio = _compute_sharpe(pnls, initial_capital)

    # ── Calmar Ratio = annualized return / max drawdown ───────────────
    if max_drawdown_pct > 0 and total_pnl != 0:
        # Estimate trading days from equity curve
        trading_days = _estimate_trading_days(equity_curve)
        if trading_days > 0:
            annualized_return_pct = (total_pnl / initial_capital) * (252.0 / trading_days) * 100.0
            calmar_ratio = annualized_return_pct / max_drawdown_pct
        else:
            calmar_ratio = 0.0
    else:
        calmar_ratio = 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
        "max_consecutive_losses": max_consec_losses,
        "avg_holding_period_min": round(avg_holding_min, 2),
        "expectancy": round(expectancy, 2),
        "calmar_ratio": round(calmar_ratio, 4),
    }


# ── Private Helpers ───────────────────────────────────────────────────────

def _compute_max_drawdown(
    equity_curve: list[tuple[int, float]] | None,
    initial_capital: float,
) -> float:
    """
    Compute maximum drawdown percentage from equity curve.

    Returns
    -------
    float
        Max drawdown as a positive percentage (e.g. 5.2 means -5.2%).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    peak = initial_capital
    max_dd_pct = 0.0

    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd_pct = ((peak - equity) / peak) * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

    return max_dd_pct


def _compute_sharpe(
    pnls: list[float],
    initial_capital: float,
    risk_free_annual: float = 0.07,
    trading_days_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio from per-trade P&L.

    Uses per-trade returns, annualized assuming avg trades per day.

    Parameters
    ----------
    pnls : list[float]
        Net P&L per trade.
    initial_capital : float
    risk_free_annual : float
        Annual risk-free rate (default 7% for India).
    trading_days_per_year : int

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(pnls) < 2:
        return 0.0

    # Per-trade returns as fraction of capital
    returns = [p / initial_capital for p in pnls]
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0

    if std_ret == 0:
        return 0.0

    # Assume average 2 trades per day for annualization
    trades_per_year = trading_days_per_year * 2
    rf_per_trade = risk_free_annual / trades_per_year

    sharpe = (mean_ret - rf_per_trade) / std_ret
    # Annualize
    sharpe_annualized = sharpe * math.sqrt(trades_per_year)

    return sharpe_annualized


def _estimate_trading_days(
    equity_curve: list[tuple[int, float]] | None,
) -> int:
    """Estimate number of unique trading days in the equity curve."""
    if not equity_curve:
        return 0

    from core.utils.time_utils import epoch_ms_to_ist

    unique_days: set[str] = set()
    for ts, _ in equity_curve:
        if ts > 0:
            dt = epoch_ms_to_ist(ts)
            unique_days.add(dt.strftime("%Y-%m-%d"))

    return len(unique_days)
