"""
NSE Trading Platform — Trade Plan Builder

Utility functions for constructing TradePlan objects with
proper position sizing and risk calculations.

Implementation: Shared across strategies.
"""

from __future__ import annotations

from core.data.models import TradePlan, Signal, Direction, SignalType
from config.settings import settings


def build_plan_from_signal(
    signal: Signal,
    risk_per_trade: float | None = None,
    lot_size: int = 1,
) -> TradePlan | None:
    """
    Build a TradePlan from a strategy signal.

    Parameters
    ----------
    signal : Signal
        Entry signal with indicator_data containing 'stoploss_price' and
        optionally 'target_price'.
    risk_per_trade : float, optional
        Max risk in INR. Defaults to settings.RISK_PER_TRADE_INR.
    lot_size : int
        Minimum lot size (for F&O). Default 1 for equity.

    Returns
    -------
    TradePlan or None
        None if plan cannot be constructed.
    """
    raise NotImplementedError("Trade plan builder — implement during Strategy 1 conversation")


def compute_position_size(
    entry_price: float,
    stoploss_price: float,
    risk_amount: float,
    lot_size: int = 1,
) -> int:
    """
    Compute position size (quantity) based on risk.

    quantity = floor(risk_amount / |entry - stoploss|)
    Rounded down to nearest lot_size.

    Parameters
    ----------
    entry_price : float
    stoploss_price : float
    risk_amount : float
        Max INR to risk.
    lot_size : int
        Minimum lot size.

    Returns
    -------
    int
        Number of shares. 0 if risk_per_share is zero or negative.
    """
    risk_per_share = abs(entry_price - stoploss_price)
    if risk_per_share <= 0:
        return 0

    raw_qty = int(risk_amount / risk_per_share)
    # Round down to lot size
    qty = (raw_qty // lot_size) * lot_size
    return max(qty, 0)
