"""
NSE Trading Platform — Deterministic ID Generators

All IDs follow the pattern: {PREFIX}_{YYYYMMDD}_{HHMMSS}_{6-char-random-hex}
This ensures:
  - IDs are sortable by time
  - Prefix identifies the entity type
  - Random suffix avoids collisions
"""

from __future__ import annotations

import secrets
from core.utils.time_utils import now_ist


def generate_id(prefix: str) -> str:
    """
    Generate a unique, time-sortable ID with a given prefix.

    Parameters
    ----------
    prefix : str
        Short identifier like 'SIG', 'ORD', 'TRD', 'PLN', 'RSK', 'CMD', 'BT', 'LLM'.

    Returns
    -------
    str
        ID in format: {PREFIX}_{YYYYMMDD}_{HHMMSS}_{hex6}
        Example: 'ORD_20260307_143022_a3f1b9'
    """
    ts = now_ist()
    date_part = ts.strftime("%Y%m%d")
    time_part = ts.strftime("%H%M%S")
    rand_part = secrets.token_hex(3)  # 6 hex chars
    return f"{prefix}_{date_part}_{time_part}_{rand_part}"


def generate_signal_id() -> str:
    """Generate ID for a strategy signal."""
    return generate_id("SIG")


def generate_order_id() -> str:
    """Generate ID for an order."""
    return generate_id("ORD")


def generate_trade_id() -> str:
    """Generate ID for a trade."""
    return generate_id("TRD")


def generate_plan_id() -> str:
    """Generate ID for a trade plan."""
    return generate_id("PLN")


def generate_risk_event_id() -> str:
    """Generate ID for a risk event."""
    return generate_id("RSK")


def generate_command_id() -> str:
    """Generate ID for a dashboard command."""
    return generate_id("CMD")


def generate_backtest_run_id() -> str:
    """Generate ID for a backtest run."""
    return generate_id("BT")


def generate_backtest_trade_id() -> str:
    """Generate ID for an individual backtest trade."""
    return generate_id("BTT")


def generate_position_id() -> str:
    """Generate ID for a live position."""
    return generate_id("POS")


def generate_llm_analysis_id() -> str:
    """Generate ID for an LLM analysis record."""
    return generate_id("LLM")
