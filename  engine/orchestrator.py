"""
NSE Trading Platform — Strategy Orchestrator

Multi-strategy orchestrator that runs the main trading loop:
  1. Pre-market scan
  2. Subscribe to ticks
  3. Build candles
  4. Feed strategies
  5. Execute signals
  6. Monitor positions
  7. End-of-day cleanup

Implementation: After Strategy 1 backtest is validated.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Multi-strategy trading orchestrator.

    Manages the lifecycle of multiple strategies running concurrently.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Orchestrator — implement after backtest validation")
