"""
NSE Trading Platform — Engine Entry Point

Starts two concurrent tasks via asyncio:
  1. FastAPI control server (health, status, kill, pause, resume)
  2. Strategy orchestrator main loop

Implementation: After Strategy 1 backtest is validated.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Start engine: control API + orchestrator."""
    raise NotImplementedError("Engine entry point — implement after backtest validation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("NSE Trading Platform — Engine Starting")
    asyncio.run(main())
