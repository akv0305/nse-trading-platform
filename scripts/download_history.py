"""
Download historical OHLCV data for all Nifty 50 stocks.

Usage:
  python scripts/download_history.py --timeframe 5m --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Download historical data."""
    parser = argparse.ArgumentParser(description="Download OHLCV history")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (1m,5m,15m,1d)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", default="NIFTY50", help="Symbol list or NIFTY50")
    args = parser.parse_args()

    logger.info(f"Downloading {args.timeframe} data from {args.start} to {args.end}")
    raise NotImplementedError("History download script — implement after broker setup")


if __name__ == "__main__":
    main()
