"""
Run a backtest for a specified strategy.

Usage:
  python scripts/run_backtest.py --strategy orb_vwap --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run backtest."""
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--strategy", required=True, help="Strategy name: orb_vwap, sd_zones")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=750000, help="Initial capital INR")
    args = parser.parse_args()

    logger.info(f"Backtest: {args.strategy} | {args.start} → {args.end} | Capital: ₹{args.capital:,.0f}")
    raise NotImplementedError("Backtest runner — implement after Strategy 1 is built")


if __name__ == "__main__":
    main()
