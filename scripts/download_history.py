"""
Download historical OHLCV data for Nifty 50 stocks and sector indices.

Populates the SQLite ohlcv_cache table via HistoricalDataManager.
Requires valid Fyers credentials in .env.

Usage:
  python scripts/download_history.py --start 2024-01-01 --end 2025-01-01
  python scripts/download_history.py --timeframe 1m --start 2024-06-01 --end 2024-12-31
  python scripts/download_history.py --symbols RELIANCE,INFY --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from core.broker.fyers_adapter import FyersAdapter
from core.data.historical import HistoricalDataManager
from core.data.universe import (
    NIFTY50,
    SECTOR_INDICES,
    to_fyers_symbol,
    validate_symbol,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_symbols(symbols_arg: str) -> list[str]:
    """
    Resolve the --symbols argument into a list of Fyers-format symbols.

    Parameters
    ----------
    symbols_arg : str
        'NIFTY50' for the full universe, or a comma-separated list of
        plain symbols like 'RELIANCE,INFY,TCS'.

    Returns
    -------
    list[str]
        Fyers-format symbols + sector indices + NIFTY 50 index.
    """
    if symbols_arg.upper() == "NIFTY50":
        equity_symbols = [to_fyers_symbol(s) for s in NIFTY50]
    else:
        raw = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
        invalid = [s for s in raw if not validate_symbol(s)]
        if invalid:
            logger.warning(
                f"Unknown symbols (not in Nifty 50): {invalid}. "
                "They will be downloaded anyway."
            )
        equity_symbols = [to_fyers_symbol(s) for s in raw]

    # Always include sector indices and NIFTY 50 index for scoring
    sector_syms = list(SECTOR_INDICES.values())
    nifty_index = ["NSE:NIFTY50-INDEX"]

    all_symbols = equity_symbols + sector_syms + nifty_index

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in all_symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def main() -> None:
    """Download historical data."""
    parser = argparse.ArgumentParser(description="Download OHLCV history")
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Candle timeframe: 1m, 3m, 5m, 15m, 30m, 1h, 1d (default: 5m)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--symbols",
        default="NIFTY50",
        help="'NIFTY50' for full universe, or comma-separated symbols (default: NIFTY50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data is cached",
    )
    args = parser.parse_args()

    logger.info(
        f"Download: {args.timeframe} candles | {args.start} → {args.end} | "
        f"Symbols: {args.symbols} | Force: {args.force}"
    )

    # 1. Resolve symbols
    symbols = _resolve_symbols(args.symbols)
    logger.info(f"Total symbols to download: {len(symbols)}")

    # 2. Connect to Fyers
    logger.info("Connecting to Fyers API...")
    try:
        adapter = FyersAdapter()
        adapter.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Fyers: {e}")
        logger.error(
            "Ensure FYERS_APP_ID and FYERS_ACCESS_TOKEN are set in .env"
        )
        sys.exit(1)

    # 3. Create HistoricalDataManager
    hdm = HistoricalDataManager(adapter)

    # 4. Download data for each symbol
    t_start = time.time()
    success_count = 0
    fail_count = 0
    total_candles = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            # Check cache first (skip if cached and not forcing)
            if not args.force and hdm.is_cached(symbol, args.timeframe, args.start, args.end):
                cached_df = hdm._load_from_cache(symbol, args.timeframe, args.start, args.end)
                candle_count = len(cached_df)
                logger.info(
                    f"  [{i}/{len(symbols)}] {symbol}: cached ({candle_count} candles) — skipped"
                )
                success_count += 1
                total_candles += candle_count
                continue

            # Download
            df = hdm.get_ohlcv(
                symbol=symbol,
                timeframe=args.timeframe,
                start_date=args.start,
                end_date=args.end,
                force_download=args.force,
            )

            candle_count = len(df)
            total_candles += candle_count

            if candle_count > 0:
                success_count += 1
                logger.info(
                    f"  [{i}/{len(symbols)}] {symbol}: ✓ {candle_count} candles"
                )
            else:
                fail_count += 1
                logger.warning(
                    f"  [{i}/{len(symbols)}] {symbol}: 0 candles returned"
                )

        except Exception as e:
            fail_count += 1
            logger.error(f"  [{i}/{len(symbols)}] {symbol}: ✗ {e}")

    # 5. Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Timeframe     : {args.timeframe}")
    print(f"  Period        : {args.start} → {args.end}")
    print(f"  Symbols       : {len(symbols)}")
    print(f"  Successful    : {success_count}")
    print(f"  Failed        : {fail_count}")
    print(f"  Total Candles : {total_candles:,}")
    print(f"  Time Elapsed  : {elapsed:.1f}s")
    print(f"  Cache DB      : {settings.DB_PATH}")
    print("=" * 60)

    # 6. Print cache stats
    try:
        stats = hdm.get_cache_stats()
        if stats:
            print(f"\n  Cache Statistics ({len(stats)} entries):")
            for key, count in sorted(stats.items()):
                print(f"    {key}: {count:,} candles")
            print()
    except Exception:
        pass

    # 7. Disconnect
    try:
        adapter.disconnect()
    except Exception:
        pass

    if fail_count > 0:
        logger.warning(f"{fail_count} symbols failed — re-run to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
