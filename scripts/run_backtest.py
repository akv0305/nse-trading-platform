"""
Run a backtest for a specified strategy.

Usage:
  python scripts/run_backtest.py --strategy orb_vwap --start 2024-01-01 --end 2025-01-01
  python scripts/run_backtest.py --strategy orb_vwap --start 2024-06-01 --end 2024-12-31 --capital 500000
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
from backtest.engine import BacktestEngine
from backtest.cost_model import FyersCostModel
from core.data.historical import HistoricalDataManager
from core.data.universe import (
    get_nifty50_fyers_symbols,
    SECTOR_INDICES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Strategy Factory ──────────────────────────────────────────────────────

def _create_strategy(name: str):
    """Instantiate a strategy by name."""
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower == "orb_vwap":
        from core.strategies.orb_vwap import ORBVWAPStrategy
        return ORBVWAPStrategy()
    elif name_lower in ("vmr_vwap", "vmr", "vwap_mean_reversion"):
        from core.strategies.vmr_strategy import VMRStrategy
        return VMRStrategy()
    else:
        raise ValueError(
            f"Unknown strategy '{name}'. Supported: orb_vwap, vmr_vwap"
        )


# ── Data Loading ──────────────────────────────────────────────────────────

def _load_data(
    start_date: str,
    end_date: str,
    timeframe: str = "5m",
) -> dict[str, "pd.DataFrame"]:
    """
    Load historical OHLCV data for all Nifty 50 stocks + sector indices.

    Attempts to load from the SQLite ohlcv_cache first.  If the cache is
    empty (no Fyers credentials / no prior download), falls back to
    generating a clear error message.

    Returns
    -------
    dict[str, pd.DataFrame]
        Fyers-symbol → DataFrame mapping.
    """
    import pandas as pd
    from core.broker.fyers_adapter import FyersAdapter

    symbols = get_nifty50_fyers_symbols()
    sector_syms = list(SECTOR_INDICES.values())
    # Also fetch NIFTY 50 index for sector scoring
    all_symbols = symbols + sector_syms + ["NSE:NIFTY50-INDEX"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for s in all_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    logger.info(
        f"Loading {timeframe} data for {len(unique_symbols)} symbols "
        f"({start_date} → {end_date})"
    )

    # Try to connect to Fyers for download; fall back to cache-only
    try:
        adapter = FyersAdapter()
        adapter.connect()
        hdm = HistoricalDataManager(adapter)
        logger.info("Fyers connected — downloading missing data from API")
    except Exception as e:
        logger.warning(
            f"Fyers connection failed ({e}); loading from cache only"
        )
        # Create a minimal adapter that raises on download
        # (HistoricalDataManager will serve from cache)
        hdm = HistoricalDataManager.__new__(HistoricalDataManager)
        hdm._db_path = str(settings.DB_PATH)
        hdm._broker = None  # type: ignore[assignment]

    data: dict[str, pd.DataFrame] = {}
    loaded = 0
    empty = 0

    for i, symbol in enumerate(unique_symbols, 1):
        try:
            # Try cache first (no broker needed)
            df = hdm._load_from_cache(symbol, timeframe, start_date, end_date)
            if df.empty and hdm._broker is not None:
                df = hdm.get_ohlcv(symbol, timeframe, start_date, end_date)
            if not df.empty:
                data[symbol] = df
                loaded += 1
            else:
                empty += 1
        except Exception as exc:
            logger.debug(f"  [{i}/{len(unique_symbols)}] {symbol}: error — {exc}")
            empty += 1

        # Progress every 10 symbols
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(unique_symbols)} symbols processed")

    logger.info(f"Data loaded: {loaded} symbols with data, {empty} empty/missing")

    if loaded == 0:
        logger.error(
            "No OHLCV data found. Run 'python scripts/download_history.py' first "
            "to populate the cache."
        )

    return data


# ── Result Printer ────────────────────────────────────────────────────────

def _print_results(result) -> None:
    """Pretty-print BacktestResult to console."""
    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULTS — {result.strategy_name}")
    print("=" * 70)
    print(f"  Period        : {result.start_date} → {result.end_date}")
    print(f"  Universe      : {result.symbol_universe}")
    print(f"  Duration      : {result.duration_sec:.1f} seconds")
    print("-" * 70)
    print(f"  Initial Capital : ₹{result.initial_capital:>12,.2f}")
    print(f"  Final Capital   : ₹{result.final_capital:>12,.2f}")
    print(f"  Total P&L       : ₹{result.total_pnl:>12,.2f}  "
          f"({result.return_pct:+.2f}%)")
    print("-" * 70)
    print(f"  Total Trades    : {result.total_trades:>6d}")
    print(f"  Winners         : {result.winning_trades:>6d}  "
          f"({result.win_rate:.1f}%)")
    print(f"  Losers          : {result.losing_trades:>6d}  "
          f"({result.losing_rate:.1f}%)")
    print(f"  Avg Trade P&L   : ₹{result.avg_trade_pnl:>10,.2f}")
    print("-" * 70)
    print(f"  Max Drawdown    : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio    : {result.sharpe_ratio:.3f}")
    print(f"  Profit Factor   : {result.profit_factor:.3f}")
    print(f"  Max Consec Loss : {result.max_consecutive_losses}")
    print("=" * 70)

    # Top 5 best & worst trades
    if result.trades:
        sorted_trades = sorted(
            result.trades, key=lambda t: t.get("pnl_net", 0), reverse=True
        )
        print("\n  Top 5 Winners:")
        for t in sorted_trades[:5]:
            print(f"    {t.get('symbol', '?'):>20s}  ₹{t.get('pnl_net', 0):>+10,.2f}  "
                  f"({t.get('exit_reason', '')})")
        print("\n  Top 5 Losers:")
        for t in sorted_trades[-5:]:
            print(f"    {t.get('symbol', '?'):>20s}  ₹{t.get('pnl_net', 0):>+10,.2f}  "
                  f"({t.get('exit_reason', '')})")
        print()

    # ── Trade diagnostics ──────────────────────────────────────────
    if result.trades:
        pnls = [t.get("pnl_net", 0) for t in result.trades]
        quantities = [t.get("quantity", 0) for t in result.trades]
        costs = [t.get("costs_total", 0) for t in result.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        print("  ── Diagnostics ──")
        print(f"  Avg winner     : ₹{sum(winners)/len(winners):>10,.2f}" if winners else "  Avg winner     : N/A")
        print(f"  Avg loser      : ₹{sum(losers)/len(losers):>10,.2f}" if losers else "  Avg loser      : N/A")
        print(f"  Avg quantity   : {sum(quantities)/len(quantities):>8,.0f} shares")
        print(f"  Total costs    : ₹{sum(costs):>10,.2f}")
        print(f"  Avg cost/trade : ₹{sum(costs)/len(costs):>8,.2f}")
        print(f"  FLATTEN exits  : {sum(1 for t in result.trades if t.get('exit_reason')=='FLATTEN')}")
        print(f"  STOPLOSS exits : {sum(1 for t in result.trades if t.get('exit_reason')=='STOPLOSS')}")
        print(f"  TARGET exits   : {sum(1 for t in result.trades if t.get('exit_reason')=='TARGET')}")
        print(f"  TRAIL exits    : {sum(1 for t in result.trades if t.get('exit_reason')=='TRAIL')}")
        print()

    # Breakdown by exit reason
    from collections import defaultdict
    reason_stats = defaultdict(lambda: {"count": 0, "total_pnl": 0.0})
    for t in result.trades:
        r = t.get("exit_reason", "UNKNOWN")
        reason_stats[r]["count"] += 1
        reason_stats[r]["total_pnl"] += t.get("pnl_net", 0)
    
    print("\n  ── P&L by Exit Reason ──")
    for reason, stats in sorted(reason_stats.items()):
        avg = stats["total_pnl"] / stats["count"] if stats["count"] else 0
        print(f"  {reason:12s}: {stats['count']:4d} trades, "
              f"total ₹{stats['total_pnl']:>+12,.2f}, "
              f"avg ₹{avg:>+8,.2f}")

# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Run backtest."""
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name: orb_vwap",
    )
    parser.add_argument(
        "--start",
        default=settings.BACKTEST_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {settings.BACKTEST_START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=settings.BACKTEST_END_DATE,
        help=f"End date YYYY-MM-DD (default: {settings.BACKTEST_END_DATE})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=settings.BACKTEST_INITIAL_CAPITAL,
        help=f"Initial capital INR (default: {settings.BACKTEST_INITIAL_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Candle timeframe (default: 5m)",
    )
    args = parser.parse_args()

    logger.info(
        f"Backtest: {args.strategy} | {args.start} → {args.end} | "
        f"Capital: ₹{args.capital:,.0f} | Timeframe: {args.timeframe}"
    )

    # 1. Create strategy
    try:
        strategy = _create_strategy(args.strategy)
        logger.info(f"Strategy created: {strategy}")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # 2. Load historical data
    t0 = time.time()
    data = _load_data(args.start, args.end, args.timeframe)
    load_time = time.time() - t0
    logger.info(f"Data loading completed in {load_time:.1f}s")

    if not data:
        logger.error("No data available — cannot run backtest.")
        sys.exit(1)

    # 3. Instantiate engine
    cost_model = FyersCostModel()
    engine = BacktestEngine(strategy=strategy, cost_model=cost_model)

    # 4. Run backtest
    logger.info("Starting backtest engine...")
    t0 = time.time()
    result = engine.run(
        data=data,
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )
    run_time = time.time() - t0
    logger.info(f"Backtest completed in {run_time:.1f}s")

    # 5. Print results
    _print_results(result)


if __name__ == "__main__":
    main()
