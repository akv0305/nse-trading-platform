#!/usr/bin/env python3
"""
NSE Trading Platform — Walk-Forward Optimisation

Splits historical data into rolling train/test windows, optimises
strategy parameters on the training set, then validates on the unseen
test set.  Reports in-sample vs out-of-sample performance to detect
curve-fitting.

Usage
-----
    python scripts/walk_forward.py
    python scripts/walk_forward.py --timeframe 5m --capital 750000
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Project path setup ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.settings import settings, Settings
from backtest.engine import BacktestEngine
from backtest.cost_model import FyersCostModel
from core.data.models import BacktestResult
from core.data.universe import (
    NIFTY50,
    SECTOR_INDICES,
    get_nifty50_fyers_symbols,
    to_fyers_symbol,
)
from core.strategies.orb_vwap import ORBVWAPStrategy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Walk-Forward Configuration
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class WFWindow:
    """One walk-forward window with train and test date ranges."""
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# 4-month training, 2-month test, rolling by 2 months
WINDOWS = [
    WFWindow("W1", "2024-01-01", "2024-04-30", "2024-05-01", "2024-06-30"),
    WFWindow("W2", "2024-03-01", "2024-06-30", "2024-07-01", "2024-08-31"),
    WFWindow("W3", "2024-05-01", "2024-08-31", "2024-09-01", "2024-10-31"),
    WFWindow("W4", "2024-07-01", "2024-10-31", "2024-11-01", "2024-12-31"),
]

# Parameter grid — kept small and meaningful based on P&L analysis
PARAM_GRID = {
    "ORB_TARGET_RR": [1.2, 1.5, 2.0],
    "ORB_TRAIL_AFTER_RR": [0.3, 0.5, 1.0],
    "ORB_TRAIL_PCT": [0.3, 0.5],
    "ENTRY_CUTOFF_IST": ["11:30", "12:30", "14:45"],
}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _build_param_combos(grid: dict) -> list[dict]:
    """Cartesian product of all parameter values."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def _apply_params(params: dict) -> dict:
    """
    Temporarily override settings with the given params.
    Returns a dict of original values for restoration.
    """
    originals = {}
    for key, value in params.items():
        originals[key] = getattr(settings, key)
        # Pydantic v2 models are somewhat immutable, use model's __dict__
        settings.__dict__[key] = value
    return originals


def _restore_params(originals: dict) -> None:
    """Restore settings to original values."""
    for key, value in originals.items():
        settings.__dict__[key] = value


def _load_data(timeframe: str = "5m") -> dict[str, pd.DataFrame]:
    """Load all cached OHLCV data from SQLite."""
    import sqlite3

    db_path = settings.DB_PATH
    if not db_path.exists():
        print(f"  ERROR: Database not found at {db_path}")
        return {}

    # Build full symbol list
    symbols = get_nifty50_fyers_symbols()
    for idx_sym in SECTOR_INDICES.values():
        if idx_sym not in symbols:
            symbols.append(idx_sym)
    nifty_idx = "NSE:NIFTY50-INDEX"
    if nifty_idx not in symbols:
        symbols.append(nifty_idx)

    conn = sqlite3.connect(str(db_path))
    data = {}

    for sym in symbols:
        try:
            query = (
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ohlcv_cache "
                "WHERE symbol = ? AND timeframe = ? "
                "ORDER BY timestamp"
            )
            df = pd.read_sql_query(query, conn, params=(sym, timeframe))
            if not df.empty:
                data[sym] = df
        except Exception:
            continue

    conn.close()
    print(f"  Loaded {len(data)} symbols from cache")
    return data



def _run_single_backtest(
    data: dict[str, pd.DataFrame],
    capital: float,
    start_date: str,
    end_date: str,
) -> BacktestResult:
    """Run one backtest with current settings."""
    strategy = ORBVWAPStrategy()
    engine = BacktestEngine(strategy, FyersCostModel())
    return engine.run(
        data=data,
        initial_capital=capital,
        start_date=start_date,
        end_date=end_date,
    )


def _score_result(result: BacktestResult) -> float:
    """
    Score a backtest result for optimisation.
    Uses Sharpe ratio as primary metric, with penalties for
    extreme drawdown and very few trades.
    """
    sharpe = result.sharpe_ratio

    # Penalise if fewer than 20 trades (unreliable stats)
    if result.total_trades < 20:
        sharpe -= 1.0

    # Penalise if drawdown exceeds 30%
    if result.max_drawdown_pct > 30:
        sharpe -= (result.max_drawdown_pct - 30) * 0.05

    return sharpe


def _format_params(params: dict) -> str:
    """Short string representation of parameters."""
    parts = []
    for k, v in params.items():
        short_key = k.replace("ORB_", "").replace("_IST", "")
        parts.append(f"{short_key}={v}")
    return ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Walk-Forward Engine
# ══════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    data: dict[str, pd.DataFrame],
    capital: float = 750_000.0,
) -> None:
    """Execute the full walk-forward optimisation."""
    combos = _build_param_combos(PARAM_GRID)
    total_combos = len(combos)

    print("\n" + "=" * 72)
    print("  WALK-FORWARD OPTIMISATION")
    print("=" * 72)
    print(f"  Parameter combinations : {total_combos}")
    print(f"  Walk-forward windows   : {len(WINDOWS)}")
    print(f"  Total backtest runs    : {total_combos * len(WINDOWS)} (train) + {len(WINDOWS)} (test)")
    print(f"  Capital                : ₹{capital:,.0f}")
    print("=" * 72)

    oos_results: list[dict] = []  # out-of-sample results

    for window in WINDOWS:
        print(f"\n{'─' * 72}")
        print(f"  {window.name}: Train {window.train_start} → {window.train_end} "
              f"| Test {window.test_start} → {window.test_end}")
        print(f"{'─' * 72}")

        # ── Training phase: test all parameter combinations ───────────
        best_score = -999.0
        best_params: dict = {}
        best_train_result: BacktestResult | None = None

        train_start_time = time.time()

        for i, params in enumerate(combos):
            originals = _apply_params(params)
            try:
                result = _run_single_backtest(
                    data, capital,
                    window.train_start, window.train_end,
                )
                score = _score_result(result)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_train_result = result

            except Exception as e:
                logger.warning(f"  Combo {i+1} failed: {e}")
            finally:
                _restore_params(originals)

            # Progress indicator
            if (i + 1) % 10 == 0 or i + 1 == total_combos:
                elapsed = time.time() - train_start_time
                print(f"    Training: {i+1}/{total_combos} combos "
                      f"({elapsed:.0f}s) best_sharpe={best_score:+.3f}", end="\r")

        train_elapsed = time.time() - train_start_time
        print(f"\n  Training complete in {train_elapsed:.0f}s")
        print(f"  Best params: {_format_params(best_params)}")

        if best_train_result:
            print(f"  In-sample:  P&L ₹{best_train_result.total_pnl:>+10,.2f} "
                  f"({best_train_result.total_pnl/capital*100:+.1f}%) "
                  f"| {best_train_result.total_trades} trades "
                  f"| Sharpe {best_train_result.sharpe_ratio:+.3f} "
                  f"| DD {best_train_result.max_drawdown_pct:.1f}%")

        # ── Test phase: apply best params to unseen data ──────────────
        if not best_params:
            print("  No valid training results — skipping test phase")
            continue

        originals = _apply_params(best_params)
        try:
            test_result = _run_single_backtest(
                data, capital,
                window.test_start, window.test_end,
            )

            print(f"  Out-of-sample: P&L ₹{test_result.total_pnl:>+10,.2f} "
                  f"({test_result.total_pnl/capital*100:+.1f}%) "
                  f"| {test_result.total_trades} trades "
                  f"| Sharpe {test_result.sharpe_ratio:+.3f} "
                  f"| DD {test_result.max_drawdown_pct:.1f}%")

            # Compute degradation
            if best_train_result and best_train_result.sharpe_ratio != 0:
                degradation = (
                    (best_train_result.sharpe_ratio - test_result.sharpe_ratio)
                    / abs(best_train_result.sharpe_ratio)
                    * 100
                )
            else:
                degradation = 0.0

            print(f"  Degradation: {degradation:+.1f}% "
                  f"({'⚠ OVERFIT' if degradation > 50 else '✓ OK' if degradation < 30 else '~ MARGINAL'})")

            oos_results.append({
                "window": window.name,
                "params": best_params.copy(),
                "train_pnl": best_train_result.total_pnl if best_train_result else 0,
                "train_sharpe": best_train_result.sharpe_ratio if best_train_result else 0,
                "train_trades": best_train_result.total_trades if best_train_result else 0,
                "test_pnl": test_result.total_pnl,
                "test_sharpe": test_result.sharpe_ratio,
                "test_trades": test_result.total_trades,
                "test_dd": test_result.max_drawdown_pct,
                "degradation": degradation,
            })

        except Exception as e:
            print(f"  Test phase failed: {e}")
        finally:
            _restore_params(originals)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 72)

    if not oos_results:
        print("  No valid results.")
        return

    print(f"\n  {'Window':<8} {'Train P&L':>12} {'Test P&L':>12} "
          f"{'Train Sharpe':>13} {'Test Sharpe':>12} {'Degrad':>8} {'Status':>10}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*13} {'─'*12} {'─'*8} {'─'*10}")

    total_oos_pnl = 0.0
    total_oos_trades = 0
    overfit_count = 0

    for r in oos_results:
        status = "⚠ OVERFIT" if r["degradation"] > 50 else "✓ OK" if r["degradation"] < 30 else "~ MARGIN"
        if r["degradation"] > 50:
            overfit_count += 1

        print(f"  {r['window']:<8} "
              f"₹{r['train_pnl']:>+10,.0f} "
              f"₹{r['test_pnl']:>+10,.0f} "
              f"{r['train_sharpe']:>+12.3f} "
              f"{r['test_sharpe']:>+11.3f} "
              f"{r['degradation']:>+7.1f}% "
              f"{status:>10}")

        total_oos_pnl += r["test_pnl"]
        total_oos_trades += r["test_trades"]

    print(f"\n  Aggregate OOS P&L    : ₹{total_oos_pnl:>+12,.2f} ({total_oos_pnl/capital*100:+.1f}%)")
    print(f"  Aggregate OOS Trades : {total_oos_trades}")
    print(f"  Overfit windows      : {overfit_count}/{len(oos_results)}")

    avg_oos_sharpe = sum(r["test_sharpe"] for r in oos_results) / len(oos_results)
    print(f"  Average OOS Sharpe   : {avg_oos_sharpe:+.3f}")

    # ── Parameter stability check ─────────────────────────────────────
    print(f"\n  ── Best Parameters Per Window ──")
    for r in oos_results:
        print(f"  {r['window']}: {_format_params(r['params'])}")

    # Check if same params won across windows
    param_sets = [tuple(sorted(r["params"].items())) for r in oos_results]
    unique_params = len(set(param_sets))
    print(f"\n  Unique parameter sets: {unique_params}/{len(oos_results)} "
          f"({'STABLE — same params work across periods' if unique_params == 1 else 'UNSTABLE — different params per period' if unique_params == len(oos_results) else 'MIXED — some consistency'})")

    # ── Final verdict ─────────────────────────────────────────────────
    print(f"\n  {'=' * 68}")
    if avg_oos_sharpe > 0 and overfit_count <= 1 and total_oos_pnl > 0:
        print("  VERDICT: Strategy shows potential. OOS results are positive.")
        print("  Consider paper trading with the most stable parameter set.")
    elif avg_oos_sharpe > -1 and overfit_count <= 2:
        print("  VERDICT: Strategy is marginal. Some windows profitable, others not.")
        print("  Consider adding market regime filters or reducing trade frequency.")
    else:
        print("  VERDICT: Strategy does NOT have a robust edge on this data.")
        print("  Recommend: pivot to Strategy 2 (S/D Zones) or add structural")
        print("  improvements (regime filter, multi-timeframe confirmation).")
    print(f"  {'=' * 68}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimisation")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--capital", type=float, default=750_000.0)
    args = parser.parse_args()

    print("Loading cached data...")
    data = _load_data(args.timeframe)

    if not data:
        print("ERROR: No cached data found. Run download_history.py first.")
        sys.exit(1)

    run_walk_forward(data, args.capital)


if __name__ == "__main__":
    main()
