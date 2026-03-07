"""
NSE Trading Platform — Live Candle Builder

Aggregates real-time ticks into OHLCV candles at a specified timeframe.
Used by the engine during live trading to feed strategies with
complete candles as they form.

Thread-safe: can receive ticks from the WebSocket thread while
the engine reads completed candles from the main loop.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from core.data.models import Candle
from core.utils.time_utils import IST, now_ist, now_epoch_ms

logger = logging.getLogger(__name__)


def _candle_start_time(dt: datetime, interval_minutes: int) -> datetime:
    """
    Compute the start time of the candle that contains datetime dt.

    Parameters
    ----------
    dt : datetime
        Current time (must have tzinfo).
    interval_minutes : int
        Candle interval in minutes (1, 3, 5, 15, 30, 60).

    Returns
    -------
    datetime
        Start of the candle period.
    """
    total_minutes = dt.hour * 60 + dt.minute
    candle_index = total_minutes // interval_minutes
    candle_start_minutes = candle_index * interval_minutes

    return dt.replace(
        hour=candle_start_minutes // 60,
        minute=candle_start_minutes % 60,
        second=0,
        microsecond=0,
    )


class LiveCandleBuilder:
    """
    Aggregates ticks into OHLCV candles in real-time.

    For each subscribed symbol, maintains a "building" candle that
    accumulates ticks.  When a tick arrives that belongs to the next
    candle period, the current candle is finalized and emitted.

    Usage
    -----
    >>> builder = LiveCandleBuilder(interval_minutes=5)
    >>> builder.on_tick("NSE:RELIANCE-EQ", 2450.50, 1500, now_epoch_ms())
    >>> completed = builder.get_completed_candles("NSE:RELIANCE-EQ")
    >>> if completed:
    ...     for candle in completed:
    ...         strategy.process_candle(candle)

    Parameters
    ----------
    interval_minutes : int
        Candle interval: 1, 3, 5, 15, 30, or 60. Default 5.
    max_completed_candles : int
        Max completed candles to keep per symbol. Default 500.
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        max_completed_candles: int = 500,
    ) -> None:
        if interval_minutes not in (1, 3, 5, 15, 30, 60):
            raise ValueError(
                f"Invalid interval: {interval_minutes}. "
                "Supported: 1, 3, 5, 15, 30, 60"
            )

        self._interval = interval_minutes
        self._max_completed = max_completed_candles
        self._lock = threading.Lock()

        # symbol → current building candle state
        self._building: dict[str, dict] = {}

        # symbol → list of completed Candle objects (newest last)
        self._completed: dict[str, list[Candle]] = defaultdict(list)

        # Timeframe string for Candle dataclass
        self._timeframe = f"{interval_minutes}m" if interval_minutes < 60 else "1h"

    def on_tick(
        self,
        symbol: str,
        price: float,
        volume: int,
        tick_epoch_ms: int,
    ) -> Candle | None:
        """
        Process a new tick.  Updates the building candle.
        If the tick starts a new candle period, finalizes the old candle.

        Parameters
        ----------
        symbol : str
        price : float
            Last traded price.
        volume : int
            Tick volume (or cumulative volume — we track delta internally).
        tick_epoch_ms : int
            Tick timestamp as epoch milliseconds.

        Returns
        -------
        Candle or None
            Returns a completed Candle if one was finalized, else None.
        """
        tick_dt = datetime.fromtimestamp(tick_epoch_ms / 1000.0, tz=IST)
        candle_start = _candle_start_time(tick_dt, self._interval)
        candle_start_ms = int(candle_start.timestamp() * 1000)

        completed_candle: Candle | None = None

        with self._lock:
            if symbol not in self._building:
                # First tick for this symbol — start a new candle
                self._building[symbol] = {
                    "start_ms": candle_start_ms,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "tick_count": 1,
                }
            else:
                current = self._building[symbol]

                if candle_start_ms > current["start_ms"]:
                    # New candle period — finalize the old one
                    completed_candle = Candle(
                        symbol=symbol,
                        timestamp=current["start_ms"],
                        open=current["open"],
                        high=current["high"],
                        low=current["low"],
                        close=current["close"],
                        volume=current["volume"],
                        timeframe=self._timeframe,
                    )

                    # Store completed candle
                    self._completed[symbol].append(completed_candle)
                    if len(self._completed[symbol]) > self._max_completed:
                        self._completed[symbol] = self._completed[symbol][-self._max_completed:]

                    # Start new candle
                    self._building[symbol] = {
                        "start_ms": candle_start_ms,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume,
                        "tick_count": 1,
                    }
                else:
                    # Same candle period — update OHLCV
                    current["high"] = max(current["high"], price)
                    current["low"] = min(current["low"], price)
                    current["close"] = price
                    current["volume"] += volume
                    current["tick_count"] += 1

        if completed_candle:
            logger.debug(
                f"Candle completed: {symbol} {self._timeframe} "
                f"O={completed_candle.open} H={completed_candle.high} "
                f"L={completed_candle.low} C={completed_candle.close} "
                f"V={completed_candle.volume}"
            )

        return completed_candle

    def get_completed_candles(
        self,
        symbol: str,
        last_n: int | None = None,
    ) -> list[Candle]:
        """
        Get completed candles for a symbol.

        Parameters
        ----------
        symbol : str
        last_n : int, optional
            Return only the last N candles. If None, return all.

        Returns
        -------
        list[Candle]
            Sorted oldest first.
        """
        with self._lock:
            candles = self._completed.get(symbol, [])
            if last_n is not None:
                return candles[-last_n:]
            return candles.copy()

    def get_building_candle(self, symbol: str) -> dict | None:
        """
        Get the currently building (incomplete) candle for a symbol.

        Returns
        -------
        dict or None
            {'start_ms': int, 'open': float, 'high': float,
             'low': float, 'close': float, 'volume': int, 'tick_count': int}
        """
        with self._lock:
            if symbol in self._building:
                return self._building[symbol].copy()
            return None

    def get_all_latest_candles(self) -> dict[str, Candle | None]:
        """
        Get the most recent completed candle for every tracked symbol.

        Returns
        -------
        dict[str, Candle | None]
        """
        with self._lock:
            result: dict[str, Candle | None] = {}
            for symbol, candles in self._completed.items():
                result[symbol] = candles[-1] if candles else None
            return result

    def reset(self, symbol: str | None = None) -> None:
        """
        Clear all state for a symbol or all symbols.

        Parameters
        ----------
        symbol : str, optional
            If None, resets everything.
        """
        with self._lock:
            if symbol:
                self._building.pop(symbol, None)
                self._completed.pop(symbol, None)
            else:
                self._building.clear()
                self._completed.clear()

    def get_stats(self) -> dict[str, dict]:
        """
        Get statistics for all tracked symbols.

        Returns
        -------
        dict[str, dict]
            symbol → {'completed': int, 'building_ticks': int}
        """
        with self._lock:
            stats: dict[str, dict] = {}
            all_symbols = set(list(self._building.keys()) + list(self._completed.keys()))
            for symbol in all_symbols:
                building = self._building.get(symbol, {})
                stats[symbol] = {
                    "completed": len(self._completed.get(symbol, [])),
                    "building_ticks": building.get("tick_count", 0),
                }
            return stats

    @property
    def interval_minutes(self) -> int:
        """Return the candle interval in minutes."""
        return self._interval

    @property
    def timeframe(self) -> str:
        """Return the timeframe string (e.g. '5m', '15m', '1h')."""
        return self._timeframe
