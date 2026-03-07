"""
NSE Trading Platform — Historical Data Manager

Downloads OHLCV data via Fyers API, handles the 100-day-per-request limit,
caches data in SQLite (ohlcv_cache table), and serves pandas DataFrames
to strategies and the backtester.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd

from config.settings import settings
from core.broker.base import BrokerAdapter
from core.data.models import Candle
from core.utils.time_utils import IST, now_ist, today_date_str

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
# Fyers limits intraday history to ~100 calendar days per request
FYERS_MAX_DAYS_PER_REQUEST = 100

# Delay between consecutive API calls to avoid rate limiting
API_CALL_DELAY_SEC = 0.5

# Resolution mapping: our labels → Fyers resolution strings
RESOLUTION_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "1d": "1D",
}


class HistoricalDataManager:
    """
    Manages historical OHLCV data: download, cache, and serve.

    Features:
      - Chunks large date ranges into 100-day windows for Fyers API
      - Caches all downloaded data in SQLite ohlcv_cache table
      - Serves data from cache when available (avoids redundant API calls)
      - Returns pandas DataFrames with proper datetime index

    Usage
    -----
    >>> from core.broker.fyers_adapter import FyersAdapter
    >>> adapter = FyersAdapter()
    >>> adapter.connect()
    >>> hdm = HistoricalDataManager(adapter)
    >>> df = hdm.get_ohlcv("NSE:RELIANCE-EQ", "5m", "2024-06-01", "2024-09-30")
    """

    def __init__(self, broker: BrokerAdapter, db_path: str | None = None) -> None:
        self._broker = broker
        self._db_path = db_path or str(settings.DB_PATH)

    def _connect_db(self) -> sqlite3.Connection:
        """Open a SQLite connection for cache operations."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public API ────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol, using cache where possible.

        Parameters
        ----------
        symbol : str
            Fyers-format symbol, e.g. 'NSE:RELIANCE-EQ'.
        timeframe : str
            Candle timeframe: '1m', '3m', '5m', '15m', '30m', '1h', '1d'.
        start_date : str
            Start date 'YYYY-MM-DD'.
        end_date : str
            End date 'YYYY-MM-DD'.
        force_download : bool
            If True, ignore cache and re-download everything.

        Returns
        -------
        pd.DataFrame
            Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex in IST.
            Sorted by timestamp ascending.
        """
        if timeframe not in RESOLUTION_MAP:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. "
                f"Supported: {list(RESOLUTION_MAP.keys())}"
            )

        if not force_download:
            cached = self._load_from_cache(symbol, timeframe, start_date, end_date)
            if not cached.empty:
                logger.info(
                    f"Cache hit: {symbol} {timeframe} {start_date}→{end_date} "
                    f"({len(cached)} candles)"
                )
                return cached

        # Download from Fyers
        candles = self._download_chunked(symbol, timeframe, start_date, end_date)

        if candles:
            # Save to cache
            self._save_to_cache(symbol, timeframe, candles)
            logger.info(
                f"Downloaded and cached: {symbol} {timeframe} {start_date}→{end_date} "
                f"({len(candles)} candles)"
            )

        # Return from cache (ensures consistency)
        return self._load_from_cache(symbol, timeframe, start_date, end_date)

    def get_ohlcv_multi(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        force_download: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple symbols.

        Parameters
        ----------
        symbols : list[str]
        timeframe : str
        start_date : str
        end_date : str
        force_download : bool

        Returns
        -------
        dict[str, pd.DataFrame]
            Symbol → DataFrame mapping.
        """
        result: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            try:
                df = self.get_ohlcv(symbol, timeframe, start_date, end_date, force_download)
                result[symbol] = df
                if i < len(symbols) - 1:
                    time.sleep(API_CALL_DELAY_SEC)
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                result[symbol] = pd.DataFrame()
        return result

    def get_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int = 100,
    ) -> pd.DataFrame:
        """
        Get the most recent N candles for a symbol.

        Calculates appropriate start_date based on num_candles and timeframe,
        then calls get_ohlcv.

        Parameters
        ----------
        symbol : str
        timeframe : str
        num_candles : int
            Number of candles to fetch.

        Returns
        -------
        pd.DataFrame
            Last num_candles rows.
        """
        # Estimate days needed based on timeframe
        candles_per_day = {
            "1m": 375,   # 6.25 hours × 60
            "3m": 125,
            "5m": 75,
            "15m": 25,
            "30m": 13,
            "1h": 7,
            "1d": 1,
        }
        cpd = candles_per_day.get(timeframe, 75)
        days_needed = max(int(num_candles / cpd) + 5, 3)  # Extra buffer for weekends/holidays

        end_date = today_date_str()
        start_dt = now_ist() - timedelta(days=days_needed)
        start_date = start_dt.strftime("%Y-%m-%d")

        df = self.get_ohlcv(symbol, timeframe, start_date, end_date)
        if not df.empty:
            return df.tail(num_candles)
        return df

    def is_cached(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> bool:
        """Check if data exists in cache for the given range."""
        df = self._load_from_cache(symbol, timeframe, start_date, end_date)
        return not df.empty

    def clear_cache(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        """
        Clear cached OHLCV data.

        Parameters
        ----------
        symbol : str, optional
            Clear only for this symbol. If None, clear all.
        timeframe : str, optional
            Clear only this timeframe. If None, clear all timeframes.

        Returns
        -------
        int
            Number of rows deleted.
        """
        conn = self._connect_db()
        try:
            if symbol and timeframe:
                cursor = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE symbol = ? AND timeframe = ?",
                    (symbol, timeframe),
                )
            elif symbol:
                cursor = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE symbol = ?", (symbol,)
                )
            elif timeframe:
                cursor = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE timeframe = ?", (timeframe,)
                )
            else:
                cursor = conn.execute("DELETE FROM ohlcv_cache")
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Cache cleared: {deleted} rows (symbol={symbol}, tf={timeframe})")
            return deleted
        finally:
            conn.close()

    # ── Download Logic ────────────────────────────────────────────────────

    def _download_chunked(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Download OHLCV data in chunks of FYERS_MAX_DAYS_PER_REQUEST days.

        Returns
        -------
        list[dict]
            Each: {'timestamp': int, 'open': float, ...}
            Timestamps are epoch seconds from Fyers.
        """
        resolution = RESOLUTION_MAP[timeframe]
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=IST)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=IST
        )

        all_candles: list[dict] = []
        chunk_start = start_dt

        while chunk_start < end_dt:
            chunk_end = min(
                chunk_start + timedelta(days=FYERS_MAX_DAYS_PER_REQUEST - 1),
                end_dt,
            )

            from_epoch = int(chunk_start.timestamp())
            to_epoch = int(chunk_end.timestamp())

            logger.debug(
                f"Downloading {symbol} {timeframe}: "
                f"{chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}"
            )

            try:
                candles = self._broker.fetch_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    from_epoch=from_epoch,
                    to_epoch=to_epoch,
                )
                all_candles.extend(candles)
            except Exception as e:
                logger.error(
                    f"Chunk download failed for {symbol} "
                    f"{chunk_start.strftime('%Y-%m-%d')}→{chunk_end.strftime('%Y-%m-%d')}: {e}"
                )

            # Rate limiting
            time.sleep(API_CALL_DELAY_SEC)
            chunk_start = chunk_end + timedelta(days=1)

        # Deduplicate by timestamp
        seen: set[int] = set()
        unique_candles: list[dict] = []
        for c in all_candles:
            ts = c["timestamp"]
            if ts not in seen:
                seen.add(ts)
                unique_candles.append(c)

        unique_candles.sort(key=lambda x: x["timestamp"])
        return unique_candles

    # ── Cache Operations ──────────────────────────────────────────────────

    def _save_to_cache(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
    ) -> None:
        """Save candles to ohlcv_cache table using INSERT OR REPLACE."""
        if not candles:
            return

        conn = self._connect_db()
        try:
            rows = [
                (
                    symbol,
                    timeframe,
                    c["timestamp"] * 1000,  # Convert epoch sec → epoch ms
                    c["open"],
                    c["high"],
                    c["low"],
                    c["close"],
                    c["volume"],
                )
                for c in candles
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv_cache
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def _load_from_cache(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load cached OHLCV data as a DataFrame.

        Returns
        -------
        pd.DataFrame
            With DatetimeIndex in IST. Empty DataFrame if no data.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=IST)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=IST
        )
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        conn = self._connect_db()
        try:
            rows = conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_cache
                WHERE symbol = ? AND timeframe = ?
                  AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (symbol, timeframe, start_ms, end_ms),
            ).fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(
                [dict(r) for r in rows],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Convert epoch ms to IST datetime index
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(
                "Asia/Kolkata"
            )
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

            return df

        finally:
            conn.close()

    # ── Utility ───────────────────────────────────────────────────────────

    def to_candle_list(self, df: pd.DataFrame, symbol: str, timeframe: str) -> list[Candle]:
        """
        Convert a DataFrame to a list of Candle dataclass instances.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: timestamp, open, high, low, close, volume.
        symbol : str
        timeframe : str

        Returns
        -------
        list[Candle]
        """
        candles: list[Candle] = []
        for _, row in df.iterrows():
            candles.append(
                Candle(
                    symbol=symbol,
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    timeframe=timeframe,
                )
            )
        return candles

    def get_cache_stats(self) -> dict[str, int]:
        """
        Return cache statistics: symbol → number of cached candles.

        Returns
        -------
        dict[str, int]
        """
        conn = self._connect_db()
        try:
            rows = conn.execute(
                """
                SELECT symbol, timeframe, COUNT(*) as cnt
                FROM ohlcv_cache
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
            ).fetchall()
            return {f"{r['symbol']}|{r['timeframe']}": r["cnt"] for r in rows}
        finally:
            conn.close()
