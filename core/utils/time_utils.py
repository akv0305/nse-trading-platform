"""
NSE Trading Platform — Time Utilities

All internal timestamps use epoch milliseconds.
All display/log timestamps use IST (Asia/Kolkata).
Market hours: 09:15 — 15:30 IST, Monday—Friday.
"""

from __future__ import annotations

import datetime
import time
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# IST timezone — Python 3.9+ built-in, no pytz needed at runtime
# ---------------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")

# NSE holidays for current FY — update annually or load from file/API.
# Format: set of "YYYY-MM-DD" strings.  Engine skips these dates entirely.
# Populated with known 2025 holidays as baseline; extend as needed.
NSE_HOLIDAYS: set[str] = {
    "2025-02-26",  # Maha Shivaratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. B.R. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti
    "2025-10-21",  # Diwali (Laxmi Pujan)
    "2025-10-22",  # Diwali (Balipratipada)
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
    # Add 2026 holidays here when NSE publishes the list
}


def now_ist() -> datetime.datetime:
    """Return current datetime in IST with timezone info."""
    return datetime.datetime.now(tz=IST)


def now_epoch_ms() -> int:
    """Return current time as epoch milliseconds."""
    return int(time.time() * 1000)


def epoch_ms_to_ist(epoch_ms: int) -> datetime.datetime:
    """Convert epoch milliseconds to IST datetime."""
    return datetime.datetime.fromtimestamp(epoch_ms / 1000.0, tz=IST)


def ist_to_epoch_ms(dt: datetime.datetime) -> int:
    """
    Convert an IST datetime to epoch milliseconds.
    If dt is naive (no tzinfo), assumes IST.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return int(dt.timestamp() * 1000)


def today_date_str() -> str:
    """Return today's date as 'YYYY-MM-DD' in IST."""
    return now_ist().strftime("%Y-%m-%d")


def today_weekday() -> int:
    """Return today's weekday: 0=Monday ... 6=Sunday."""
    return now_ist().weekday()


def is_trading_day(date_str: str | None = None) -> bool:
    """
    Check if a given date (YYYY-MM-DD) is a trading day.
    Must be Monday—Friday and not in NSE_HOLIDAYS.
    If date_str is None, checks today.
    """
    if date_str is None:
        dt = now_ist()
        date_str = dt.strftime("%Y-%m-%d")
    else:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=IST)

    # Weekend check: Saturday=5, Sunday=6
    if dt.weekday() >= 5:
        return False

    # Holiday check
    if date_str in NSE_HOLIDAYS:
        return False

    return True


def is_market_hours(
    open_time: str = "09:15",
    close_time: str = "15:30",
) -> bool:
    """
    Check if current IST time is within market hours.

    Parameters
    ----------
    open_time : str
        Market open in 'HH:MM' format (default '09:15').
    close_time : str
        Market close in 'HH:MM' format (default '15:30').

    Returns
    -------
    bool
        True if now is between open_time and close_time on a trading day.
    """
    current = now_ist()

    if not is_trading_day():
        return False

    open_h, open_m = map(int, open_time.split(":"))
    close_h, close_m = map(int, close_time.split(":"))

    market_open = current.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = current.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    return market_open <= current <= market_close


def is_past_time(time_str: str) -> bool:
    """
    Check if current IST time is past a given 'HH:MM' time today.

    Parameters
    ----------
    time_str : str
        Time in 'HH:MM' format.

    Returns
    -------
    bool
    """
    current = now_ist()
    h, m = map(int, time_str.split(":"))
    target = current.replace(hour=h, minute=m, second=0, microsecond=0)
    return current >= target


def time_until_market_open(open_time: str = "09:15") -> datetime.timedelta:
    """
    Return timedelta until next market open.
    If market is already open or closed for today, returns time until
    next trading day's open.
    """
    current = now_ist()
    open_h, open_m = map(int, open_time.split(":"))

    target = current.replace(hour=open_h, minute=open_m, second=0, microsecond=0)

    if current < target and is_trading_day():
        return target - current

    # Move to next day and find next trading day
    next_day = current + datetime.timedelta(days=1)
    for _ in range(10):  # Max 10 days ahead (covers long holidays)
        date_str = next_day.strftime("%Y-%m-%d")
        if is_trading_day(date_str):
            target = next_day.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
            return target - current
        next_day += datetime.timedelta(days=1)

    # Fallback — should not happen
    return datetime.timedelta(hours=24)


def parse_time_str(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute) tuple."""
    parts = time_str.strip().split(":")
    return int(parts[0]), int(parts[1])


def date_range_strings(start: str, end: str) -> list[str]:
    """
    Generate a list of 'YYYY-MM-DD' strings from start to end (inclusive).

    Parameters
    ----------
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    list[str]
    """
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
    dates: list[str] = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += datetime.timedelta(days=1)
    return dates


def trading_days_in_range(start: str, end: str) -> list[str]:
    """
    Return only trading days (weekdays, non-holidays) in a date range.

    Parameters
    ----------
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    list[str]
    """
    return [d for d in date_range_strings(start, end) if is_trading_day(d)]
