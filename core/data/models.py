"""
NSE Trading Platform — Data Models

Immutable dataclasses used across the entire platform.
All prices in INR. All timestamps in epoch milliseconds (IST).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NO_ACTION = "NO_ACTION"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PLACED = "PLACED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOPLOSS = "STOPLOSS"
    TRAIL = "TRAIL"
    FLATTEN = "FLATTEN"
    MANUAL = "MANUAL"
    KILL = "KILL"
    TIME_EXIT = "TIME_EXIT"


class ZoneType(str, Enum):
    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"


class EngineStatus(str, Enum):
    STOPPED = "STOPPED"
    WARMING_UP = "WARMING_UP"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    KILLED = "KILLED"


# ── Candle ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Candle:
    """
    Single OHLCV candle.

    Attributes
    ----------
    symbol : str
        Fyers symbol, e.g. 'NSE:RELIANCE-EQ'.
    timestamp : int
        Candle open time as epoch milliseconds (IST).
    open : float
    high : float
    low : float
    close : float
    volume : int
    timeframe : str
        Candle timeframe: '1m', '3m', '5m', '15m', '1h', '1d'.
    """
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1d"

    @property
    def typical_price(self) -> float:
        """(High + Low + Close) / 3 — used in VWAP calculation."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def body_size(self) -> float:
        """Absolute candle body size."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Full candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open


# ── Signal ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Signal:
    """
    Output from a strategy's signal generation.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy that produced this signal.
    symbol : str
        Target symbol.
    signal_type : SignalType
        BUY, SELL, EXIT_LONG, EXIT_SHORT, or NO_ACTION.
    strength : float
        Confidence score from 0.0 (weakest) to 1.0 (strongest).
    price_at_signal : float
        Market price when signal was generated.
    timestamp : int
        Epoch milliseconds when signal was created.
    indicator_data : dict
        Snapshot of all indicator values at signal time.
    skip_reason : str
        If NO_ACTION, explains why (e.g. 'risk_limit', 'no_breakout').
    """
    strategy_name: str
    symbol: str
    signal_type: SignalType
    strength: float
    price_at_signal: float
    timestamp: int
    indicator_data: dict = field(default_factory=dict)
    skip_reason: str = ""


# ── Zone (Supply / Demand) ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Zone:
    """
    A supply or demand zone identified from price action.

    Attributes
    ----------
    symbol : str
        Stock symbol.
    zone_type : ZoneType
        SUPPLY or DEMAND.
    price_high : float
        Upper boundary of the zone.
    price_low : float
        Lower boundary of the zone.
    origin_timestamp : int
        When the zone was first formed (epoch ms).
    strength_score : float
        Zone quality score (0.0 to 5.0).
    touch_count : int
        Number of times price has revisited this zone.
    is_fresh : bool
        True if zone has never been tested since formation.
    metadata : dict
        Extra info: formation candle count, departure speed, etc.
    """
    symbol: str
    zone_type: ZoneType
    price_high: float
    price_low: float
    origin_timestamp: int
    strength_score: float = 0.0
    touch_count: int = 0
    is_fresh: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        """Midpoint price of the zone."""
        return (self.price_high + self.price_low) / 2.0

    @property
    def width(self) -> float:
        """Zone width in absolute price terms."""
        return self.price_high - self.price_low

    @property
    def width_pct(self) -> float:
        """Zone width as percentage of midpoint."""
        mid = self.midpoint
        if mid == 0:
            return 0.0
        return (self.width / mid) * 100.0


# ── Trade Plan ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TradePlan:
    """
    Pre-trade intent — created before any order is placed.
    Captures the complete plan: entry, SL, target, quantity, risk.

    Attributes
    ----------
    strategy_name : str
    symbol : str
    direction : Direction
        LONG or SHORT.
    entry_price : float
        Planned entry price.
    stoploss_price : float
        Stoploss price.
    target_price : float or None
        Target price (None if trailing only).
    quantity : int
        Number of shares/lots.
    risk_amount : float
        INR at risk = quantity × |entry - stoploss|.
    reward_amount : float or None
        INR potential reward.
    rr_ratio : float or None
        Reward:Risk ratio.
    metadata : dict
        Strategy-specific data (ORB levels, zone info, etc.).
    """
    strategy_name: str
    symbol: str
    direction: Direction
    entry_price: float
    stoploss_price: float
    target_price: float | None
    quantity: int
    risk_amount: float
    reward_amount: float | None = None
    rr_ratio: float | None = None
    metadata: dict = field(default_factory=dict)

    def compute_rr(self) -> float | None:
        """Calculate and set reward:risk ratio."""
        if self.target_price is None:
            return None
        risk = abs(self.entry_price - self.stoploss_price)
        if risk == 0:
            return None
        reward = abs(self.target_price - self.entry_price)
        self.rr_ratio = round(reward / risk, 2)
        self.reward_amount = self.quantity * reward
        return self.rr_ratio


# ── Backtest Result ───────────────────────────────────────────────────────

@dataclass(slots=True)
class BacktestResult:
    """
    Summary of a completed backtest run.

    Attributes
    ----------
    strategy_name : str
    symbol_universe : str
        Comma-separated symbols or 'NIFTY50'.
    start_date : str
        YYYY-MM-DD.
    end_date : str
    initial_capital : float
    final_capital : float
    total_trades : int
    winning_trades : int
    losing_trades : int
    win_rate : float
        Percentage (0-100).
    total_pnl : float
        Net P&L in INR.
    max_drawdown_pct : float
        Maximum drawdown percentage.
    sharpe_ratio : float
    profit_factor : float
        Gross profit / gross loss.
    avg_trade_pnl : float
    max_consecutive_losses : int
    params : dict
        Strategy parameters used for this run.
    trades : list
        List of individual trade dicts.
    equity_curve : list
        List of (timestamp, equity) tuples.
    duration_sec : float
        How long the backtest took to execute.
    """
    strategy_name: str
    symbol_universe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    max_consecutive_losses: int = 0
    params: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def return_pct(self) -> float:
        """Total return as percentage of initial capital."""
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100.0

    @property
    def losing_rate(self) -> float:
        """Loss rate percentage."""
        return 100.0 - self.win_rate if self.win_rate else 0.0


# ── Tick (for live data) ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Tick:
    """
    A single price tick from the broker WebSocket.

    Attributes
    ----------
    symbol : str
    ltp : float
        Last traded price.
    timestamp : int
        Epoch milliseconds.
    volume : int
        Cumulative day volume (if available).
    bid : float
        Best bid (if available).
    ask : float
        Best ask (if available).
    """
    symbol: str
    ltp: float
    timestamp: int
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
