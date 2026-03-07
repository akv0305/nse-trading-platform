"""
NSE Trading Platform — Abstract Strategy Base Class

All strategies (ORB+VWAP, S/D Zones, future strategies) must
implement this interface.  The engine's orchestrator calls these
methods in sequence during each tick cycle.

Strategy lifecycle per day:
  1. pre_market_scan() — run before market open, select stocks
  2. on_candle() — called for each completed candle during market hours
  3. should_exit() — called on each tick/candle for open positions
  4. end_of_day() — called after market close for cleanup/logging
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.data.models import Signal, TradePlan, Candle, SignalType


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.

    Attributes
    ----------
    name : str
        Unique strategy identifier (e.g. 'orb_vwap', 'sd_zones').
    version : str
        Strategy version for tracking parameter changes.

    Subclass Requirements
    ---------------------
    Must implement:
      - pre_market_scan()
      - on_candle()
      - should_exit()
      - get_params()
    """

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        self._name = name
        self._version = version
        self._active = True
        self._watchlist: list[str] = []   # Symbols selected by pre_market_scan

    @property
    def name(self) -> str:
        """Unique strategy name."""
        return self._name

    @property
    def version(self) -> str:
        """Strategy version string."""
        return self._version

    @property
    def is_active(self) -> bool:
        """True if strategy is enabled for trading."""
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._active = value

    @property
    def watchlist(self) -> list[str]:
        """Current list of symbols this strategy is watching."""
        return self._watchlist.copy()

    # ── Abstract Methods (must implement) ─────────────────────────────────

    @abstractmethod
    def pre_market_scan(
        self,
        universe: list[str],
        historical_data: dict[str, pd.DataFrame],
    ) -> list[str]:
        """
        Run before market open.  Scans the universe and returns
        a shortlist of symbols to watch today.

        Parameters
        ----------
        universe : list[str]
            Full list of tradeable symbols (e.g. Nifty 50 in Fyers format).
        historical_data : dict[str, pd.DataFrame]
            Pre-loaded historical data for each symbol.
            DataFrame columns: timestamp, open, high, low, close, volume.

        Returns
        -------
        list[str]
            Shortlisted symbols for today's session.
            Sets self._watchlist internally.
        """
        ...

    @abstractmethod
    def on_candle(
        self,
        symbol: str,
        candle: Candle,
        candle_history: list[Candle],
        current_position: dict | None,
    ) -> Signal:
        """
        Called when a new candle completes for a watched symbol.
        This is the main signal generation method.

        Parameters
        ----------
        symbol : str
            Fyers-format symbol.
        candle : Candle
            The just-completed candle.
        candle_history : list[Candle]
            Recent candle history (oldest first), including the new candle.
        current_position : dict or None
            If the strategy has an open position in this symbol, the
            position dict from DBWriter.  None if no position.

        Returns
        -------
        Signal
            Signal object with signal_type, strength, indicator_data, etc.
            Return Signal with signal_type=NO_ACTION if no trade.
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        symbol: str,
        current_price: float,
        position: dict,
        candle: Candle | None = None,
    ) -> Signal:
        """
        Check if an open position should be exited.
        Called on each tick or candle for symbols with open positions.

        Parameters
        ----------
        symbol : str
        current_price : float
            Current LTP.
        position : dict
            Position data from DB: direction, entry_price, stoploss_price,
            target_price, quantity, etc.
        candle : Candle, optional
            Latest candle if available.

        Returns
        -------
        Signal
            EXIT_LONG, EXIT_SHORT, or NO_ACTION.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return current strategy parameters as a dict.
        Used for logging, backtest records, and parameter optimization.

        Returns
        -------
        dict
            All tunable parameters and their current values.
        """
        ...

    # ── Optional Overrides ────────────────────────────────────────────────

    def end_of_day(self) -> None:
        """
        Called after market close.  Override for daily cleanup,
        state reset, performance logging, etc.
        Default: clears watchlist.
        """
        self._watchlist.clear()

    def build_trade_plan(
        self,
        signal: Signal,
        risk_per_trade: float,
    ) -> TradePlan | None:
        """
        Convert a signal into a TradePlan with position sizing.
        Default implementation handles basic LONG/SHORT plans.
        Override for custom position sizing logic.

        Parameters
        ----------
        signal : Signal
            The entry signal (BUY or SELL).
        risk_per_trade : float
            Max risk in INR for this trade.

        Returns
        -------
        TradePlan or None
            None if plan cannot be constructed (e.g. missing data).
        """
        from core.data.models import Direction

        indicator_data = signal.indicator_data

        # Extract required fields from indicator_data
        stoploss = indicator_data.get("stoploss_price")
        target = indicator_data.get("target_price")

        if stoploss is None:
            return None

        entry_price = signal.price_at_signal

        # Determine direction
        if signal.signal_type == SignalType.BUY:
            direction = Direction.LONG
            risk_per_share = abs(entry_price - stoploss)
        elif signal.signal_type == SignalType.SELL:
            direction = Direction.SHORT
            risk_per_share = abs(stoploss - entry_price)
        else:
            return None

        if risk_per_share <= 0:
            return None

        # Position sizing
        quantity = int(risk_per_trade / risk_per_share)
        if quantity <= 0:
            return None

        risk_amount = quantity * risk_per_share
        reward_amount = quantity * abs(target - entry_price) if target else None
        rr_ratio = abs(target - entry_price) / risk_per_share if target and risk_per_share > 0 else None

        plan = TradePlan(
            strategy_name=self._name,
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_price,
            stoploss_price=stoploss,
            target_price=target,
            quantity=quantity,
            risk_amount=round(risk_amount, 2),
            reward_amount=round(reward_amount, 2) if reward_amount else None,
            rr_ratio=round(rr_ratio, 2) if rr_ratio else None,
            metadata=signal.indicator_data,
        )

        return plan

    # ── Utility Methods ───────────────────────────────────────────────────

    def no_action_signal(
        self,
        symbol: str,
        price: float,
        timestamp: int,
        reason: str = "",
        indicator_data: dict | None = None,
    ) -> Signal:
        """
        Convenience method to create a NO_ACTION signal.

        Parameters
        ----------
        symbol : str
        price : float
        timestamp : int
        reason : str
        indicator_data : dict, optional

        Returns
        -------
        Signal
        """
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=price,
            timestamp=timestamp,
            indicator_data=indicator_data or {},
            skip_reason=reason,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self._name}' v{self._version} active={self._active}>"
