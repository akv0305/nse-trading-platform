"""
NSE Trading Platform — Abstract Broker Adapter

All broker implementations (Fyers, Paper, future brokers) must
implement this interface.  The engine and execution layer only
interact through this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class BrokerAdapter(ABC):
    """
    Abstract base class for broker connections.

    Each user gets their own adapter instance.  The adapter handles:
      - Authentication & connection lifecycle
      - Real-time tick streaming (WebSocket)
      - Order placement, cancellation, status
      - Position and holdings queries
      - Historical data fetching
    """

    # ── Connection Lifecycle ──────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """
        Authenticate with the broker and establish session.

        Raises
        ------
        ConnectionError
            If authentication fails or broker is unreachable.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close all connections and clean up resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if broker session is active and healthy."""
        ...

    # ── Real-Time Data ────────────────────────────────────────────────────

    @abstractmethod
    def subscribe_ticks(
        self,
        symbols: list[str],
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        """
        Subscribe to real-time tick data for given symbols.

        Parameters
        ----------
        symbols : list[str]
            Fyers-format symbols, e.g. ['NSE:RELIANCE-EQ', 'NSE:INFY-EQ'].
        on_tick : callable, optional
            Callback invoked with tick data dict on each tick.
            If None, ticks are stored internally and retrieved via get_ltp().
        """
        ...

    @abstractmethod
    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe from tick data for given symbols."""
        ...

    @abstractmethod
    def get_ltp(self, symbol: str) -> float | None:
        """
        Get last traded price for a symbol.

        Returns None if no tick received yet or data is stale.

        Parameters
        ----------
        symbol : str
            Fyers-format symbol.

        Returns
        -------
        float or None
        """
        ...

    @abstractmethod
    def get_ltp_bulk(self, symbols: list[str]) -> dict[str, float | None]:
        """
        Get LTPs for multiple symbols in one call.

        Parameters
        ----------
        symbols : list[str]

        Returns
        -------
        dict[str, float | None]
            Symbol → LTP mapping.  None for symbols with no data.
        """
        ...

    # ── Order Management ──────────────────────────────────────────────────

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        limit_price: float | None = None,
        trigger_price: float | None = None,
        product_type: str = "INTRADAY",
        tag: str = "",
    ) -> dict[str, Any]:
        """
        Place an order with the broker.

        Parameters
        ----------
        symbol : str
            Fyers-format symbol.
        side : str
            'BUY' or 'SELL'.
        order_type : str
            'MARKET', 'LIMIT', 'SL', 'SL-M'.
        quantity : int
            Number of shares.
        limit_price : float, optional
            Required for LIMIT and SL orders.
        trigger_price : float, optional
            Required for SL and SL-M orders.
        product_type : str
            'INTRADAY' (MIS) or 'CNC' (delivery) or 'MARGIN'.
        tag : str
            Optional order tag for identification.

        Returns
        -------
        dict
            Must contain at minimum:
              - 'order_id': str (broker-assigned)
              - 'status': str ('PLACED', 'REJECTED', etc.)
              - 'message': str (broker message)

        Raises
        ------
        ConnectionError
            If broker is unreachable.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel a pending order.

        Parameters
        ----------
        order_id : str

        Returns
        -------
        dict
            {'order_id': str, 'status': str, 'message': str}
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get current status of an order.

        Returns
        -------
        dict
            Must contain: 'order_id', 'status', 'filled_qty',
            'avg_fill_price', 'message'.
        """
        ...

    @abstractmethod
    def get_open_orders(self) -> list[dict[str, Any]]:
        """Return list of all currently open/pending orders."""
        ...

    # ── Positions ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """
        Get all current positions from the broker.

        Returns
        -------
        list[dict]
            Each dict must contain: 'symbol', 'side', 'quantity',
            'avg_price', 'ltp', 'pnl', 'product_type'.
        """
        ...

    @abstractmethod
    def flatten_all(self, product_type: str = "INTRADAY") -> list[dict[str, Any]]:
        """
        Close all open positions by placing counter orders.

        Parameters
        ----------
        product_type : str
            Filter by product type. Default 'INTRADAY'.

        Returns
        -------
        list[dict]
            List of order results for each closing order.
        """
        ...

    # ── Historical Data ───────────────────────────────────────────────────

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        resolution: str,
        from_epoch: int,
        to_epoch: int,
    ) -> list[dict]:
        """
        Fetch historical OHLCV candles.

        Parameters
        ----------
        symbol : str
            Fyers-format symbol.
        resolution : str
            Candle resolution: '1', '3', '5', '15', '30', '60', '1D'.
        from_epoch : int
            Start time as Unix epoch seconds.
        to_epoch : int
            End time as Unix epoch seconds.

        Returns
        -------
        list[dict]
            Each dict: {'timestamp': int, 'open': float, 'high': float,
            'low': float, 'close': float, 'volume': int}
        """
        ...
