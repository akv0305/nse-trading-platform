"""
NSE Trading Platform — Paper Trading Adapter

Wraps a real BrokerAdapter (typically FyersAdapter) for simulated
order execution.  Uses real market LTP from the underlying adapter
but simulates fills with configurable slippage.

Perfect for strategy validation before going live.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from core.broker.base import BrokerAdapter
from core.utils.ids import generate_id
from core.utils.time_utils import now_epoch_ms, now_ist

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
TICK_SIZE = 0.05  # NSE equity tick size (5 paise)


class PaperAdapter(BrokerAdapter):
    """
    Paper trading adapter.

    Uses a real adapter for market data (ticks, LTP, historical data)
    but simulates all order execution locally with slippage modeling.

    Parameters
    ----------
    real_adapter : BrokerAdapter
        Underlying adapter for real market data.
    base_slippage_ticks : int
        Number of ticks of slippage to apply. Default 1 (= 0.05).
    volatile_slippage_ticks : int
        Extra slippage during volatile periods (open, close). Default 2.

    Usage
    -----
    >>> real = FyersAdapter()
    >>> real.connect()
    >>> paper = PaperAdapter(real)
    >>> paper.connect()  # No-op, delegates to real
    >>> result = paper.place_order('NSE:RELIANCE-EQ', 'BUY', 'MARKET', 100)
    """

    def __init__(
        self,
        real_adapter: BrokerAdapter,
        base_slippage_ticks: int = 1,
        volatile_slippage_ticks: int = 2,
    ) -> None:
        self._real = real_adapter
        self._base_slippage_ticks = base_slippage_ticks
        self._volatile_slippage_ticks = volatile_slippage_ticks

        # Simulated state
        self._orders: dict[str, dict[str, Any]] = {}       # order_id → order data
        self._positions: dict[str, dict[str, Any]] = {}     # symbol → position
        self._lock = threading.Lock()

    # ── Connection (delegated to real adapter) ────────────────────────────

    def connect(self) -> None:
        """Delegates to real adapter. Paper adapter has no own connection."""
        if not self._real.is_connected():
            self._real.connect()
        logger.info("Paper adapter connected (using real adapter for market data)")

    def disconnect(self) -> None:
        """Paper adapter disconnect — does not disconnect real adapter."""
        logger.info("Paper adapter disconnected")

    def is_connected(self) -> bool:
        """Paper adapter is connected if real adapter is."""
        return self._real.is_connected()

    # ── Tick Data (delegated to real adapter) ─────────────────────────────

    def subscribe_ticks(
        self,
        symbols: list[str],
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        """Delegates tick subscription to real adapter."""
        self._real.subscribe_ticks(symbols, on_tick)

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Delegates to real adapter."""
        self._real.unsubscribe_ticks(symbols)

    def get_ltp(self, symbol: str) -> float | None:
        """Get real LTP from underlying adapter."""
        return self._real.get_ltp(symbol)

    def get_ltp_bulk(self, symbols: list[str]) -> dict[str, float | None]:
        """Get real LTPs from underlying adapter."""
        return self._real.get_ltp_bulk(symbols)

    # ── Slippage Model ────────────────────────────────────────────────────

    def _compute_slippage(self, side: str) -> float:
        """
        Compute slippage in price terms based on time of day.

        During volatile periods (first 10 min after open, last 40 min before
        close), slippage is doubled.

        Parameters
        ----------
        side : str
            'BUY' or 'SELL'.

        Returns
        -------
        float
            Signed slippage: positive for BUY (price goes up),
            negative for SELL (price goes down).
        """
        current = now_ist()
        h, m = current.hour, current.minute

        # Volatile periods: 09:15-09:25 and 14:50-15:30
        is_volatile = (
            (h == 9 and 15 <= m <= 25)
            or (h == 14 and m >= 50)
            or (h == 15 and m <= 30)
        )

        ticks = self._volatile_slippage_ticks if is_volatile else self._base_slippage_ticks
        slippage = ticks * TICK_SIZE

        # BUY → fill higher (worse), SELL → fill lower (worse)
        return slippage if side == "BUY" else -slippage

    @staticmethod
    def _round_to_tick(price: float) -> float:
        """Round price to nearest tick size."""
        return round(round(price / TICK_SIZE) * TICK_SIZE, 2)

    # ── Order Management (simulated) ──────────────────────────────────────

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
        Simulate order placement.

        For MARKET orders: fills immediately at LTP ± slippage.
        For LIMIT orders: fills at limit price (optimistic simulation).
        """
        order_id = generate_id("PAPER")
        now = now_epoch_ms()

        ltp = self.get_ltp(symbol)
        if ltp is None:
            logger.warning(f"Paper order: no LTP for {symbol}, rejecting")
            order = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "status": "REJECTED",
                "filled_qty": 0,
                "avg_fill_price": 0.0,
                "message": "No market data available",
                "placed_at": now,
            }
            with self._lock:
                self._orders[order_id] = order
            return {"order_id": order_id, "status": "REJECTED", "message": order["message"]}

        # Calculate fill price
        if order_type == "MARKET":
            slippage = self._compute_slippage(side)
            fill_price = self._round_to_tick(ltp + slippage)
        elif order_type == "LIMIT" and limit_price is not None:
            fill_price = limit_price
        else:
            fill_price = self._round_to_tick(ltp)

        # Create filled order
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "limit_price": limit_price,
            "trigger_price": trigger_price,
            "product_type": product_type,
            "tag": tag,
            "status": "FILLED",
            "filled_qty": quantity,
            "avg_fill_price": fill_price,
            "message": "Paper fill",
            "placed_at": now,
            "filled_at": now,
        }

        with self._lock:
            self._orders[order_id] = order
            self._update_position(symbol, side, quantity, fill_price, product_type)

        logger.info(
            f"Paper {side} {quantity} {symbol} @ {fill_price:.2f} "
            f"(LTP={ltp:.2f}, slip={fill_price - ltp:+.2f}) | ID: {order_id}"
        )

        return {
            "order_id": order_id,
            "status": "FILLED",
            "message": f"Paper fill at {fill_price:.2f}",
        }

    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        product_type: str,
    ) -> None:
        """Update internal position tracking after a fill."""
        signed_qty = quantity if side == "BUY" else -quantity

        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos["net_qty"]
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position fully closed
                del self._positions[symbol]
            else:
                # Update average price for adding to position
                if (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                    # Adding to same direction
                    total_cost = pos["avg_price"] * abs(old_qty) + fill_price * quantity
                    pos["avg_price"] = total_cost / abs(new_qty)
                pos["net_qty"] = new_qty
        else:
            if signed_qty != 0:
                self._positions[symbol] = {
                    "symbol": symbol,
                    "net_qty": signed_qty,
                    "avg_price": fill_price,
                    "product_type": product_type,
                }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a paper order (only if still pending — unlikely in paper mode)."""
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                if order["status"] == "PENDING":
                    order["status"] = "CANCELLED"
                    return {"order_id": order_id, "status": "CANCELLED", "message": "Paper cancelled"}
                return {
                    "order_id": order_id,
                    "status": order["status"],
                    "message": f"Cannot cancel: order already {order['status']}",
                }
        return {"order_id": order_id, "status": "UNKNOWN", "message": "Order not found"}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of a paper order."""
        with self._lock:
            if order_id in self._orders:
                o = self._orders[order_id]
                return {
                    "order_id": order_id,
                    "status": o["status"],
                    "filled_qty": o["filled_qty"],
                    "avg_fill_price": o["avg_fill_price"],
                    "message": o["message"],
                }
        return {
            "order_id": order_id,
            "status": "UNKNOWN",
            "filled_qty": 0,
            "avg_fill_price": 0.0,
            "message": "Paper order not found",
        }

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Return pending paper orders (rare in paper mode since fills are instant)."""
        with self._lock:
            return [
                {
                    "order_id": o["order_id"],
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "quantity": o["quantity"],
                    "order_type": o["order_type"],
                    "status": o["status"],
                }
                for o in self._orders.values()
                if o["status"] == "PENDING"
            ]

    # ── Positions ─────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all paper positions."""
        with self._lock:
            positions = []
            for sym, pos in self._positions.items():
                ltp = self.get_ltp(sym) or pos["avg_price"]
                net_qty = pos["net_qty"]
                pnl = (ltp - pos["avg_price"]) * net_qty
                positions.append({
                    "symbol": sym,
                    "side": "BUY" if net_qty > 0 else "SELL",
                    "quantity": abs(net_qty),
                    "avg_price": pos["avg_price"],
                    "ltp": ltp,
                    "pnl": round(pnl, 2),
                    "product_type": pos["product_type"],
                })
            return positions

    def flatten_all(self, product_type: str = "INTRADAY") -> list[dict[str, Any]]:
        """Close all paper positions."""
        positions = self.get_positions()
        results: list[dict[str, Any]] = []

        for pos in positions:
            if product_type and pos["product_type"] != product_type:
                continue

            counter_side = "SELL" if pos["side"] == "BUY" else "BUY"
            result = self.place_order(
                symbol=pos["symbol"],
                side=counter_side,
                order_type="MARKET",
                quantity=pos["quantity"],
                product_type=pos["product_type"],
                tag="FLATTEN",
            )
            results.append(result)

        return results

    # ── Historical Data (delegated) ───────────────────────────────────────

    def fetch_historical_data(
        self,
        symbol: str,
        resolution: str,
        from_epoch: int,
        to_epoch: int,
    ) -> list[dict]:
        """Delegates to real adapter."""
        return self._real.fetch_historical_data(symbol, resolution, from_epoch, to_epoch)
