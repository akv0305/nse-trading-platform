"""
NSE Trading Platform — Fyers API v3 Broker Adapter

Production-ready adapter with:
  - WebSocket tick streaming with watchdog & auto-reconnect
  - REST fallback for LTP when WebSocket is stale
  - Order placement with retry logic & rate-limit handling
  - Historical data fetching with Fyers 100-day limit awareness
  - Thread-safe LTP cache
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from config.settings import settings
from core.broker.base import BrokerAdapter
from core.utils.time_utils import now_epoch_ms, now_ist, is_market_hours

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
STALE_DATA_THRESHOLD_SEC = 30      # Seconds before tick data is considered stale
MAX_RECONNECT_ATTEMPTS = 10        # Max WebSocket reconnection tries
RECONNECT_DELAY_SEC = 5            # Seconds between reconnection attempts
TICK_SIZE = 0.05                   # NSE equity tick size


class FyersAdapter(BrokerAdapter):
    """
    Fyers API v3 broker adapter.

    Usage
    -----
    >>> adapter = FyersAdapter()
    >>> adapter.connect()
    >>> adapter.subscribe_ticks(['NSE:RELIANCE-EQ'])
    >>> ltp = adapter.get_ltp('NSE:RELIANCE-EQ')
    """

    def __init__(self) -> None:
        self._fyers: fyersModel.FyersModel | None = None
        self._ws: data_ws.FyersDataSocket | None = None
        self._connected: bool = False
        self._ws_connected: bool = False

        # Thread-safe LTP cache: symbol → {'ltp': float, 'ts': epoch_ms}
        self._ltp_cache: dict[str, dict[str, float | int]] = {}
        self._ltp_lock = threading.Lock()

        # WebSocket management
        self._subscribed_symbols: list[str] = []
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_running: bool = False
        self._last_tick_time: float = 0.0  # time.time() of last tick
        self._reconnect_count: int = 0

        # Tick callback (optional)
        self._on_tick_callback: Callable[[dict], None] | None = None

    # ── Connection Lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate with Fyers using access token."""
        if not settings.FYERS_APP_ID or not settings.FYERS_ACCESS_TOKEN:
            raise ConnectionError(
                "FYERS_APP_ID and FYERS_ACCESS_TOKEN must be set in .env"
            )

        try:
            self._fyers = fyersModel.FyersModel(
                client_id=settings.FYERS_APP_ID,
                is_async=False,
                token=settings.FYERS_ACCESS_TOKEN,
                log_path="",
            )

            # Validate connection with a profile request
            profile = self._fyers.get_profile()
            if profile.get("s") == "ok":
                self._connected = True
                name = profile.get("data", {}).get("name", "Unknown")
                logger.info(f"Fyers connected: {name}")
            else:
                msg = profile.get("message", "Unknown error")
                raise ConnectionError(f"Fyers auth failed: {msg}")

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Fyers connection error: {e}") from e

    def disconnect(self) -> None:
        """Stop WebSocket, watchdog, and clean up."""
        self._watchdog_running = False
        self._kill_websocket()
        self._connected = False
        self._fyers = None
        logger.info("Fyers disconnected")

    def is_connected(self) -> bool:
        """Return True if Fyers session is active."""
        return self._connected and self._fyers is not None

    # ── WebSocket Tick Streaming ──────────────────────────────────────────

    def subscribe_ticks(
        self,
        symbols: list[str],
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        """Start WebSocket and subscribe to tick data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Fyers. Call connect() first.")

        self._subscribed_symbols = symbols
        self._on_tick_callback = on_tick
        self._start_websocket()

    def unsubscribe_ticks(self, symbols: list[str]) -> None:
        """Unsubscribe symbols. If no symbols left, stops WebSocket."""
        for s in symbols:
            if s in self._subscribed_symbols:
                self._subscribed_symbols.remove(s)

        if not self._subscribed_symbols:
            self._kill_websocket()

    def _start_websocket(self) -> None:
        """Initialize and start the Fyers WebSocket connection."""
        try:
            self._ws = data_ws.FyersDataSocket(
                access_token=f"{settings.FYERS_APP_ID}:{settings.FYERS_ACCESS_TOKEN}",
                log_path="",
                litemode=True,
                write_to_file=False,
                reconnect=True,
                on_connect=self._ws_on_connect,
                on_close=self._ws_on_close,
                on_error=self._ws_on_error,
                on_message=self._ws_on_message,
            )

            self._ws.subscribe(symbols=self._subscribed_symbols, data_type="SymbolUpdate")
            self._ws.keep_running()

            self._ws_connected = True
            self._reconnect_count = 0
            logger.info(f"WebSocket started, subscribed to {len(self._subscribed_symbols)} symbols")

            # Start watchdog
            self._start_watchdog()

        except Exception as e:
            logger.error(f"WebSocket start failed: {e}")
            self._ws_connected = False

    def _kill_websocket(self) -> None:
        """Force-close the WebSocket connection."""
        self._watchdog_running = False
        if self._ws is not None:
            try:
                self._ws.unsubscribe(symbols=self._subscribed_symbols, data_type="SymbolUpdate")
            except Exception:
                pass
            self._ws = None
        self._ws_connected = False
        logger.info("WebSocket killed")

    def _ws_on_connect(self) -> None:
        """WebSocket connected callback."""
        self._ws_connected = True
        self._reconnect_count = 0
        logger.info("WebSocket connected")

    def _ws_on_close(self, code: Any = None, reason: Any = None) -> None:
        """WebSocket disconnected callback."""
        self._ws_connected = False
        logger.warning(f"WebSocket closed: code={code}, reason={reason}")

    def _ws_on_error(self, error: Any) -> None:
        """WebSocket error callback."""
        logger.error(f"WebSocket error: {error}")

    def _ws_on_message(self, message: dict) -> None:
        """
        Process incoming tick message.
        Updates LTP cache and calls user callback if set.
        """
        try:
            if isinstance(message, dict):
                symbol = message.get("symbol", "")
                ltp = message.get("ltp", 0.0)

                if symbol and ltp:
                    now = now_epoch_ms()
                    with self._ltp_lock:
                        self._ltp_cache[symbol] = {"ltp": float(ltp), "ts": now}
                    self._last_tick_time = time.time()

                    if self._on_tick_callback:
                        self._on_tick_callback(message)

        except Exception as e:
            logger.error(f"Tick processing error: {e}")

    # ── Watchdog ──────────────────────────────────────────────────────────

    def _start_watchdog(self) -> None:
        """Start a background thread that monitors WebSocket health."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="FyersWatchdog",
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """
        Monitor WebSocket health. If no ticks for STALE_DATA_THRESHOLD_SEC,
        attempt reconnection.  Only runs during market hours.
        """
        while self._watchdog_running:
            try:
                time.sleep(STALE_DATA_THRESHOLD_SEC)

                if not self._watchdog_running:
                    break

                # Only monitor during market hours
                if not is_market_hours():
                    continue

                # Check if ticks are stale
                elapsed = time.time() - self._last_tick_time
                if elapsed > STALE_DATA_THRESHOLD_SEC and self._subscribed_symbols:
                    logger.warning(
                        f"No ticks for {elapsed:.0f}s, attempting reconnect "
                        f"(attempt {self._reconnect_count + 1}/{MAX_RECONNECT_ATTEMPTS})"
                    )

                    if self._reconnect_count < MAX_RECONNECT_ATTEMPTS:
                        self._reconnect_count += 1
                        self._kill_websocket()
                        time.sleep(RECONNECT_DELAY_SEC)
                        self._start_websocket()
                    else:
                        logger.error(
                            f"Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached. "
                            "Falling back to REST."
                        )
                        self._watchdog_running = False

            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    # ── LTP Access ────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float | None:
        """
        Get LTP from cache.  Falls back to REST if cache is stale.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        float or None
        """
        with self._ltp_lock:
            cached = self._ltp_cache.get(symbol)

        if cached:
            age_sec = (now_epoch_ms() - cached["ts"]) / 1000.0
            if age_sec < STALE_DATA_THRESHOLD_SEC:
                return cached["ltp"]

        # Fallback to REST
        return self._rest_get_ltp(symbol)

    def get_ltp_bulk(self, symbols: list[str]) -> dict[str, float | None]:
        """Get LTPs for multiple symbols."""
        result: dict[str, float | None] = {}
        stale_symbols: list[str] = []

        for symbol in symbols:
            with self._ltp_lock:
                cached = self._ltp_cache.get(symbol)
            if cached:
                age_sec = (now_epoch_ms() - cached["ts"]) / 1000.0
                if age_sec < STALE_DATA_THRESHOLD_SEC:
                    result[symbol] = cached["ltp"]
                    continue
            stale_symbols.append(symbol)

        # Bulk REST fallback for stale/missing symbols
        if stale_symbols:
            rest_ltps = self._rest_get_ltp_bulk(stale_symbols)
            result.update(rest_ltps)

        return result

    def _rest_get_ltp(self, symbol: str) -> float | None:
        """Fetch LTP via REST API as fallback."""
        if not self._fyers:
            return None
        try:
            data = {"symbols": symbol}
            response = self._fyers.quotes(data=data)
            if response.get("s") == "ok":
                quotes = response.get("d", [])
                if quotes:
                    ltp = quotes[0].get("v", {}).get("lp")
                    if ltp:
                        now = now_epoch_ms()
                        with self._ltp_lock:
                            self._ltp_cache[symbol] = {"ltp": float(ltp), "ts": now}
                        return float(ltp)
        except Exception as e:
            logger.error(f"REST LTP fetch failed for {symbol}: {e}")
        return None

    def _rest_get_ltp_bulk(self, symbols: list[str]) -> dict[str, float | None]:
        """Fetch LTPs for multiple symbols via REST."""
        result: dict[str, float | None] = {s: None for s in symbols}
        if not self._fyers:
            return result
        try:
            data = {"symbols": ",".join(symbols)}
            response = self._fyers.quotes(data=data)
            if response.get("s") == "ok":
                now = now_epoch_ms()
                for quote in response.get("d", []):
                    sym = quote.get("n", "")
                    ltp = quote.get("v", {}).get("lp")
                    if sym and ltp:
                        ltp_f = float(ltp)
                        result[sym] = ltp_f
                        with self._ltp_lock:
                            self._ltp_cache[sym] = {"ltp": ltp_f, "ts": now}
        except Exception as e:
            logger.error(f"REST bulk LTP fetch failed: {e}")
        return result

    # ── Order Management ──────────────────────────────────────────────────

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
        Place an order with Fyers.  Retries on rate-limit errors.

        Returns
        -------
        dict
            {'order_id': str, 'status': str, 'message': str}
        """
        if not self._fyers:
            raise ConnectionError("Fyers not connected")

        # Map to Fyers API constants
        side_map = {"BUY": 1, "SELL": -1}
        type_map = {"MARKET": 2, "LIMIT": 1, "SL": 3, "SL-M": 4}
        product_map = {"INTRADAY": "INTRADAY", "CNC": "CNC", "MARGIN": "MARGIN"}

        order_data = {
            "symbol": symbol,
            "qty": quantity,
            "type": type_map.get(order_type, 2),
            "side": side_map.get(side, 1),
            "productType": product_map.get(product_type, "INTRADAY"),
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "orderTag": tag,
        }

        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        else:
            order_data["limitPrice"] = 0

        if trigger_price is not None:
            order_data["stopPrice"] = trigger_price
        else:
            order_data["stopPrice"] = 0

        # Retry logic
        for attempt in range(1, settings.ORDER_RETRY_MAX + 1):
            try:
                response = self._fyers.place_order(data=order_data)

                if response.get("s") == "ok":
                    order_id = response.get("id", "")
                    logger.info(
                        f"Order placed: {side} {quantity} {symbol} @ {order_type} "
                        f"| ID: {order_id}"
                    )
                    return {
                        "order_id": order_id,
                        "status": "PLACED",
                        "message": response.get("message", ""),
                    }
                else:
                    msg = response.get("message", "Unknown error")
                    code = response.get("code", 0)

                    # Rate limit — retry after delay
                    if code == -16:
                        logger.warning(
                            f"Rate limited on attempt {attempt}, "
                            f"retrying in {settings.ORDER_RETRY_DELAY_SEC}s..."
                        )
                        time.sleep(settings.ORDER_RETRY_DELAY_SEC)
                        continue

                    logger.error(f"Order rejected: {msg} (code={code})")
                    return {
                        "order_id": "",
                        "status": "REJECTED",
                        "message": msg,
                    }

            except Exception as e:
                logger.error(f"Order placement error (attempt {attempt}): {e}")
                if attempt < settings.ORDER_RETRY_MAX:
                    time.sleep(settings.ORDER_RETRY_DELAY_SEC)
                    continue
                return {
                    "order_id": "",
                    "status": "REJECTED",
                    "message": str(e),
                }

        return {"order_id": "", "status": "REJECTED", "message": "Max retries exceeded"}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a pending order."""
        if not self._fyers:
            raise ConnectionError("Fyers not connected")
        try:
            response = self._fyers.cancel_order(data={"id": order_id})
            status = "CANCELLED" if response.get("s") == "ok" else "FAILED"
            return {
                "order_id": order_id,
                "status": status,
                "message": response.get("message", ""),
            }
        except Exception as e:
            logger.error(f"Cancel order failed for {order_id}: {e}")
            return {"order_id": order_id, "status": "FAILED", "message": str(e)}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get current status of a specific order."""
        if not self._fyers:
            raise ConnectionError("Fyers not connected")
        try:
            response = self._fyers.orderBook()
            if response.get("s") == "ok":
                for order in response.get("orderBook", []):
                    if order.get("id") == order_id:
                        return {
                            "order_id": order_id,
                            "status": self._map_fyers_status(order.get("status", 0)),
                            "filled_qty": order.get("filledQty", 0),
                            "avg_fill_price": order.get("tradedPrice", 0.0),
                            "message": order.get("message", ""),
                        }
            return {
                "order_id": order_id,
                "status": "UNKNOWN",
                "filled_qty": 0,
                "avg_fill_price": 0.0,
                "message": "Order not found in order book",
            }
        except Exception as e:
            logger.error(f"Get order status failed: {e}")
            return {
                "order_id": order_id,
                "status": "UNKNOWN",
                "filled_qty": 0,
                "avg_fill_price": 0.0,
                "message": str(e),
            }

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Return all open/pending orders."""
        if not self._fyers:
            return []
        try:
            response = self._fyers.orderBook()
            if response.get("s") == "ok":
                orders = []
                for order in response.get("orderBook", []):
                    status = order.get("status", 0)
                    # Fyers status 6 = pending/open
                    if status in (6, 1, 4):
                        orders.append({
                            "order_id": order.get("id", ""),
                            "symbol": order.get("symbol", ""),
                            "side": "BUY" if order.get("side") == 1 else "SELL",
                            "quantity": order.get("qty", 0),
                            "order_type": order.get("type", ""),
                            "status": self._map_fyers_status(status),
                        })
                return orders
        except Exception as e:
            logger.error(f"Get open orders failed: {e}")
        return []

    # ── Positions ─────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all current positions."""
        if not self._fyers:
            return []
        try:
            response = self._fyers.positions()
            if response.get("s") == "ok":
                positions = []
                for pos in response.get("netPositions", []):
                    net_qty = pos.get("netQty", 0)
                    if net_qty != 0:
                        positions.append({
                            "symbol": pos.get("symbol", ""),
                            "side": "BUY" if net_qty > 0 else "SELL",
                            "quantity": abs(net_qty),
                            "avg_price": pos.get("avgPrice", 0.0),
                            "ltp": pos.get("ltp", 0.0),
                            "pnl": pos.get("pl", 0.0),
                            "product_type": pos.get("productType", ""),
                        })
                return positions
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
        return []

    def flatten_all(self, product_type: str = "INTRADAY") -> list[dict[str, Any]]:
        """Close all open positions by placing counter orders."""
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
            logger.info(
                f"Flatten: {counter_side} {pos['quantity']} {pos['symbol']} → {result['status']}"
            )

        return results

    # ── Historical Data ───────────────────────────────────────────────────

    def fetch_historical_data(
        self,
        symbol: str,
        resolution: str,
        from_epoch: int,
        to_epoch: int,
    ) -> list[dict]:
        """
        Fetch historical OHLCV candles from Fyers.

        Note: Fyers limits to 100 days per request for intraday data.
        For longer ranges, the caller (HistoricalDataManager) should
        chunk the requests.

        Parameters
        ----------
        symbol : str
        resolution : str
            '1', '3', '5', '15', '30', '60', '1D'
        from_epoch : int
            Start time, Unix epoch seconds.
        to_epoch : int
            End time, Unix epoch seconds.

        Returns
        -------
        list[dict]
            Each: {'timestamp': int, 'open': float, 'high': float,
            'low': float, 'close': float, 'volume': int}
        """
        if not self._fyers:
            raise ConnectionError("Fyers not connected")

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",  # epoch
            "range_from": str(from_epoch),
            "range_to": str(to_epoch),
            "cont_flag": "1",
        }

        try:
            response = self._fyers.history(data=data)
            if response.get("s") == "ok":
                candles = response.get("candles", [])
                result = []
                for c in candles:
                    if len(c) >= 6:
                        result.append({
                            "timestamp": int(c[0]),
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": int(c[5]),
                        })
                return result
            else:
                logger.error(
                    f"History fetch failed for {symbol}: {response.get('message', '')}"
                )
                return []

        except Exception as e:
            logger.error(f"History fetch error for {symbol}: {e}")
            return []

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _map_fyers_status(status_code: int) -> str:
        """Map Fyers numeric order status to string."""
        status_map = {
            1: "PLACED",
            2: "FILLED",
            3: "PARTIAL",
            4: "PENDING",
            5: "CANCELLED",
            6: "PENDING",
            7: "REJECTED",
        }
        return status_map.get(status_code, "UNKNOWN")
