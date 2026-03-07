"""
NSE Trading Platform — Order Manager

Handles the full order lifecycle: plan → place → monitor → fill/cancel.
Bridges strategy signals with broker adapter.

Implementation: Shared across strategies.
"""

from __future__ import annotations

import logging
from typing import Any

from core.broker.base import BrokerAdapter
from core.data.models import TradePlan, OrderSide, OrderType
from core.utils.logger import DBWriter
from core.utils.time_utils import now_epoch_ms

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages order lifecycle.

    Responsibilities:
      - Convert TradePlan → broker order
      - Place order via adapter (live or paper)
      - Track order status
      - Record fills in database
      - Handle partial fills and rejections

    Usage
    -----
    >>> om = OrderManager(broker_adapter, db_writer)
    >>> result = om.execute_plan(user_id, trade_plan)
    """

    def __init__(self, broker: BrokerAdapter, db: DBWriter) -> None:
        self._broker = broker
        self._db = db

    def execute_plan(
        self,
        user_id: str,
        plan: TradePlan,
        order_type: str = "MARKET",
        product_type: str = "INTRADAY",
    ) -> dict[str, Any]:
        """
        Execute a trade plan by placing an entry order.

        Parameters
        ----------
        user_id : str
        plan : TradePlan
        order_type : str
            'MARKET' or 'LIMIT'.
        product_type : str
            'INTRADAY' or 'CNC'.

        Returns
        -------
        dict
            {'success': bool, 'order_id': str, 'trade_id': str, 'message': str}
        """
        raise NotImplementedError("Order execution — implement during engine build")

    def place_exit_order(
        self,
        user_id: str,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        trigger_price: float | None = None,
        product_type: str = "INTRADAY",
        exit_reason: str = "",
    ) -> dict[str, Any]:
        """
        Place an exit order for an open position.

        Parameters
        ----------
        user_id : str
        trade_id : str
        symbol : str
        side : str
            Counter-side: 'SELL' for long exit, 'BUY' for short exit.
        quantity : int
        order_type : str
        limit_price : float, optional
        trigger_price : float, optional
        product_type : str
        exit_reason : str

        Returns
        -------
        dict
            {'success': bool, 'order_id': str, 'message': str}
        """
        raise NotImplementedError("Exit order — implement during engine build")

    def check_and_update_orders(self, user_id: str) -> list[dict]:
        """
        Poll broker for order status updates and sync with database.

        Returns
        -------
        list[dict]
            List of orders that changed status.
        """
        raise NotImplementedError("Order status sync — implement during engine build")
