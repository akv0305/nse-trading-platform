"""
NSE Trading Platform — Risk Manager

Enforces all risk rules: per-trade, per-strategy, daily limits,
position limits, trading hours, and kill switch.

Used by the engine before every order placement.
"""

from __future__ import annotations

import logging

from config.settings import settings
from core.utils.logger import DBWriter
from core.utils.time_utils import now_epoch_ms, is_past_time, is_market_hours

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralized risk management.

    Checks performed before every trade:
      1. Daily P&L within limit
      2. Per-strategy daily P&L within limit
      3. Open positions under max concurrent limit
      4. Trading within allowed hours
      5. Kill switch not active
      6. Risk per trade within budget

    Usage
    -----
    >>> rm = RiskManager(db_writer)
    >>> allowed, reason = rm.can_trade("default_user", "orb_vwap", risk_amount=3750.0)
    >>> if not allowed:
    ...     print(f"Trade blocked: {reason}")
    """

    def __init__(self, db: DBWriter) -> None:
        self._db = db
        self._kill_switch: dict[str, bool] = {}  # user_id → active

    def can_trade(
        self,
        user_id: str,
        strategy_name: str,
        risk_amount: float,
    ) -> tuple[bool, str]:
        """
        Check if a new trade is allowed.

        Parameters
        ----------
        user_id : str
        strategy_name : str
        risk_amount : float
            INR at risk for the proposed trade.

        Returns
        -------
        tuple[bool, str]
            (allowed, reason). reason is empty if allowed.
        """
        # Kill switch check
        if self._kill_switch.get(user_id, False):
            return False, "Kill switch is active"

        # Trading hours check
        if not is_market_hours(settings.MARKET_OPEN, settings.MARKET_CLOSE):
            return False, "Outside market hours"

        # Entry cutoff check
        if is_past_time(settings.ENTRY_CUTOFF_IST):
            return False, f"Past entry cutoff ({settings.ENTRY_CUTOFF_IST})"

        # Daily P&L check
        daily_pnl = self._db.get_todays_daily_pnl(user_id)
        if daily_pnl <= -settings.DAILY_LOSS_LIMIT_INR:
            self.activate_kill_switch(user_id, "Daily loss limit breached")
            return False, f"Daily loss limit breached: ₹{daily_pnl:,.2f}"

        # Per-strategy daily loss check
        strategy_pnl = self._get_strategy_daily_pnl(user_id, strategy_name)
        if strategy_pnl <= -settings.PER_STRATEGY_DAILY_LOSS:
            return False, f"Strategy '{strategy_name}' daily loss limit breached: ₹{strategy_pnl:,.2f}"

        # Position count check
        open_count = self._db.count_open_positions(user_id)
        if open_count >= settings.MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached"

        # Risk per trade check
        max_risk = settings.RISK_PER_TRADE_INR
        if risk_amount > max_risk:
            return False, f"Risk ₹{risk_amount:,.2f} exceeds max ₹{max_risk:,.2f}"

        return True, ""

    def should_flatten(self, user_id: str) -> tuple[bool, str]:
        """
        Check if all positions should be flattened.

        Returns
        -------
        tuple[bool, str]
            (should_flatten, reason)
        """
        if self._kill_switch.get(user_id, False):
            return True, "Kill switch active"

        if is_past_time(settings.FLATTEN_TIME_IST):
            return True, f"Flatten time reached ({settings.FLATTEN_TIME_IST})"

        daily_pnl = self._db.get_todays_daily_pnl(user_id)
        if daily_pnl <= -settings.DAILY_LOSS_LIMIT_INR:
            self.activate_kill_switch(user_id, "Daily loss limit — flatten all")
            return True, "Daily loss limit breached"

        return False, ""

    def activate_kill_switch(self, user_id: str, reason: str) -> None:
        """Activate kill switch for a user."""
        self._kill_switch[user_id] = True
        self._db.write_risk_event(
            user_id=user_id,
            event_type="KILL_SWITCH",
            description=reason,
            daily_pnl=self._db.get_todays_daily_pnl(user_id),
            threshold=settings.DAILY_LOSS_LIMIT_INR,
            action_taken="KILL",
        )
        self._db.upsert_system_state(user_id=user_id, kill_switch_active=True)
        logger.critical(f"KILL SWITCH ACTIVATED for {user_id}: {reason}")

    def deactivate_kill_switch(self, user_id: str) -> None:
        """Manually deactivate kill switch (e.g. next trading day)."""
        self._kill_switch[user_id] = False
        self._db.upsert_system_state(user_id=user_id, kill_switch_active=False)
        logger.info(f"Kill switch deactivated for {user_id}")

    def is_kill_switch_active(self, user_id: str) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch.get(user_id, False)

    def _get_strategy_daily_pnl(self, user_id: str, strategy_name: str) -> float:
        """Get today's net P&L for a specific strategy."""
        trades = self._db.get_todays_trades(user_id)
        return sum(
            t.get("pnl_net", 0.0) or 0.0
            for t in trades
            if t.get("strategy_name") == strategy_name and t.get("exit_price") is not None
        )
