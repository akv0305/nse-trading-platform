"""
NSE Trading Platform — Logging Utilities

Two logging systems:
  1. DBWriter  — Thread-safe SQLite writer for structured records.
  2. JSONLLogger — Append-only JSONL file logger for audit trail.

Both are used by the engine, strategies, and risk manager.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import orjson

from config.settings import settings
from core.utils.time_utils import now_epoch_ms, today_date_str
from core.utils.ids import (
    generate_signal_id,
    generate_order_id,
    generate_trade_id,
    generate_plan_id,
    generate_risk_event_id,
    generate_command_id,
    generate_position_id,
)


class DBWriter:
    """
    Thread-safe SQLite writer.

    On init, creates the database (if missing), loads the schema,
    and ensures the default user exists.  Every write method is
    protected by a threading lock.

    Usage
    -----
    >>> db = DBWriter()
    >>> db.write_signal(user_id="default_user", strategy_name="orb_vwap", ...)
    """

    def __init__(self, db_path: Path | None = None, schema_path: Path | None = None) -> None:
        self._db_path = db_path or settings.DB_PATH
        self._schema_path = schema_path or settings.SCHEMA_PATH
        self._lock = threading.Lock()

        # Ensure storage directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create DB and load schema
        self._init_db()

    def _init_db(self) -> None:
        """Create database and execute schema if tables don't exist."""
        with self._lock:
            conn = self._connect()
            try:
                # Check if schema is already loaded
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
                )
                if cursor.fetchone() is None:
                    # Load and execute schema
                    if self._schema_path.exists():
                        schema_sql = self._schema_path.read_text(encoding="utf-8")
                        conn.executescript(schema_sql)
                        conn.commit()
                    else:
                        raise FileNotFoundError(
                            f"Schema file not found: {self._schema_path}"
                        )
                conn.close()
            except Exception:
                conn.close()
                raise

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode and pragmas."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write Methods ─────────────────────────────────────────────────────

    def write_signal(
        self,
        user_id: str,
        strategy_name: str,
        symbol: str,
        signal_type: str,
        strength: float,
        price_at_signal: float,
        indicator_data: dict | None = None,
        skip_reason: str = "",
        acted_on: bool = False,
    ) -> str:
        """
        Insert a new signal record.

        Returns
        -------
        str
            The generated signal_id.
        """
        signal_id = generate_signal_id()
        now = now_epoch_ms()
        ind_json = json.dumps(indicator_data or {})

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO signals
                        (signal_id, user_id, strategy_name, symbol, signal_type,
                         strength, price_at_signal, indicator_data, skip_reason,
                         acted_on, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id, user_id, strategy_name, symbol, signal_type,
                        strength, price_at_signal, ind_json, skip_reason,
                        1 if acted_on else 0, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return signal_id

    def write_trade_plan(
        self,
        user_id: str,
        strategy_name: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stoploss_price: float,
        target_price: float | None,
        quantity: int,
        risk_amount: float,
        reward_amount: float | None = None,
        rr_ratio: float | None = None,
        plan_metadata: dict | None = None,
    ) -> str:
        """
        Insert a new trade plan.

        Returns
        -------
        str
            The generated plan_id.
        """
        plan_id = generate_plan_id()
        now = now_epoch_ms()
        meta_json = json.dumps(plan_metadata or {})

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO trade_plans
                        (plan_id, user_id, strategy_name, symbol, direction,
                         entry_price, stoploss_price, target_price, quantity,
                         risk_amount, reward_amount, rr_ratio, plan_metadata,
                         status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                    """,
                    (
                        plan_id, user_id, strategy_name, symbol, direction,
                        entry_price, stoploss_price, target_price, quantity,
                        risk_amount, reward_amount, rr_ratio, meta_json, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return plan_id

    def write_order(
        self,
        user_id: str,
        plan_id: str | None,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        limit_price: float | None = None,
        trigger_price: float | None = None,
        order_id: str | None = None,
    ) -> str:
        """
        Insert a new order record.

        Parameters
        ----------
        order_id : str, optional
            If provided (e.g. broker-assigned ID), use it.
            Otherwise, generate one.

        Returns
        -------
        str
            The order_id.
        """
        if order_id is None:
            order_id = generate_order_id()
        now = now_epoch_ms()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO orders
                        (order_id, user_id, plan_id, symbol, side, order_type,
                         quantity, limit_price, trigger_price, status, placed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                    """,
                    (
                        order_id, user_id, plan_id, symbol, side, order_type,
                        quantity, limit_price, trigger_price, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return order_id

    def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_qty: int = 0,
        avg_fill_price: float | None = None,
        broker_status: str = "",
        broker_msg: str = "",
    ) -> None:
        """Update an existing order's status and fill info."""
        now = now_epoch_ms()
        filled_at = now if status in ("FILLED", "PARTIAL") else None
        cancelled_at = now if status in ("CANCELLED", "REJECTED") else None

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE orders
                    SET status = ?, filled_qty = ?, avg_fill_price = ?,
                        broker_status = ?, broker_msg = ?,
                        filled_at = COALESCE(?, filled_at),
                        cancelled_at = COALESCE(?, cancelled_at)
                    WHERE order_id = ?
                    """,
                    (
                        status, filled_qty, avg_fill_price,
                        broker_status, broker_msg,
                        filled_at, cancelled_at, order_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def write_trade(
        self,
        user_id: str,
        strategy_name: str,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        entry_time: int,
        stoploss_price: float | None = None,
        target_price: float | None = None,
        entry_order_id: str | None = None,
        plan_id: str | None = None,
    ) -> str:
        """
        Insert a new trade (at entry time).  Exit fields filled later.

        Returns
        -------
        str
            The generated trade_id.
        """
        trade_id = generate_trade_id()
        now = now_epoch_ms()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO trades
                        (trade_id, user_id, strategy_name, symbol, direction,
                         quantity, entry_price, entry_time, stoploss_price,
                         target_price, entry_order_id, plan_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, user_id, strategy_name, symbol, direction,
                        quantity, entry_price, entry_time, stoploss_price,
                        target_price, entry_order_id, plan_id, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: int,
        pnl_gross: float,
        pnl_net: float,
        costs_total: float,
        exit_reason: str,
        exit_order_id: str | None = None,
        trade_metadata: dict | None = None,
    ) -> None:
        """Update a trade with exit details and P&L."""
        meta_json = json.dumps(trade_metadata or {})

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE trades
                    SET exit_price = ?, exit_time = ?, pnl_gross = ?,
                        pnl_net = ?, costs_total = ?, exit_reason = ?,
                        exit_order_id = ?, trade_metadata = ?
                    WHERE trade_id = ?
                    """,
                    (
                        exit_price, exit_time, pnl_gross, pnl_net,
                        costs_total, exit_reason, exit_order_id,
                        meta_json, trade_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def upsert_position(
        self,
        user_id: str,
        strategy_name: str,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        current_price: float | None = None,
        unrealised_pnl: float | None = None,
        stoploss_price: float | None = None,
        target_price: float | None = None,
        trade_id: str | None = None,
        position_id: str | None = None,
    ) -> str:
        """
        Insert or update a live position.

        Returns
        -------
        str
            The position_id.
        """
        if position_id is None:
            position_id = generate_position_id()
        now = now_epoch_ms()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO positions
                        (position_id, user_id, strategy_name, symbol, direction,
                         quantity, entry_price, current_price, unrealised_pnl,
                         stoploss_price, target_price, trade_id, opened_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(position_id) DO UPDATE SET
                        current_price = excluded.current_price,
                        unrealised_pnl = excluded.unrealised_pnl,
                        stoploss_price = excluded.stoploss_price,
                        target_price = excluded.target_price,
                        updated_at = excluded.updated_at
                    """,
                    (
                        position_id, user_id, strategy_name, symbol, direction,
                        quantity, entry_price, current_price, unrealised_pnl,
                        stoploss_price, target_price, trade_id, now, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return position_id

    def remove_position(self, position_id: str) -> None:
        """Delete a closed position from the positions table."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM positions WHERE position_id = ?", (position_id,))
                conn.commit()
            finally:
                conn.close()

    def write_risk_event(
        self,
        user_id: str,
        event_type: str,
        description: str,
        daily_pnl: float | None = None,
        threshold: float | None = None,
        action_taken: str = "",
        strategy_name: str = "",
    ) -> str:
        """
        Log a risk event.

        Returns
        -------
        str
            The generated event_id.
        """
        event_id = generate_risk_event_id()
        now = now_epoch_ms()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO risk_events
                        (event_id, user_id, event_type, strategy_name,
                         description, daily_pnl, threshold, action_taken, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, user_id, event_type, strategy_name,
                        description, daily_pnl, threshold, action_taken, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return event_id

    def write_command(
        self,
        user_id: str,
        command_type: str,
        payload: dict | None = None,
    ) -> str:
        """
        Insert a control command from the dashboard.

        Returns
        -------
        str
            The generated command_id.
        """
        command_id = generate_command_id()
        now = now_epoch_ms()
        payload_json = json.dumps(payload or {})

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO commands
                        (command_id, user_id, command_type, payload, status, created_at)
                    VALUES (?, ?, ?, ?, 'PENDING', ?)
                    """,
                    (command_id, user_id, command_type, payload_json, now),
                )
                conn.commit()
            finally:
                conn.close()

        return command_id

    def upsert_system_state(
        self,
        user_id: str,
        engine_status: str = "",
        active_strategy: str = "",
        current_positions: int = 0,
        daily_pnl: float = 0.0,
        daily_trades: int = 0,
        kill_switch_active: bool = False,
        last_tick_at: int | None = None,
        last_signal: str = "",
        last_signal_at: int | None = None,
    ) -> None:
        """Upsert the system state row for a user."""
        now = now_epoch_ms()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO system_state
                        (user_id, engine_status, active_strategy, current_positions,
                         daily_pnl, daily_trades, kill_switch_active,
                         last_tick_at, last_signal, last_signal_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        engine_status = excluded.engine_status,
                        active_strategy = excluded.active_strategy,
                        current_positions = excluded.current_positions,
                        daily_pnl = excluded.daily_pnl,
                        daily_trades = excluded.daily_trades,
                        kill_switch_active = excluded.kill_switch_active,
                        last_tick_at = COALESCE(excluded.last_tick_at, system_state.last_tick_at),
                        last_signal = excluded.last_signal,
                        last_signal_at = COALESCE(excluded.last_signal_at, system_state.last_signal_at),
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id, engine_status, active_strategy, current_positions,
                        daily_pnl, daily_trades, 1 if kill_switch_active else 0,
                        last_tick_at, last_signal, last_signal_at, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # ── Read Methods ──────────────────────────────────────────────────────

    def get_system_state(self, user_id: str) -> dict | None:
        """Fetch current system state for a user. Returns dict or None."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM system_state WHERE user_id = ?", (user_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_pending_commands(self, user_id: str) -> list[dict]:
        """Fetch all pending commands for a user, oldest first."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM commands
                    WHERE user_id = ? AND status = 'PENDING'
                    ORDER BY created_at ASC
                    """,
                    (user_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def mark_command_processed(self, command_id: str, status: str = "PROCESSED") -> None:
        """Mark a command as processed or failed."""
        now = now_epoch_ms()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE commands SET status = ?, processed_at = ? WHERE command_id = ?",
                    (status, now, command_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_todays_trades(self, user_id: str) -> list[dict]:
        """Fetch all trades entered today for a user."""
        today = today_date_str()
        # Compute epoch ms for start of today IST
        from core.utils.time_utils import IST
        import datetime

        start_of_day = datetime.datetime.strptime(today, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=IST
        )
        start_ms = int(start_of_day.timestamp() * 1000)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM trades
                    WHERE user_id = ? AND entry_time >= ?
                    ORDER BY entry_time ASC
                    """,
                    (user_id, start_ms),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_todays_daily_pnl(self, user_id: str) -> float:
        """Sum net P&L of all closed trades today for a user."""
        trades = self.get_todays_trades(user_id)
        return sum(t.get("pnl_net", 0.0) or 0.0 for t in trades if t.get("exit_price") is not None)

    def get_open_positions(self, user_id: str) -> list[dict]:
        """Fetch all open positions for a user."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE user_id = ? ORDER BY opened_at ASC",
                    (user_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_todays_signals(self, user_id: str) -> list[dict]:
        """Fetch all signals generated today for a user."""
        today = today_date_str()
        from core.utils.time_utils import IST
        import datetime

        start_of_day = datetime.datetime.strptime(today, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=IST
        )
        start_ms = int(start_of_day.timestamp() * 1000)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM signals
                    WHERE user_id = ? AND created_at >= ?
                    ORDER BY created_at ASC
                    """,
                    (user_id, start_ms),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_todays_risk_events(self, user_id: str) -> list[dict]:
        """Fetch all risk events from today for a user."""
        today = today_date_str()
        from core.utils.time_utils import IST
        import datetime

        start_of_day = datetime.datetime.strptime(today, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=IST
        )
        start_ms = int(start_of_day.timestamp() * 1000)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM risk_events
                    WHERE user_id = ? AND created_at >= ?
                    ORDER BY created_at ASC
                    """,
                    (user_id, start_ms),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_open_positions(self, user_id: str) -> int:
        """Return count of open positions for a user."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM positions WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()


class JSONLLogger:
    """
    Append-only JSONL file logger for audit trail.

    Creates one file per day: events_YYYYMMDD.jsonl
    Each line is a JSON object with timestamp, user_id, level, event, message, context.

    Usage
    -----
    >>> logger = JSONLLogger()
    >>> logger.log("default_user", "INFO", "ENGINE_START", "Engine started successfully")
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or settings.LOG_PATH
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_file_path(self) -> Path:
        """Return today's log file path."""
        date_str = today_date_str().replace("-", "")
        return self._log_dir / f"events_{date_str}.jsonl"

    def log(
        self,
        user_id: str,
        level: str,
        event: str,
        message: str,
        context: dict | None = None,
    ) -> None:
        """
        Write a single log entry as a JSON line.

        Parameters
        ----------
        user_id : str
            User associated with this event.
        level : str
            Log level: DEBUG | INFO | WARNING | ERROR | CRITICAL.
        event : str
            Event type identifier (e.g. 'ENGINE_START', 'SIGNAL_GENERATED').
        message : str
            Human-readable message.
        context : dict, optional
            Additional structured data to include.
        """
        record = {
            "ts": now_epoch_ms(),
            "user_id": user_id,
            "level": level,
            "event": event,
            "msg": message,
        }
        if context:
            record["ctx"] = context

        line = orjson.dumps(record).decode("utf-8") + "\n"

        with self._lock:
            file_path = self._get_file_path()
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line)
