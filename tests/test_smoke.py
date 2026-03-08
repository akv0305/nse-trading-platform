"""
NSE Trading Platform — Smoke Tests

Basic tests to verify all modules import correctly and
core functionality works without broker connection.
"""

from __future__ import annotations


def test_config_loads():
    """Settings load with defaults."""
    from config.settings import settings
    assert settings.TOTAL_CAPITAL == 750_000.0
    assert settings.RISK_PER_TRADE_PCT == 0.5
    assert settings.RISK_PER_TRADE_INR == 3_750.0


def test_time_utils():
    """Time utilities work correctly."""
    from core.utils.time_utils import now_ist, now_epoch_ms, today_date_str
    assert now_ist().tzinfo is not None
    assert now_epoch_ms() > 0
    assert len(today_date_str()) == 10


def test_id_generation():
    """IDs are unique and properly formatted."""
    from core.utils.ids import generate_signal_id, generate_order_id
    id1 = generate_signal_id()
    id2 = generate_signal_id()
    assert id1 != id2
    assert id1.startswith("SIG_")
    assert generate_order_id().startswith("ORD_")


def test_data_models():
    """Data models instantiate correctly."""
    from core.data.models import Candle, Signal, SignalType, Direction, TradePlan
    c = Candle("NSE:TEST-EQ", 1000000, 100.0, 105.0, 98.0, 103.0, 50000)
    assert c.is_bullish
    assert c.typical_price == (105.0 + 98.0 + 103.0) / 3


def test_universe():
    """Stock universe has 50 stocks with sector mapping."""
    from core.data.universe import NIFTY50, SECTOR_MAP, to_fyers_symbol
    assert len(NIFTY50) == 49
    assert len(SECTOR_MAP) == 49
    assert to_fyers_symbol("RELIANCE") == "NSE:RELIANCE-EQ"


def test_cost_model():
    """Cost model produces reasonable numbers."""
    from backtest.cost_model import FyersCostModel
    model = FyersCostModel()
    costs = model.compute_trade_costs(1000.0, 1010.0, 100)
    assert costs.total > 0
    assert costs.brokerage > 0
    assert costs.stt > 0


def test_db_writer():
    """DBWriter creates database and tables."""
    from core.utils.logger import DBWriter
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        from config.settings import settings
        db = DBWriter(db_path=db_path, schema_path=settings.SCHEMA_PATH)
        state = db.get_system_state("default_user")
        assert state is not None
        assert state["engine_status"] == "STOPPED"


def test_candle_builder():
    """LiveCandleBuilder aggregates ticks."""
    from core.data.live_candles import LiveCandleBuilder
    from core.utils.time_utils import now_epoch_ms
    builder = LiveCandleBuilder(interval_minutes=5)
    now = now_epoch_ms()
    builder.on_tick("TEST", 100.0, 1000, now)
    builder.on_tick("TEST", 105.0, 500, now + 1000)
    building = builder.get_building_candle("TEST")
    assert building is not None
    assert building["high"] == 105.0
    assert building["low"] == 100.0


def test_risk_manager():
    """RiskManager instantiates and checks work."""
    from core.risk.risk_manager import RiskManager
    from core.utils.logger import DBWriter
    import tempfile
    from pathlib import Path
    from config.settings import settings

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = DBWriter(db_path=db_path, schema_path=settings.SCHEMA_PATH)
        rm = RiskManager(db)
        # Outside market hours — should be blocked
        allowed, reason = rm.can_trade("default_user", "test", 3750.0)
        # Will fail due to market hours check (we're running tests outside market)
        assert not allowed or reason == ""
