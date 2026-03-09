"""
NSE Trading Platform — Central Configuration
Uses Pydantic BaseSettings to read from .env file with sensible defaults.
All monetary values in INR. All timestamps in IST (Asia/Kolkata).
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ---------------------------------------------------------------------------
# Project root = parent of 'config/' directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Single source of truth for every tunable parameter.
    Values load in order: .env file → environment variables → defaults below.
    """

    # ── Fyers Broker Credentials ──────────────────────────────────────────
    FYERS_APP_ID: str = Field(default="", description="Fyers API v3 app ID (e.g. XXXXXXXX-100)")
    FYERS_SECRET_KEY: str = Field(default="", description="Fyers API secret key")
    FYERS_REDIRECT_URI: str = Field(default="https://127.0.0.1", description="OAuth redirect URI registered in Fyers app")
    FYERS_ACCESS_TOKEN: str = Field(default="", description="Current session access token (refreshed daily)")
    FYERS_PIN: str = Field(default="", description="4-digit PIN for Fyers login automation (optional)")
    FYERS_TOTP_KEY: str = ""         # External 2FA TOTP secret key from myaccount.fyers.in
    FYERS_USER_ID: str = ""          # Fyers client ID (e.g., "AB1234")

    # ── Capital & Risk ────────────────────────────────────────────────────
    TOTAL_CAPITAL: float = Field(default=750_000.0, description="Total trading capital in INR")
    RISK_PER_TRADE_PCT: float = Field(default=0.5, description="Max risk per single trade as % of capital")
    MAX_CONCURRENT_POSITIONS: int = Field(default=3, description="Max open positions across all strategies")
    DAILY_LOSS_LIMIT_PCT: float = Field(default=2.0, description="Hard daily loss limit as % of capital — kill switch triggers")
    PER_STRATEGY_DAILY_LOSS: float = Field(default=6_000.0, description="Max daily loss per strategy in INR")

    # ── Trading Hours (IST, 24h format) ───────────────────────────────────
    MARKET_OPEN: str = Field(default="09:15", description="NSE market open time IST")
    MARKET_CLOSE: str = Field(default="15:30", description="NSE market close time IST")
    PRE_MARKET_BUFFER_MIN: int = Field(default=10, description="Minutes before market open to start engine")
    ENTRY_CUTOFF_IST: str = Field(default="14:45", description="No new entries after this time IST")
    FLATTEN_TIME_IST: str = Field(default="15:22", description="Force-flatten all intraday positions by this time IST")
    NO_TRADE_FIRST_MIN: int = Field(default=3, description="Skip first N minutes after open for data settling")

    # ── ORB + VWAP Strategy Defaults ──────────────────────────────────────
    ORB_PERIOD_MINUTES: int = Field(default=15, description="Opening range duration in minutes (15 or 30)")
    ORB_BREAKOUT_BUFFER_PCT: float = Field(default=0.05, description="Buffer % above/below ORB high/low for breakout trigger")
    VWAP_BAND_STD_MULTIPLIER: float = Field(default=1.5, description="VWAP band = VWAP ± (multiplier × std-dev)")
    ORB_STOPLOSS_MODE: str = Field(default="ORB_OPPOSITE", description="SL mode: ORB_OPPOSITE | ATR_BASED | FIXED_PCT")
    ORB_TARGET_RR: float = Field(default=2.0, description="Default reward:risk ratio for ORB targets")
    ORB_TRAIL_AFTER_RR: float = Field(default=1.0, description="Start trailing SL after this R:R achieved")
    ORB_TRAIL_PCT: float = Field(default=0.3, description="Trailing SL as % of move from entry")

    # ── Supply/Demand Zone Strategy Defaults ──────────────────────────────
    SD_LOOKBACK_DAYS: int = Field(default=60, description="Days of history to scan for S/D zones")
    SD_MIN_ZONE_STRENGTH: float = Field(default=2.0, description="Min score for a zone to be tradeable (0-5 scale)")
    SD_ZONE_FRESHNESS_DAYS: int = Field(default=20, description="Zones older than this get score penalty")
    SD_RISK_REWARD_MIN: float = Field(default=2.0, description="Min R:R to take a zone trade")
    SD_MAX_ZONE_WIDTH_PCT: float = Field(default=3.0, description="Ignore zones wider than this % of price")

    # ── Scanner / Universe ────────────────────────────────────────────────
    SCAN_UNIVERSE: str = Field(default="NIFTY50", description="Stock universe: NIFTY50 | NIFTY100 | NIFTY200 | CUSTOM")
    SCAN_TOP_N: int = Field(default=10, description="Max stocks to shortlist from scanner")
    SECTOR_INDICES: list[str] = Field(
        default=[
            "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO",
            "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY", "NIFTY ENERGY",
            "NIFTY INFRA", "NIFTY PSE", "NIFTY MEDIA", "NIFTY FIN SERVICE",
            "NIFTY PRIVATE BANK",
        ],
        description="Sector indices to track for rotation analysis",
    )

    # ── VWAP Mean-Reversion Strategy Defaults ─────────────────────────────
    VMR_BAND_TIMEFRAME: str = Field(default="5m", description="Timeframe for VWAP band computation")
    VMR_SIGNAL_TIMEFRAME: str = Field(default="5m", description="Timeframe for pin bar detection")
    VMR_TRADE_TIMEFRAME: str = Field(default="5m", description="Timeframe for trade management")
    VMR_BAND_SD_THRESHOLD: float = Field(default=2.0, description="Min SD from VWAP for band touch")
    VMR_MIN_WICK_BODY_RATIO: float = Field(default=2.0, description="Min wick-to-body ratio for pin bar")
    VMR_MIN_VOLUME_RATIO: float = Field(default=1.5, description="Min volume vs 10-candle avg")
    VMR_VOLUME_LOOKBACK: int = Field(default=10, description="Candles to average for volume filter")
    VMR_SL_LOOKBACK_CANDLES: int = Field(default=2, description="Prior candles for structure-based SL")
    VMR_TRAIL_ACTIVATE_PCT: float = Field(default=0.5, description="Activate trailing stop after this % favorable move")
    VMR_TRAIL_PCT: float = Field(default=0.3, description="Trail SL as % of move from entry")
    VMR_MAX_GAP_PCT: float = Field(default=2.0, description="Skip stocks gapped > this % from prev close")
    VMR_ENTRY_CUTOFF_IST: str = Field(default="14:00", description="No new VMR entries after this time")
    VMR_FLATTEN_TIME_IST: str = Field(default="15:22", description="Force-flatten VMR positions by this time")
    VMR_SKIP_FIRST_MINUTES: int = Field(default=30, description="Skip first N minutes after open")
    VMR_TARGET_MODE: str = Field(default="VWAP", description="Target: VWAP | HALF_SD | FIXED_RR")
    VMR_FIXED_TARGET_RR: float = Field(default=2.0, description="R:R for FIXED_RR target mode")
    VMR_MIN_SIGNAL_SCORE: float = Field(default=0.45, description="Min composite signal quality score")
    VMR_MAX_CONCURRENT: int = Field(default=3, description="Max concurrent VMR positions")
    VMR_SECTOR_FILTER: bool = Field(default=True, description="Enable sector bias filter for VMR")
    VMR_INDEX_FILTER: bool = Field(default=True, description="Skip if NIFTY50 itself is > 1.5SD from its VWAP")
    VMR_FIXED_LOSS_PER_TRADE: float = Field(default=1000.0, description="Fixed max loss per trade in INR for position sizing")
    VMR_SL_TICKS_BEYOND_PIN: int = Field(default=3, description="Number of ticks beyond pin bar extreme for SL")
    VMR_VOLUME_LOOKBACK: int = Field(default=20, description="SMA period for volume filter")
    VMR_MIN_ATR_MULTIPLIER: float = Field(default=1.0, description="Min candle range as multiple of ATR-14")
    VMR_PRIOR_TREND_LOOKBACK: int = Field(default=5, description="Candles to check for prior trend")
    VMR_MAX_NOSE_PCT: float = Field(default=0.10, description="Max nose wick as fraction of candle range")
    VMR_REQUIRE_CONFIRMATION: bool = Field(default=True, description="Wait for next candle to confirm")

    # ── Execution / Slippage ──────────────────────────────────────────────
    SLIPPAGE_PCT: float = Field(default=0.05, description="Assumed slippage % for backtesting")
    PASSIVE_TIMEOUT_MS: int = Field(default=3_000, description="Wait ms for passive fill before going aggressive")
    ORDER_RETRY_MAX: int = Field(default=3, description="Max order placement retries")
    ORDER_RETRY_DELAY_SEC: float = Field(default=2.0, description="Seconds between retries")

    # ── Brokerage / Cost Model (Fyers) ────────────────────────────────────
    BROKERAGE_FLAT_PER_ORDER: float = Field(default=20.0, description="Fyers flat brokerage per executed order INR")
    BROKERAGE_INTRADAY_PCT: float = Field(default=0.03, description="Fyers intraday brokerage % (whichever lower)")
    STT_SELL_INTRADAY_PCT: float = Field(default=0.025, description="STT on sell side for intraday equity %")
    EXCHANGE_TXN_PCT: float = Field(default=0.00345, description="NSE transaction charge %")
    SEBI_TURNOVER_PCT: float = Field(default=0.0001, description="SEBI turnover fee %")
    GST_PCT: float = Field(default=18.0, description="GST on brokerage + exchange + SEBI charges %")
    STAMP_DUTY_BUY_PCT: float = Field(default=0.003, description="Stamp duty on buy side %")
    IPFT_PCT: float = Field(default=0.0001, description="IPFT charge %")

    # ── Backtest ──────────────────────────────────────────────────────────
    BACKTEST_START_DATE: str = Field(default="2024-01-01", description="Default backtest start YYYY-MM-DD")
    BACKTEST_END_DATE: str = Field(default="2025-01-01", description="Default backtest end YYYY-MM-DD")
    BACKTEST_INITIAL_CAPITAL: float = Field(default=750_000.0, description="Backtest starting capital INR")

    # ── Claude LLM Integration (Phase 4) ──────────────────────────────────
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key for Claude LLM analyst")
    LLM_MODEL: str = Field(default="claude-sonnet-4-20250514", description="Claude model string for pre-market analyst")
    LLM_MAX_TOKENS: int = Field(default=2_048, description="Max response tokens for LLM calls")
    LLM_TEMPERATURE: float = Field(default=0.3, description="LLM temperature — low for deterministic analysis")
    LLM_ENABLED: bool = Field(default=False, description="Master switch for LLM analyst features")

    # ── Dashboard ─────────────────────────────────────────────────────────
    DASHBOARD_PORT: int = Field(default=8501, description="Streamlit dashboard port")
    DASHBOARD_REFRESH_SEC: int = Field(default=5, description="Auto-refresh interval for live dashboard")

    # ── Engine / API ──────────────────────────────────────────────────────
    ENGINE_API_PORT: int = Field(default=8100, description="FastAPI control server port")
    ENGINE_API_TOKEN: str = Field(default="changeme", description="Bearer token for engine control API")
    ENGINE_TICK_INTERVAL_SEC: int = Field(default=60, description="Main loop tick interval seconds")

    # ── Database ──────────────────────────────────────────────────────────
    DB_NAME: str = Field(default="trade.db", description="SQLite database filename")
    DEFAULT_USER_ID: str = Field(default="default_user", description="Default user ID for single-user mode")

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG | INFO | WARNING | ERROR")
    LOG_DIR: str = Field(default="storage/logs", description="Directory for JSONL log files")

    # ── Telegram Alerts (Optional) ────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram bot token for alerts")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Telegram chat ID for alerts")

    # ── Derived Properties ────────────────────────────────────────────────
    @property
    def PROJECT_ROOT(self) -> Path:
        """Absolute path to project root directory."""
        return _PROJECT_ROOT

    @property
    def DB_PATH(self) -> Path:
        """Absolute path to SQLite database file."""
        return _PROJECT_ROOT / "storage" / self.DB_NAME

    @property
    def SCHEMA_PATH(self) -> Path:
        """Absolute path to SQL schema file."""
        return _PROJECT_ROOT / "storage" / "schema.sql"

    @property
    def LOG_PATH(self) -> Path:
        """Absolute path to log directory."""
        return _PROJECT_ROOT / self.LOG_DIR

    @property
    def RISK_PER_TRADE_INR(self) -> float:
        """Max risk per trade in INR."""
        return self.TOTAL_CAPITAL * (self.RISK_PER_TRADE_PCT / 100.0)

    @property
    def DAILY_LOSS_LIMIT_INR(self) -> float:
        """Hard daily loss limit in INR."""
        return self.TOTAL_CAPITAL * (self.DAILY_LOSS_LIMIT_PCT / 100.0)

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Module-level singleton — import this wherever config is needed
# ---------------------------------------------------------------------------
settings = Settings()
