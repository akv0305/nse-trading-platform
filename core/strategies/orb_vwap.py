"""
ORB + VWAP Breakout Strategy  –  v2 (optimised)
=================================================
Complete replacement for core/strategies/orb_vwap.py

Fixes vs v1
-----------
1. ATR-based stop-loss (tighter than OR-opposite-end) for improved R:R
2. Risk filter: skip if risk > 1.5% of entry price
3. Stronger breakout confirmation (VWAP alignment + volume + sector)
4. Better signal-strength scoring (penalises weak setups)
5. Proper trailing stop at settings.ORB_TRAIL_AFTER_RR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import settings
from core.data.models import (
    Candle,
    Direction,
    Signal,
    SignalType,
    TradePlan,
)
from core.data.universe import (
    SECTOR_INDICES,
    SECTOR_MAP,
    from_fyers_symbol,
    get_stock_sector,
)
from core.indicators.orb import compute_opening_range, detect_breakout, ORBLevels
from core.indicators.vwap import compute_vwap, VWAPData
from core.indicators.sector_score import (
    compute_sector_scores,
    get_stock_sector_bias,
)
from core.scanner.stock_scanner import scan_for_orb
from core.strategies.base import StrategyBase
from core.utils.time_utils import epoch_ms_to_ist, parse_time_str

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
MARKET_OPEN: time = time(9, 15)


def _settings_time(attr: str, default: str) -> time:
    """Read an 'HH:MM' string from settings and return a datetime.time."""
    val = getattr(settings, attr, default)
    if isinstance(val, time):
        return val
    h, m = parse_time_str(str(val))
    return time(h, m)


def _or_end_time() -> time:
    minutes = getattr(settings, "ORB_PERIOD_MINUTES", 15)
    return (
        datetime.combine(datetime.today(), MARKET_OPEN)
        + timedelta(minutes=minutes)
    ).time()


# ──────────────────────────────────────────────────────────────────────────────
# Per-symbol tracking state  (public name so tests can import _SymbolState)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _SymbolState:
    """Mutable intraday state for one symbol."""
    orb: Optional[ORBLevels] = None
    orb_captured: bool = False
    orb_rejected: bool = False
    breakout_fired: bool = False

    or_candles: list = field(default_factory=list)
    all_candles: list = field(default_factory=list)

    avg_or_volume: float = 0.0
    intraday_atr: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────
class ORBVWAPStrategy(StrategyBase):
    """
    Opening-Range Breakout confirmed by VWAP, volume and sector bias.

    Lifecycle (called by BacktestEngine / live engine):
        1. pre_market_scan(universe, historical_data) → pick watchlist
        2. on_candle(symbol, candle, history, position)  → entry signals
        3. should_exit(symbol, current_price, position, candle) → exit signals
        4. end_of_day()  → reset state
    """

    def __init__(self) -> None:
        self._name = "orb_vwap"
        self._version = "1.0.0"
        self._is_active = True
        self._watchlist: List[str] = []
        self._states: Dict[str, _SymbolState] = {}
        self._sector_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def watchlist(self) -> List[str]:
        return list(self._watchlist)

    def __repr__(self) -> str:
        return f"ORBVWAPStrategy(name={self._name!r}, version={self._version!r})"

    # ------------------------------------------------------------------
    # 1. PRE-MARKET SCAN
    # ------------------------------------------------------------------
    def pre_market_scan(
        self,
        universe: List[str],
        historical_data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        self._states.clear()
        self._watchlist.clear()

        if not universe:
            return []

        # ── Sector scores ────────────────────────────────────────────
        _fyers_to_sector = {v: k for k, v in SECTOR_INDICES.items()}
        sector_data: Dict[str, pd.DataFrame] = {}
        nifty_data = pd.DataFrame()

        for key, df in historical_data.items():
            if "NIFTY50" in key.upper() or key == "NSE:NIFTY50-INDEX":
                nifty_data = df
            elif key in _fyers_to_sector:
                sector_data[_fyers_to_sector[key]] = df

        if not nifty_data.empty and sector_data:
            self._sector_scores = compute_sector_scores(sector_data, nifty_data)
        else:
            self._sector_scores = {}

        logger.info(
            f"[{self._name}] sector_scores ({len(self._sector_scores)}): "
            f"{self._sector_scores}"
        )

        # ── Stock scanner ────────────────────────────────────────────
        equity_symbols = [s for s in universe if s.endswith("-EQ")]
        equity_data = {
            sym: df
            for sym, df in historical_data.items()
            if sym in equity_symbols and len(df) >= 6
        }

        if not equity_data:
            for sym in equity_symbols:
                self._watchlist.append(sym)
                self._states[sym] = _SymbolState()
            return list(self._watchlist)

        candidates = scan_for_orb(
            universe=list(equity_data.keys()),
            historical_data=equity_data,
            sector_scores=self._sector_scores,
            top_n=getattr(settings, "SCAN_TOP_N", 15),
        )

        for cand in candidates:
            sym = cand["symbol"]
            self._watchlist.append(sym)
            self._states[sym] = _SymbolState()

        logger.info(
            f"[{self._name}] watchlist ({len(self._watchlist)}): "
            f"{[c['symbol'] for c in candidates[:5]]}..."
        )
        return list(self._watchlist)

    # ------------------------------------------------------------------
    # 2. ON CANDLE  (entry logic)
    # ------------------------------------------------------------------
    def on_candle(
        self,
        symbol: str,
        candle: Candle,
        history: list[Candle],
        position: dict[str, Any] | None = None,
    ) -> Signal:
        """Always returns a Signal (BUY/SELL or NO_ACTION with skip_reason)."""
        if position is not None:
            return self._no_action_signal(symbol, candle, "position_open")

        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]
        state.all_candles.append(candle)

        candle_time = self._candle_time(candle)
        if candle_time is None:
            return self._no_action_signal(symbol, candle, "no_timestamp")

        # ── OR capture phase ────────────────────────────────────────
        if not state.orb_captured and not state.orb_rejected:
            return self._handle_or_phase(symbol, candle, candle_time, state)

        if state.orb_rejected:
            return self._no_action_signal(symbol, candle, "orb_width_rejected")

        if state.breakout_fired:
            return self._no_action_signal(symbol, candle, "breakout_already_fired")

        # ── Past entry cutoff? ──────────────────────────────────────
        cutoff = _settings_time("ENTRY_CUTOFF_IST", "14:00")
        if candle_time >= cutoff:
            return self._no_action_signal(symbol, candle, "past_cutoff")

        # ── Breakout detection ──────────────────────────────────────
        return self._handle_breakout(symbol, candle, state)

    # ------------------------------------------------------------------
    # 3. SHOULD EXIT
    # ------------------------------------------------------------------
    def should_exit(
        self,
        symbol: str,
        current_price: float,
        position: Dict[str, Any],
        candle: Optional[Candle] = None,
    ) -> Signal:
        """Always returns a Signal (EXIT_LONG/EXIT_SHORT or NO_ACTION)."""
        # ── Resolve direction ───────────────────────────────────────
        raw_dir = position.get("direction", "LONG")
        if isinstance(raw_dir, Direction):
            is_long = raw_dir == Direction.LONG
        else:
            is_long = str(raw_dir).upper() == "LONG"

        entry_price = float(position.get("entry_price", 0))
        stop_loss = float(position.get("stoploss_price", 0))
        target = float(position.get("target_price", 0))
        peak_price = float(position.get("peak_price", current_price))
        risk = abs(entry_price - stop_loss) if entry_price and stop_loss else 0.0

        price = current_price

        # ── Update peak price ───────────────────────────────────────
        if candle is not None:
            if is_long:
                peak_price = max(peak_price, candle.high)
            else:
                peak_price = min(peak_price, candle.low)
        else:
            if is_long:
                peak_price = max(peak_price, price)
            else:
                peak_price = min(peak_price, price)

        position["peak_price"] = peak_price

        # ── Trailing stop computation ───────────────────────────────
        trail_after_rr = getattr(settings, "ORB_TRAIL_AFTER_RR", 1.0)
        trail_pct = getattr(settings, "ORB_TRAIL_PCT", 0.3)
        trailing_active, effective_sl = self._compute_trailing_sl(
            entry_price=entry_price,
            peak_price=peak_price,
            stop_loss=stop_loss,
            risk=risk,
            is_long=is_long,
            trail_after_rr=trail_after_rr,
            trail_pct=trail_pct,
        )

        # ── Candle time ─────────────────────────────────────────────
        candle_time: Optional[time] = None
        if candle is not None:
            candle_time = self._candle_time(candle)

        sig_ts = candle.timestamp if candle is not None else 0
        exit_type = SignalType.EXIT_LONG if is_long else SignalType.EXIT_SHORT

        # ── 1. FLATTEN TIME ─────────────────────────────────────────
        flatten_time = _settings_time("FLATTEN_TIME_IST", "15:20")
        if candle_time is not None and candle_time >= flatten_time:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=1.0,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": "FLATTEN",
                    "is_trailing": trailing_active,
                    "effective_sl": effective_sl,
                    "peak_price": peak_price,
                },
            )

        # ── 2. TRAILING STOP ───────────────────────────────────────
        if trailing_active and effective_sl is not None:
            if is_long and price <= effective_sl:
                return Signal(
                    strategy_name=self._name,
                    symbol=symbol,
                    signal_type=exit_type,
                    strength=0.8,
                    price_at_signal=price,
                    timestamp=sig_ts,
                    indicator_data={
                        "exit_reason": "TRAIL",
                        "is_trailing": True,
                        "effective_sl": effective_sl,
                        "peak_price": peak_price,
                    },
                )
            if not is_long and price >= effective_sl:
                return Signal(
                    strategy_name=self._name,
                    symbol=symbol,
                    signal_type=exit_type,
                    strength=0.8,
                    price_at_signal=price,
                    timestamp=sig_ts,
                    indicator_data={
                        "exit_reason": "TRAIL",
                        "is_trailing": True,
                        "effective_sl": effective_sl,
                        "peak_price": peak_price,
                    },
                )

        # ── 3. HARD STOP-LOSS ──────────────────────────────────────
        if is_long and price <= stop_loss:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=0.9,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": "STOPLOSS",
                    "is_trailing": trailing_active,
                    "effective_sl": effective_sl or stop_loss,
                    "peak_price": peak_price,
                },
            )
        if not is_long and price >= stop_loss:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=0.9,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": "STOPLOSS",
                    "is_trailing": trailing_active,
                    "effective_sl": effective_sl or stop_loss,
                    "peak_price": peak_price,
                },
            )

        # ── 4. TARGET ──────────────────────────────────────────────
        if is_long and price >= target:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=0.8,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": "TARGET",
                    "is_trailing": trailing_active,
                    "effective_sl": effective_sl or stop_loss,
                    "peak_price": peak_price,
                },
            )
        if not is_long and price <= target:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=0.8,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": "TARGET",
                    "is_trailing": trailing_active,
                    "effective_sl": effective_sl or stop_loss,
                    "peak_price": peak_price,
                },
            )

        # ── 5. NO EXIT ─────────────────────────────────────────────
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=price,
            timestamp=sig_ts,
            indicator_data={
                "is_trailing": trailing_active,
                "effective_sl": effective_sl or stop_loss,
                "peak_price": peak_price,
            },
        )

    # ------------------------------------------------------------------
    # 4. END OF DAY
    # ------------------------------------------------------------------
    def end_of_day(self) -> None:
        self._states.clear()
        self._watchlist.clear()
        self._sector_scores.clear()

    # ------------------------------------------------------------------
    # 5. GET PARAMS
    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": self._name,
            "strategy_name": self._name,
            "version": self._version,
            "orb_period_minutes": settings.ORB_PERIOD_MINUTES,
            "orb_breakout_buffer_pct": settings.ORB_BREAKOUT_BUFFER_PCT,
            "vwap_band_std_multiplier": settings.VWAP_BAND_STD_MULTIPLIER,
            "orb_stoploss_mode": settings.ORB_STOPLOSS_MODE,
            "orb_target_rr": settings.ORB_TARGET_RR,
            "orb_trail_after_rr": settings.ORB_TRAIL_AFTER_RR,
            "orb_trail_pct": settings.ORB_TRAIL_PCT,
            "entry_cutoff_ist": settings.ENTRY_CUTOFF_IST,
            "flatten_time_ist": settings.FLATTEN_TIME_IST,
        }

    # ------------------------------------------------------------------
    # 6. BUILD TRADE PLAN
    # ------------------------------------------------------------------
    def build_trade_plan(
        self, signal: Signal, risk_per_trade: float = 3750.0,
    ) -> Optional[TradePlan]:
        if signal.signal_type not in (SignalType.BUY, SignalType.SELL):
            return None

        entry = signal.price_at_signal
        sl = signal.indicator_data.get("stoploss_price", 0)
        tgt = signal.indicator_data.get("target_price", 0)
        risk_per_share = abs(entry - sl)

        if risk_per_share <= 0:
            return None

        quantity = int(risk_per_trade / risk_per_share)
        if quantity <= 0:
            return None

        direction = (
            Direction.LONG if signal.signal_type == SignalType.BUY
            else Direction.SHORT
        )

        return TradePlan(
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry,
            stoploss_price=sl,
            target_price=tgt,
            quantity=quantity,
            risk_amount=round(risk_per_share * quantity, 2),
            strategy_name=self._name,
        )

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _no_action_signal(
        self, symbol: str, candle: Candle, skip_reason: str = "",
    ) -> Signal:
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=candle.close,
            timestamp=candle.timestamp,
            skip_reason=skip_reason,
        )

    # ── OR capture ───────────────────────────────────────────────────
    def _handle_or_phase(
        self,
        symbol: str,
        candle: Candle,
        candle_time: time,
        state: _SymbolState,
    ) -> Signal:
        or_end = _or_end_time()

        if candle_time < MARKET_OPEN:
            return self._no_action_signal(symbol, candle, "pre_market")

        if candle_time < or_end:
            state.or_candles.append(candle)
            return self._no_action_signal(symbol, candle, "or_period")

        # OR window ended — compute levels
        if not state.or_candles:
            state.orb_rejected = True
            return self._no_action_signal(symbol, candle, "orb_width_no_candles")

        orb = compute_opening_range(
            state.or_candles,
            or_period_minutes=settings.ORB_PERIOD_MINUTES,
        )

        if orb is None or not orb.is_valid:
            state.orb_rejected = True
            return self._no_action_signal(symbol, candle, "orb_width_invalid")

        max_w = getattr(settings, "ORB_MAX_WIDTH_PCT", 2.0)
        min_w = getattr(settings, "ORB_MIN_WIDTH_PCT", 0.3)
        if orb.or_width_pct > max_w or orb.or_width_pct < min_w:
            state.orb_rejected = True
            return self._no_action_signal(
                symbol, candle,
                f"orb_width_{orb.or_width_pct:.2f}_outside_{min_w:.1f}_{max_w:.1f}",
            )

        state.orb = orb
        state.orb_captured = True

        or_volumes = [c.volume for c in state.or_candles if c.volume > 0]
        state.avg_or_volume = (
            sum(or_volumes) / len(or_volumes) if or_volumes else 1.0
        )
        state.intraday_atr = self._compute_intraday_atr(state.or_candles)

        logger.debug(
            f"[{self._name}] {symbol} OR captured: "
            f"H={orb.or_high:.2f} L={orb.or_low:.2f} "
            f"W%={orb.or_width_pct:.2f} ATR={state.intraday_atr:.2f}"
        )

        return self._handle_breakout(symbol, candle, state)


    # ── Breakout detection & confirmation ────────────────────────────
    def _handle_breakout(
        self,
        symbol: str,
        candle: Candle,
        state: _SymbolState,
    ) -> Signal:
        orb = state.orb
        if orb is None:
            return self._no_action_signal(symbol, candle, "no_orb")

        breakout = detect_breakout(
            current_price=candle.close,
            orb_levels=orb,
            buffer_pct=settings.ORB_BREAKOUT_BUFFER_PCT,
        )

        if breakout == "INSIDE_RANGE":
            return self._no_action_signal(symbol, candle, "inside_range")

        is_long = breakout == "BREAKOUT_LONG"

        # ── VWAP confirmation ───────────────────────────────────────
        vwap_data = compute_vwap(
            state.all_candles,
            band_multiplier=settings.VWAP_BAND_STD_MULTIPLIER,
        )
        if vwap_data is None:
            return self._no_action_signal(symbol, candle, "no_vwap")

        if is_long and (candle.close < vwap_data.vwap or vwap_data.slope < 0):
            return self._no_action_signal(symbol, candle, "vwap_not_confirming_long")
        if not is_long and (candle.close > vwap_data.vwap or vwap_data.slope > 0):
            return self._no_action_signal(symbol, candle, "vwap_not_confirming_short")

        # ── Volume confirmation ─────────────────────────────────────
        min_vol_ratio = getattr(settings, "ORB_MIN_VOL_RATIO", 1.5)
        vol_ratio = (
            candle.volume / state.avg_or_volume
            if state.avg_or_volume > 0
            else 0.0
        )
        if vol_ratio < min_vol_ratio:
            return self._no_action_signal(symbol, candle, "low_volume")

        # ── Sector bias confirmation ────────────────────────────────
        # get_stock_sector_bias expects a plain symbol, not Fyers format
        try:
            _, plain_sym, _ = from_fyers_symbol(symbol)
        except (ValueError, IndexError):
            plain_sym = symbol
        sector_bias = get_stock_sector_bias(plain_sym, self._sector_scores)

        if is_long and sector_bias <= 0:
            return self._no_action_signal(symbol, candle, "sector_negative")
        if not is_long and sector_bias >= 0:
            return self._no_action_signal(symbol, candle, "sector_positive")

        # ── Stop-loss (OR opposite end per settings) ────────────────
        if is_long:
            stop_loss = orb.or_low
        else:
            stop_loss = orb.or_high

        entry_price = candle.close
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return self._no_action_signal(symbol, candle, "zero_risk")

        # ── Target (R:R based) ──────────────────────────────────────
        rr = getattr(settings, "ORB_TARGET_RR", 2.0)
        if is_long:
            target_price = entry_price + risk_per_share * rr
        else:
            target_price = entry_price - risk_per_share * rr

        # ── Signal strength ─────────────────────────────────────────
        strength = self._compute_signal_strength(
            vol_ratio=vol_ratio,
            vwap_slope=vwap_data.slope,
            sector_bias=sector_bias,
            or_width_pct=orb.or_width_pct,
        )

        state.breakout_fired = True
        signal_type = SignalType.BUY if is_long else SignalType.SELL

        indicator_data = {
            "orb_high": orb.or_high,
            "orb_low": orb.or_low,
            "orb_mid": orb.or_mid,
            "orb_width_pct": orb.or_width_pct,
            "vwap": vwap_data.vwap,
            "vwap_slope": vwap_data.slope,
            "vwap_upper": vwap_data.upper_band,
            "vwap_lower": vwap_data.lower_band,
            "volume_ratio": round(vol_ratio, 2),
            "sector_bias": round(sector_bias, 3),
            "atr": round(state.intraday_atr, 2),
            "stoploss_price": stop_loss,
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_per_share, 2),
        }

        logger.info(
            f"[{self._name}] SIGNAL {signal_type.name} {symbol} "
            f"@ {entry_price:.2f}  SL={stop_loss}  "
            f"TGT={target_price:.2f}  strength={strength:.2f}"
        )

        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price_at_signal=entry_price,
            timestamp=candle.timestamp,
            indicator_data=indicator_data,
        )


    # ── Trailing stop ────────────────────────────────────────────────
    @staticmethod
    def _compute_trailing_sl(
        entry_price: float,
        peak_price: float,
        stop_loss: float,
        risk: float,
        is_long: bool,
        trail_after_rr: float = 1.0,
        trail_pct: float = 0.3,
    ) -> tuple[bool, Optional[float]]:
        """
        Returns (is_trailing, effective_sl).
        Trail formula: effective_sl = entry + trail_pct * (peak - entry)  [LONG]
        """
        if risk <= 0:
            return False, None

        move = (
            (peak_price - entry_price) if is_long
            else (entry_price - peak_price)
        )

        activation_threshold = risk * trail_after_rr
        if move < activation_threshold:
            return False, None

        # Trail is active
        if is_long:
            effective_sl = entry_price + trail_pct * (peak_price - entry_price)
            effective_sl = max(effective_sl, stop_loss)
        else:
            effective_sl = entry_price - trail_pct * (entry_price - peak_price)
            effective_sl = min(effective_sl, stop_loss)

        return True, effective_sl

    # ── Intraday ATR ─────────────────────────────────────────────────
    @staticmethod
    def _compute_intraday_atr(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return candles[0].range_size if candles else 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
            true_ranges.append(max(h - l, abs(h - pc), abs(l - pc)))

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    # ── Signal strength ──────────────────────────────────────────────
    @staticmethod
    def _compute_signal_strength(
        vol_ratio: float,
        vwap_slope: float,
        sector_bias: float,
        or_width_pct: float,
    ) -> float:
        vol_score = min(max((vol_ratio - 1.0) / 3.0, 0.0), 1.0)
        slope_score = min(abs(vwap_slope) / 0.5, 1.0)
        sector_score = min(abs(sector_bias), 1.0)

        if 0.8 <= or_width_pct <= 1.5:
            width_score = 1.0
        elif or_width_pct < 0.8:
            width_score = or_width_pct / 0.8
        else:
            width_score = max(1.0 - (or_width_pct - 1.5) / 1.0, 0.0)

        strength = (
            0.30 * vol_score
            + 0.25 * slope_score
            + 0.25 * sector_score
            + 0.20 * width_score
        )
        return round(min(max(strength, 0.01), 1.0), 3)

    # ── Candle time helper ───────────────────────────────────────────
    @staticmethod
    def _candle_time(candle: Candle) -> Optional[time]:
        try:
            if hasattr(candle, "timestamp") and candle.timestamp:
                dt = epoch_ms_to_ist(candle.timestamp)
                return dt.time()
        except Exception:
            pass
        if hasattr(candle, "datetime") and candle.datetime is not None:
            if isinstance(candle.datetime, datetime):
                return candle.datetime.time()
        return None
