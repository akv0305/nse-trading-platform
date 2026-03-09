"""
VWAP Mean-Reversion Strategy (VMR)
===================================
Fades extreme moves away from VWAP using wick-rejection (pin bar)
confirmation at VWAP standard-deviation bands.

Entry Logic (LONG):
  - Price low touches ≤ -1.5 SD VWAP band
  - Bullish wick rejection (hammer): lower wick ≥ 2× body, body in upper 1/3
  - Volume ≥ avg of last 10 candles
  - Sector not strongly negative
  - NIFTY50 not below its own -1.5 SD VWAP
  - No opening gap > 2%
  - Enter at pin bar close

Entry Logic (SHORT): Mirror at upper band with shooting star.

Exit Priority: (1) Flatten at 15:22, (2) Trailing stop (0.3% activation,
0.3% trail), (3) Hard stop (lowest/highest of 3 candles ± 2 ticks),
(4) Target at VWAP or 0.5 SD band.

Supports configurable timeframes: band TF, signal TF, trade TF.
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
from core.indicators.wick_rejection import (
    RejectionType,
    VWAPBands,
    WickRejection,
    compute_formation_stop,
    compute_vwap_bands,
    detect_wick_rejection,
    resample_candles,
)
from core.indicators.sector_score import (
    compute_sector_scores,
    get_stock_sector_bias,
)
from core.strategies.base import StrategyBase
from core.utils.time_utils import IST, epoch_ms_to_ist, parse_time_str

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
MARKET_OPEN = time(9, 15)
TICK_SIZE = 0.05


def _settings_time(attr: str, default: str) -> time:
    """Read an 'HH:MM' string from settings and return a datetime.time."""
    val = getattr(settings, attr, default)
    if isinstance(val, time):
        return val
    h, m = parse_time_str(str(val))
    return time(h, m)


def _tf_minutes(tf_str: str) -> int:
    """Convert timeframe string like '3m', '5m', '15m' to int minutes."""
    tf = tf_str.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    return 5  # default


# ──────────────────────────────────────────────────────────────────────────
# Per-symbol state
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class _VMRState:
    """Mutable intraday state for one symbol."""
    candles: list = field(default_factory=list)         # all candles today
    band_candles: list = field(default_factory=list)    # resampled band-TF candles
    signal_fired: bool = False
    prev_day_close: float = 0.0
    gap_pct: float = 0.0
    gap_rejected: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────
class VMRStrategy(StrategyBase):
    """
    VWAP Mean-Reversion with wick-rejection confirmation.

    Lifecycle:
      1. pre_market_scan(universe, historical_data) → pick watchlist
      2. on_candle(symbol, candle, history, position) → entry signals
      3. should_exit(symbol, current_price, position, candle) → exit signals
      4. end_of_day() → reset state
    """

    def __init__(self) -> None:
        super().__init__(name="vmr_vwap", version="1.0.0")
        self._states: Dict[str, _VMRState] = {}
        self._sector_scores: Dict[str, float] = {}
        self._nifty_candles: List[Candle] = []

    # ------------------------------------------------------------------
    # Properties (inherited from StrategyBase, listed for clarity)
    # ------------------------------------------------------------------
    # name, version, is_active, watchlist — all inherited

    def __repr__(self) -> str:
        return f"VMRStrategy(name={self._name!r}, version={self._version!r})"

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
        self._nifty_candles.clear()

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

        # ── Select liquid equities ───────────────────────────────────
        equity_symbols = [s for s in universe if s.endswith("-EQ")]

        candidates = []
        for sym in equity_symbols:
            df = historical_data.get(sym)
            if df is None or df.empty or len(df) < 6:
                continue

            closes = df["close"].values
            volumes = df["volume"].values
            current_price = float(closes[-1])
            prev_close = float(closes[-2]) if len(closes) >= 2 else current_price

            # Relative volume
            avg_vol = float(volumes[-6:-1].mean()) if len(volumes) >= 6 else float(volumes[:-1].mean())
            if avg_vol <= 0:
                continue
            rel_vol = float(volumes[-1]) / avg_vol

            # ATR % (need some volatility for mean-reversion)
            import numpy as np
            highs = df["high"].values
            lows = df["low"].values
            tr_vals = []
            for i in range(-min(5, len(closes) - 1), 0):
                h, l_, pc = float(highs[i]), float(lows[i]), float(closes[i - 1])
                tr_vals.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
            atr_pct = (np.mean(tr_vals) / current_price * 100.0) if tr_vals and current_price > 0 else 0

            # Score: prefer liquid, moderate volatility stocks
            if atr_pct < 0.5 or atr_pct > 5.0:
                continue

            vol_score = min(rel_vol / 2.0, 1.0)
            # Ideal ATR for mean-rev: 1-2.5%
            atr_score = 1.0 - abs(atr_pct - 1.5) / 2.0
            atr_score = max(0.0, min(1.0, atr_score))

            composite = 0.5 * vol_score + 0.5 * atr_score
            candidates.append({
                "symbol": sym,
                "score": composite,
                "prev_close": prev_close,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_n = getattr(settings, "SCAN_TOP_N", 10)

        for cand in candidates[:top_n]:
            sym = cand["symbol"]
            self._watchlist.append(sym)
            state = _VMRState()
            state.prev_day_close = cand["prev_close"]
            self._states[sym] = state

        logger.info(
            f"[{self._name}] watchlist ({len(self._watchlist)}): "
            f"{self._watchlist[:5]}..."
        )
        return list(self._watchlist)

    # ------------------------------------------------------------------
    # 2. ON CANDLE (entry logic)
    # ------------------------------------------------------------------
    def on_candle(
        self,
        symbol: str,
        candle: Candle,
        candle_history: List[Candle],
        current_position: dict | None = None,
    ) -> Signal:
        if current_position is not None:
            return self._no_signal(symbol, candle, "position_open")

        if symbol not in self._states:
            self._states[symbol] = _VMRState()
        state = self._states[symbol]

        # Track all candles
        state.candles.append(candle)

        candle_time = self._candle_time(candle)
        if candle_time is None:
            return self._no_signal(symbol, candle, "no_timestamp")

        # ── Skip first N minutes ────────────────────────────────────
        skip_min = getattr(settings, "VMR_SKIP_FIRST_MINUTES", 15)
        skip_end = (
            datetime.combine(datetime.today(), MARKET_OPEN)
            + timedelta(minutes=skip_min)
        ).time()
        if candle_time < skip_end:
            return self._no_signal(symbol, candle, "warmup_period")

        # ── Gap filter (run once) ───────────────────────────────────
        if not state.gap_rejected and state.prev_day_close > 0 and len(state.candles) <= 10:
            first_open = state.candles[0].open
            gap = abs(first_open - state.prev_day_close) / state.prev_day_close * 100.0
            state.gap_pct = gap
            max_gap = getattr(settings, "VMR_MAX_GAP_PCT", 2.0)
            if gap > max_gap:
                state.gap_rejected = True

        if state.gap_rejected:
            return self._no_signal(symbol, candle, "gap_too_wide")

        # ── Entry cutoff ────────────────────────────────────────────
        cutoff = _settings_time("VMR_ENTRY_CUTOFF_IST", "14:45")
        if candle_time >= cutoff:
            return self._no_signal(symbol, candle, "past_cutoff")

        # ── Already fired today ─────────────────────────────────────
        if state.signal_fired:
            return self._no_signal(symbol, candle, "signal_already_fired")

        # ── Need enough candles for VWAP bands ──────────────────────
        if len(state.candles) < 5:
            return self._no_signal(symbol, candle, "insufficient_candles")

        # ── Compute VWAP bands ──────────────────────────────────────
        band_tf_min = _tf_minutes(getattr(settings, "VMR_BAND_TIMEFRAME", "5m"))
        signal_tf_min = _tf_minutes(getattr(settings, "VMR_SIGNAL_TIMEFRAME", "5m"))
        candle_tf_min = _tf_minutes(candle.timeframe) if candle.timeframe else 5

        # Resample for band computation if needed
        if band_tf_min > candle_tf_min:
            band_candles = resample_candles(state.candles, candle_tf_min, band_tf_min)
        else:
            band_candles = state.candles

        bands = compute_vwap_bands(band_candles)
        if bands is None or bands.std_dev <= 0:
            return self._no_signal(symbol, candle, "no_vwap_bands")

        # ── Detect wick rejection ───────────────────────────────────
        min_wick = getattr(settings, "VMR_MIN_WICK_BODY_RATIO", 2.0)
        vol_lookback = getattr(settings, "VMR_VOLUME_LOOKBACK", 10)
        band_sd = getattr(settings, "VMR_BAND_SD_THRESHOLD", 1.5)

        recent = state.candles[:-1]  # exclude current candle
        wr = detect_wick_rejection(
            candle=candle,
            vwap_bands=bands,
            recent_candles=recent,
            volume_lookback=vol_lookback,
            min_wick_ratio=min_wick,
            band_threshold_sd=band_sd,
        )

        if wr.rejection_type == RejectionType.NONE:
            return self._no_signal(symbol, candle, wr.details.get("reason", "no_pattern"))

        # ── Determine direction ─────────────────────────────────────
        is_long = wr.rejection_type == RejectionType.BULLISH_HAMMER

        # ── Sector filter ───────────────────────────────────────────
        if getattr(settings, "VMR_SECTOR_FILTER", True):
            try:
                _, plain_sym, _ = from_fyers_symbol(symbol)
            except (ValueError, IndexError):
                plain_sym = symbol
            sector_bias = get_stock_sector_bias(plain_sym, self._sector_scores)

            # For mean-reversion: don't fade a move that sector supports
            # i.e., don't go LONG if sector is strongly negative (price *should* go down)
            # and don't go SHORT if sector is strongly positive
            if is_long and sector_bias < -0.5:
                return self._no_signal(symbol, candle, "sector_against_long")
            if not is_long and sector_bias > 0.5:
                return self._no_signal(symbol, candle, "sector_against_short")

        # ── Signal quality threshold ────────────────────────────────
        min_score = getattr(settings, "VMR_MIN_SIGNAL_SCORE", 0.45)
        if wr.score < min_score:
            return self._no_signal(symbol, candle, "low_signal_score")

        entry_price = candle.close

        # ── Stop-loss: pin bar wick extreme + buffer ticks ──────────
        sl_ticks = getattr(settings, "VMR_SL_TICKS_BEYOND_PIN", 3)
        tick = TICK_SIZE
        if is_long:
            stop_price = round(candle.low - sl_ticks * tick, 2)
            risk_per_share = entry_price - stop_price
        else:
            stop_price = round(candle.high + sl_ticks * tick, 2)
            risk_per_share = stop_price - entry_price

        if risk_per_share <= 0:
            return self._no_signal(symbol, candle, "zero_risk")

        # ── Target: VWAP ────────────────────────────────────────────
        target_mode = getattr(settings, "VMR_TARGET_MODE", "VWAP")
        if target_mode == "VWAP":
            target_price = bands.vwap
        elif target_mode == "HALF_SD":
            target_price = bands.lower_1sd if is_long else bands.upper_1sd
        else:
            rr = getattr(settings, "VMR_FIXED_TARGET_RR", 2.0)
            target_price = entry_price + risk_per_share * rr if is_long else entry_price - risk_per_share * rr

        # Ensure target is on the correct side
        if is_long and target_price <= entry_price:
            target_price = entry_price + risk_per_share * 1.5
        if not is_long and target_price >= entry_price:
            target_price = entry_price - risk_per_share * 1.5

        # ── R:R filter ──────────────────────────────────────────────
        reward_per_share = abs(target_price - entry_price)
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio < 1.5:
            return self._no_signal(symbol, candle, f"rr_too_low_{rr_ratio:.1f}")

        # ── SL width cap: reject if SL > 1.5% of price ─────────────
        sl_pct = (risk_per_share / entry_price) * 100.0
        if sl_pct > 1.5:
            return self._no_signal(symbol, candle, "sl_too_wide")

        # ── Position sizing: fixed loss per trade ───────────────────
        fixed_loss = getattr(settings, "VMR_FIXED_LOSS_PER_TRADE", 1000.0)
        quantity = int(fixed_loss / risk_per_share)
        if quantity <= 0:
            return self._no_signal(symbol, candle, "quantity_zero")

        state.signal_fired = True
        signal_type = SignalType.BUY if is_long else SignalType.SELL

        indicator_data = {
            "vwap": bands.vwap,
            "vwap_upper_1_5sd": bands.upper_1_5sd,
            "vwap_lower_1_5sd": bands.lower_1_5sd,
            "vwap_std_dev": bands.std_dev,
            "rejection_type": wr.rejection_type.value,
            "wick_body_ratio": wr.wick_body_ratio,
            "body_position": wr.body_position,
            "volume_ratio": wr.volume_ratio,
            "band_touch": wr.band_touch,
            "vwap_distance_sd": wr.vwap_distance_sd,
            "signal_score": wr.score,
            "stoploss_price": stop_price,
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "rr_ratio": round(rr_ratio, 2),
            "sl_pct": round(sl_pct, 2),
            "quantity": quantity,
        }

        logger.info(
            f"[{self._name}] SIGNAL {signal_type.name} {symbol} "
            f"@ {entry_price:.2f}  SL={stop_price:.2f}  "
            f"TGT={target_price:.2f}  R:R={rr_ratio:.1f}  "
            f"qty={quantity}  risk=₹{fixed_loss:.0f}  "
            f"score={wr.score:.2f}  band={wr.band_touch}"
        )

        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=signal_type,
            strength=wr.score,
            price_at_signal=entry_price,
            timestamp=candle.timestamp,
            indicator_data=indicator_data,
        )


    # ------------------------------------------------------------------
    # 3. SHOULD EXIT
    # ------------------------------------------------------------------
    def should_exit(
        self,
        symbol: str,
        current_price: float,
        position: Dict[str, Any],
        candle: Candle | None = None,
    ) -> Signal:
        raw_dir = position.get("direction", "LONG")
        if isinstance(raw_dir, Direction):
            is_long = raw_dir == Direction.LONG
        else:
            is_long = str(raw_dir).upper() == "LONG"

        entry_price = float(position.get("entry_price", 0))
        stop_loss = float(position.get("stoploss_price", 0))
        target = float(position.get("target_price", 0))
        peak_price = float(position.get("peak_price", current_price))

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

        # ── Candle time ─────────────────────────────────────────────
        candle_time = self._candle_time(candle) if candle else None
        sig_ts = candle.timestamp if candle else 0
        exit_type = SignalType.EXIT_LONG if is_long else SignalType.EXIT_SHORT

        # ── Trailing stop computation ───────────────────────────────
        # VMR_TRAIL_ACTIVATE_PCT and VMR_TRAIL_PCT are in % terms
        # e.g., 0.3 means 0.3% of entry price
        trail_activate_pct = getattr(settings, "VMR_TRAIL_ACTIVATE_PCT", 0.3)
        trail_lock_pct = getattr(settings, "VMR_TRAIL_PCT", 0.3)

        trailing_active = False
        effective_sl = stop_loss

        if entry_price > 0:
            # Convert % thresholds to absolute price moves
            activate_move = entry_price * (trail_activate_pct / 100.0)
            
            if is_long:
                favorable_move = peak_price - entry_price
                if favorable_move >= activate_move:
                    trailing_active = True
                    # Trail SL = lock in trail_lock_pct% of entry price
                    # below the peak, but never below original SL
                    trail_distance = entry_price * (trail_lock_pct / 100.0)
                    trail_sl = peak_price - trail_distance
                    effective_sl = max(trail_sl, stop_loss)
            else:
                favorable_move = entry_price - peak_price
                if favorable_move >= activate_move:
                    trailing_active = True
                    trail_distance = entry_price * (trail_lock_pct / 100.0)
                    trail_sl = peak_price + trail_distance
                    effective_sl = min(trail_sl, stop_loss)

        def _exit(reason: str, strength: float = 0.8) -> Signal:
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=strength,
                price_at_signal=price,
                timestamp=sig_ts,
                indicator_data={
                    "exit_reason": reason,
                    "is_trailing": trailing_active,
                    "effective_sl": round(effective_sl, 2),
                    "peak_price": peak_price,
                },
            )

        # ── 1. FLATTEN TIME ─────────────────────────────────────────
        flatten_time = _settings_time("VMR_FLATTEN_TIME_IST", "15:22")
        if candle_time is not None and candle_time >= flatten_time:
            return _exit("FLATTEN", 1.0)

        # ── 2. TRAILING STOP ───────────────────────────────────────
        if trailing_active:
            if is_long and price <= effective_sl:
                return _exit("TRAIL")
            if not is_long and price >= effective_sl:
                return _exit("TRAIL")

        # ── 3. HARD STOP-LOSS ──────────────────────────────────────
        if is_long and price <= stop_loss:
            return _exit("STOPLOSS", 0.9)
        if not is_long and price >= stop_loss:
            return _exit("STOPLOSS", 0.9)

        # ── 4. TARGET ──────────────────────────────────────────────
        if is_long and price >= target:
            return _exit("TARGET")
        if not is_long and price <= target:
            return _exit("TARGET")

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
                "effective_sl": round(effective_sl, 2),
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
        self._nifty_candles.clear()

    # ------------------------------------------------------------------
    # 5. GET PARAMS
    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": self._name,
            "strategy_name": self._name,
            "version": self._version,
            "band_timeframe": getattr(settings, "VMR_BAND_TIMEFRAME", "5m"),
            "signal_timeframe": getattr(settings, "VMR_SIGNAL_TIMEFRAME", "5m"),
            "trade_timeframe": getattr(settings, "VMR_TRADE_TIMEFRAME", "5m"),
            "band_sd_threshold": getattr(settings, "VMR_BAND_SD_THRESHOLD", 1.5),
            "min_wick_body_ratio": getattr(settings, "VMR_MIN_WICK_BODY_RATIO", 2.0),
            "min_volume_ratio": getattr(settings, "VMR_MIN_VOLUME_RATIO", 1.0),
            "trail_activate_pct": getattr(settings, "VMR_TRAIL_ACTIVATE_PCT", 0.3),
            "trail_pct": getattr(settings, "VMR_TRAIL_PCT", 0.3),
            "entry_cutoff_ist": getattr(settings, "VMR_ENTRY_CUTOFF_IST", "14:45"),
            "flatten_time_ist": getattr(settings, "VMR_FLATTEN_TIME_IST", "15:22"),
            "target_mode": getattr(settings, "VMR_TARGET_MODE", "VWAP"),
            "sector_filter": getattr(settings, "VMR_SECTOR_FILTER", True),
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _no_signal(self, symbol: str, candle: Candle, reason: str = "") -> Signal:
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=SignalType.NO_ACTION,
            strength=0.0,
            price_at_signal=candle.close,
            timestamp=candle.timestamp,
            skip_reason=reason,
        )

    @staticmethod
    def _candle_time(candle: Candle | None) -> time | None:
        if candle is None:
            return None
        try:
            if hasattr(candle, "timestamp") and candle.timestamp:
                dt = epoch_ms_to_ist(candle.timestamp)
                return dt.time()
        except Exception:
            pass
        return None
