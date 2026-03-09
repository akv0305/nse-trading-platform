"""
VWAP Mean-Reversion Strategy (VMR) — v2.0
==========================================
Fades extreme moves away from VWAP using wick-rejection (pin bar)
confirmation at VWAP standard-deviation bands.

Entry Logic (LONG example):
  1. Price low touches ≤ -1.5 SD VWAP band
  2. Bullish wick rejection (hammer) detected:
     - Lower wick ≥ 2× body, body in upper 1/3, nose ≤ 10% of range
     - Volume ≥ 20-candle SMA
     - Candle range ≥ 1.0× ATR-14
     - Prior 5 candles show downward drift
  3. Sector not strongly negative
  4. Pin bar sets a PENDING ORDER at pin_bar.high + 1 tick
  5. Next candle: if high crosses trigger price → ENTRY confirmed
     Entry price = trigger price (pin_bar.high + 1 tick)
     SL = pin_bar.low − 3 ticks
     Target = VWAP (mean reversion)
  6. Position size = Fixed_Loss_Per_Trade / risk_per_share

Entry Logic (SHORT): Mirror at upper band with shooting star.
  Trigger = pin_bar.low − 1 tick. Confirm when next candle low ≤ trigger.

Exit Priority: (1) Flatten at 15:22, (2) Trailing stop,
  (3) Hard stop (pin bar wick ± ticks), (4) Target at VWAP.

Supports configurable timeframes: band TF, signal TF, trade TF.
"""

from __future__ import annotations

from core.indicators.daily_levels import (
    DailyLevels,
    compute_daily_levels_from_intraday,
    is_near_support,
    is_near_resistance,
)

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
    compute_atr,
    compute_formation_stop,
    compute_vwap_bands,
    detect_wick_rejection,
    has_prior_trend,
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
    candles: list = field(default_factory=list)
    band_candles: list = field(default_factory=list)
    signal_fired: bool = False
    prev_day_close: float = 0.0
    gap_pct: float = 0.0
    gap_rejected: bool = False
    #daily_levels: Optional[Any] = field(default=None)  # DailyLevels

    # ── Pending order (confirmation candle logic) ──────────────────
    # When a pin bar is detected, we don't enter immediately.
    # We set a pending trigger price. On the NEXT candle, if price
    # crosses the trigger, the entry is confirmed.
    pending_trigger: Optional[Dict[str, Any]] = field(default=None)


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

            avg_vol = float(volumes[-6:-1].mean()) if len(volumes) >= 6 else float(volumes[:-1].mean())
            if avg_vol <= 0:
                continue
            rel_vol = float(volumes[-1]) / avg_vol

            import numpy as np
            highs = df["high"].values
            lows = df["low"].values
            tr_vals = []
            for i in range(-min(5, len(closes) - 1), 0):
                h, l_, pc = float(highs[i]), float(lows[i]), float(closes[i - 1])
                tr_vals.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
            atr_pct = (np.mean(tr_vals) / current_price * 100.0) if tr_vals and current_price > 0 else 0

            if atr_pct < 0.5 or atr_pct > 5.0:
                continue

            vol_score = min(rel_vol / 2.0, 1.0)
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
            # Also cancel any pending trigger past cutoff
            state.pending_trigger = None
            return self._no_signal(symbol, candle, "past_cutoff")

        # ── Already fired today ─────────────────────────────────────
        if state.signal_fired:
            return self._no_signal(symbol, candle, "signal_already_fired")

        # ══════════════════════════════════════════════════════════════
        # STEP A: Check if there's a PENDING trigger from previous
        #         pin bar that needs confirmation on THIS candle
        # ══════════════════════════════════════════════════════════════
        if state.pending_trigger is not None:
            return self._check_pending_confirmation(symbol, candle, state)

        # ══════════════════════════════════════════════════════════════
        # STEP B: No pending trigger — look for a NEW pin bar
        # ══════════════════════════════════════════════════════════════

        # ── Need enough candles for VWAP bands ──────────────────────
        if len(state.candles) < 5:
            return self._no_signal(symbol, candle, "insufficient_candles")

        # ── Compute VWAP bands ──────────────────────────────────────
        band_tf_min = _tf_minutes(getattr(settings, "VMR_BAND_TIMEFRAME", "5m"))
        candle_tf_min = _tf_minutes(candle.timeframe) if candle.timeframe else 5

        if band_tf_min > candle_tf_min:
            band_candles = resample_candles(state.candles, candle_tf_min, band_tf_min)
        else:
            band_candles = state.candles

        bands = compute_vwap_bands(band_candles)
        if bands is None or bands.std_dev <= 0:
            return self._no_signal(symbol, candle, "no_vwap_bands")

        # ── Detect wick rejection ───────────────────────────────────
        min_wick = getattr(settings, "VMR_MIN_WICK_BODY_RATIO", 2.0)
        vol_lookback = getattr(settings, "VMR_VOLUME_LOOKBACK", 20)
        band_sd = getattr(settings, "VMR_BAND_SD_THRESHOLD", 1.5)
        min_atr_mult = getattr(settings, "VMR_MIN_ATR_MULTIPLIER", 1.0)
        max_nose = getattr(settings, "VMR_MAX_NOSE_PCT", 0.10)
        trend_lookback = getattr(settings, "VMR_PRIOR_TREND_LOOKBACK", 5)

        recent = state.candles[:-1]  # exclude current candle
        wr = detect_wick_rejection(
            candle=candle,
            vwap_bands=bands,
            recent_candles=recent,
            volume_lookback=vol_lookback,
            min_wick_ratio=min_wick,
            band_threshold_sd=band_sd,
            min_atr_multiplier=min_atr_mult,
            max_nose_pct=max_nose,
            prior_trend_lookback=trend_lookback,
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

            if is_long and sector_bias < -0.5:
                return self._no_signal(symbol, candle, "sector_against_long")
            if not is_long and sector_bias > 0.5:
                return self._no_signal(symbol, candle, "sector_against_short")

        # ── Daily S/R level filter ──────────────────────────────────
        if getattr(settings, "VMR_DAILY_LEVEL_FILTER", True):
            candle_time_dt = epoch_ms_to_ist(candle.timestamp)
            today_str = candle_time_dt.strftime("%Y-%m-%d")
            dl = compute_daily_levels_from_intraday(
                candle_history, today_str,
            )
            if dl is not None:
                buffer = getattr(settings, "VMR_DAILY_LEVEL_BUFFER_PCT", 0.5)
                if is_long and not is_near_support(candle.low, dl, buffer):
                    return self._no_signal(symbol, candle, "not_near_daily_support")
                if not is_long and not is_near_resistance(candle.high, dl, buffer):
                    return self._no_signal(symbol, candle, "not_near_daily_resistance")

        # ── Signal quality threshold ────────────────────────────────
        min_score = getattr(settings, "VMR_MIN_SIGNAL_SCORE", 0.45)
        if wr.score < min_score:
            return self._no_signal(symbol, candle, "low_signal_score")

        # ══════════════════════════════════════════════════════════════
        # PIN BAR DETECTED — Set pending trigger for next candle
        # ══════════════════════════════════════════════════════════════
        # Entry trigger: 1 tick above pin bar high (LONG)
        #                1 tick below pin bar low  (SHORT)
        #
        # SL: pin bar wick extreme ± buffer ticks
        #   LONG SL  = pin_bar.low  − sl_ticks × tick_size
        #   SHORT SL = pin_bar.high + sl_ticks × tick_size

        sl_ticks = getattr(settings, "VMR_SL_TICKS_BEYOND_PIN", 3)
        tick = TICK_SIZE

        if is_long:
            trigger_price = round(candle.high + 1 * tick, 2)
            stop_price = round(candle.low - sl_ticks * tick, 2)
        else:
            trigger_price = round(candle.low - 1 * tick, 2)
            stop_price = round(candle.high + sl_ticks * tick, 2)

        # Pre-compute risk for R:R and sizing validation
        risk_per_share = abs(trigger_price - stop_price)
        if risk_per_share <= 0:
            return self._no_signal(symbol, candle, "zero_risk")

        # ── Target: VWAP (mean reversion) ───────────────────────────
        #target_mode = getattr(settings, "VMR_TARGET_MODE", "VWAP")
        #if target_mode == "VWAP":
        #    target_price = bands.vwap
        #elif target_mode == "HALF_SD":
        #    target_price = bands.lower_1sd if is_long else bands.upper_1sd
        #else:
        #    rr = getattr(settings, "VMR_FIXED_TARGET_RR", 2.0)
        #    if is_long:
        #        target_price = trigger_price + risk_per_share * rr
        #    else:
        #        target_price = trigger_price - risk_per_share * rr
        target_price = 0.0  # No fixed target; exit via trail or flatten only

        # Ensure target is on the correct side
        if is_long and target_price <= trigger_price:
            target_price = trigger_price + risk_per_share * 1.5
        if not is_long and target_price >= trigger_price:
            target_price = trigger_price - risk_per_share * 1.5

        # ── R:R filter ──────────────────────────────────────────────
        reward_per_share = abs(target_price - trigger_price)
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio < 1.5:
            return self._no_signal(symbol, candle, f"rr_too_low_{rr_ratio:.1f}")

        # ── SL width cap ────────────────────────────────────────────
        sl_pct = (risk_per_share / trigger_price) * 100.0
        if sl_pct > 1.5:
            return self._no_signal(symbol, candle, "sl_too_wide")

        # ── Position sizing: fixed loss per trade ───────────────────
        fixed_loss = getattr(settings, "VMR_FIXED_LOSS_PER_TRADE", 1000.0)
        quantity = int(fixed_loss / risk_per_share)
        if quantity <= 0:
            return self._no_signal(symbol, candle, "quantity_zero")

        # Cap position value at a reasonable limit to avoid engine override.
        # Use VMR_MAX_POSITION_VALUE_PCT of capital (default 15%).
        max_pos_pct = getattr(settings, "VMR_MAX_POSITION_VALUE_PCT", 15.0)
        max_pos_value = settings.TOTAL_CAPITAL * (max_pos_pct / 100.0)
        position_value = trigger_price * quantity
        if position_value > max_pos_value:
            quantity = int(max_pos_value / trigger_price)
            if quantity <= 0:
                return self._no_signal(symbol, candle, "quantity_zero_after_cap")

        # ── Store pending trigger ───────────────────────────────────
        state.pending_trigger = {
            "is_long": is_long,
            "trigger_price": trigger_price,
            "stop_price": stop_price,
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "rr_ratio": round(rr_ratio, 2),
            "sl_pct": round(sl_pct, 2),
            "quantity": quantity,
            "pin_high": candle.high,
            "pin_low": candle.low,
            "pin_close": candle.close,
            "pin_timestamp": candle.timestamp,
            "wr_score": wr.score,
            "wr_rejection_type": wr.rejection_type.value,
            "wr_wick_body_ratio": wr.wick_body_ratio,
            "wr_body_position": wr.body_position,
            "wr_volume_ratio": wr.volume_ratio,
            "wr_band_touch": wr.band_touch,
            "wr_vwap_distance_sd": wr.vwap_distance_sd,
            "vwap": bands.vwap,
            "vwap_upper_1_5sd": bands.upper_1_5sd,
            "vwap_lower_1_5sd": bands.lower_1_5sd,
            "vwap_std_dev": bands.std_dev,
        }

        logger.info(
            f"[{self._name}] PENDING {'LONG' if is_long else 'SHORT'} "
            f"{symbol} trigger={trigger_price:.2f}  "
            f"SL={stop_price:.2f}  TGT={target_price:.2f}  "
            f"R:R={rr_ratio:.1f}  qty={quantity}  "
            f"score={wr.score:.2f}  band={wr.band_touch}"
        )

        return self._no_signal(symbol, candle, "awaiting_confirmation")

    # ------------------------------------------------------------------
    # PENDING CONFIRMATION CHECK
    # ------------------------------------------------------------------
    def _check_pending_confirmation(
        self,
        symbol: str,
        candle: Candle,
        state: _VMRState,
    ) -> Signal:
        """
        Check if the current candle confirms the pending pin-bar trigger.

        LONG:  candle.high >= trigger_price → confirmed, enter at trigger
        SHORT: candle.low  <= trigger_price → confirmed, enter at trigger

        The pending trigger expires after ONE candle. If not confirmed,
        it is cancelled.
        """
        pt = state.pending_trigger
        is_long = pt["is_long"]
        trigger_price = pt["trigger_price"]

        confirmed = False
        if is_long and candle.high >= trigger_price:
            confirmed = True
        elif not is_long and candle.low <= trigger_price:
            confirmed = True

        # Cancel the pending trigger regardless (one-candle window)
        state.pending_trigger = None

        if not confirmed:
            return self._no_signal(symbol, candle, "confirmation_failed")

        # ── CONFIRMED — fire the signal ─────────────────────────────
        state.signal_fired = True

        entry_price = trigger_price
        stop_price = pt["stop_price"]
        target_price = pt["target_price"]
        risk_per_share = pt["risk_per_share"]
        rr_ratio = pt["rr_ratio"]
        sl_pct = pt["sl_pct"]
        quantity = pt["quantity"]

        signal_type = SignalType.BUY if is_long else SignalType.SELL

        indicator_data = {
            "vwap": pt["vwap"],
            "vwap_upper_1_5sd": pt["vwap_upper_1_5sd"],
            "vwap_lower_1_5sd": pt["vwap_lower_1_5sd"],
            "vwap_std_dev": pt["vwap_std_dev"],
            "rejection_type": pt["wr_rejection_type"],
            "wick_body_ratio": pt["wr_wick_body_ratio"],
            "body_position": pt["wr_body_position"],
            "volume_ratio": pt["wr_volume_ratio"],
            "band_touch": pt["wr_band_touch"],
            "vwap_distance_sd": pt["wr_vwap_distance_sd"],
            "signal_score": pt["wr_score"],
            "stoploss_price": stop_price,
            "target_price": target_price,
            "risk_per_share": risk_per_share,
            "rr_ratio": rr_ratio,
            "sl_pct": sl_pct,
            "quantity": quantity,
            "trigger_price": trigger_price,
            "pin_high": pt["pin_high"],
            "pin_low": pt["pin_low"],
            "confirmed_by_candle_ts": candle.timestamp,
        }

        logger.info(
            f"[{self._name}] CONFIRMED {signal_type.name} {symbol} "
            f"@ {entry_price:.2f}  SL={stop_price:.2f}  "
            f"TGT={target_price:.2f}  R:R={rr_ratio:.1f}  "
            f"qty={quantity}  score={pt['wr_score']:.2f}  "
            f"band={pt['wr_band_touch']}"
        )

        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            signal_type=signal_type,
            strength=pt["wr_score"],
            price_at_signal=entry_price,
            timestamp=candle.timestamp,
            indicator_data=indicator_data,
        )

    # ------------------------------------------------------------------
    # 3. SHOULD EXIT
    # ------------------------------------------------------------------
    def should_exit(self, candle, position, state) -> Optional[Signal]:
    """
    Exit logic — only 3 exits:
      1. STOPLOSS  — hard stop (unchanged)
      2. TRAIL     — activates at 1R, trails using swing lows/highs
      3. FLATTEN   — forced close at 13:22 IST
    No TARGET exit.
    """
    direction = position.direction  # "LONG" or "SHORT"
    entry_price = position.entry_price
    sl_price = position.stoploss_price
    risk_per_share = abs(entry_price - sl_price)

    # ── 1. FLATTEN at 13:22 IST ──
    flatten_time = _parse_time(settings.VMR_FLATTEN_TIME)  # 13:22
    candle_time = _candle_ist_time(candle)
    if candle_time >= flatten_time:
        exit_price = candle.close
        return Signal(
            signal_type=SignalType.EXIT,
            strength=1.0,
            price=exit_price,
            timestamp=candle.timestamp,
            indicator_data={
                "exit_reason": "FLATTEN",
                "exit_price": exit_price,
            },
        )

    # ── 2. HARD STOPLOSS ──
    if direction == "LONG" and candle.low <= sl_price:
        return Signal(
            signal_type=SignalType.EXIT,
            strength=1.0,
            price=sl_price,
            timestamp=candle.timestamp,
            indicator_data={
                "exit_reason": "STOPLOSS",
                "exit_price": sl_price,
            },
        )
    elif direction == "SHORT" and candle.high >= sl_price:
        return Signal(
            signal_type=SignalType.EXIT,
            strength=1.0,
            price=sl_price,
            timestamp=candle.timestamp,
            indicator_data={
                "exit_reason": "STOPLOSS",
                "exit_price": sl_price,
            },
        )

    # ── 3. TRAILING STOP (R-multiple activation, swing trail) ──
    # Track max favorable excursion (MFE)
    if not hasattr(state, 'trail_active'):
        state.trail_active = False
        state.mfe = 0.0
        state.trail_stop = sl_price  # starts at original SL
        state.recent_candles = []

    # Keep recent candles for swing trail
    state.recent_candles.append(candle)
    lookback = settings.VMR_TRAIL_SWING_LOOKBACK  # default 3
    if len(state.recent_candles) > lookback + 1:
        state.recent_candles = state.recent_candles[-(lookback + 1):]

    # Calculate current favorable excursion
    if direction == "LONG":
        current_excursion = candle.high - entry_price
        state.mfe = max(state.mfe, current_excursion)
    else:
        current_excursion = entry_price - candle.low
        state.mfe = max(state.mfe, current_excursion)

    # Check if trail should activate (1R move in our favor)
    activate_distance = risk_per_share * settings.VMR_TRAIL_ACTIVATE_R  # 1.0R
    if not state.trail_active and state.mfe >= activate_distance:
        state.trail_active = True
        # Immediately move stop to breakeven (entry price)
        if settings.VMR_TRAIL_LOCK_BREAKEVEN:
            state.trail_stop = entry_price

    # Once trail is active, update trail stop using swing method
    if state.trail_active:
        trail_candles = state.recent_candles[:-1]  # exclude current candle
        if trail_candles:
            if direction == "LONG":
                # Trail to the lowest low of last N candles (swing low)
                swing_low = min(c.low for c in trail_candles)
                # Trail stop can only move UP, never down
                new_trail = max(state.trail_stop, swing_low)
                # Floor at breakeven
                if settings.VMR_TRAIL_LOCK_BREAKEVEN:
                    new_trail = max(new_trail, entry_price)
                state.trail_stop = new_trail

                # Check if trailed out
                if candle.low <= state.trail_stop:
                    return Signal(
                        signal_type=SignalType.EXIT,
                        strength=1.0,
                        price=state.trail_stop,
                        timestamp=candle.timestamp,
                        indicator_data={
                            "exit_reason": "TRAIL",
                            "exit_price": state.trail_stop,
                            "mfe": state.mfe,
                            "trail_r_multiple": state.mfe / risk_per_share if risk_per_share > 0 else 0,
                        },
                    )

            else:  # SHORT
                # Trail to the highest high of last N candles (swing high)
                swing_high = max(c.high for c in trail_candles)
                # Trail stop can only move DOWN, never up
                new_trail = min(state.trail_stop, swing_high)
                # Floor at breakeven
                if settings.VMR_TRAIL_LOCK_BREAKEVEN:
                    new_trail = min(new_trail, entry_price)
                state.trail_stop = new_trail

                # Check if trailed out
                if candle.high >= state.trail_stop:
                    return Signal(
                        signal_type=SignalType.EXIT,
                        strength=1.0,
                        price=state.trail_stop,
                        timestamp=candle.timestamp,
                        indicator_data={
                            "exit_reason": "TRAIL",
                            "exit_price": state.trail_stop,
                            "mfe": state.mfe,
                            "trail_r_multiple": state.mfe / risk_per_share if risk_per_share > 0 else 0,
                        },
                    )

    return None  # No exit signal — hold the position

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
            "fixed_loss_per_trade": getattr(settings, "VMR_FIXED_LOSS_PER_TRADE", 1000.0),
            "sl_ticks_beyond_pin": getattr(settings, "VMR_SL_TICKS_BEYOND_PIN", 3),
            "min_atr_multiplier": getattr(settings, "VMR_MIN_ATR_MULTIPLIER", 1.0),
            "prior_trend_lookback": getattr(settings, "VMR_PRIOR_TREND_LOOKBACK", 5),
            "max_nose_pct": getattr(settings, "VMR_MAX_NOSE_PCT", 0.10),
            "require_confirmation": getattr(settings, "VMR_REQUIRE_CONFIRMATION", True),
            "daily_level_filter": getattr(settings, "VMR_DAILY_LEVEL_FILTER", True),
            "daily_level_buffer_pct": getattr(settings, "VMR_DAILY_LEVEL_BUFFER_PCT", 0.5),
            "max_position_value_pct": getattr(settings, "VMR_MAX_POSITION_VALUE_PCT", 30.0),
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
