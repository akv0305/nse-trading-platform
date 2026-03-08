"""
NSE Trading Platform — ORB + VWAP Breakout Strategy

Intraday equity strategy for Nifty 50 stocks.

Lifecycle:
  1. pre_market_scan()  → shortlist 10 stocks via scanner
  2. on_candle()        → capture ORB, detect breakouts, confirm with VWAP
  3. should_exit()      → check SL, target, trailing stop, flatten time
  4. end_of_day()       → reset ORB levels and internal state
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from config.settings import settings
from core.data.models import Candle, Signal, SignalType, Direction
from core.data.universe import from_fyers_symbol
from core.indicators.orb import ORBLevels, compute_opening_range, detect_breakout
from core.indicators.vwap import VWAPData, compute_vwap
from core.indicators.sector_score import compute_sector_scores, get_stock_sector_bias
from core.scanner.stock_scanner import scan_for_orb
from core.strategies.base import StrategyBase
from core.utils.time_utils import epoch_ms_to_ist, ist_to_epoch_ms, IST, now_epoch_ms

logger = logging.getLogger(__name__)


@dataclass
class _SymbolState:
    """Internal per-symbol tracking state for one trading day."""
    orb: ORBLevels | None = None          # OR levels once captured
    orb_captured: bool = False             # True after OR period ends
    orb_rejected: bool = False             # True if OR was captured but rejected (width filter)
    breakout_fired: bool = False           # True after first breakout signal
    or_candles: list[Candle] = field(default_factory=list)  # candles during OR window
    all_candles: list[Candle] = field(default_factory=list)  # all candles today
    avg_or_volume: float = 0.0            # avg volume during OR period
    peak_price_long: float = 0.0          # high watermark since entry (for trailing SL)
    peak_price_short: float = float("inf")  # low watermark since entry (for trailing SL)


class ORBVWAPStrategy(StrategyBase):
    """
    Opening Range Breakout + VWAP confirmation strategy.

    Entry:
      LONG  — close > ORB_HIGH + buffer, price > VWAP, VWAP slope up,
              volume > 1.5× OR avg, sector positive.
      SHORT — close < ORB_LOW − buffer, price < VWAP, VWAP slope down,
              volume > 1.5× OR avg, sector negative.

    Exit:
      Stoploss (ORB opposite), target (R:R), trailing stop, flatten time.
    """

    def __init__(self) -> None:
        super().__init__(name="orb_vwap", version="1.0.0")

        # ── Per-symbol state, reset daily via end_of_day() ────────────
        self._states: dict[str, _SymbolState] = {}

        # ── Sector scores, computed once during pre_market_scan() ─────
        self._sector_scores: dict[str, float] = {}

    # ══════════════════════════════════════════════════════════════════
    # StrategyBase abstract methods
    # ══════════════════════════════════════════════════════════════════

    def pre_market_scan(
        self,
        universe: list[str],
        historical_data: dict[str, pd.DataFrame],
    ) -> list[str]:
        """
        Shortlist top N stocks for today's ORB trading session.

        Steps:
          1. Compute sector scores from last 5 days of sector indices.
          2. Run scan_for_orb() on universe with historical data.
          3. Set watchlist to top SCAN_TOP_N symbols.
        """
        # ── Compute sector scores ─────────────────────────────────────
        sector_data: dict[str, pd.DataFrame] = {}
        nifty_data: pd.DataFrame = pd.DataFrame()

        for key, df in historical_data.items():
            if "NIFTY" in key.upper() and "INDEX" in key.upper():
                try:
                    _, name_part, _ = from_fyers_symbol(key)
                except (ValueError, IndexError):
                    pass

                if "NIFTY50" in key.upper() or key.upper() == "NSE:NIFTY50-INDEX":
                    nifty_data = df
                else:
                    sector_data[key] = df

        if not sector_data or nifty_data.empty:
            self._sector_scores = {}
        else:
            self._sector_scores = compute_sector_scores(
                sector_data, nifty_data, lookback_days=5,
            )

        # ── Run scanner ───────────────────────────────────────────────
        candidates = scan_for_orb(
            universe=universe,
            historical_data=historical_data,
            sector_scores=self._sector_scores,
            top_n=settings.SCAN_TOP_N,
        )

        self._watchlist = [c["symbol"] for c in candidates]

        # Initialize per-symbol state
        self._states = {sym: _SymbolState() for sym in self._watchlist}

        logger.info(
            f"[{self._name}] pre_market_scan: {len(self._watchlist)} stocks "
            f"shortlisted from {len(universe)} universe"
        )

        return self._watchlist.copy()

    def on_candle(
        self,
        symbol: str,
        candle: Candle,
        candle_history: list[Candle],
        current_position: dict | None,
    ) -> Signal:
        """
        Main signal generation.

        Phase 1: During OR period → accumulate candles, capture ORB levels.
        Phase 2: After OR period → detect breakouts, confirm with VWAP.

        Returns BUY, SELL, or NO_ACTION.
        """
        ts = candle.timestamp
        candle_dt = epoch_ms_to_ist(ts)

        # ── Ensure state exists ───────────────────────────────────────
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        state = self._states[symbol]

        # ── Accumulate all candles for VWAP ───────────────────────────
        state.all_candles.append(candle)

        # ── Update peak price tracking for trailing stop ──────────────
        if current_position is not None:
            state.peak_price_long = max(state.peak_price_long, candle.high)
            state.peak_price_short = min(state.peak_price_short, candle.low)

        # ── Phase 1: OR Capture ───────────────────────────────────────
        if not state.orb_captured:
            return self._handle_or_phase(symbol, candle, state)

        # ── ORB rejected (width filter) → no signals for this symbol ──
        if state.orb_rejected:
            return self.no_action_signal(
                symbol, candle.close, ts, reason="orb_width_rejected",
            )

        # ── Already have a position → no new entry ────────────────────
        if current_position is not None:
            return self.no_action_signal(
                symbol, candle.close, ts, reason="position_open",
            )

        # ── Already fired one breakout for this symbol today ──────────
        if state.breakout_fired:
            return self.no_action_signal(
                symbol, candle.close, ts, reason="breakout_already_fired",
            )

        # ── Entry cutoff check ────────────────────────────────────────
        cutoff_h, cutoff_m = map(int, settings.ENTRY_CUTOFF_IST.split(":"))
        if candle_dt.hour > cutoff_h or (candle_dt.hour == cutoff_h and candle_dt.minute >= cutoff_m):
            return self.no_action_signal(
                symbol, candle.close, ts, reason="past_entry_cutoff",
            )

        # ── Phase 2: Breakout Detection ───────────────────────────────
        return self._handle_breakout_phase(symbol, candle, state)

    def should_exit(
        self,
        symbol: str,
        current_price: float,
        position: dict,
        candle: Candle | None = None,
    ) -> Signal:
        """
        Check exit conditions for an open position.

        Priority order:
          1. Flatten time (force close)
          2. Stoploss hit (including trailing)
          3. Target hit
        """
        ts = candle.timestamp if candle else now_epoch_ms()
        candle_dt = epoch_ms_to_ist(ts)
        direction = position.get("direction", "LONG")
        entry_price = float(position.get("entry_price", 0))
        stoploss_price = float(position.get("stoploss_price", 0))
        target_price = position.get("target_price")
        if target_price is not None:
            target_price = float(target_price)

        # ── Update peak price from position if provided ───────────────
        # The backtest engine / live engine should pass peak_price in
        # the position dict.  If not present, use current_price as
        # fallback (which is the candle-by-candle behavior).
        if direction == "LONG":
            peak_price = float(position.get("peak_price", current_price))
            # Also update from candle high if available
            if candle is not None:
                peak_price = max(peak_price, candle.high)
            peak_price = max(peak_price, current_price)
        else:
            peak_price = float(position.get("peak_price", current_price))
            if candle is not None:
                peak_price = min(peak_price, candle.low)
            peak_price = min(peak_price, current_price)

        # ── 1. Flatten Time ───────────────────────────────────────────
        flatten_h, flatten_m = map(int, settings.FLATTEN_TIME_IST.split(":"))
        if candle_dt.hour > flatten_h or (candle_dt.hour == flatten_h and candle_dt.minute >= flatten_m):
            exit_type = SignalType.EXIT_LONG if direction == "LONG" else SignalType.EXIT_SHORT
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=1.0,
                price_at_signal=current_price,
                timestamp=ts,
                indicator_data={"exit_reason": "FLATTEN", "flatten_time": settings.FLATTEN_TIME_IST},
                skip_reason="",
            )

        # ── Compute trailing stop from peak price (high watermark) ────
        trailing_sl = self._compute_trailing_sl(
            direction, entry_price, stoploss_price, peak_price,
        )
        # Use the tighter of original SL and trailing SL
        effective_sl = stoploss_price
        is_trailing = False
        if trailing_sl is not None:
            if direction == "LONG" and trailing_sl > stoploss_price:
                effective_sl = trailing_sl
                is_trailing = True
            elif direction == "SHORT" and trailing_sl < stoploss_price:
                effective_sl = trailing_sl
                is_trailing = True

        # ── 2. Stoploss Check ─────────────────────────────────────────
        sl_hit = False
        if direction == "LONG" and current_price <= effective_sl:
            sl_hit = True
        elif direction == "SHORT" and current_price >= effective_sl:
            sl_hit = True

        if sl_hit:
            exit_type = SignalType.EXIT_LONG if direction == "LONG" else SignalType.EXIT_SHORT
            exit_reason = "TRAIL" if is_trailing else "STOPLOSS"
            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=exit_type,
                strength=1.0,
                price_at_signal=current_price,
                timestamp=ts,
                indicator_data={
                    "exit_reason": exit_reason,
                    "stoploss_price": effective_sl,
                    "is_trailing": is_trailing,
                    "peak_price": peak_price,
                },
                skip_reason="",
            )

        # ── 3. Target Check ───────────────────────────────────────────
        if target_price is not None:
            target_hit = False
            if direction == "LONG" and current_price >= target_price:
                target_hit = True
            elif direction == "SHORT" and current_price <= target_price:
                target_hit = True

            if target_hit:
                exit_type = SignalType.EXIT_LONG if direction == "LONG" else SignalType.EXIT_SHORT
                return Signal(
                    strategy_name=self._name,
                    symbol=symbol,
                    signal_type=exit_type,
                    strength=1.0,
                    price_at_signal=current_price,
                    timestamp=ts,
                    indicator_data={
                        "exit_reason": "TARGET",
                        "target_price": target_price,
                    },
                    skip_reason="",
                )

        # ── No exit ───────────────────────────────────────────────────
        return self.no_action_signal(
            symbol, current_price, ts,
            reason="no_exit_condition",
            indicator_data={
                "effective_sl": effective_sl,
                "is_trailing": is_trailing,
                "peak_price": peak_price,
            },
        )

    def get_params(self) -> dict:
        """Return all ORB+VWAP strategy parameters."""
        return {
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
            "risk_per_trade_inr": settings.RISK_PER_TRADE_INR,
            "max_concurrent_positions": settings.MAX_CONCURRENT_POSITIONS,
            "scan_top_n": settings.SCAN_TOP_N,
        }

    def end_of_day(self) -> None:
        """Reset all per-symbol state and watchlist for the next day."""
        self._states.clear()
        self._sector_scores.clear()
        super().end_of_day()

    # ══════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════

    def _handle_or_phase(
        self,
        symbol: str,
        candle: Candle,
        state: _SymbolState,
    ) -> Signal:
        """
        Accumulate OR candles and capture ORB levels once OR period ends.
        """
        ts = candle.timestamp
        candle_dt = epoch_ms_to_ist(ts)

        open_h, open_m = map(int, settings.MARKET_OPEN.split(":"))
        open_minutes = open_h * 60 + open_m
        or_end_minutes = open_minutes + settings.ORB_PERIOD_MINUTES
        candle_minutes = candle_dt.hour * 60 + candle_dt.minute

        # ── Still within OR window → accumulate ───────────────────────
        if candle_minutes < or_end_minutes:
            state.or_candles.append(candle)
            return self.no_action_signal(
                symbol, candle.close, ts, reason="or_period_in_progress",
            )

        # ── OR window just ended → compute ORB levels ────────────────
        orb = compute_opening_range(
            candles=state.or_candles,
            or_period_minutes=settings.ORB_PERIOD_MINUTES,
            market_open_time=settings.MARKET_OPEN,
        )

        state.orb = orb
        state.orb_captured = True

        if orb is None or not orb.is_valid:
            state.orb_rejected = True
            return self.no_action_signal(
                symbol, candle.close, ts,
                reason="orb_capture_failed",
                indicator_data={"orb": None},
            )

        # ── OR width filter (skip if too wide or too narrow) ──────────
        if orb.or_width_pct > 2.0 or orb.or_width_pct < 0.3:
            state.orb_rejected = True
            return self.no_action_signal(
                symbol, candle.close, ts,
                reason=f"orb_width_out_of_range({orb.or_width_pct:.2f}%)",
                indicator_data={"or_high": orb.or_high, "or_low": orb.or_low,
                                "or_width_pct": orb.or_width_pct},
            )

        # ── Compute avg volume during OR for later comparison ─────────
        or_volumes = [c.volume for c in state.or_candles if c.volume > 0]
        state.avg_or_volume = sum(or_volumes) / len(or_volumes) if or_volumes else 0.0

        logger.info(
            f"[{self._name}] ORB captured for {symbol}: "
            f"H={orb.or_high} L={orb.or_low} W={orb.or_width_pct:.2f}%"
        )

        # ── Now process this first post-OR candle for breakout ────────
        return self._handle_breakout_phase(symbol, candle, state)

    def _handle_breakout_phase(
        self,
        symbol: str,
        candle: Candle,
        state: _SymbolState,
    ) -> Signal:
        """
        Detect breakout and confirm with VWAP + volume + sector.
        """
        ts = candle.timestamp
        orb = state.orb

        if orb is None or not orb.is_valid:
            return self.no_action_signal(
                symbol, candle.close, ts, reason="orb_not_valid",
            )

        # ── 1. Breakout detection (candle close vs ORB levels) ────────
        breakout = detect_breakout(
            current_price=candle.close,
            orb_levels=orb,
            buffer_pct=settings.ORB_BREAKOUT_BUFFER_PCT,
        )

        if breakout == "INSIDE_RANGE":
            return self.no_action_signal(
                symbol, candle.close, ts,
                reason="inside_range",
                indicator_data={"or_high": orb.or_high, "or_low": orb.or_low},
            )

        # ── 2. VWAP confirmation ──────────────────────────────────────
        vwap_data = compute_vwap(
            candles=state.all_candles,
            band_multiplier=settings.VWAP_BAND_STD_MULTIPLIER,
            slope_lookback=5,
        )

        if vwap_data is None:
            return self.no_action_signal(
                symbol, candle.close, ts, reason="vwap_unavailable",
            )

        # ── 3. Volume confirmation ────────────────────────────────────
        vol_ratio = (candle.volume / state.avg_or_volume) if state.avg_or_volume > 0 else 0.0

        # ── 4. Sector bias ────────────────────────────────────────────
        try:
            _, plain_symbol, _ = from_fyers_symbol(symbol)
        except (ValueError, IndexError):
            plain_symbol = symbol
        sector_bias = get_stock_sector_bias(plain_symbol, self._sector_scores)

        # ── Build indicator snapshot ──────────────────────────────────
        indicator_data = {
            "or_high": orb.or_high,
            "or_low": orb.or_low,
            "or_mid": orb.or_mid,
            "or_width_pct": orb.or_width_pct,
            "vwap": vwap_data.vwap,
            "vwap_upper": vwap_data.upper_band,
            "vwap_lower": vwap_data.lower_band,
            "vwap_slope": vwap_data.slope,
            "breakout_volume": candle.volume,
            "avg_or_volume": state.avg_or_volume,
            "volume_ratio": round(vol_ratio, 2),
            "sector_bias": sector_bias,
        }

        # ── 5. LONG entry conditions ─────────────────────────────────
        if breakout == "BREAKOUT_LONG":
            conditions_met = (
                candle.close > vwap_data.vwap        # price above VWAP
                and vwap_data.slope > 0              # VWAP rising
                and vol_ratio >= 1.5                 # volume confirmation
                and sector_bias > 0                  # sector positive
            )

            if not conditions_met:
                reasons = []
                if candle.close <= vwap_data.vwap:
                    reasons.append("below_vwap")
                if vwap_data.slope <= 0:
                    reasons.append("vwap_slope_down")
                if vol_ratio < 1.5:
                    reasons.append(f"low_volume({vol_ratio:.1f}x)")
                if sector_bias <= 0:
                    reasons.append(f"sector_negative({sector_bias:.2f})")
                return self.no_action_signal(
                    symbol, candle.close, ts,
                    reason="long_conditions_failed:" + ",".join(reasons),
                    indicator_data=indicator_data,
                )

            # ── Compute SL, target, strength ──────────────────────────
            stoploss = orb.or_low  # ORB_OPPOSITE mode
            risk = candle.close - stoploss
            if risk <= 0:
                return self.no_action_signal(
                    symbol, candle.close, ts,
                    reason="zero_risk_long",
                    indicator_data=indicator_data,
                )

            target = candle.close + (risk * settings.ORB_TARGET_RR)
            strength = self._compute_signal_strength(
                vol_ratio, vwap_data.slope, sector_bias, orb.or_width_pct,
            )

            indicator_data["stoploss_price"] = round(stoploss, 2)
            indicator_data["target_price"] = round(target, 2)
            indicator_data["risk_per_share"] = round(risk, 2)

            state.breakout_fired = True

            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price_at_signal=candle.close,
                timestamp=ts,
                indicator_data=indicator_data,
                skip_reason="",
            )

        # ── 6. SHORT entry conditions ─────────────────────────────────
        if breakout == "BREAKOUT_SHORT":
            conditions_met = (
                candle.close < vwap_data.vwap        # price below VWAP
                and vwap_data.slope < 0              # VWAP falling
                and vol_ratio >= 1.5                 # volume confirmation
                and sector_bias < 0                  # sector negative
            )

            if not conditions_met:
                reasons = []
                if candle.close >= vwap_data.vwap:
                    reasons.append("above_vwap")
                if vwap_data.slope >= 0:
                    reasons.append("vwap_slope_up")
                if vol_ratio < 1.5:
                    reasons.append(f"low_volume({vol_ratio:.1f}x)")
                if sector_bias >= 0:
                    reasons.append(f"sector_positive({sector_bias:.2f})")
                return self.no_action_signal(
                    symbol, candle.close, ts,
                    reason="short_conditions_failed:" + ",".join(reasons),
                    indicator_data=indicator_data,
                )

            stoploss = orb.or_high  # ORB_OPPOSITE mode
            risk = stoploss - candle.close
            if risk <= 0:
                return self.no_action_signal(
                    symbol, candle.close, ts,
                    reason="zero_risk_short",
                    indicator_data=indicator_data,
                )

            target = candle.close - (risk * settings.ORB_TARGET_RR)
            strength = self._compute_signal_strength(
                vol_ratio, abs(vwap_data.slope), abs(sector_bias), orb.or_width_pct,
            )

            indicator_data["stoploss_price"] = round(stoploss, 2)
            indicator_data["target_price"] = round(target, 2)
            indicator_data["risk_per_share"] = round(risk, 2)

            state.breakout_fired = True

            return Signal(
                strategy_name=self._name,
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                price_at_signal=candle.close,
                timestamp=ts,
                indicator_data=indicator_data,
                skip_reason="",
            )

        # ── Fallback ──────────────────────────────────────────────────
        return self.no_action_signal(
            symbol, candle.close, ts, reason="no_signal",
            indicator_data=indicator_data,
        )

    def _compute_trailing_sl(
        self,
        direction: str,
        entry_price: float,
        original_sl: float,
        peak_price: float,
    ) -> float | None:
        """
        Compute trailing stop level based on the high watermark (peak_price).

        The trailing stop only ratchets in the favorable direction:
          LONG:  trail_sl = entry + trail_pct × (peak_high - entry)
          SHORT: trail_sl = entry - trail_pct × (entry - peak_low)

        Trailing activates only after ORB_TRAIL_AFTER_RR is achieved
        (measured from peak, not current price).

        Parameters
        ----------
        direction : str
            'LONG' or 'SHORT'.
        entry_price : float
        original_sl : float
        peak_price : float
            Highest price since entry (LONG) or lowest (SHORT).

        Returns
        -------
        float or None
            Trailing SL price, or None if trailing not yet active.
        """
        risk = abs(entry_price - original_sl)
        if risk <= 0:
            return None

        if direction == "LONG":
            move = peak_price - entry_price
            rr_achieved = move / risk
            if rr_achieved < settings.ORB_TRAIL_AFTER_RR:
                return None
            trailing_sl = entry_price + (settings.ORB_TRAIL_PCT * move)
            return round(trailing_sl, 2)

        else:  # SHORT
            move = entry_price - peak_price
            rr_achieved = move / risk
            if rr_achieved < settings.ORB_TRAIL_AFTER_RR:
                return None
            trailing_sl = entry_price - (settings.ORB_TRAIL_PCT * move)
            return round(trailing_sl, 2)

    def _compute_signal_strength(
        self,
        vol_ratio: float,
        vwap_slope_abs: float,
        sector_bias_abs: float,
        or_width_pct: float,
    ) -> float:
        """
        Compute a composite signal strength score (0.0 to 1.0).

        Factors:
          - Volume ratio (higher = stronger)
          - VWAP slope magnitude (steeper = stronger)
          - Sector directional strength
          - OR width (moderate is ideal)
        """
        # Volume: 1.5x=0.5, 2.0x=0.67, 3.0x=1.0, cap at 1.0
        vol_score = min(vol_ratio / 3.0, 1.0)

        # Sector: already 0..1 range
        sector_score = min(abs(sector_bias_abs), 1.0)

        # OR width: ideal around 0.8-1.2%, penalize extremes
        if 0.6 <= or_width_pct <= 1.5:
            width_score = 1.0
        elif 0.3 <= or_width_pct <= 2.0:
            width_score = 0.6
        else:
            width_score = 0.3

        # VWAP slope: normalize (small numbers, so scale up)
        slope_score = min(abs(vwap_slope_abs) * 100.0, 1.0)

        strength = (
            vol_score * 0.30
            + slope_score * 0.25
            + sector_score * 0.25
            + width_score * 0.20
        )
        return round(max(0.0, min(1.0, strength)), 4)
