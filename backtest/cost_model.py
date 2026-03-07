"""
NSE Trading Platform — Fyers Cost Model

Computes exact transaction costs for Indian equity markets via Fyers broker.

Cost components (as of 2025):
  - Brokerage: ₹20 flat per executed order OR 0.03% (whichever is lower)
  - STT: 0.025% on sell side (intraday equity)
  - Exchange Transaction: 0.00345% on turnover (NSE)
  - SEBI Turnover Fee: 0.0001% on turnover
  - GST: 18% on (brokerage + exchange + SEBI charges)
  - Stamp Duty: 0.003% on buy side
  - IPFT: 0.0001% on turnover

All percentages are of turnover (price × quantity).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """
    Breakdown of all costs for a single trade (entry + exit).

    All values in INR.
    """
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    gst: float
    stamp_duty: float
    ipft: float
    total: float

    def as_dict(self) -> dict[str, float]:
        """Return all cost components as a dict."""
        return {
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange_txn": self.exchange_txn,
            "sebi_fee": self.sebi_fee,
            "gst": self.gst,
            "stamp_duty": self.stamp_duty,
            "ipft": self.ipft,
            "total": self.total,
        }


class FyersCostModel:
    """
    Computes transaction costs for equity trades on NSE via Fyers.

    Supports:
      - Intraday (MIS): STT on sell side only
      - Delivery (CNC): STT on both sides (different rates)

    Usage
    -----
    >>> model = FyersCostModel()
    >>> costs = model.compute_trade_costs(
    ...     entry_price=2450.0, exit_price=2480.0,
    ...     quantity=100, trade_type="INTRADAY"
    ... )
    >>> print(f"Total costs: ₹{costs.total:.2f}")
    >>> print(f"Net P&L: ₹{(2480-2450)*100 - costs.total:.2f}")
    """

    def __init__(
        self,
        brokerage_flat: float | None = None,
        brokerage_pct: float | None = None,
        stt_sell_intraday_pct: float | None = None,
        exchange_txn_pct: float | None = None,
        sebi_pct: float | None = None,
        gst_pct: float | None = None,
        stamp_duty_buy_pct: float | None = None,
        ipft_pct: float | None = None,
    ) -> None:
        """
        Initialize cost model. Uses settings defaults if not overridden.

        All pct parameters are in percentage terms (e.g. 0.03 means 0.03%).
        """
        self.brokerage_flat = brokerage_flat if brokerage_flat is not None else settings.BROKERAGE_FLAT_PER_ORDER
        self.brokerage_pct = brokerage_pct if brokerage_pct is not None else settings.BROKERAGE_INTRADAY_PCT
        self.stt_sell_intraday_pct = stt_sell_intraday_pct if stt_sell_intraday_pct is not None else settings.STT_SELL_INTRADAY_PCT
        self.exchange_txn_pct = exchange_txn_pct if exchange_txn_pct is not None else settings.EXCHANGE_TXN_PCT
        self.sebi_pct = sebi_pct if sebi_pct is not None else settings.SEBI_TURNOVER_PCT
        self.gst_pct = gst_pct if gst_pct is not None else settings.GST_PCT
        self.stamp_duty_buy_pct = stamp_duty_buy_pct if stamp_duty_buy_pct is not None else settings.STAMP_DUTY_BUY_PCT
        self.ipft_pct = ipft_pct if ipft_pct is not None else settings.IPFT_PCT

    def _compute_brokerage(self, turnover: float) -> float:
        """
        Compute brokerage for one leg (buy or sell).

        Fyers charges: min(₹20 flat, 0.03% of turnover) per order.

        Parameters
        ----------
        turnover : float
            price × quantity for this leg.

        Returns
        -------
        float
            Brokerage in INR.
        """
        pct_brokerage = turnover * (self.brokerage_pct / 100.0)
        return min(self.brokerage_flat, pct_brokerage)

    def compute_trade_costs(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        trade_type: str = "INTRADAY",
    ) -> TradeCosts:
        """
        Compute full round-trip costs for a trade.

        Parameters
        ----------
        entry_price : float
            Buy/entry price per share.
        exit_price : float
            Sell/exit price per share.
        quantity : int
            Number of shares traded.
        trade_type : str
            'INTRADAY' or 'DELIVERY'.

        Returns
        -------
        TradeCosts
            Complete cost breakdown.

        Examples
        --------
        >>> model = FyersCostModel()
        >>> costs = model.compute_trade_costs(2450.0, 2480.0, 100)
        >>> print(f"Total: ₹{costs.total:.2f}")
        """
        buy_turnover = entry_price * quantity
        sell_turnover = exit_price * quantity
        total_turnover = buy_turnover + sell_turnover

        # ── Brokerage (both legs) ─────────────────────────────────────────
        brokerage_buy = self._compute_brokerage(buy_turnover)
        brokerage_sell = self._compute_brokerage(sell_turnover)
        brokerage = brokerage_buy + brokerage_sell

        # ── STT ───────────────────────────────────────────────────────────
        if trade_type == "INTRADAY":
            # Intraday: STT only on sell side
            stt = sell_turnover * (self.stt_sell_intraday_pct / 100.0)
        else:
            # Delivery: 0.1% on both sides
            stt = total_turnover * (0.1 / 100.0)

        # ── Exchange Transaction Charges ──────────────────────────────────
        exchange_txn = total_turnover * (self.exchange_txn_pct / 100.0)

        # ── SEBI Turnover Fee ─────────────────────────────────────────────
        sebi_fee = total_turnover * (self.sebi_pct / 100.0)

        # ── GST (18% on brokerage + exchange + SEBI) ─────────────────────
        gst_base = brokerage + exchange_txn + sebi_fee
        gst = gst_base * (self.gst_pct / 100.0)

        # ── Stamp Duty (on buy side only) ─────────────────────────────────
        stamp_duty = buy_turnover * (self.stamp_duty_buy_pct / 100.0)

        # ── IPFT ──────────────────────────────────────────────────────────
        ipft = total_turnover * (self.ipft_pct / 100.0)

        # ── Total ─────────────────────────────────────────────────────────
        total = brokerage + stt + exchange_txn + sebi_fee + gst + stamp_duty + ipft

        return TradeCosts(
            brokerage=round(brokerage, 2),
            stt=round(stt, 2),
            exchange_txn=round(exchange_txn, 2),
            sebi_fee=round(sebi_fee, 2),
            gst=round(gst, 2),
            stamp_duty=round(stamp_duty, 2),
            ipft=round(ipft, 2),
            total=round(total, 2),
        )

    def compute_net_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: str = "LONG",
        trade_type: str = "INTRADAY",
    ) -> dict[str, float]:
        """
        Compute gross P&L, costs, and net P&L for a trade.

        Parameters
        ----------
        entry_price : float
        exit_price : float
        quantity : int
        direction : str
            'LONG' or 'SHORT'.
        trade_type : str
            'INTRADAY' or 'DELIVERY'.

        Returns
        -------
        dict
            {'gross_pnl': float, 'costs': float, 'net_pnl': float,
             'cost_breakdown': dict}
        """
        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * quantity
            costs = self.compute_trade_costs(entry_price, exit_price, quantity, trade_type)
        else:
            # SHORT: entry is sell, exit is buy
            gross_pnl = (entry_price - exit_price) * quantity
            costs = self.compute_trade_costs(exit_price, entry_price, quantity, trade_type)

        net_pnl = gross_pnl - costs.total

        return {
            "gross_pnl": round(gross_pnl, 2),
            "costs": costs.total,
            "net_pnl": round(net_pnl, 2),
            "cost_breakdown": costs.as_dict(),
        }

    def compute_breakeven_move(
        self,
        price: float,
        quantity: int,
        trade_type: str = "INTRADAY",
    ) -> float:
        """
        Compute the minimum price move needed to break even after costs.

        Parameters
        ----------
        price : float
            Entry price.
        quantity : int
        trade_type : str

        Returns
        -------
        float
            Breakeven price move per share in INR.
        """
        # Compute costs for a flat trade (entry == exit)
        costs = self.compute_trade_costs(price, price, quantity, trade_type)
        if quantity > 0:
            return round(costs.total / quantity, 2)
        return 0.0

    def get_params(self) -> dict[str, float]:
        """Return all cost model parameters."""
        return {
            "brokerage_flat": self.brokerage_flat,
            "brokerage_pct": self.brokerage_pct,
            "stt_sell_intraday_pct": self.stt_sell_intraday_pct,
            "exchange_txn_pct": self.exchange_txn_pct,
            "sebi_pct": self.sebi_pct,
            "gst_pct": self.gst_pct,
            "stamp_duty_buy_pct": self.stamp_duty_buy_pct,
            "ipft_pct": self.ipft_pct,
        }
