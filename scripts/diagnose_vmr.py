#!/usr/bin/env python3
"""Diagnose VMR backtest - check actual risk, R:R, and filter effectiveness."""
import sqlite3
import pandas as pd
from config.settings import settings
from core.strategies.vmr_strategy import VMRStrategy
from backtest.engine import BacktestEngine
from backtest.cost_model import FyersCostModel
from core.data.universe import get_nifty50_fyers_symbols

db = sqlite3.connect(str(settings.DB_PATH))
symbols = get_nifty50_fyers_symbols()

data = {}
for sym in symbols:
    df = pd.read_sql(
        "SELECT * FROM ohlcv_cache WHERE symbol=? AND timeframe='5m' ORDER BY timestamp",
        db, params=(sym,),
    )
    if not df.empty:
        data[sym] = df

strategy = VMRStrategy()
engine = BacktestEngine(strategy, FyersCostModel())
result = engine.run(data, initial_capital=750000.0,
                    start_date="2024-01-01", end_date="2025-01-01")

trades = result.trades
if not trades:
    print("No trades!")
    db.close()
    exit()

print(f"\nTotal trades: {len(trades)}")
print(f"\n{'='*90}")
print(f"{'Symbol':<22} {'Dir':<6} {'Entry':>8} {'SL':>8} {'TGT':>8} "
      f"{'Qty':>5} {'Risk$':>7} {'PnL$':>8} {'Exit':<10} {'R:R':>5}")
print(f"{'='*90}")

sl_losses = []
tgt_wins = []
actual_risks = []
planned_rrs = []

for t in trades[:50]:
    ind = t.get("indicator_data", {})
    sl = t.get("stoploss_price", 0) or ind.get("stoploss_price", 0)
    tgt = t.get("target_price", 0) or ind.get("target_price", 0)
    qty = t["quantity"]
    rps = ind.get("risk_per_share", 0)
    rr = ind.get("rr_ratio", 0)
    planned_risk = rps * qty if rps else abs(t["entry_price"] - sl) * qty

    print(f"{t['symbol']:<22} {t['direction']:<6} {t['entry_price']:>8.2f} "
          f"{sl:>8.2f} {tgt:>8.2f} {qty:>5} {planned_risk:>7.0f} "
          f"{t['pnl_net']:>8.2f} {t['exit_reason']:<10} {rr:>5.1f}")

    actual_risks.append(planned_risk)
    planned_rrs.append(rr)
    if t["exit_reason"] == "STOPLOSS":
        sl_losses.append(t["pnl_net"])
    elif t["exit_reason"] == "TARGET":
        tgt_wins.append(t["pnl_net"])

print(f"\n{'='*90}")
print(f"Avg planned risk : Rs.{sum(actual_risks)/len(actual_risks):.0f}")
print(f"Avg planned R:R  : {sum(planned_rrs)/len(planned_rrs):.2f}")
if sl_losses:
    print(f"Avg SL loss      : Rs.{sum(sl_losses)/len(sl_losses):.2f}")
if tgt_wins:
    print(f"Avg TGT win      : Rs.{sum(tgt_wins)/len(tgt_wins):.2f}")

reduced = [t for t in trades
           if t.get("indicator_data", {}).get("quantity", 0) > t["quantity"]]
print(f"\nTrades with qty reduced by engine: {len(reduced)} / {len(trades)}")

db.close()
