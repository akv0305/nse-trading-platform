"""
NSE Trading Platform — LLM Prompt Templates

Structured prompts for Claude API calls.
Phase 4 implementation.
"""

from __future__ import annotations

PRE_MARKET_SYSTEM_PROMPT = """You are an expert NSE (National Stock Exchange of India) 
pre-market analyst. You analyze market conditions and provide structured 
trading recommendations for an algorithmic trading system.

Your analysis must be objective, data-driven, and risk-aware.
Always consider: global cues, sector rotation, FII/DII flows, 
option chain data, and technical levels.

Respond in JSON format only."""


PRE_MARKET_USER_TEMPLATE = """Analyze the following market context for today's 
trading session ({date}) and provide recommendations.

## Market Context
{market_context}

## Required Output (JSON)
{{
  "market_bias": "BULLISH|BEARISH|NEUTRAL",
  "confidence": 0.0-1.0,
  "nifty_levels": {{
    "support_1": float,
    "support_2": float, 
    "resistance_1": float,
    "resistance_2": float
  }},
  "sector_picks": [
    {{"sector": "name", "bias": "LONG|SHORT|AVOID", "reason": "..."}}
  ],
  "stock_watchlist": [
    {{"symbol": "name", "bias": "LONG|SHORT", "reason": "..."}}
  ],
  "risk_warnings": ["warning1", "warning2"],
  "trade_approval": true|false,
  "max_position_size_pct": 0.0-100.0,
  "reasoning": "2-3 sentence summary"
}}"""


EOD_REVIEW_TEMPLATE = """Review today's trading performance and provide 
insights for tomorrow.

## Today's Trades
{trades_summary}

## Market Performance
{market_summary}

## Required Output (JSON)
{{
  "performance_rating": "GOOD|AVERAGE|POOR",
  "key_observations": ["obs1", "obs2"],
  "mistakes_identified": ["mistake1"],
  "tomorrow_adjustments": ["adj1", "adj2"],
  "summary": "2-3 sentence review"
}}"""
