"""
NSE Trading Platform — FastAPI Control Server

Provides HTTP endpoints for the dashboard to control the engine:
  /health, /status, /kill, /pause, /resume, /flatten

Implementation: After Strategy 1 backtest is validated.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="NSE Trading Engine", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "engine": "nse-trading-platform"}


# Additional endpoints: /status, /kill, /pause, /resume, /flatten
# Implementation: After backtest validation.
