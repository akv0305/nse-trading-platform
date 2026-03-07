# ═══════════════════════════════════════════════════════════════════════════
# NSE Trading Platform — Docker Image
# Multi-stage build for production deployment on Hostinger VPS
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p storage/logs backtest/reports

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the trading engine
CMD ["python", "engine/run_engine.py"]

# ── Alternative run targets ───────────────────────────────────────────────
# Dashboard:   docker run -p 8501:8501 nse-platform streamlit run dashboard/app.py --server.port 8501
# Engine API:  docker run -p 8100:8100 nse-platform python engine/run_engine.py
# Backtest:    docker run nse-platform python scripts/run_backtest.py --strategy orb_vwap --start 2024-01-01 --end 2025-01-01
