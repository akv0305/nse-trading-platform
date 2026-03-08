"""
NSE Trading Platform — Step 2 Tests: Scanners

Tests for stock_scanner.scan_for_orb() and sector_analyzer.analyze_sector_rotation().
Run: pytest tests/test_step2_scanners.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.universe import NIFTY50, to_fyers_symbol


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build synthetic daily OHLCV DataFrames
# ═══════════════════════════════════════════════════════════════════════════

def _make_daily_df(
    closes: list[float],
    volumes: list[int] | None = None,
    spread_pct: float = 1.5,
) -> pd.DataFrame:
    """
    Build a daily OHLCV DataFrame from a list of close prices.
    High/Low are derived from close ± spread_pct/2.
    Open is close of previous day (or first close).
    """
    n = len(closes)
    if volumes is None:
        volumes = [100000] * n

    rows = []
    for i, c in enumerate(closes):
        half_spread = c * (spread_pct / 200.0)
        h = round(c + half_spread, 2)
        l = round(c - half_spread, 2)
        o = round(closes[i - 1] if i > 0 else c, 2)
        rows.append({
            "timestamp": 1704067200000 + i * 86400000,  # dummy epoch ms
            "open": o,
            "high": h,
            "low": l,
            "close": round(c, 2),
            "volume": volumes[i],
        })
    return pd.DataFrame(rows)


def _make_universe_data(
    n_stocks: int = 10,
    base_price: float = 1000.0,
    days: int = 25,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """
    Build a synthetic universe of Fyers symbols + their daily OHLCV data.
    Uses actual Nifty50 symbols.
    """
    symbols = [to_fyers_symbol(s) for s in NIFTY50[:n_stocks]]
    data: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols):
        # Each stock has a slightly different trend/volatility
        drift = (i - n_stocks // 2) * 0.3  # some up, some down
        prices = [base_price + drift * d + np.sin(d / 3.0) * 10 for d in range(days)]

        # Vary volume: last day gets a boost for some stocks
        vols = [100000 + (i * 5000)] * days
        if i % 3 == 0:
            vols[-1] = int(vols[-1] * 2.5)  # high relative volume

        data[sym] = _make_daily_df(prices, vols, spread_pct=1.0 + i * 0.3)

    return symbols, data


def _make_sector_scores() -> dict[str, float]:
    """Build a realistic sector scores dict."""
    return {
        "NIFTY BANK": 0.6,
        "NIFTY IT": -0.4,
        "NIFTY PHARMA": 0.3,
        "NIFTY AUTO": 0.1,
        "NIFTY FMCG": -0.2,
        "NIFTY METAL": 0.8,
        "NIFTY ENERGY": 0.5,
        "NIFTY INFRA": -0.1,
        "NIFTY FIN SERVICE": 0.4,
        "NIFTY REALTY": -0.5,
        "NIFTY MEDIA": -0.3,
        "NIFTY PSE": 0.2,
        "NIFTY PRIVATE BANK": 0.55,
    }


# ═══════════════════════════════════════════════════════════════════════════
# scan_for_orb Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScanForOrb:
    """Tests for stock_scanner.scan_for_orb()."""

    def test_returns_list(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data()
        result = scan_for_orb(symbols, data, _make_sector_scores())
        assert isinstance(result, list)

    def test_respects_top_n(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data(n_stocks=20)
        result = scan_for_orb(symbols, data, _make_sector_scores(), top_n=5)
        assert len(result) <= 5

    def test_each_candidate_has_required_keys(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data()
        result = scan_for_orb(symbols, data, _make_sector_scores())
        for candidate in result:
            assert "symbol" in candidate
            assert "score" in candidate
            assert "reason" in candidate
            assert isinstance(candidate["score"], float)
            assert isinstance(candidate["reason"], str)
            assert len(candidate["reason"]) > 0

    def test_sorted_by_score_descending(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data(n_stocks=15)
        result = scan_for_orb(symbols, data, _make_sector_scores())
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_universe(self):
        from core.scanner.stock_scanner import scan_for_orb
        result = scan_for_orb([], {}, {})
        assert result == []

    def test_empty_historical_data(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols = [to_fyers_symbol(s) for s in NIFTY50[:5]]
        result = scan_for_orb(symbols, {}, _make_sector_scores())
        assert result == []

    def test_skips_stocks_with_insufficient_data(self):
        from core.scanner.stock_scanner import scan_for_orb
        sym = to_fyers_symbol("RELIANCE")
        # Only 3 rows — less than the 6 required
        data = {sym: _make_daily_df([100, 101, 102])}
        result = scan_for_orb([sym], data, _make_sector_scores())
        assert len(result) == 0

    def test_skips_zero_volume_stocks(self):
        from core.scanner.stock_scanner import scan_for_orb
        sym = to_fyers_symbol("RELIANCE")
        data = {sym: _make_daily_df(
            [100, 101, 102, 103, 104, 105, 106],
            volumes=[0, 0, 0, 0, 0, 0, 0],
        )}
        result = scan_for_orb([sym], data, _make_sector_scores())
        assert len(result) == 0

    def test_high_volume_stock_scores_higher(self):
        """A stock with high relative volume on last day should score better."""
        from core.scanner.stock_scanner import scan_for_orb
        sym_high = to_fyers_symbol("HDFCBANK")
        sym_low = to_fyers_symbol("ICICIBANK")

        base_prices = [1000 + i * 5 for i in range(25)]

        # High vol: last day 3x average
        vols_high = [100000] * 24 + [300000]
        # Low vol: last day below average
        vols_low = [100000] * 24 + [50000]

        data = {
            sym_high: _make_daily_df(base_prices, vols_high, spread_pct=2.0),
            sym_low: _make_daily_df(base_prices, vols_low, spread_pct=2.0),
        }
        scores = {
            "NIFTY BANK": 0.5,
            "NIFTY IT": 0.5,
            "NIFTY FIN SERVICE": 0.5,
        }
        result = scan_for_orb([sym_high, sym_low], data, scores, top_n=10)

        if len(result) >= 2:
            score_high = next(c["score"] for c in result if c["symbol"] == sym_high)
            score_low = next(c["score"] for c in result if c["symbol"] == sym_low)
            assert score_high > score_low

    def test_candidate_symbols_are_from_universe(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data()
        result = scan_for_orb(symbols, data, _make_sector_scores())
        result_symbols = {c["symbol"] for c in result}
        assert result_symbols.issubset(set(symbols))

    def test_scores_between_zero_and_one(self):
        from core.scanner.stock_scanner import scan_for_orb
        symbols, data = _make_universe_data(n_stocks=20)
        result = scan_for_orb(symbols, data, _make_sector_scores())
        for c in result:
            assert 0.0 <= c["score"] <= 1.0, f"{c['symbol']} score={c['score']}"


# ═══════════════════════════════════════════════════════════════════════════
# analyze_sector_rotation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyzeSectorRotation:
    """Tests for sector_analyzer.analyze_sector_rotation()."""

    def test_returns_dict_for_all_sectors(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i * 0.5 for i in range(25)])
        sectors = {
            "NIFTY BANK": _make_daily_df([100 + i * 1.0 for i in range(25)]),
            "NIFTY IT": _make_daily_df([100 - i * 0.3 for i in range(25)]),
        }
        result = analyze_sector_rotation(sectors, nifty)
        assert "NIFTY BANK" in result
        assert "NIFTY IT" in result

    def test_result_has_required_keys(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i for i in range(25)])
        sectors = {"NIFTY BANK": _make_daily_df([100 + i * 1.5 for i in range(25)])}
        result = analyze_sector_rotation(sectors, nifty)

        for sector_name, info in result.items():
            assert "momentum" in info
            assert "trend" in info
            assert "relative_strength" in info
            assert "rotation_phase" in info

    def test_rotation_phases_valid(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i for i in range(25)])
        sectors = {
            "NIFTY BANK": _make_daily_df([100 + i * 2 for i in range(25)]),
            "NIFTY IT": _make_daily_df([100 - i * 0.5 for i in range(25)]),
            "NIFTY AUTO": _make_daily_df([100 + i * 0.01 for i in range(25)]),
        }
        result = analyze_sector_rotation(sectors, nifty)
        valid_phases = {"LEADING", "WEAKENING", "LAGGING", "IMPROVING"}
        for info in result.values():
            assert info["rotation_phase"] in valid_phases

    def test_trend_values_valid(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100] * 25)
        sectors = {
            "NIFTY BANK": _make_daily_df([100 + i * 2 for i in range(25)]),
            "NIFTY IT": _make_daily_df([100 - i * 2 for i in range(25)]),
            "NIFTY FMCG": _make_daily_df([100 + i * 0.01 for i in range(25)]),
        }
        result = analyze_sector_rotation(sectors, nifty)
        valid_trends = {"BULLISH", "BEARISH", "NEUTRAL"}
        for info in result.values():
            assert info["trend"] in valid_trends

    def test_outperformer_has_positive_rs(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i * 0.5 for i in range(25)])  # +12% over 20d
        bank = _make_daily_df([100 + i * 2.0 for i in range(25)])   # +48% over 20d
        result = analyze_sector_rotation({"NIFTY BANK": bank}, nifty)
        assert result["NIFTY BANK"]["relative_strength"] > 0

    def test_underperformer_has_negative_rs(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i * 1.0 for i in range(25)])
        it = _make_daily_df([100 - i * 0.5 for i in range(25)])
        result = analyze_sector_rotation({"NIFTY IT": it}, nifty)
        assert result["NIFTY IT"]["relative_strength"] < 0

    def test_leading_phase_conditions(self):
        """Sector beating Nifty in both long and short term → LEADING."""
        from core.scanner.sector_analyzer import analyze_sector_rotation
        # Nifty: flat
        nifty = _make_daily_df([100.0] * 25)
        # Sector: consistently rising → RS > 0, momentum > 0
        bank = _make_daily_df([100 + i * 1.5 for i in range(25)])
        result = analyze_sector_rotation({"NIFTY BANK": bank}, nifty,
                                          short_period=5, long_period=20)
        assert result["NIFTY BANK"]["rotation_phase"] == "LEADING"

    def test_lagging_phase_conditions(self):
        """Sector underperforming in both timeframes → LAGGING."""
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i * 1.0 for i in range(25)])
        # Sector: declining while Nifty rises
        it = _make_daily_df([100 - i * 1.0 for i in range(25)])
        result = analyze_sector_rotation({"NIFTY IT": it}, nifty,
                                          short_period=5, long_period=20)
        assert result["NIFTY IT"]["rotation_phase"] == "LAGGING"

    def test_empty_sector_data(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i for i in range(25)])
        result = analyze_sector_rotation({}, nifty)
        assert result == {}

    def test_insufficient_nifty_data_returns_neutral(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100])  # only 1 row
        sectors = {"NIFTY BANK": _make_daily_df([100, 105, 110])}
        result = analyze_sector_rotation(sectors, nifty)
        assert result["NIFTY BANK"]["rotation_phase"] == "LAGGING"
        assert result["NIFTY BANK"]["relative_strength"] == 0.0

    def test_momentum_is_float(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i for i in range(25)])
        sectors = {"NIFTY BANK": _make_daily_df([100 + i * 1.5 for i in range(25)])}
        result = analyze_sector_rotation(sectors, nifty)
        assert isinstance(result["NIFTY BANK"]["momentum"], float)

    def test_multiple_sectors_simultaneously(self):
        from core.scanner.sector_analyzer import analyze_sector_rotation
        nifty = _make_daily_df([100 + i * 0.5 for i in range(25)])
        sectors = {
            "NIFTY BANK": _make_daily_df([100 + i * 2.0 for i in range(25)]),
            "NIFTY IT": _make_daily_df([100 - i * 0.3 for i in range(25)]),
            "NIFTY AUTO": _make_daily_df([100 + i * 0.8 for i in range(25)]),
            "NIFTY PHARMA": _make_daily_df([100 + i * 0.4 for i in range(25)]),
            "NIFTY METAL": _make_daily_df([100 + i * 3.0 for i in range(25)]),
        }
        result = analyze_sector_rotation(sectors, nifty)
        assert len(result) == 5
        # Metal should have highest RS (strongest outperformer)
        rs_values = {k: v["relative_strength"] for k, v in result.items()}
        assert rs_values["NIFTY METAL"] == max(rs_values.values())
