"""
NSE Trading Platform — Stock Universe

Nifty 50 constituents, sector mapping, and helper functions
to convert between display symbols and Fyers API format.

Update the NIFTY50 list periodically when NSE rebalances (March & September).
Last updated: March 2025.
"""

from __future__ import annotations


# ── Nifty 50 Constituents ─────────────────────────────────────────────────
# Symbol: NSE trading symbol (without exchange prefix)

NIFTY50: list[str] = [
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]


# ── Sector Mapping ────────────────────────────────────────────────────────
# Maps each Nifty 50 stock to its primary sector.
# Used by scanner for sector rotation analysis.

SECTOR_MAP: dict[str, str] = {
    "ADANIPORTS": "Infrastructure",
    "APOLLOHOSP": "Healthcare",
    "ASIANPAINT": "Consumer",
    "AXISBANK": "Banking",
    "BAJAJ-AUTO": "Auto",
    "BAJAJFINSV": "Financial Services",
    "BAJFINANCE": "Financial Services",
    "BHARTIARTL": "Telecom",
    "BPCL": "Energy",
    "BRITANNIA": "FMCG",
    "CIPLA": "Pharma",
    "COALINDIA": "Metals & Mining",
    "DRREDDY": "Pharma",
    "EICHERMOT": "Auto",
    "ETERNAL": "Consumer",
    "GRASIM": "Cement",
    "HCLTECH": "IT",
    "HDFC": "Financial Services",
    "HDFCBANK": "Banking",
    "HDFCLIFE": "Insurance",
    "HEROMOTOCO": "Auto",
    "HINDALCO": "Metals & Mining",
    "HINDUNILVR": "FMCG",
    "ICICIBANK": "Banking",
    "INDUSINDBK": "Banking",
    "INFY": "IT",
    "ITC": "FMCG",
    "JSWSTEEL": "Metals & Mining",
    "KOTAKBANK": "Banking",
    "LT": "Infrastructure",
    "M&M": "Auto",
    "MARUTI": "Auto",
    "NESTLEIND": "FMCG",
    "NTPC": "Energy",
    "ONGC": "Energy",
    "POWERGRID": "Energy",
    "RELIANCE": "Conglomerate",
    "SBILIFE": "Insurance",
    "SBIN": "Banking",
    "SHRIRAMFIN": "Financial Services",
    "SUNPHARMA": "Pharma",
    "TATACONSUM": "FMCG",
    "TATAMOTORS": "Auto",
    "TATASTEEL": "Metals & Mining",
    "TCS": "IT",
    "TECHM": "IT",
    "TITAN": "Consumer",
    "TRENT": "Consumer",
    "ULTRACEMCO": "Cement",
    "WIPRO": "IT",
}


# ── Sector Index Symbols (Fyers format) ───────────────────────────────────

SECTOR_INDICES: dict[str, str] = {
    "NIFTY BANK": "NSE:NIFTYBANK-INDEX",
    "NIFTY IT": "NSE:NIFTYIT-INDEX",
    "NIFTY PHARMA": "NSE:NIFTYPHARMA-INDEX",
    "NIFTY AUTO": "NSE:NIFTYAUTO-INDEX",
    "NIFTY FMCG": "NSE:NIFTYFMCG-INDEX",
    "NIFTY METAL": "NSE:NIFTYMETAL-INDEX",
    "NIFTY REALTY": "NSE:NIFTYREALTY-INDEX",
    "NIFTY ENERGY": "NSE:NIFTYENERGY-INDEX",
    "NIFTY INFRA": "NSE:NIFTYINFRA-INDEX",
    "NIFTY PSE": "NSE:NIFTYPSE-INDEX",
    "NIFTY MEDIA": "NSE:NIFTYMEDIA-INDEX",
    "NIFTY FIN SERVICE": "NSE:NIFTYFINSERVICE-INDEX",
    "NIFTY PRIVATE BANK": "NSE:NIFTYPRIVATEBANK-INDEX",
}


# ── All Unique Sectors ────────────────────────────────────────────────────

ALL_SECTORS: list[str] = sorted(set(SECTOR_MAP.values()))


# ── Helper Functions ──────────────────────────────────────────────────────

def to_fyers_symbol(symbol: str, exchange: str = "NSE", segment: str = "EQ") -> str:
    """
    Convert a plain symbol to Fyers API format.

    Parameters
    ----------
    symbol : str
        Plain symbol like 'RELIANCE'.
    exchange : str
        Exchange prefix, default 'NSE'.
    segment : str
        Segment suffix, default 'EQ' for equity.
        Use 'FUT' for futures, 'CE'/'PE' for options.

    Returns
    -------
    str
        Fyers format: 'NSE:RELIANCE-EQ'

    Examples
    --------
    >>> to_fyers_symbol("RELIANCE")
    'NSE:RELIANCE-EQ'
    >>> to_fyers_symbol("NIFTY", segment="FUT")
    'NSE:NIFTY-FUT'
    """
    return f"{exchange}:{symbol}-{segment}"


def from_fyers_symbol(fyers_symbol: str) -> tuple[str, str, str]:
    """
    Parse a Fyers symbol into components.

    Parameters
    ----------
    fyers_symbol : str
        e.g. 'NSE:RELIANCE-EQ'

    Returns
    -------
    tuple[str, str, str]
        (exchange, symbol, segment) e.g. ('NSE', 'RELIANCE', 'EQ')
    """
    exchange, rest = fyers_symbol.split(":", 1)
    parts = rest.rsplit("-", 1)
    symbol = parts[0]
    segment = parts[1] if len(parts) > 1 else "EQ"
    return exchange, symbol, segment


def get_nifty50_fyers_symbols(segment: str = "EQ") -> list[str]:
    """
    Return all Nifty 50 symbols in Fyers format.

    Parameters
    ----------
    segment : str
        Default 'EQ'. Use 'FUT' for futures.

    Returns
    -------
    list[str]
        e.g. ['NSE:ADANIPORTS-EQ', 'NSE:APOLLOHOSP-EQ', ...]
    """
    return [to_fyers_symbol(s, segment=segment) for s in NIFTY50]


def get_sector_stocks(sector: str) -> list[str]:
    """
    Return all Nifty 50 stocks belonging to a sector.

    Parameters
    ----------
    sector : str
        Sector name like 'Banking', 'IT', 'Pharma'.

    Returns
    -------
    list[str]
        Plain symbols, e.g. ['AXISBANK', 'HDFCBANK', 'ICICIBANK', ...]
    """
    return [s for s, sec in SECTOR_MAP.items() if sec == sector]


def get_stock_sector(symbol: str) -> str:
    """
    Look up the sector for a given stock symbol.

    Parameters
    ----------
    symbol : str
        Plain symbol like 'RELIANCE'.

    Returns
    -------
    str
        Sector name, or 'Unknown' if not in Nifty 50.
    """
    return SECTOR_MAP.get(symbol, "Unknown")


def get_sector_index_symbol(sector_name: str) -> str | None:
    """
    Get the Fyers index symbol for a sector name.

    Parameters
    ----------
    sector_name : str
        e.g. 'NIFTY BANK', 'NIFTY IT'

    Returns
    -------
    str or None
        Fyers symbol or None if not found.
    """
    return SECTOR_INDICES.get(sector_name)


def get_universe(universe_name: str = "NIFTY50") -> list[str]:
    """
    Return stock list for a named universe.

    Parameters
    ----------
    universe_name : str
        'NIFTY50' (only supported universe currently).

    Returns
    -------
    list[str]
        Plain symbol list.

    Raises
    ------
    ValueError
        If universe_name is not recognized.
    """
    if universe_name == "NIFTY50":
        return NIFTY50.copy()
    else:
        raise ValueError(
            f"Unknown universe: {universe_name}. Supported: NIFTY50"
        )


def validate_symbol(symbol: str) -> bool:
    """
    Check if a symbol is in the known Nifty 50 universe.

    Parameters
    ----------
    symbol : str
        Plain symbol like 'RELIANCE'.

    Returns
    -------
    bool
    """
    return symbol in NIFTY50


def get_sector_summary() -> dict[str, int]:
    """
    Return a dict of sector → number of stocks.

    Returns
    -------
    dict[str, int]
        e.g. {'Banking': 5, 'IT': 5, ...}
    """
    summary: dict[str, int] = {}
    for sector in SECTOR_MAP.values():
        summary[sector] = summary.get(sector, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
