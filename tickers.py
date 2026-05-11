"""
Sector ETF tickers for rotation strategy.

11 S&P 500 sector ETFs + SPY as benchmark.
SPY is the benchmark — relative returns are sector minus SPY.
"""

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLB": "Materials",
}

BENCHMARK = "SPY"


def get_all_tickers() -> list:
    """All tickers: SPY first, then sectors alphabetically."""
    sectors = sorted(SECTOR_ETFS.keys())
    return [BENCHMARK] + sectors


def get_sector_names() -> dict:
    """Map ticker to sector name."""
    return {**SECTOR_ETFS, BENCHMARK: "S&P 500"}


def get_sector_tickers() -> list:
    """Just the sector ETFs (no SPY)."""
    return sorted(SECTOR_ETFS.keys())
