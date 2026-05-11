"""
Download weekly OHLCV data for sector ETFs from Yahoo Finance.

Usage:
    python download_data.py
"""
import argparse
from pathlib import Path

import pandas as pd

from config import Config
from tickers import get_all_tickers, get_sector_names


def download_weekly_data(cfg: Config):
    """Download weekly OHLCV for all sector ETFs + SPY."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        return

    tickers = get_all_tickers()
    names = get_sector_names()
    start = cfg.data.start_date
    end = cfg.data.end_date
    output_dir = cfg.data.raw_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weekly data for {len(tickers)} tickers")
    print(f"Date range: {start} to {end}")
    print(f"Tickers: {tickers}\n")

    all_data = {}

    for ticker in tickers:
        print(f"  {ticker} ({names.get(ticker, '')})...", end=" ")
        try:
            data = yf.download(
                ticker, start=start, end=end,
                interval="1wk", auto_adjust=True, progress=False,
            )
            if data is not None and len(data) > 0:
                data = data[["Open", "High", "Low", "Close", "Volume"]]
                data.columns = ["open", "high", "low", "close", "volume"]
                data["ticker"] = ticker
                data.index.name = "date"
                all_data[ticker] = data
                print(f"{len(data)} weeks")
            else:
                print("No data")
        except Exception as e:
            print(f"Error: {e}")

    if not all_data:
        print("No data downloaded!")
        return

    # Combine and save
    combined = pd.concat(all_data.values(), axis=0)
    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path = output_dir / "weekly_sector_data.parquet"
    combined.to_parquet(output_path)

    n_tickers = combined["ticker"].nunique()
    n_weeks = combined["date"].nunique()
    date_range = f"{combined['date'].min().date()} to {combined['date'].max().date()}"

    print(f"\nSaved: {output_path}")
    print(f"  {n_tickers} tickers, {n_weeks} weeks")
    print(f"  Date range: {date_range}")


def main():
    parser = argparse.ArgumentParser(description="Download weekly sector data")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date

    download_weekly_data(cfg)


if __name__ == "__main__":
    main()
