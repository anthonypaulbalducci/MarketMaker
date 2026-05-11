"""
Preprocess weekly sector ETF data into feature tensors.

Input:  weekly_sector_data.parquet (from download_data.py)
Output: features.npy — (weeks, n_sectors+1, n_features)
        relative_returns.npy — (weeks, n_sectors) — sector return minus SPY
        metadata.json

Features per variate (adapted for weekly):
0. returns — weekly close-to-close return
1. log_returns — log(close/prev_close)
2. weekly_range — (high - low) / close
3. open_close_return — (close - open) / open
4. log_volume — log(volume + 1)
5. relative_volume — volume / 4-week rolling mean
6. rsi_4 — 4-week RSI (analogous to RSI-14 on daily)
7. price_vs_sma4 — close / 4-week SMA - 1
8. realized_vol_4 — 4-week rolling std of returns
9. return_rank — cross-sectional rank of weekly return

Usage:
    python preprocess.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from tickers import get_all_tickers, BENCHMARK


def compute_rsi(returns: pd.Series, period: int = 4) -> pd.Series:
    """RSI adapted for weekly (4-week period)."""
    gains = returns.clip(lower=0)
    losses = (-returns).clip(lower=0)
    avg_gain = gains.rolling(period, min_periods=1).mean()
    avg_loss = losses.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0 - 0.5  # Center around 0


def compute_features(ohlcv_wide: dict, tickers: list) -> np.ndarray:
    """
    Compute features for each ticker from weekly OHLCV.

    Args:
        ohlcv_wide: dict with keys 'open','high','low','close','volume',
                    each a DataFrame indexed by date with ticker columns
        tickers: ordered list of tickers

    Returns:
        features_3d: (n_weeks, n_tickers, n_features) array
    """
    n_days = len(ohlcv_wide["close"])
    n_tickers = len(tickers)
    n_features = 10
    features = np.full((n_days, n_tickers, n_features), np.nan, dtype=np.float32)

    for i, ticker in enumerate(tickers):
        c = ohlcv_wide["close"][ticker]
        o = ohlcv_wide["open"][ticker]
        h = ohlcv_wide["high"][ticker]
        lo = ohlcv_wide["low"][ticker]
        v = ohlcv_wide["volume"][ticker].astype(float)

        # 0: returns
        ret = c.pct_change()
        features[:, i, 0] = ret.values

        # 1: log returns
        log_ret = np.log(c / c.shift(1))
        features[:, i, 1] = log_ret.values

        # 2: weekly range
        rng = (h - lo) / c
        features[:, i, 2] = rng.values

        # 3: open-to-close return
        oc = (c - o) / o
        features[:, i, 3] = oc.values

        # 4: log volume
        log_vol = np.log(v + 1)
        features[:, i, 4] = log_vol.values

        # 5: relative volume (vs 4-week rolling mean)
        vol_ma = v.rolling(4, min_periods=1).mean()
        rel_vol = v / (vol_ma + 1e-10)
        features[:, i, 5] = rel_vol.values

        # 6: RSI (4-week)
        rsi = compute_rsi(ret, period=4)
        features[:, i, 6] = rsi.values

        # 7: price vs 4-week SMA
        sma = c.rolling(4, min_periods=1).mean()
        vs_sma = c / sma - 1
        features[:, i, 7] = vs_sma.values

        # 8: realized vol (4-week rolling std)
        rvol = ret.rolling(4, min_periods=2).std()
        features[:, i, 8] = rvol.values

    # 9: cross-sectional return rank (computed across tickers per week)
    returns_matrix = features[:, :, 0]  # (weeks, tickers)
    for t in range(n_days):
        row = returns_matrix[t]
        if not np.all(np.isnan(row)):
            valid = ~np.isnan(row)
            ranks = np.zeros_like(row)
            ranks[valid] = pd.Series(row[valid]).rank(pct=True).values - 0.5
            features[t, :, 9] = ranks

    return features


def preprocess(cfg: Config):
    """Full preprocessing pipeline."""
    data_path = cfg.data.raw_data_dir / "weekly_sector_data.parquet"
    if not data_path.exists():
        print(f"No data at {data_path}. Run download_data.py first.")
        sys.exit(1)

    print("Loading weekly sector data...")
    raw = pd.read_parquet(data_path)
    raw["date"] = pd.to_datetime(raw["date"])

    tickers = get_all_tickers()
    available = raw["ticker"].unique().tolist()
    missing = [t for t in tickers if t not in available]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
        tickers = [t for t in tickers if t in available]

    print(f"Tickers: {tickers}")
    print(f"Weeks: {raw['date'].nunique()}")

    # Pivot to wide format
    ohlcv_wide = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivot = raw.pivot_table(index="date", columns="ticker", values=field)
        pivot = pivot[tickers]  # Consistent ordering
        pivot = pivot.sort_index()
        pivot = pivot.ffill()   # Forward-fill gaps
        ohlcv_wide[field] = pivot

    dates = ohlcv_wide["close"].index.tolist()
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    # Compute features
    print("\nComputing features...")
    features_3d = compute_features(ohlcv_wide, tickers)
    print(f"Feature tensor: {features_3d.shape} (weeks, tickers, features)")

    # Trim leading NaN rows
    valid_mask = ~np.isnan(features_3d).any(axis=(1, 2))
    first_valid = np.argmax(valid_mask)
    features_3d = features_3d[first_valid:]
    dates = dates[first_valid:]
    print(f"After trimming: {features_3d.shape}")

    np.nan_to_num(features_3d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute relative returns (sector return - SPY return)
    spy_idx = tickers.index(BENCHMARK)
    sector_indices = [i for i, t in enumerate(tickers) if t != BENCHMARK]
    sector_tickers = [t for t in tickers if t != BENCHMARK]

    # Weekly returns for all tickers (feature index 0)
    all_returns = features_3d[:, :, 0]  # (weeks, n_tickers)
    spy_returns = all_returns[:, spy_idx]  # (weeks,)

    # Relative returns: sector - SPY
    relative_returns = all_returns[:, sector_indices] - spy_returns[:, np.newaxis]
    # Shape: (weeks, n_sectors)

    print(f"\nRelative returns: {relative_returns.shape} ({len(sector_tickers)} sectors)")
    for i, t in enumerate(sector_tickers):
        mean_rel = relative_returns[:, i].mean() * 100
        print(f"  {t}: mean relative return = {mean_rel:+.3f}% per week")

    # Save
    output_dir = cfg.data.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "features.npy", features_3d.astype(np.float32))
    np.save(output_dir / "relative_returns.npy", relative_returns.astype(np.float32))
    np.save(output_dir / "spy_returns.npy", spy_returns.astype(np.float32))

    feature_names = list(cfg.features.features)
    metadata = {
        "shape": list(features_3d.shape),
        "tickers": tickers,
        "sector_tickers": sector_tickers,
        "spy_index": spy_idx,
        "sector_indices": sector_indices,
        "dates": [str(d) for d in dates],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_variates": len(tickers),
        "n_sectors": len(sector_tickers),
        "n_weeks": len(dates),
        "frequency": "weekly",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(f"\nSaved to {output_dir}/")
    print(f"  features.npy: {features_3d.shape}")
    print(f"  relative_returns.npy: {relative_returns.shape}")
    print(f"  spy_returns.npy: {spy_returns.shape}")
    print(f"  SPY index: {spy_idx}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess weekly sector data")
    args = parser.parse_args()
    cfg = Config()
    preprocess(cfg)


if __name__ == "__main__":
    main()
