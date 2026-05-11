"""
Generate this week's sector rotation picks.

Loads the trained model, fetches the latest 12 weeks of sector data,
runs inference, and outputs which sectors to go long and short.

Usage:
    python predict.py
    python predict.py --n-seeds 5    # Average across 5 seeds for stability
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from config import Config
from tickers import get_all_tickers, get_sector_names, BENCHMARK
from model import build_model
from preprocess import compute_features


def get_latest_data(cfg, lookback_weeks=12):
    """Download the most recent weeks of sector data."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return None, None

    tickers = get_all_tickers()
    # Grab extra weeks to account for feature computation warmup
    extra = 10
    total_weeks_needed = lookback_weeks + extra

    print(f"Downloading latest {total_weeks_needed} weeks of data...")

    ohlcv_wide = {"open": {}, "high": {}, "low": {}, "close": {}, "volume": {}}

    for ticker in tickers:
        data = yf.download(ticker, period=f"{total_weeks_needed + 2}wk",
                           interval="1wk", auto_adjust=True, progress=False)
        if data is None or len(data) == 0:
            print(f"  WARNING: No data for {ticker}")
            continue

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.columns = ["open", "high", "low", "close", "volume"]

        for field in ohlcv_wide:
            ohlcv_wide[field][ticker] = data[field]

    # Align all tickers to same dates
    import pandas as pd
    for field in ohlcv_wide:
        ohlcv_wide[field] = pd.DataFrame(ohlcv_wide[field]).sort_index().ffill()

    dates = ohlcv_wide["close"].index.tolist()
    print(f"  Got {len(dates)} weeks ({dates[0].date()} to {dates[-1].date()})")

    # Compute features
    features = compute_features(ohlcv_wide, tickers)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Take the last lookback_weeks
    if features.shape[0] > lookback_weeks:
        features = features[-lookback_weeks:]
        dates = dates[-lookback_weeks:]

    return features, dates


def predict(cfg, n_seeds=1):
    """Generate this week's sector picks."""
    data_dir = cfg.data.processed_data_dir
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        print("No metadata found. Run the full pipeline first: python run.py")
        return

    metadata = json.loads(meta_path.read_text())
    tickers = metadata["tickers"]
    sector_tickers = metadata["sector_tickers"]
    spy_index = metadata["spy_index"]
    n_sectors = len(sector_tickers)
    names = get_sector_names()

    # Get normalization stats from training data
    features_full = np.load(data_dir / "features.npy")
    train_end = int(features_full.shape[0] * cfg.train.train_ratio)
    train_data = features_full[:train_end]
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0

    # Get latest data
    lookback = cfg.model.lookback_len
    latest_features, latest_dates = get_latest_data(cfg, lookback_weeks=lookback)

    if latest_features is None:
        return

    # Normalize using training stats
    latest_features = (latest_features - mean) / std
    np.nan_to_num(latest_features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare input tensor
    x = torch.from_numpy(latest_features).unsqueeze(0)  # (1, lookback, N, F)

    cfg.model.num_variates = latest_features.shape[1]
    cfg.model.n_features = latest_features.shape[2]

    # Run prediction (optionally across multiple seeds)
    ckpt_path = cfg.train.checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"No trained model at {ckpt_path}. Run python run.py first.")
        return

    all_preds = []

    for seed_idx in range(n_seeds):
        seed = cfg.train.seed + seed_idx * 137
        torch.manual_seed(seed)

        model = build_model(cfg, spy_index=spy_index, n_sectors=n_sectors)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            pred = model(x).squeeze(0).numpy()  # (n_sectors,)
        all_preds.append(pred)

    # Average predictions across seeds
    avg_pred = np.mean(all_preds, axis=0)

    # Rank sectors
    ranked_indices = np.argsort(avg_pred)[::-1]  # High to low

    # Display results
    today = datetime.now().strftime("%Y-%m-%d")
    data_through = latest_dates[-1].strftime("%Y-%m-%d") if hasattr(latest_dates[-1], 'strftime') else str(latest_dates[-1])[:10]

    print(f"\n{'='*60}")
    print(f"  SECTOR ROTATION PICKS")
    print(f"  Generated: {today}")
    print(f"  Data through: {data_through}")
    if n_seeds > 1:
        print(f"  Averaged across {n_seeds} model seeds")
    print(f"{'='*60}")

    print(f"\n  📈 LONG (top 3 predicted outperformers):")
    print(f"  {'':>4s} {'Ticker':>6s}  {'Sector':<28s}  {'Predicted':>10s}")
    print(f"  {'':>4s} {'------':>6s}  {'----------------------------':<28s}  {'----------':>10s}")
    for rank, idx in enumerate(ranked_indices[:3]):
        ticker = sector_tickers[idx]
        sector = names.get(ticker, "")
        pred_val = avg_pred[idx] * 100
        print(f"  {rank+1:>3d}. {ticker:>6s}  {sector:<28s}  {pred_val:>+9.3f}%")

    print(f"\n  📉 SHORT (bottom 3 predicted underperformers):")
    print(f"  {'':>4s} {'Ticker':>6s}  {'Sector':<28s}  {'Predicted':>10s}")
    print(f"  {'':>4s} {'------':>6s}  {'----------------------------':<28s}  {'----------':>10s}")
    for rank, idx in enumerate(ranked_indices[-3:][::-1]):
        ticker = sector_tickers[idx]
        sector = names.get(ticker, "")
        pred_val = avg_pred[idx] * 100
        print(f"  {rank+1:>3d}. {ticker:>6s}  {sector:<28s}  {pred_val:>+9.3f}%")

    print(f"\n  Full ranking (best to worst):")
    print(f"  {'Rank':>4s}  {'Ticker':>6s}  {'Sector':<28s}  {'Predicted':>10s}")
    print(f"  {'----':>4s}  {'------':>6s}  {'----------------------------':<28s}  {'----------':>10s}")
    for rank, idx in enumerate(ranked_indices):
        ticker = sector_tickers[idx]
        sector = names.get(ticker, "")
        pred_val = avg_pred[idx] * 100
        marker = " ◀ LONG" if rank < 3 else " ◀ SHORT" if rank >= n_sectors - 3 else ""
        print(f"  {rank+1:>4d}  {ticker:>6s}  {sector:<28s}  {pred_val:>+9.3f}%{marker}")

    print(f"\n  Strategy: Equal-weight long top 3, short bottom 3")
    print(f"  Hold for 1 week, then re-run this script")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate weekly sector picks")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds to average (more = more stable)")
    args = parser.parse_args()

    cfg = Config()
    predict(cfg, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
