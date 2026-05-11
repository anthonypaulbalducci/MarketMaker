"""
Dataset for weekly sector rotation.

Input:  (lookback_len, n_variates, n_features) — all tickers including SPY
Target: (n_sectors,) — relative returns for sector ETFs (sector - SPY)

The model sees all tickers (SPY + sectors) as input, but only predicts
the relative returns of the sector ETFs.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class SectorRotationDataset(Dataset):
    """Weekly sector rotation dataset."""

    def __init__(self, features, relative_returns, lookback_len):
        self.features = features.astype(np.float32)
        self.targets = relative_returns.astype(np.float32)
        self.lookback_len = lookback_len
        self.n_samples = len(features) - lookback_len

        if self.n_samples <= 0:
            raise ValueError(f"Data too short: {len(features)} weeks for lookback={lookback_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.lookback_len]  # (lookback, N, F)
        y = self.targets[idx + self.lookback_len]          # (n_sectors,)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_and_split_data(cfg: Config) -> dict:
    """Load processed features and create train/val/test datasets."""
    data_dir = cfg.data.processed_data_dir
    features = np.load(data_dir / "features.npy")
    relative_returns = np.load(data_dir / "relative_returns.npy")
    spy_returns = np.load(data_dir / "spy_returns.npy")
    metadata = json.loads((data_dir / "metadata.json").read_text())

    tickers = metadata["tickers"]
    sector_tickers = metadata["sector_tickers"]
    feature_names = metadata["feature_names"]
    spy_index = metadata["spy_index"]
    dates = metadata["dates"]

    T, N, F = features.shape
    n_sectors = relative_returns.shape[1]
    print(f"Loaded: {T} weeks, {N} variates, {F} features, {n_sectors} sectors")
    print(f"Sectors: {sector_tickers}")

    # Chronological split
    train_end = int(T * cfg.train.train_ratio)
    val_end = train_end + int(T * cfg.train.val_ratio)

    train_features = features[:train_end]
    val_features = features[train_end:val_end]
    test_features = features[val_end:]

    train_targets = relative_returns[:train_end]
    val_targets = relative_returns[train_end:val_end]
    test_targets = relative_returns[val_end:]

    print(f"\nTrain: {len(train_features)} weeks ({dates[0]} to {dates[train_end-1]})")
    print(f"Val:   {len(val_features)} weeks ({dates[train_end]} to {dates[val_end-1]})")
    print(f"Test:  {len(test_features)} weeks ({dates[val_end]} to {dates[-1]})")

    # Z-score normalization
    if cfg.features.normalization == "zscore":
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        std[std < 1e-8] = 1.0
        train_features = (train_features - mean) / std
        val_features = (val_features - mean) / std
        test_features = (test_features - mean) / std
        print("Applied z-score normalization")

    for arr in [train_features, val_features, test_features]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    lookback = cfg.model.lookback_len

    train_ds = SectorRotationDataset(train_features, train_targets, lookback)
    val_ds = SectorRotationDataset(val_features, val_targets, lookback)
    test_ds = SectorRotationDataset(test_features, test_targets, lookback)

    print(f"Samples — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    cfg.model.num_variates = N
    cfg.model.n_features = F

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tickers": tickers,
        "sector_tickers": sector_tickers,
        "feature_names": feature_names,
        "spy_index": spy_index,
        "n_sectors": n_sectors,
        "num_variates": N,
        "n_features": F,
        "dates": dates,
        "spy_returns": spy_returns,
        "train_end": train_end,
        "val_end": val_end,
    }
