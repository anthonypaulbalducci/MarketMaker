"""
Walk-forward training for sector rotation.

Trains on a rolling window of weekly data, tests on the next period,
slides forward. Each fold gets fresh normalization from its own
training window — no data leakage.

Usage:
    python walk_forward.py
    python walk_forward.py --train-weeks 400 --test-weeks 26
    python walk_forward.py --n-seeds 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from config import Config
from dataset import SectorRotationDataset
from model import build_model
from train import (
    get_device, EarlyStopping,
    train_one_epoch, evaluate,
)


def fit_one_fold(
    cfg: Config,
    train_feat: np.ndarray,
    val_feat: np.ndarray,
    test_feat,
    train_tgt: np.ndarray,
    val_tgt: np.ndarray,
    test_tgt,
    lookback: int,
    device,
    n_sectors: int,
    spy_index: int,
    epochs: int,
    patience: int,
    seed: int,
    trial=None,
    fold_idx: int = 0,
) -> dict:
    """Train one fold and report metrics.

    test_feat / test_tgt may be None during hyperparameter tuning, in which
    case the test loader is skipped and only val metrics are returned.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SectorRotationDataset(train_feat, train_tgt, lookback)
    val_ds = SectorRotationDataset(val_feat, val_tgt, lookback)

    train_ld = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    test_ld = None
    if test_feat is not None and test_tgt is not None:
        test_ds = SectorRotationDataset(test_feat, test_tgt, lookback)
        test_ld = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg, spy_index=spy_index, n_sectors=n_sectors)
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg.train.learning_rate,
                                   weight_decay=cfg.train.weight_decay)
    total_steps = max(epochs * len(train_ld), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_val_rank_corr = -float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_ld, optimizer, scheduler,
                        loss_fn, device, cfg.train.max_grad_norm, 0)
        val_m = evaluate(model, val_ld, loss_fn, device)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_val_rank_corr = val_m["rank_corr"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping(val_m["loss"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    out = {
        "best_val_loss": float(best_val_loss),
        "best_val_rank_corr": float(best_val_rank_corr),
        "predictions": None,
        "targets": None,
        "test_rank_corr": None,
        "test_top3_accuracy": None,
    }

    if test_ld is not None:
        test_m = evaluate(model, test_ld, loss_fn, device)
        out["predictions"] = test_m["predictions"]
        out["targets"] = test_m["targets"]
        out["test_rank_corr"] = float(test_m["rank_corr"])
        out["test_top3_accuracy"] = float(test_m["top3_accuracy"])

    return out


def _normalize_fold(features, t_s, t_e, v_s, v_e, te_s, te_e):
    """Z-score using the fold's training stats. te_s/te_e may equal v_e (no test slice)."""
    train_feat = features[t_s:t_e].copy()
    val_feat = features[v_s:v_e].copy()
    test_feat = features[te_s:te_e].copy() if te_e > te_s else None

    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0)
    std[std < 1e-8] = 1.0

    train_feat = (train_feat - mean) / std
    val_feat = (val_feat - mean) / std
    if test_feat is not None:
        test_feat = (test_feat - mean) / std

    arrays = [train_feat, val_feat] + ([test_feat] if test_feat is not None else [])
    for arr in arrays:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return train_feat, val_feat, test_feat


def walk_forward(
    cfg: Config,
    train_weeks: int = 400,
    val_weeks: int = 52,
    test_weeks: int = 26,
    epochs_per_fold: int = 80,
    patience: int = 12,
    n_seeds: int = 1,
):
    """Walk-forward training and evaluation for sector rotation."""
    device = get_device(cfg.train.device)
    lookback = cfg.model.lookback_len
    total_window = train_weeks + val_weeks + test_weeks

    # Load full dataset
    data_dir = cfg.data.processed_data_dir
    features = np.load(data_dir / "features.npy")
    relative_returns = np.load(data_dir / "relative_returns.npy")
    spy_returns = np.load(data_dir / "spy_returns.npy")
    metadata = json.loads((data_dir / "metadata.json").read_text())

    T, N, F_dim = features.shape
    n_sectors = relative_returns.shape[1]
    sector_tickers = metadata["sector_tickers"]
    spy_index = metadata["spy_index"]
    dates = metadata["dates"]

    cfg.model.num_variates = N
    cfg.model.n_features = F_dim

    print(f"Device: {device}")
    print(f"Dataset: {T} weeks, {N} variates, {F_dim} features, {n_sectors} sectors")
    print(f"Walk-forward: {train_weeks}w train, {val_weeks}w val, {test_weeks}w test")
    print(f"Seeds per fold: {n_seeds}")

    # Calculate folds
    folds = []
    start = 0
    while start + total_window <= T:
        train_end = start + train_weeks
        val_end = train_end + val_weeks
        test_end = min(val_end + test_weeks, T)
        folds.append({
            "train": (start, train_end),
            "val": (train_end, val_end),
            "test": (val_end, test_end),
        })
        start += test_weeks

    print(f"Total folds: {len(folds)}\n")

    all_preds = []
    all_targets = []
    all_dates = []
    fold_results = []

    for fold_idx, fold in enumerate(folds):
        t_s, t_e = fold["train"]
        v_s, v_e = fold["val"]
        te_s, te_e = fold["test"]

        print(f"{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(folds)}  "
              f"Train: {dates[t_s]}–{dates[t_e-1]}  "
              f"Test: {dates[te_s]}–{dates[te_e-1]}")

        train_feat, val_feat, test_feat = _normalize_fold(
            features, t_s, t_e, v_s, v_e, te_s, te_e)

        train_tgt = relative_returns[t_s:t_e]
        val_tgt = relative_returns[v_s:v_e]
        test_tgt = relative_returns[te_s:te_e]

        # Multi-seed for this fold
        fold_preds = []
        fold_target = None
        t0 = time.time()

        for seed_idx in range(n_seeds):
            seed = cfg.train.seed + seed_idx * 137 + fold_idx * 31

            fold_out = fit_one_fold(
                cfg=cfg,
                train_feat=train_feat, val_feat=val_feat, test_feat=test_feat,
                train_tgt=train_tgt, val_tgt=val_tgt, test_tgt=test_tgt,
                lookback=lookback, device=device,
                n_sectors=n_sectors, spy_index=spy_index,
                epochs=epochs_per_fold, patience=patience, seed=seed,
                fold_idx=fold_idx,
            )
            fold_preds.append(fold_out["predictions"])
            fold_target = fold_out["targets"]

            if n_seeds > 1:
                print(f"    Seed {seed_idx+1}: RankCorr={fold_out['test_rank_corr']:.4f} "
                      f"Top3={fold_out['test_top3_accuracy']:.4f}")

        fold_time = time.time() - t0

        # Average predictions across seeds for this fold
        if n_seeds > 1:
            fold_pred = np.mean(fold_preds, axis=0)
        else:
            fold_pred = fold_preds[0]

        # Compute fold metrics on averaged predictions
        n_long, n_short = 3, 3
        fold_returns = []
        for t in range(len(fold_pred)):
            ranks = np.argsort(fold_pred[t])
            ret = fold_target[t, ranks[-n_long:]].mean() - fold_target[t, ranks[:n_short]].mean()
            fold_returns.append(ret)
        fold_returns = np.array(fold_returns)

        fold_sharpe = 0.0
        if fold_returns.std() > 0:
            fold_sharpe = fold_returns.mean() / fold_returns.std() * np.sqrt(52)

        # Rank correlation
        rcs = []
        for t in range(len(fold_pred)):
            if np.std(fold_pred[t]) > 1e-10 and np.std(fold_target[t]) > 1e-10:
                c = np.corrcoef(fold_pred[t], fold_target[t])[0, 1]
                if not np.isnan(c):
                    rcs.append(c)
        fold_rank_corr = np.mean(rcs) if rcs else 0.0

        print(f"  Fold {fold_idx+1}: Sharpe={fold_sharpe:.2f} "
              f"RankCorr={fold_rank_corr:.4f} "
              f"WinRate={(fold_returns > 0).mean()*100:.0f}% ({fold_time:.0f}s)\n")

        all_preds.append(fold_pred)
        all_targets.append(fold_target)
        test_dates = dates[te_s + lookback: te_s + lookback + len(fold_pred)]
        all_dates.extend(test_dates)

        fold_results.append({
            "fold": fold_idx + 1,
            "train_dates": [dates[t_s], dates[t_e-1]],
            "test_dates": [dates[te_s], dates[te_e-1]],
            "sharpe": float(fold_sharpe),
            "rank_corr": float(fold_rank_corr),
            "win_rate": float((fold_returns > 0).mean()),
            "n_test_weeks": len(fold_pred),
            "time": fold_time,
        })

    # Combine all folds
    combined_preds = np.concatenate(all_preds, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)
    n_total = len(combined_preds)

    # Overall strategy
    n_long, n_short = 3, 3
    overall_returns = np.zeros(n_total)
    for t in range(n_total):
        ranks = np.argsort(combined_preds[t])
        overall_returns[t] = (combined_targets[t, ranks[-n_long:]].mean() -
                              combined_targets[t, ranks[:n_short]].mean())

    overall_sharpe = 0.0
    if overall_returns.std() > 0:
        overall_sharpe = overall_returns.mean() / overall_returns.std() * np.sqrt(52)

    print(f"{'='*60}")
    print(f"WALK-FORWARD RESULTS ({len(folds)} folds, {n_total} test weeks)")
    print(f"{'='*60}")
    print(f"Sharpe:     {overall_sharpe:.2f}")
    print(f"Win rate:   {(overall_returns > 0).mean()*100:.0f}%")
    print(f"Total return: {np.cumsum(overall_returns)[-1]*100:.1f}%")
    print(f"\nPer-fold:")
    for fr in fold_results:
        print(f"  Fold {fr['fold']:2d}: Sharpe={fr['sharpe']:+.2f}  "
              f"RankCorr={fr['rank_corr']:.3f}  "
              f"WinRate={fr['win_rate']*100:.0f}%  "
              f"({fr['test_dates'][0]} to {fr['test_dates'][1]})")

    # Save
    results_dir = cfg.train.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(results_dir / "test_predictions.npz",
             predictions=combined_preds, targets=combined_targets,
             sector_tickers=np.array(sector_tickers),
             spy_returns=spy_returns)

    wf_results = {
        "walk_forward": True,
        "n_folds": len(folds),
        "n_seeds_per_fold": n_seeds,
        "total_test_weeks": n_total,
        "overall_sharpe": float(overall_sharpe),
        "fold_results": fold_results,
        "sector_tickers": sector_tickers,
        "test_rank_corr": float(np.mean([fr["rank_corr"] for fr in fold_results])),
        "test_top3_acc": 0.0,
        "best_epoch": 0,
        "history": {"val_rank_corr": [fr["rank_corr"] for fr in fold_results]},
    }
    (results_dir / "training_results.json").write_text(
        json.dumps(wf_results, indent=2, default=str))

    _plot_walk_forward(combined_preds, combined_targets, overall_returns,
                       fold_results, sector_tickers, overall_sharpe, results_dir)

    print(f"\nResults saved to {results_dir}/")


def _plot_walk_forward(preds, targets, returns, fold_results,
                       sector_tickers, sharpe, save_dir):

    n_total = len(returns)
    cum_strategy = np.cumsum(returns)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("Walk-Forward Sector Rotation", fontsize=14, fontweight="bold")

    # 1. Cumulative returns with fold boundaries
    ax = axes[0]
    ax.plot(cum_strategy, label=f"Long/Short Top-3 (Sharpe: {sharpe:.2f})",
            alpha=0.9, linewidth=1.5, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    cum_idx = 0
    for fr in fold_results:
        cum_idx += fr["n_test_weeks"]
        if cum_idx < n_total:
            ax.axvline(x=cum_idx, color="gray", linestyle=":", alpha=0.3)

    win_rate = (returns > 0).mean() * 100
    ax.annotate(f"Win rate: {win_rate:.0f}%", xy=(0.02, 0.82),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns (vertical lines = fold boundaries)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Per-fold Sharpe
    ax = axes[1]
    fold_nums = [fr["fold"] for fr in fold_results]
    fold_sharpes = [fr["sharpe"] for fr in fold_results]
    colors = ["green" if s > 0 else "red" for s in fold_sharpes]
    ax.bar(fold_nums, fold_sharpes, color=colors, alpha=0.7, edgecolor="white")
    ax.axhline(y=sharpe, color="steelblue", linewidth=2, linestyle="--",
               label=f"Overall: {sharpe:.2f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Fold"); ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Per Fold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # 3. Per-sector prediction correlation across all folds
    ax = axes[2]
    n_sectors = preds.shape[1]
    per_sector_corr = []
    for i in range(n_sectors):
        c = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        per_sector_corr.append(c if not np.isnan(c) else 0.0)

    colors = ["green" if c > 0 else "red" for c in per_sector_corr]
    ax.bar(range(len(sector_tickers)), per_sector_corr, color=colors,
           alpha=0.7, edgecolor="white")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(sector_tickers)))
    ax.set_xticklabels(sector_tickers, rotation=45, ha="right")
    ax.set_ylabel("Correlation"); ax.set_title("Per-Sector Prediction Correlation (all folds)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "walk_forward_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Standalone cumulative
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cum_strategy, label=f"Sector Rotation (Sharpe: {sharpe:.2f})",
            alpha=0.9, linewidth=1.5, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Walk-Forward Sector Rotation Strategy")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved walk_forward_results.png and cumulative_returns.png")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward sector rotation")
    parser.add_argument("--train-weeks", type=int, default=400,
                        help="Training window in weeks (~8 years)")
    parser.add_argument("--val-weeks", type=int, default=52,
                        help="Validation window in weeks (1 year)")
    parser.add_argument("--test-weeks", type=int, default=26,
                        help="Test window in weeks (6 months)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Seeds per fold (ensemble within each fold)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.train.device = args.device

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    walk_forward(cfg, train_weeks=args.train_weeks, val_weeks=args.val_weeks,
                 test_weeks=args.test_weeks, epochs_per_fold=args.epochs,
                 patience=args.patience, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
