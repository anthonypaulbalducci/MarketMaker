"""
Multi-seed ensemble for sector rotation.

Trains N models with different seeds, averages their sector predictions,
then runs the long/short strategy on the averaged signal.

Usage:
    python ensemble.py
    python ensemble.py --n-seeds 10
    python ensemble.py --n-seeds 5 --max-days 52
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

from config import Config
from dataset import load_and_split_data
from model import build_model
from train import (
    get_device, EarlyStopping, save_checkpoint,
    train_one_epoch, evaluate,
)


def train_single_seed(cfg, seed, data, device):
    """Train one model with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(cfg, spy_index=data["spy_index"],
                        n_sectors=data["n_sectors"])
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                   weight_decay=cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(data["train_loader"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=cfg.train.patience)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.train.epochs + 1):
        train_m = train_one_epoch(model, data["train_loader"], optimizer,
                                   scheduler, loss_fn, device, cfg.train.max_grad_norm, 0)
        val_m = evaluate(model, data["val_loader"], loss_fn, device)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"      Epoch {epoch:3d} | Train: {train_m['loss']:.6f} | "
                  f"Val: {val_m['loss']:.6f} | RankCorr: {val_m['rank_corr']:.4f}")

        if early_stopping(val_m["loss"]):
            print(f"      Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    test_m = evaluate(model, data["test_loader"], loss_fn, device)
    return test_m


def run_ensemble(cfg, seeds, max_days=0):
    """Train multiple models and ensemble."""
    device = get_device(cfg.train.device)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}\n")

    data = load_and_split_data(cfg)
    sector_tickers = data["sector_tickers"]
    n_sectors = data["n_sectors"]

    all_predictions = []
    all_targets = None
    seed_results = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")

        t0 = time.time()
        test_m = train_single_seed(cfg, seed, data, device)
        elapsed = time.time() - t0

        all_predictions.append(test_m["predictions"])
        if all_targets is None:
            all_targets = test_m["targets"]

        print(f"  Seed {seed}: RankCorr={test_m['rank_corr']:.4f} "
              f"Top3Acc={test_m['top3_accuracy']:.4f} ({elapsed:.0f}s)")

        seed_results.append({
            "seed": seed,
            "rank_corr": test_m["rank_corr"],
            "top3_acc": test_m["top3_accuracy"],
            "time": elapsed,
        })

    # Stack and truncate
    pred_matrix = np.stack(all_predictions, axis=0)  # (n_seeds, n_weeks, n_sectors)

    if max_days > 0 and pred_matrix.shape[1] > max_days:
        pred_matrix = pred_matrix[:, :max_days, :]
        all_targets = all_targets[:max_days]
        print(f"\nLimiting evaluation to first {max_days} test weeks")

    ensemble_pred = pred_matrix.mean(axis=0)  # (n_weeks, n_sectors)
    n_weeks = ensemble_pred.shape[0]

    # Ensemble metrics
    rank_corrs = []
    for t in range(n_weeks):
        if np.std(ensemble_pred[t]) > 1e-10 and np.std(all_targets[t]) > 1e-10:
            c = np.corrcoef(ensemble_pred[t], all_targets[t])[0, 1]
            if not np.isnan(c):
                rank_corrs.append(c)
    ensemble_rank_corr = np.mean(rank_corrs) if rank_corrs else 0.0

    top_n = 3
    correct = 0
    for t in range(n_weeks):
        pred_top = set(np.argsort(ensemble_pred[t])[-top_n:])
        actual_top = set(np.argsort(all_targets[t])[-top_n:])
        correct += len(pred_top & actual_top)
    ensemble_top3 = correct / (n_weeks * top_n)

    # Strategy returns
    n_long, n_short = 3, 3
    weekly_returns = np.zeros(n_weeks)
    for t in range(n_weeks):
        ranks = np.argsort(ensemble_pred[t])
        long_ret = all_targets[t, ranks[-n_long:]].mean()
        short_ret = all_targets[t, ranks[:n_short]].mean()
        weekly_returns[t] = long_ret - short_ret

    sharpe = 0.0
    if weekly_returns.std() > 0:
        sharpe = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)

    # Per-seed strategy sharpes (on truncated data)
    individual_sharpes = []
    for s in range(pred_matrix.shape[0]):
        s_returns = np.zeros(n_weeks)
        for t in range(n_weeks):
            ranks = np.argsort(pred_matrix[s, t])
            s_returns[t] = all_targets[t, ranks[-n_long:]].mean() - all_targets[t, ranks[:n_short]].mean()
        s_sharpe = s_returns.mean() / s_returns.std() * np.sqrt(52) if s_returns.std() > 0 else 0.0
        individual_sharpes.append(s_sharpe)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS ({len(seeds)} seeds, {n_weeks} weeks)")
    print(f"{'='*60}")
    print(f"Rank Correlation: {ensemble_rank_corr:.4f}")
    print(f"Top-3 Accuracy:   {ensemble_top3:.4f} (random: {3/11:.4f})")
    print(f"Sharpe:           {sharpe:.2f}")
    print(f"Win rate:         {(weekly_returns > 0).mean()*100:.0f}%")
    print(f"Total return:     {np.cumsum(weekly_returns)[-1]*100:.1f}%")

    print(f"\nPer-seed:")
    for sr, sh in zip(seed_results, individual_sharpes):
        print(f"  Seed {sr['seed']}: RankCorr={sr['rank_corr']:.4f} "
              f"Top3={sr['top3_acc']:.4f} Sharpe={sh:.2f}")
    print(f"  Ensemble:  RankCorr={ensemble_rank_corr:.4f} "
          f"Top3={ensemble_top3:.4f} Sharpe={sharpe:.2f}")

    # Save
    results_dir = cfg.train.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(results_dir / "test_predictions.npz",
             predictions=ensemble_pred, targets=all_targets,
             individual_predictions=pred_matrix,
             spy_returns=data["spy_returns"],
             sector_tickers=np.array(sector_tickers))

    ensemble_results = {
        "ensemble": True, "n_seeds": len(seeds), "seeds": seeds,
        "ensemble_rank_corr": float(ensemble_rank_corr),
        "ensemble_top3_acc": float(ensemble_top3),
        "ensemble_sharpe": float(sharpe),
        "seed_results": seed_results,
        "individual_sharpes": individual_sharpes,
        "sector_tickers": sector_tickers,
        "history": {"val_rank_corr": [sr["rank_corr"] for sr in seed_results]},
    }
    (results_dir / "training_results.json").write_text(
        json.dumps(ensemble_results, indent=2, default=str))

    _plot_ensemble(ensemble_pred, all_targets, pred_matrix, seeds,
                   individual_sharpes, sharpe, seed_results,
                   sector_tickers, weekly_returns, data, results_dir)

    print(f"\nResults saved to {results_dir}/")


def _plot_ensemble(ensemble_pred, targets, pred_matrix, seeds,
                   individual_sharpes, ensemble_sharpe, seed_results,
                   sector_tickers, weekly_returns, data, save_dir):

    n_seeds = pred_matrix.shape[0]
    n_weeks = ensemble_pred.shape[0]
    n_long, n_short = 3, 3

    cum_strategy = np.cumsum(weekly_returns)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f"Sector Rotation Ensemble ({len(seeds)} seeds)",
                 fontsize=14, fontweight="bold")

    # 1. Cumulative returns: ensemble vs individual seeds
    ax = axes[0]
    for s in range(n_seeds):
        s_returns = np.zeros(n_weeks)
        for t in range(n_weeks):
            ranks = np.argsort(pred_matrix[s, t])
            s_returns[t] = targets[t, ranks[-n_long:]].mean() - targets[t, ranks[:n_short]].mean()
        ax.plot(np.cumsum(s_returns), color="lightgray", alpha=0.5, linewidth=0.8)

    ax.plot(cum_strategy, label=f"Ensemble (Sharpe: {ensemble_sharpe:.2f})",
            alpha=0.9, linewidth=2.0, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    win_rate = (weekly_returns > 0).mean() * 100
    ax.annotate(f"Win rate: {win_rate:.0f}%", xy=(0.02, 0.82),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Ensemble vs Individual Seeds (gray)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Per-seed Sharpe
    ax = axes[1]
    colors = ["green" if s > 0 else "red" for s in individual_sharpes]
    ax.bar(range(len(seeds)), individual_sharpes, color=colors, alpha=0.7, edgecolor="white")
    ax.axhline(y=ensemble_sharpe, color="steelblue", linewidth=2, linestyle="--",
               label=f"Ensemble: {ensemble_sharpe:.2f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"Seed\n{s}" for s in seeds], fontsize=8)
    ax.set_ylabel("Sharpe Ratio"); ax.set_title("Individual Seed Sharpe vs Ensemble")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # 3. Seed agreement on top-3 picks
    ax = axes[2]
    agreement = np.zeros((n_seeds, n_seeds))
    for i in range(n_seeds):
        for j in range(n_seeds):
            overlap = 0
            for t in range(n_weeks):
                top_i = set(np.argsort(pred_matrix[i, t])[-3:])
                top_j = set(np.argsort(pred_matrix[j, t])[-3:])
                overlap += len(top_i & top_j) / 3.0
            agreement[i, j] = overlap / n_weeks

    sns.heatmap(agreement, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=[f"S{s}" for s in seeds],
                yticklabels=[f"S{s}" for s in seeds],
                ax=ax, vmin=0.2, vmax=1.0, center=0.5)
    ax.set_title("Seed Agreement (fraction overlap in top-3 picks)")

    plt.tight_layout()
    plt.savefig(save_dir / "ensemble_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Standalone cumulative returns
    fig, ax = plt.subplots(figsize=(14, 6))
    for s in range(n_seeds):
        s_returns = np.zeros(n_weeks)
        for t in range(n_weeks):
            ranks = np.argsort(pred_matrix[s, t])
            s_returns[t] = targets[t, ranks[-n_long:]].mean() - targets[t, ranks[:n_short]].mean()
        ax.plot(np.cumsum(s_returns), color="lightgray", alpha=0.5, linewidth=0.8)
    ax.plot(cum_strategy, label=f"Ensemble (Sharpe: {ensemble_sharpe:.2f})",
            alpha=0.9, linewidth=2.0, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Sector Rotation Ensemble")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ensemble_results.png and cumulative_returns.png")


def main():
    parser = argparse.ArgumentParser(description="Sector rotation ensemble")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.train.device = args.device

    seeds = args.seeds or [42, 123, 456, 789, 1024][:args.n_seeds]
    if args.n_seeds > 5 and not args.seeds:
        seeds = [42 + i * 137 for i in range(args.n_seeds)]

    run_ensemble(cfg, seeds, max_days=args.max_days)


if __name__ == "__main__":
    main()
