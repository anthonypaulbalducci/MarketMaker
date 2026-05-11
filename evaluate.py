"""
Evaluation for sector rotation strategy.

Strategy: Each week, go long top-3 sectors and short bottom-3 sectors
based on model's predicted relative returns.

Generates:
- Training curves (loss, rank correlation, top-3 accuracy)
- Sector rotation strategy cumulative returns
- Per-sector prediction quality
- Sector allocation heatmap over time
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


def plot_training_curves(results: dict, save_dir: Path):
    """Training curves for sector rotation."""
    history = results["history"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sector Rotation Training History", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", alpha=0.8)
    ax.plot(history["val_loss"], label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["val_rank_corr"], color="green", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rank Correlation")
    ax.set_title("Cross-Sector Rank Correlation")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["val_top3_acc"], color="purple", alpha=0.8)
    random_baseline = 3.0 / 11.0
    ax.axhline(y=random_baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Random ({random_baseline:.1%})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Top-3 Accuracy")
    ax.set_title("Top-3 Sector Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.text(0.5, 0.5,
            f"Test Rank Corr: {results.get('test_rank_corr', 0):.4f}\n"
            f"Test Top3 Acc: {results.get('test_top3_acc', 0):.4f}\n"
            f"Best Epoch: {results.get('best_epoch', 0)}",
            transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.set_title("Test Results"); ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved training_curves.png")


def plot_strategy(predictions, targets, sector_tickers, spy_returns,
                  save_dir, val_end=0, max_days=0):
    """
    Sector rotation strategy simulation.

    Strategy: each week, long top-3 predicted sectors, short bottom-3.
    Equal weight within each group. The strategy is market-neutral
    (longs and shorts cancel out the market exposure).
    """
    n_weeks, n_sectors = predictions.shape

    if max_days > 0 and n_weeks > max_days:
        predictions = predictions[:max_days]
        targets = targets[:max_days]
        if spy_returns is not None:
            spy_returns = spy_returns[:max_days]
        n_weeks = max_days

    n_long = 3
    n_short = 3

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    fig.suptitle("Sector Rotation Strategy", fontsize=14, fontweight="bold")

    # --- Panel 1: Strategy cumulative returns ---
    ax = axes[0]

    weekly_returns = np.zeros(n_weeks)
    for t in range(n_weeks):
        pred_ranks = np.argsort(predictions[t])  # Low to high
        short_idx = pred_ranks[:n_short]
        long_idx = pred_ranks[-n_long:]

        # P&L: average of long sectors minus average of short sectors (relative returns)
        long_return = targets[t, long_idx].mean()
        short_return = targets[t, short_idx].mean()
        weekly_returns[t] = long_return - short_return

    cum_strategy = np.cumsum(weekly_returns)

    # SPY buy & hold for comparison (use actual SPY returns aligned to test period)
    if spy_returns is not None and len(spy_returns) >= val_end + 12 + n_weeks:
        test_spy = spy_returns[val_end + 12 : val_end + 12 + n_weeks]
        cum_spy = np.cumsum(test_spy)
    else:
        test_spy = np.zeros(n_weeks)
        cum_spy = np.zeros(n_weeks)

    # Equal-weight all sectors (naive benchmark)
    equal_weight_returns = targets.mean(axis=1)  # Average relative return
    cum_equal = np.cumsum(equal_weight_returns)

    ax.plot(cum_strategy, label="Long/Short Top-3 vs Bottom-3",
            alpha=0.9, linewidth=1.5, color="steelblue")
    ax.plot(cum_spy, label="SPY Buy & Hold",
            alpha=0.7, linewidth=1.2, color="orange")
    ax.plot(cum_equal, label="Equal Weight All Sectors",
            alpha=0.5, linewidth=1.0, color="gray", linestyle="--")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    if weekly_returns.std() > 0:
        sharpe = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)
        ax.annotate(f"Annualized Sharpe: {sharpe:.2f}", xy=(0.02, 0.92),
                    xycoords="axes fraction", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    hit_rate = (weekly_returns > 0).mean() * 100
    ax.annotate(f"Win rate: {hit_rate:.0f}%", xy=(0.02, 0.82),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Long Top-3 / Short Bottom-3 Sectors")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Panel 2: Per-sector prediction quality ---
    ax = axes[1]

    per_sector_corr = []
    for i, ticker in enumerate(sector_tickers):
        corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        per_sector_corr.append(corr)

    colors = ["green" if c > 0 else "red" for c in per_sector_corr]
    bars = ax.bar(range(len(sector_tickers)), per_sector_corr,
                  color=colors, alpha=0.7, edgecolor="white")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(sector_tickers)))
    ax.set_xticklabels(sector_tickers, rotation=45, ha="right")
    ax.set_ylabel("Correlation"); ax.set_title("Per-Sector Prediction Correlation")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Sector allocation heatmap ---
    ax = axes[2]

    # Show which sectors the model picks each week
    n_show = min(52, n_weeks)  # Show up to 1 year
    allocation = np.zeros((len(sector_tickers), n_show))
    for t in range(n_show):
        pred_ranks = np.argsort(predictions[t])
        for idx in pred_ranks[-n_long:]:
            allocation[idx, t] = 1     # Long
        for idx in pred_ranks[:n_short]:
            allocation[idx, t] = -1    # Short

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(allocation, ax=ax, cmap=cmap, center=0,
                yticklabels=sector_tickers,
                xticklabels=[str(i) if i % 4 == 0 else "" for i in range(n_show)],
                cbar_kws={"label": "Position (1=Long, -1=Short)"})
    ax.set_xlabel("Week"); ax.set_title(f"Sector Allocations (first {n_show} weeks)")

    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved cumulative_returns.png")
    if weekly_returns.std() > 0:
        print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Win rate: {hit_rate:.0f}%")
    print(f"  Mean weekly return: {weekly_returns.mean()*100:.3f}%")
    print(f"  Strategy total: {cum_strategy[-1]*100:.1f}%")


def generate_all_plots(cfg: Config, max_days: int = 0):
    """Generate all plots from saved results."""
    results_dir = cfg.train.results_dir

    results_path = results_dir / "training_results.json"
    if not results_path.exists():
        print(f"No results at {results_path}. Run train.py first.")
        return

    results = json.loads(results_path.read_text())
    plot_training_curves(results, results_dir)

    pred_path = results_dir / "test_predictions.npz"
    if pred_path.exists():
        data = np.load(pred_path, allow_pickle=True)
        predictions = data["predictions"]
        targets = data["targets"]
        spy_returns = data["spy_returns"] if "spy_returns" in data else None
        sector_tickers = list(data["sector_tickers"]) if "sector_tickers" in data else results.get("sector_tickers", [])

        # Get val_end from metadata
        meta_path = cfg.data.processed_data_dir / "metadata.json"
        val_end = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            n_weeks = meta["n_weeks"]
            val_end = int(n_weeks * (cfg.train.train_ratio + cfg.train.val_ratio))

        plot_strategy(predictions, targets, sector_tickers, spy_returns,
                      results_dir, val_end, max_days)
    else:
        print("No predictions found.")

    print(f"\nAll plots saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sector rotation")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--max-days", type=int, default=0,
                        help="Limit to first N test weeks (0 = all)")
    args = parser.parse_args()

    cfg = Config()
    if args.results_dir:
        cfg.train.results_dir = Path(args.results_dir)

    generate_all_plots(cfg, max_days=args.max_days)


if __name__ == "__main__":
    main()
