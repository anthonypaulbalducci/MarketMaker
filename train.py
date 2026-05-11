"""
Training loop for sector rotation model.

Loss: MSE on predicted vs actual relative returns across all sectors.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from dataset import load_and_split_data
from model import build_model


def get_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    device, max_grad_norm, log_interval) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)       # (B, n_sectors)
        loss = loss_fn(pred, y)
        loss.backward()

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "time": time.time() - t0}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)    # (n_samples, n_sectors)
    targets = np.concatenate(all_targets, axis=0)

    # Rank correlation: does the model rank sectors correctly?
    rank_corrs = []
    for t in range(len(preds)):
        if np.std(preds[t]) > 1e-10 and np.std(targets[t]) > 1e-10:
            corr = np.corrcoef(preds[t], targets[t])[0, 1]
            if not np.isnan(corr):
                rank_corrs.append(corr)

    mean_rank_corr = np.mean(rank_corrs) if rank_corrs else 0.0

    # Top-N accuracy: is the model's top sector actually a top performer?
    top_n = 3
    correct_top = 0
    for t in range(len(preds)):
        pred_top = set(np.argsort(preds[t])[-top_n:])
        actual_top = set(np.argsort(targets[t])[-top_n:])
        correct_top += len(pred_top & actual_top)
    top_accuracy = correct_top / (len(preds) * top_n) if len(preds) > 0 else 0.0

    return {
        "loss": total_loss / max(n_batches, 1),
        "rank_corr": float(mean_rank_corr),
        "top3_accuracy": float(top_accuracy),
        "predictions": preds,
        "targets": targets,
    }


def train(cfg: Config):
    """Full training pipeline."""
    device = get_device(cfg.train.device)
    print(f"Device: {device}")

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    data = load_and_split_data(cfg)

    model = build_model(
        cfg,
        spy_index=data["spy_index"],
        sector_indices=None,
        n_sectors=data["n_sectors"],
    )
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    total_steps = cfg.train.epochs * len(data["train_loader"])
    warmup_steps = cfg.train.warmup_epochs * len(data["train_loader"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6,
    )

    early_stopping = EarlyStopping(patience=cfg.train.patience)
    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_rank_corr": [], "val_top3_acc": [],
    }

    print(f"\n{'='*70}")
    print(f"Training for {cfg.train.epochs} epochs | "
          f"Variates: {cfg.model.num_variates} | Sectors: {data['n_sectors']}")
    print(f"Lookback: {cfg.model.lookback_len} weeks | "
          f"d_model: {cfg.model.d_model} | lstm_layers: {cfg.model.lstm_layers}")
    print(f"{'='*70}\n")

    for epoch in range(1, cfg.train.epochs + 1):
        train_m = train_one_epoch(
            model, data["train_loader"], optimizer, scheduler,
            loss_fn, device, cfg.train.max_grad_norm, 0,
        )
        val_m = evaluate(model, data["val_loader"], loss_fn, device)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_rank_corr"].append(val_m["rank_corr"])
        history["val_top3_acc"].append(val_m["top3_accuracy"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_m['loss']:.6f} | "
                  f"Val: {val_m['loss']:.6f} | "
                  f"RankCorr: {val_m['rank_corr']:.4f} | "
                  f"Top3Acc: {val_m['top3_accuracy']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, val_m["loss"],
                            cfg.train.checkpoint_dir / "best_model.pt")

        if early_stopping(val_m["loss"]):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest val_loss={best_val_loss:.6f} at epoch {best_epoch}")

    # Load best and test
    load_checkpoint(cfg.train.checkpoint_dir / "best_model.pt", model)
    model = model.to(device)

    test_m = evaluate(model, data["test_loader"], loss_fn, device)
    print(f"\nTest Results:")
    print(f"  Loss:      {test_m['loss']:.6f}")
    print(f"  RankCorr:  {test_m['rank_corr']:.4f}")
    print(f"  Top3Acc:   {test_m['top3_accuracy']:.4f}")

    # Save
    results = {
        "best_epoch": best_epoch,
        "test_loss": test_m["loss"],
        "test_rank_corr": test_m["rank_corr"],
        "test_top3_acc": test_m["top3_accuracy"],
        "history": history,
        "sector_tickers": data["sector_tickers"],
    }

    results_dir = cfg.train.results_dir
    (results_dir / "training_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    np.savez(
        results_dir / "test_predictions.npz",
        predictions=test_m["predictions"],
        targets=test_m["targets"],
        spy_returns=data["spy_returns"],
        sector_tickers=np.array(data["sector_tickers"]),
    )

    print(f"Results saved to {results_dir}/")
    return model, history, test_m


def main():
    parser = argparse.ArgumentParser(description="Train sector rotation model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.device:
        cfg.train.device = args.device

    train(cfg)


if __name__ == "__main__":
    main()
