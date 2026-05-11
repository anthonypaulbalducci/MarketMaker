"""
Optuna hyperparameter tuning for sector rotation.

Search space covers TFT architecture + AdamW optimization. Each trial fits
the model on the last N walk-forward folds (val-only — test slices are
untouched to prevent leakage) and returns the mean val rank correlation.

Usage:
    python tune.py                              # 30 trials, last 4 folds
    python tune.py --n-trials 100 --timeout 7200
    python tune.py --n-folds 6 --epochs-per-fold 40
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from config import Config
from train import get_device
from walk_forward import fit_one_fold, _normalize_fold


SEARCH_SPACE_DESCRIPTION = """
Architecture:
  d_model       ∈ {32, 64, 128}
  n_heads       ∈ {2, 4, 8}            (constrained: d_model % n_heads == 0)
  lstm_layers   ∈ {1, 2, 3}
  dropout       ∈ [0.0, 0.4]
  lookback_len  ∈ {8, 12, 16, 24}
Optimization:
  learning_rate ∈ [1e-5, 1e-2]  (log)
  weight_decay  ∈ [1e-6, 1e-2]  (log)
  batch_size    ∈ {16, 32, 64}
  warmup_epochs ∈ {0, 3, 5, 10}
"""


def sample_params(trial):
    """Sample one hyperparameter configuration."""
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lookback_len": trial.suggest_categorical("lookback_len", [8, 12, 16, 24]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_epochs": trial.suggest_categorical("warmup_epochs", [0, 3, 5, 10]),
    }


def apply_params(cfg: Config, params: dict) -> Config:
    """Return a Config copy with sampled params applied."""
    new_cfg = deepcopy(cfg)
    new_cfg.model.d_model = params["d_model"]
    new_cfg.model.n_heads = params["n_heads"]
    new_cfg.model.lstm_layers = params["lstm_layers"]
    new_cfg.model.dropout = params["dropout"]
    new_cfg.model.lookback_len = params["lookback_len"]
    new_cfg.train.learning_rate = params["learning_rate"]
    new_cfg.train.weight_decay = params["weight_decay"]
    new_cfg.train.batch_size = params["batch_size"]
    new_cfg.train.warmup_epochs = params["warmup_epochs"]
    return new_cfg


def build_tuning_folds(T: int, train_weeks: int, val_weeks: int,
                       test_weeks: int, n_folds: int):
    """Reproduce walk_forward.py's fold layout and keep only the last N.

    test_weeks is part of the stride so the layout matches walk_forward.py,
    but the test slice is never read by the tuner.
    """
    total_window = train_weeks + val_weeks + test_weeks
    folds = []
    start = 0
    while start + total_window <= T:
        train_end = start + train_weeks
        val_end = train_end + val_weeks
        folds.append({"train": (start, train_end), "val": (train_end, val_end)})
        start += test_weeks
    return folds[-n_folds:] if n_folds > 0 else folds


def make_objective(cfg: Config, features, relative_returns, n_sectors,
                   spy_index, dates, device, folds, epochs_per_fold,
                   patience, base_seed):
    """Build the Optuna objective closure."""

    def objective(trial):
        import optuna

        params = sample_params(trial)
        trial_cfg = apply_params(cfg, params)
        lookback = trial_cfg.model.lookback_len

        fold_corrs = []
        for fold_idx, fold in enumerate(folds):
            t_s, t_e = fold["train"]
            v_s, v_e = fold["val"]

            train_feat, val_feat, _ = _normalize_fold(
                features, t_s, t_e, v_s, v_e, v_e, v_e,
            )
            train_tgt = relative_returns[t_s:t_e]
            val_tgt = relative_returns[v_s:v_e]

            try:
                fold_out = fit_one_fold(
                    cfg=trial_cfg,
                    train_feat=train_feat, val_feat=val_feat, test_feat=None,
                    train_tgt=train_tgt, val_tgt=val_tgt, test_tgt=None,
                    lookback=lookback, device=device,
                    n_sectors=n_sectors, spy_index=spy_index,
                    epochs=epochs_per_fold, patience=patience,
                    seed=base_seed + fold_idx * 31,
                    fold_idx=fold_idx,
                )
            except ValueError as e:
                # e.g. dataset too short for sampled lookback
                raise optuna.TrialPruned(f"fold {fold_idx} invalid: {e}")

            fold_corrs.append(fold_out["best_val_rank_corr"])

            # Report running mean to enable pruning after at least one fold
            running_mean = float(np.mean(fold_corrs))
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_corrs))

    return objective


def tune(
    cfg: Config,
    n_trials: int = 30,
    timeout: int = None,
    n_folds: int = 4,
    train_weeks: int = 400,
    val_weeks: int = 52,
    test_weeks: int = 26,
    epochs_per_fold: int = 30,
    patience: int = 8,
    study_name: str = "sector_rotation",
    storage: str = None,
):
    """Run hyperparameter search and save best config."""
    import optuna

    device = get_device(cfg.train.device)
    print(f"Device: {device}")
    print(SEARCH_SPACE_DESCRIPTION)

    data_dir = cfg.data.processed_data_dir
    features = np.load(data_dir / "features.npy")
    relative_returns = np.load(data_dir / "relative_returns.npy")
    metadata = json.loads((data_dir / "metadata.json").read_text())

    T, N, F_dim = features.shape
    n_sectors = relative_returns.shape[1]
    spy_index = metadata["spy_index"]
    dates = metadata["dates"]

    cfg.model.num_variates = N
    cfg.model.n_features = F_dim

    folds = build_tuning_folds(T, train_weeks, val_weeks, test_weeks, n_folds)
    if len(folds) == 0:
        raise RuntimeError(
            f"No valid folds: T={T} < total_window={train_weeks + val_weeks + test_weeks}")

    print(f"Tuning on {len(folds)} fold(s); {epochs_per_fold} epochs each")
    for i, f in enumerate(folds):
        t_s, t_e = f["train"]
        v_s, v_e = f["val"]
        print(f"  Fold {i+1}: train {dates[t_s]}–{dates[t_e-1]}  "
              f"val {dates[v_s]}–{dates[v_e-1]}")

    results_dir = cfg.train.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if storage is None:
        storage = f"sqlite:///{results_dir / 'optuna_study.db'}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=cfg.train.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = make_objective(
        cfg, features, relative_returns, n_sectors, spy_index, dates,
        device, folds, epochs_per_fold, patience, base_seed=cfg.train.seed,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False, gc_after_trial=True)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Tuning finished in {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean val rank_corr: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Top 10
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\nTop {min(10, len(completed))} trials:")
    for t in completed[:10]:
        print(f"  #{t.number}: rank_corr={t.value:.4f}  {t.params}")

    best_path = results_dir / "best_params.json"
    best_path.write_text(json.dumps({
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_folds": len(folds),
        "epochs_per_fold": epochs_per_fold,
        "elapsed_sec": elapsed,
    }, indent=2))
    print(f"\nSaved best params to {best_path}")
    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for sector rotation")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Wall-clock budget in seconds (overrides --n-trials when hit)")
    parser.add_argument("--n-folds", type=int, default=4,
                        help="Number of most-recent walk-forward folds to use")
    parser.add_argument("--train-weeks", type=int, default=400)
    parser.add_argument("--val-weeks", type=int, default=52)
    parser.add_argument("--test-weeks", type=int, default=26,
                        help="Used only to keep fold stride consistent with walk_forward.py")
    parser.add_argument("--epochs-per-fold", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--study-name", type=str, default="sector_rotation")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL; defaults to sqlite in results dir")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.train.device = args.device

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    tune(
        cfg,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_folds=args.n_folds,
        train_weeks=args.train_weeks,
        val_weeks=args.val_weeks,
        test_weeks=args.test_weeks,
        epochs_per_fold=args.epochs_per_fold,
        patience=args.patience,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
