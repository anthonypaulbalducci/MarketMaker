"""
Run the full sector rotation pipeline.

Usage:
    python run.py                    # Full pipeline
    python run.py --skip-download    # Skip data download (use existing)
    python run.py --skip-preprocess  # Skip preprocessing too
    python run.py --tune             # Run Optuna tuning before training
"""
import argparse
import json

from config import Config


def _apply_best_params(cfg: Config, best_params: dict) -> None:
    """Mirror tune.apply_params but in-place on cfg."""
    model_keys = {"d_model", "n_heads", "lstm_layers", "dropout", "lookback_len"}
    train_keys = {"learning_rate", "weight_decay", "batch_size", "warmup_epochs"}
    for k, v in best_params.items():
        if k in model_keys:
            setattr(cfg.model, k, v)
        elif k in train_keys:
            setattr(cfg.train, k, v)


def main():
    parser = argparse.ArgumentParser(description="Sector Rotation Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before final training")
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--tune-timeout", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.train.device = args.device

    # Step 1: Download
    if not args.skip_download and not args.skip_preprocess:
        print("\n" + "="*70)
        print("STEP 1: Download weekly sector data")
        print("="*70)
        from download_data import download_weekly_data
        download_weekly_data(cfg)

    # Step 2: Preprocess
    if not args.skip_preprocess:
        print("\n" + "="*70)
        print("STEP 2: Preprocess features")
        print("="*70)
        from preprocess import preprocess
        preprocess(cfg)

    # Step 2.5: Tune (optional)
    if args.tune:
        print("\n" + "="*70)
        print("STEP 2.5: Hyperparameter tuning (Optuna)")
        print("="*70)
        from tune import tune
        tune(cfg, n_trials=args.tune_trials, timeout=args.tune_timeout)

        best_path = cfg.train.results_dir / "best_params.json"
        best = json.loads(best_path.read_text())
        print(f"\nApplying best params from {best_path}")
        _apply_best_params(cfg, best["best_params"])

    # Step 3: Train
    print("\n" + "="*70)
    print("STEP 3: Train model")
    print("="*70)
    from train import train
    train(cfg)

    # Step 4: Evaluate
    print("\n" + "="*70)
    print("STEP 4: Evaluate")
    print("="*70)
    from evaluate import generate_all_plots
    generate_all_plots(cfg)


if __name__ == "__main__":
    main()
