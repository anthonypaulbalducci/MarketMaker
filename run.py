"""
Run the full sector rotation pipeline.

Usage:
    python run.py                    # Full pipeline
    python run.py --skip-download    # Skip data download (use existing)
    python run.py --skip-preprocess  # Skip preprocessing too
"""
import argparse
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Sector Rotation Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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
