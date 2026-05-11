"""
Generate picks.json for the Preceptron MarketMaker website.

This script runs the trained TFT model, identifies the top-3 long and
bottom-3 short sector picks for the upcoming week, and writes a lightweight
JSON file consumed by marketmaker.html. Optionally uploads to S3.

Usage:
    python generate_picks.py
        Writes picks.json to the current directory.

    python generate_picks.py --s3
        Writes picks.json locally AND uploads to s3://preceptron.com/picks.json

    python generate_picks.py --s3 --bucket preceptron.com --key picks.json
        Same as above with explicit overrides.

Requires:
    - A trained model (run.py must have been run at least once)
    - simulation.py and tickers.py in the same directory
    - For S3 upload: boto3 + AWS credentials configured (aws configure)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def build_payload():
    """Run the model and build the picks.json payload."""
    try:
        from simulation import get_predictions
        from tickers import get_sector_names
    except ImportError as e:
        print(f"ERROR: Could not import from simulation.py / tickers.py: {e}")
        print("Make sure you are running this from the OpusTFTTransformer directory.")
        return None

    print("Loading the trained TFT model and generating predictions...")
    longs, shorts, predictions = get_predictions()
    if longs is None:
        print("ERROR: Failed to generate predictions.")
        print("Has the model been trained? Run 'python run.py' first.")
        return None

    names = get_sector_names()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": "TFT",
        "longs": [
            {
                "ticker": t,
                "sector": names.get(t, ""),
                "predicted_return": round(float(predictions[t]), 6),
            }
            for t in longs
        ],
        "shorts": [
            {
                "ticker": t,
                "sector": names.get(t, ""),
                "predicted_return": round(float(predictions[t]), 6),
            }
            for t in shorts
        ],
        "all_predictions": {
            t: round(float(predictions[t]), 6) for t in predictions
        },
    }
    return payload


def upload_to_s3(local_path, bucket, key):
    """Upload picks.json to S3 with cache-control headers set for freshness."""
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print("ERROR: boto3 is not installed.")
        print("Install it with:  pip install boto3")
        return False

    s3 = boto3.client("s3")
    try:
        s3.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "application/json",
                "CacheControl": "no-cache, max-age=0",
            },
        )
        print(f"Uploaded -> s3://{bucket}/{key}")
        print(f"Public URL (if bucket is configured for static hosting):")
        print(f"  http://{bucket}.s3-website-us-east-1.amazonaws.com/{key}")
        print(f"  https://{bucket}/{key}  (once HTTPS/CloudFront is live)")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"S3 upload failed: {e}")
        print("Check that 'aws configure' has been run and the IAM user has S3 write access.")
        return False


def print_picks(payload):
    """Pretty-print the picks to the terminal."""
    print()
    print("=" * 60)
    print(f"  TFT SECTOR ROTATION PICKS  -  {payload['date']}")
    print("=" * 60)
    print()
    print("  LONG (calls):")
    for p in payload["longs"]:
        print(f"    {p['ticker']:>5s}  {p['sector']:<28s}  {p['predicted_return']*100:+.3f}%")
    print()
    print("  SHORT (puts):")
    for p in payload["shorts"]:
        print(f"    {p['ticker']:>5s}  {p['sector']:<28s}  {p['predicted_return']*100:+.3f}%")
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate picks.json for the Preceptron website"
    )
    parser.add_argument(
        "--output", default="picks.json",
        help="Local output path (default: picks.json)"
    )
    parser.add_argument(
        "--s3", action="store_true",
        help="Also upload to S3 after writing locally"
    )
    parser.add_argument(
        "--bucket", default="preceptron.com",
        help="S3 bucket name (default: preceptron.com)"
    )
    parser.add_argument(
        "--key", default="picks.json",
        help="S3 object key (default: picks.json)"
    )
    args = parser.parse_args()

    payload = build_payload()
    if payload is None:
        sys.exit(1)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path.resolve()}")

    print_picks(payload)

    if args.s3:
        print()
        print("Uploading to S3...")
        ok = upload_to_s3(output_path, args.bucket, args.key)
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()
