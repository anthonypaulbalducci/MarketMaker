"""
Plot simulation performance vs SPY buy-and-hold.

Reads simulation_portfolio.json, reconstructs the portfolio equity curve
from closed_trades, fetches SPY data for the same period via yfinance,
and produces a chart with two panels:

    Top:    Equity curves (your portfolio vs SPY at the same starting capital)
    Bottom: Weekly P&L bars

Output is a PNG sized for web display, styled to match the Preceptron
website (dark navy background).

Usage:
    python plot_performance.py
        Writes performance.png to the current directory.

    python plot_performance.py --output chart.png

    python plot_performance.py --upload
        Also uploads to s3://preceptron.com/performance.png so the
        website can embed it.

    python plot_performance.py --upload-picks
        Writes the latest weekly pick to picks.json and uploads to
        s3://preceptron.com/picks.json.

Requires:
    pip install matplotlib pandas yfinance
    pip install boto3   # only if using --upload / --upload-picks
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PORTFOLIO_FILE = Path("simulation_portfolio.json")


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_portfolio():
    if not PORTFOLIO_FILE.exists():
        print(f"Error: {PORTFOLIO_FILE} not found.")
        print("Run this from the same directory as simulation.py.")
        return None
    return json.loads(PORTFOLIO_FILE.read_text())


def parse_iso(s):
    """Parse an ISO timestamp string defensively."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        # Strip timezone for consistent comparisons
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except (ValueError, TypeError):
        return None


def build_equity_curve(portfolio):
    """
    Reconstruct portfolio value over time from closed trades.

    Each closed trade locks in its P&L on its close_date. We start at
    initial_capital and walk through close events chronologically.
    """
    initial = float(portfolio.get("initial_capital", 10000))
    created = parse_iso(portfolio.get("created"))
    closed = portfolio.get("closed_trades", [])

    events = []
    for t in closed:
        close_dt = parse_iso(t.get("close_date"))
        pnl = float(t.get("pnl", 0))
        if close_dt is not None:
            events.append((close_dt, pnl))

    if not events:
        return [(created or datetime.now(), initial)]

    events.sort(key=lambda x: x[0])

    # Group events by date so trades closed on the same day collapse to one point
    by_date = defaultdict(float)
    for dt, pnl in events:
        by_date[dt.date()] += pnl

    curve = [(created or events[0][0] - timedelta(days=1), initial)]
    running = initial
    for d, total_pnl in sorted(by_date.items()):
        running += total_pnl
        curve.append((datetime(d.year, d.month, d.day), running))
    return curve


def aggregate_weekly_pnl(portfolio):
    """Group closed trades by ISO week. Returns sorted dict {(year, week): pnl}."""
    closed = portfolio.get("closed_trades", [])
    weekly = defaultdict(float)
    for t in closed:
        close_dt = parse_iso(t.get("close_date"))
        if close_dt is None:
            continue
        year, week, _ = close_dt.isocalendar()
        weekly[(year, week)] += float(t.get("pnl", 0))
    return dict(sorted(weekly.items()))


def fetch_spy(start, end, initial):
    """Fetch SPY closes between start and end, scaled to the same initial capital."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — skipping SPY benchmark.")
        print("Install with:  pip install yfinance")
        return None

    try:
        # Pad both ends so we have data even if start/end fall on weekends
        spy = yf.Ticker("SPY").history(
            start=(start - timedelta(days=4)).strftime("%Y-%m-%d"),
            end=(end + timedelta(days=2)).strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
    except Exception as e:
        print(f"Failed to fetch SPY: {e}")
        return None

    if spy.empty:
        print("SPY data was empty — skipping benchmark.")
        return None

    # Drop timezone if present
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    # Normalize `start` to date-only. SPY index entries are at midnight,
    # but `start` may carry an afternoon timestamp (e.g. when setup was
    # run after market close). Without this, we'd skip the first session
    # and the SPY line would start one trading day late.
    start_day = datetime(start.year, start.month, start.day)
    on_or_after = spy[spy.index >= start_day]
    if on_or_after.empty:
        return None
    base_price = on_or_after["Close"].iloc[0]

    equity = on_or_after["Close"] / base_price * initial
    return equity


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_chart(portfolio, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    initial = float(portfolio.get("initial_capital", 10000))

    # Color palette matching the website
    BG       = "#0a0e1b"
    PANEL    = "#1c1f2e"
    GRID     = "#2a2d40"
    FG       = "#ffffff"
    MUTED    = "#8b8fa3"
    LONG     = "#3ba776"
    SHORT    = "#e25555"
    SPY_CLR  = "#c9a961"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8.5),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.32},
    )
    fig.patch.set_facecolor(BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.tick_params(colors=MUTED, which="both")
        ax.grid(True, color=GRID, alpha=0.45, linewidth=0.6)

    # ===== Top: Equity curves =====
    curve = build_equity_curve(portfolio)
    dates = [c[0] for c in curve]
    values = [c[1] for c in curve]

    final_value = values[-1]
    total_return_pct = (final_value / initial - 1) * 100
    weekly = aggregate_weekly_pnl(portfolio)
    n_weeks = len(weekly)

    if len(curve) >= 2:
        ax1.plot(
            dates, values,
            color=LONG, linewidth=2.6,
            marker="o", markersize=7, markerfacecolor=LONG, markeredgecolor=BG,
            label="MarketMaker", zorder=3,
        )
        # Light fill under the curve
        ax1.fill_between(dates, initial, values, color=LONG, alpha=0.10, zorder=1)
        ax1.axhline(y=initial, color=MUTED, linestyle="--", alpha=0.45, linewidth=1, zorder=1)

        # SPY benchmark
        spy = fetch_spy(dates[0], dates[-1], initial)
        if spy is not None and not spy.empty:
            ax1.plot(
                spy.index, spy.values,
                color=SPY_CLR, linewidth=2, alpha=0.9,
                label="SPY buy & hold", zorder=2,
            )
            spy_final = float(spy.iloc[-1])
            spy_return = (spy_final / initial - 1) * 100
        else:
            spy_return = None

        leg = ax1.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, framealpha=0.85)
        for text in leg.get_texts():
            text.set_color(FG)
    else:
        ax1.text(
            0.5, 0.5,
            "Run the simulation through at least one full week\nto see your equity curve.",
            ha="center", va="center", color=MUTED, fontsize=13,
            transform=ax1.transAxes,
        )
        spy_return = None

    ax1.set_title("Portfolio Value vs SPY Buy & Hold", color=FG, fontsize=15, pad=14, weight="bold")
    ax1.set_ylabel("Portfolio value ($)", color=MUTED, fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))

    # Summary stats overlay
    summary_lines = [
        f"Starting capital  ${initial:>10,.0f}",
        f"Current value     ${final_value:>10,.2f}",
        f"Total return      {total_return_pct:>+10.2f}%",
        f"Weeks complete    {n_weeks:>10d}",
    ]
    if spy_return is not None:
        summary_lines.append(f"SPY return        {spy_return:>+10.2f}%")
        summary_lines.append(f"Alpha             {total_return_pct - spy_return:>+10.2f}%")

    ax1.text(
        0.985, 0.04, "\n".join(summary_lines),
        transform=ax1.transAxes, ha="right", va="bottom",
        color=FG, fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL, edgecolor=GRID, alpha=0.92),
    )

    # ===== Bottom: Weekly P&L bars =====
    if weekly:
        labels = []
        values_w = []
        for (year, week), pnl in weekly.items():
            try:
                monday = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
            except ValueError:
                monday = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")
            labels.append(monday)
            values_w.append(pnl)

        colors = [LONG if v >= 0 else SHORT for v in values_w]
        x = np.arange(len(labels))
        bars = ax2.bar(x, values_w, color=colors, width=0.55, edgecolor=BG, linewidth=1.5)

        max_abs = max(abs(v) for v in values_w) if values_w else 1
        for bar, v in zip(bars, values_w):
            offset = max_abs * 0.06 * (1 if v >= 0 else -1)
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"${v:+,.0f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                color=FG, fontsize=10, weight="bold",
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels([d.strftime("Week of\n%b %d") for d in labels], color=MUTED)
        ax2.axhline(y=0, color=MUTED, alpha=0.5, linewidth=1)

        # Pad y-limits so labels don't get clipped
        ymin = min(values_w) if values_w else 0
        ymax = max(values_w) if values_w else 0
        pad = max(abs(ymin), abs(ymax)) * 0.35 + 1
        ax2.set_ylim(ymin - pad, ymax + pad)
    else:
        ax2.text(
            0.5, 0.5, "No completed weeks yet",
            ha="center", va="center", color=MUTED, fontsize=12,
            transform=ax2.transAxes,
        )

    ax2.set_title("Weekly P&L", color=FG, fontsize=12, pad=10, weight="bold")
    ax2.set_ylabel("P&L ($)", color=MUTED, fontsize=11)

    # Generated-on stamp
    fig.text(
        0.99, 0.005,
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="right", va="bottom", color=MUTED, fontsize=8, family="monospace",
    )

    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path.resolve()}")


# ----------------------------------------------------------------------
# Optional S3 upload
# ----------------------------------------------------------------------

def upload_to_s3(local_path, bucket, key, content_type="image/png"):
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print("boto3 not installed.  pip install boto3")
        return False

    s3 = boto3.client("s3")
    try:
        s3.upload_file(
            str(local_path), bucket, key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "no-cache, max-age=0",
            },
        )
        print(f"Uploaded -> s3://{bucket}/{key}")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"S3 upload failed: {e}")
        return False


def write_latest_picks(portfolio, output_path):
    """Extract the most recent weekly pick into its own JSON file.

    Excludes portfolio state (cash, positions, history) so this file is safe
    to publish on a public bucket.
    """
    picks = portfolio.get("weekly_picks") or []
    if not picks:
        print("No weekly_picks in portfolio — nothing to upload.")
        return False

    latest = picks[-1]
    try:
        from tickers import get_sector_names
        sector_names = get_sector_names()
    except Exception:
        sector_names = {}

    payload = {
        "date": latest.get("date"),
        "longs": latest.get("longs", []),
        "shorts": latest.get("shorts", []),
        "predictions": latest.get("predictions", {}),
        "executed": latest.get("executed", False),
        "sector_names": sector_names,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot simulation performance vs SPY")
    parser.add_argument("--output", default="performance.png", help="Output PNG path")
    parser.add_argument("--upload", action="store_true", help="Upload chart PNG to S3")
    parser.add_argument("--bucket", default="preceptron.com", help="S3 bucket")
    parser.add_argument("--key", default="performance.png", help="S3 object key for chart")
    parser.add_argument("--upload-picks", action="store_true",
                        help="Also write latest picks to picks.json and upload to S3")
    parser.add_argument("--picks-output", default="picks.json", help="Local picks JSON path")
    parser.add_argument("--picks-key", default="picks.json", help="S3 object key for picks JSON")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart generation (useful with --upload-picks only)")
    args = parser.parse_args()

    portfolio = load_portfolio()
    if portfolio is None:
        sys.exit(1)

    exit_code = 0

    if not args.no_chart:
        output_path = Path(args.output)
        plot_chart(portfolio, output_path)
        if args.upload:
            if not upload_to_s3(output_path, args.bucket, args.key, "image/png"):
                exit_code = 2

    if args.upload_picks:
        picks_path = Path(args.picks_output)
        if write_latest_picks(portfolio, picks_path):
            if not upload_to_s3(picks_path, args.bucket, args.picks_key, "application/json"):
                exit_code = 2
        else:
            exit_code = 3

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
