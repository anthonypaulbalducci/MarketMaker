"""
Paper trading simulation for the sector rotation strategy.

Simulates weekly options trading based on model predictions:
- Long picks → buy calls (configurable OTM strike, ~1 month expiration)
- Short picks → buy puts (configurable OTM strike, ~1 month expiration)
- Positions held for 1 week, then closed and rebalanced

State is persisted in portfolio.json so you can run this week after week.

Usage:
    python simulation.py setup --capital 10000             # Initialize with $10k, $1 OTM
    python simulation.py setup --capital 10000 --otm 0.5   # $0.50 OTM (more leverage)
    python simulation.py setup --capital 10000 --commissions 0.65   # $0.65/contract
    python simulation.py predict                    # Get this week's picks
    python simulation.py execute                    # "Buy" options Monday
    python simulation.py status                     # Check current positions
    python simulation.py close                      # Close positions Friday
    python simulation.py history                    # Show full trade history

Weekly workflow:
    Friday:   python simulation.py predict
    Monday:   python simulation.py execute
    Friday:   python simulation.py close
    Friday:   python simulation.py predict    (for next week)
    Monday:   python simulation.py execute
    ...

Or run the full weekly cycle at once:
    python simulation.py week                       # close + predict + execute
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PORTFOLIO_FILE = Path("simulation_portfolio.json")


def get_default_portfolio():
    return {
        "initial_capital": 10000.0,
        "cash": 10000.0,
        "otm_amount": 1.0,         # Strike distance OTM in dollars (configurable at setup)
        "commission_per_contract": 0.0,  # Per-contract commission, charged on buy AND sell
        "positions": [],           # Active option positions
        "closed_trades": [],       # Historical closed trades
        "weekly_picks": [],        # History of weekly predictions
        "pnl_history": [],         # Weekly P&L snapshots
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


def load_portfolio():
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return None


def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().isoformat()
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2, default=str))


def get_option_data(ticker, option_type="call", otm_amount=1.0, min_dte=20, max_dte=40):
    """
    Find a suitable option contract from real market data.
    
    Args:
        ticker: ETF ticker (e.g., 'XLK')
        option_type: 'call' or 'put'
        otm_amount: dollars out of the money for strike selection
        min_dte: minimum days to expiration
        max_dte: maximum days to expiration
    
    Returns:
        dict with contract details, or None if unavailable
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return None

    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info.get("lastPrice", None)
        if current_price is None:
            hist = stock.history(period="5d")
            if hist.empty:
                print(f"  WARNING: No price data for {ticker}")
                return None
            current_price = hist["Close"].iloc[-1]

        # Find expiration ~1 month out
        expirations = stock.options
        if not expirations:
            print(f"  WARNING: No options available for {ticker}")
            return None

        target_date = datetime.now() + timedelta(days=30)
        best_exp = None
        best_diff = float("inf")

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - datetime.now()).days
            if min_dte <= dte <= max_dte:
                diff = abs(dte - 30)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str

        # Fallback: nearest expiration >= min_dte
        if best_exp is None:
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                if dte >= min_dte:
                    best_exp = exp_str
                    break

        if best_exp is None:
            print(f"  WARNING: No suitable expiration for {ticker}")
            return None

        # Get option chain
        chain = stock.option_chain(best_exp)
        if option_type == "call":
            options = chain.calls
            target_strike = current_price + otm_amount
        else:
            options = chain.puts
            target_strike = current_price - otm_amount

        if options.empty:
            print(f"  WARNING: No {option_type}s for {ticker} at {best_exp}")
            return None

        # Find closest strike to target
        options = options.copy()
        options["strike_diff"] = abs(options["strike"] - target_strike)
        best_row = options.loc[options["strike_diff"].idxmin()]

        strike = float(best_row["strike"])
        last_price = float(best_row["lastPrice"]) if best_row["lastPrice"] > 0 else None
        bid = float(best_row["bid"]) if best_row["bid"] > 0 else None
        ask = float(best_row["ask"]) if best_row["ask"] > 0 else None
        
        # Use mid price if last is stale, fallback to ask
        if ask and bid:
            entry_price = (bid + ask) / 2
        elif ask:
            entry_price = ask
        elif last_price:
            entry_price = last_price
        else:
            print(f"  WARNING: No pricing for {ticker} {option_type} {strike}")
            return None

        exp_date = datetime.strptime(best_exp, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        iv = float(best_row.get("impliedVolatility", 0))
        volume = int(best_row.get("volume", 0)) if pd.notna(best_row.get("volume")) else 0
        oi = int(best_row.get("openInterest", 0)) if pd.notna(best_row.get("openInterest")) else 0

        return {
            "ticker": ticker,
            "type": option_type,
            "strike": round(strike, 2),
            "expiration": best_exp,
            "dte": dte,
            "underlying_price": round(current_price, 2),
            "entry_price": round(entry_price, 2),
            "bid": round(bid, 2) if bid else None,
            "ask": round(ask, 2) if ask else None,
            "last_price": round(last_price, 2) if last_price else None,
            "implied_vol": round(iv, 4),
            "volume": volume,
            "open_interest": oi,
            "contract_symbol": str(best_row.get("contractSymbol", "")),
        }

    except Exception as e:
        print(f"  ERROR getting options for {ticker}: {e}")
        return None


def get_current_option_price(contract_symbol, ticker, option_type, strike, expiration):
    """Look up current price of an existing option position."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        options = chain.calls if option_type == "call" else chain.puts

        match = options[options["strike"] == strike]
        if match.empty:
            # Try closest strike
            options = options.copy()
            options["diff"] = abs(options["strike"] - strike)
            match = options.loc[[options["diff"].idxmin()]]

        row = match.iloc[0]
        bid = float(row["bid"]) if row["bid"] > 0 else 0
        ask = float(row["ask"]) if row["ask"] > 0 else 0
        last = float(row["lastPrice"]) if row["lastPrice"] > 0 else 0

        if bid and ask:
            return round((bid + ask) / 2, 2)
        elif last:
            return round(last, 2)
        return None
    except Exception:
        return None


def get_predictions():
    """Run the model and get sector picks."""
    from config import Config
    from tickers import get_all_tickers, get_sector_names, BENCHMARK
    from model import build_model
    from preprocess import compute_features

    cfg = Config()
    data_dir = cfg.data.processed_data_dir
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        print("No trained model found. Run 'python run.py' first.")
        return None, None, None

    metadata = json.loads(meta_path.read_text())
    tickers = metadata["tickers"]
    sector_tickers = metadata["sector_tickers"]
    spy_index = metadata["spy_index"]
    n_sectors = len(sector_tickers)
    names = get_sector_names()
    lookback = cfg.model.lookback_len

    # Get normalization stats
    features_full = np.load(data_dir / "features.npy")
    train_end = int(features_full.shape[0] * cfg.train.train_ratio)
    mean = features_full[:train_end].mean(axis=0)
    std = features_full[:train_end].std(axis=0)
    std[std < 1e-8] = 1.0

    # Download latest data
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return None, None, None

    print("Downloading latest sector data...")
    ohlcv_wide = {"open": {}, "high": {}, "low": {}, "close": {}, "volume": {}}
    extra = 10
    for ticker in tickers:
        data = yf.download(ticker, period=f"{lookback + extra + 2}wk",
                           interval="1wk", auto_adjust=True, progress=False)
        if data is not None and len(data) > 0:
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data.columns = ["open", "high", "low", "close", "volume"]
            for field in ohlcv_wide:
                ohlcv_wide[field][ticker] = data[field]

    import pandas as pd
    for field in ohlcv_wide:
        ohlcv_wide[field] = pd.DataFrame(ohlcv_wide[field]).sort_index().ffill()

    features = compute_features(ohlcv_wide, tickers)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if features.shape[0] > lookback:
        features = features[-lookback:]

    features = (features - mean) / std
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    import torch
    x = torch.from_numpy(features).unsqueeze(0)

    cfg.model.num_variates = features.shape[1]
    cfg.model.n_features = features.shape[2]

    ckpt_path = cfg.train.checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run 'python run.py' first.")
        return None, None, None

    model = build_model(cfg, spy_index=spy_index, n_sectors=n_sectors)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(x).squeeze(0).numpy()

    ranked = np.argsort(pred)[::-1]
    longs = [sector_tickers[i] for i in ranked[:3]]
    shorts = [sector_tickers[i] for i in ranked[-3:][::-1]]

    predictions = {sector_tickers[i]: float(pred[i]) for i in range(n_sectors)}

    return longs, shorts, predictions


# ============================================================
# Commands
# ============================================================

def cmd_setup(args):
    """Initialize the paper trading portfolio."""
    capital = args.capital
    otm = args.otm
    commissions = args.commissions
    portfolio = get_default_portfolio()
    portfolio["initial_capital"] = capital
    portfolio["cash"] = capital
    portfolio["otm_amount"] = otm
    portfolio["commission_per_contract"] = commissions
    save_portfolio(portfolio)
    print(f"\n{'='*60}")
    print(f"  Paper Trading Portfolio Initialized")
    print(f"  Starting capital: ${capital:,.2f}")
    print(f"  OTM strike:       ${otm:g} from underlying")
    print(f"  Commission:       ${commissions:.2f} per contract (each leg)")
    print(f"  Saved to: {PORTFOLIO_FILE}")
    print(f"{'='*60}\n")
    print("Next step: Run 'python simulation.py predict' to get this week's picks")


def cmd_predict(args):
    """Generate predictions and store them."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found. Run 'python simulation.py setup' first.")
        return

    print("\nGenerating sector rotation predictions...\n")
    longs, shorts, predictions = get_predictions()

    if longs is None:
        return

    from tickers import get_sector_names
    names = get_sector_names()

    pick = {
        "date": datetime.now().isoformat(),
        "longs": longs,
        "shorts": shorts,
        "predictions": predictions,
        "executed": False,
    }
    portfolio["weekly_picks"].append(pick)
    save_portfolio(portfolio)

    otm = portfolio.get("otm_amount", 1.0)

    print(f"\n{'='*60}")
    print(f"  SECTOR ROTATION PICKS — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    print(f"\n  \U0001F4C8 LONG (buy CALLS ${otm:g} OTM, ~1 month expiry):")
    for i, t in enumerate(longs):
        print(f"    {i+1}. {t:>6s}  {names.get(t, ''):<28s}  pred: {predictions[t]*100:+.3f}%")

    print(f"\n  \U0001F4C9 SHORT (buy PUTS ${otm:g} OTM, ~1 month expiry):")
    for i, t in enumerate(shorts):
        print(f"    {i+1}. {t:>6s}  {names.get(t, ''):<28s}  pred: {predictions[t]*100:+.3f}%")

    print(f"\n  Cash available: ${portfolio['cash']:,.2f}")
    print(f"  Active positions: {len(portfolio['positions'])}")
    print(f"\n  Next step: Run 'python simulation.py execute' on Monday to buy options")
    print(f"{'='*60}\n")


def cmd_execute(args):
    """Execute trades: close stale positions, open new ones."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found. Run 'python simulation.py setup' first.")
        return

    if not portfolio["weekly_picks"]:
        print("No predictions found. Run 'python simulation.py predict' first.")
        return

    latest_pick = portfolio["weekly_picks"][-1]
    if latest_pick["executed"]:
        print("Latest picks already executed. Run 'python simulation.py predict' first for new picks.")
        return

    longs = latest_pick["longs"]
    shorts = latest_pick["shorts"]
    all_picks = set(longs + shorts)

    from tickers import get_sector_names
    names = get_sector_names()

    print(f"\n{'='*60}")
    print(f"  EXECUTING TRADES — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    # Step 1: Close positions not in new picks
    commission = portfolio.get("commission_per_contract", 0.0)
    positions_to_keep = []
    for pos in portfolio["positions"]:
        ticker = pos["ticker"]
        # Keep if still in picks with same direction
        keep = False
        if ticker in longs and pos["type"] == "call":
            keep = True
        elif ticker in shorts and pos["type"] == "put":
            keep = True

        if keep:
            print(f"\n  HOLD: {ticker} {pos['type'].upper()} ${pos['strike']} "
                  f"exp {pos['expiration']} (still in picks)")
            positions_to_keep.append(pos)
        else:
            # Close this position
            current_price = get_current_option_price(
                pos.get("contract_symbol", ""), pos["ticker"],
                pos["type"], pos["strike"], pos["expiration"])

            if current_price is None:
                current_price = pos["entry_price"] * 0.8  # Estimate 20% decay
                print(f"\n  CLOSE: {ticker} {pos['type'].upper()} ${pos['strike']} "
                      f"(price unavailable, estimated ${current_price:.2f})")
            else:
                print(f"\n  CLOSE: {ticker} {pos['type'].upper()} ${pos['strike']} "
                      f"@ ${current_price:.2f} (was ${pos['entry_price']:.2f})")

            sell_commission = round(commission * pos["contracts"], 2)
            gross_proceeds = current_price * pos["contracts"] * 100
            net_proceeds = gross_proceeds - sell_commission
            # P&L = what we got back minus what we paid (both include commissions)
            pnl = net_proceeds - pos["total_cost"]
            portfolio["cash"] += net_proceeds

            closed = {**pos,
                      "close_date": datetime.now().isoformat(),
                      "close_price": current_price,
                      "sell_commission": sell_commission,
                      "pnl": round(pnl, 2)}
            portfolio["closed_trades"].append(closed)
            extra = f" (after ${sell_commission:.2f} commission)" if sell_commission > 0 else ""
            print(f"         P&L: ${pnl:+,.2f}{extra}")

    portfolio["positions"] = positions_to_keep

    # Step 2: Determine allocation per new position
    existing_tickers = {(p["ticker"], p["type"]) for p in portfolio["positions"]}
    new_trades = []
    for t in longs:
        if (t, "call") not in existing_tickers:
            new_trades.append((t, "call"))
    for t in shorts:
        if (t, "put") not in existing_tickers:
            new_trades.append((t, "put"))

    if new_trades:
        # Allocate available cash equally across new positions
        # Reserve 10% cash buffer
        available = portfolio["cash"] * 0.9
        per_position = available / max(len(new_trades), 1)
        otm = portfolio.get("otm_amount", 1.0)
        commission = portfolio.get("commission_per_contract", 0.0)

        msg = f"\n  Allocating ${per_position:,.2f} per new position " \
              f"({len(new_trades)} new trades) at ${otm:g} OTM"
        if commission > 0:
            msg += f", ${commission:.2f}/contract commission"
        print(msg)

        for ticker, opt_type in new_trades:
            print(f"\n  OPEN: {ticker} {opt_type.upper()} ${otm:g} OTM...")
            option = get_option_data(ticker, opt_type, otm_amount=otm)

            if option is None:
                print(f"         Skipped (no options data available)")
                continue

            # Effective cost per contract = option premium (×100) + commission
            premium_per_contract = option["entry_price"] * 100  # Options are per 100 shares
            cost_per_contract = premium_per_contract + commission
            if premium_per_contract <= 0:
                print(f"         Skipped (zero price)")
                continue

            n_contracts = max(1, int(per_position / cost_per_contract))
            total_cost = n_contracts * cost_per_contract

            if total_cost > portfolio["cash"]:
                n_contracts = max(1, int(portfolio["cash"] * 0.9 / cost_per_contract))
                total_cost = n_contracts * cost_per_contract

            if total_cost > portfolio["cash"]:
                print(f"         Skipped (insufficient cash: need ${total_cost:,.2f}, "
                      f"have ${portfolio['cash']:,.2f})")
                continue

            buy_commission = round(commission * n_contracts, 2)
            portfolio["cash"] -= total_cost

            position = {
                "ticker": ticker,
                "type": opt_type,
                "strike": option["strike"],
                "expiration": option["expiration"],
                "dte": option["dte"],
                "underlying_price": option["underlying_price"],
                "entry_price": option["entry_price"],
                "contracts": n_contracts,
                "total_cost": round(total_cost, 2),
                "buy_commission": buy_commission,
                "entry_date": datetime.now().isoformat(),
                "contract_symbol": option.get("contract_symbol", ""),
                "implied_vol": option.get("implied_vol", 0),
            }
            portfolio["positions"].append(position)

            print(f"         {names.get(ticker, '')} {opt_type.upper()} "
                  f"${option['strike']} exp {option['expiration']} "
                  f"({option['dte']}d)")
            cost_str = (f"{n_contracts} contract(s) @ ${option['entry_price']:.2f} "
                        f"= ${total_cost:,.2f}")
            if buy_commission > 0:
                cost_str += f" (incl. ${buy_commission:.2f} commission)"
            print(f"         {cost_str}")
            print(f"         Underlying: ${option['underlying_price']:.2f}  "
                  f"IV: {option['implied_vol']*100:.1f}%")

    latest_pick["executed"] = True
    save_portfolio(portfolio)

    # Summary
    total_invested = sum(p["total_cost"] for p in portfolio["positions"])
    print(f"\n{'='*60}")
    print(f"  EXECUTION SUMMARY")
    print(f"  Positions: {len(portfolio['positions'])}")
    print(f"  Invested:  ${total_invested:,.2f}")
    print(f"  Cash:      ${portfolio['cash']:,.2f}")
    print(f"  Total:     ${total_invested + portfolio['cash']:,.2f}")
    print(f"\n  Next step: Hold all week. Run 'python simulation.py status' to check.")
    print(f"  Friday: Run 'python simulation.py close' then 'python simulation.py predict'")
    print(f"{'='*60}\n")


def cmd_close(args):
    """Close all current positions."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found.")
        return

    if not portfolio["positions"]:
        print("No open positions to close.")
        return

    from tickers import get_sector_names
    names = get_sector_names()

    print(f"\n{'='*60}")
    print(f"  CLOSING ALL POSITIONS — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    commission = portfolio.get("commission_per_contract", 0.0)
    total_pnl = 0
    total_commissions = 0
    for pos in portfolio["positions"]:
        current_price = get_current_option_price(
            pos.get("contract_symbol", ""), pos["ticker"],
            pos["type"], pos["strike"], pos["expiration"])

        if current_price is None:
            # Estimate: decay proportional to time held
            days_held = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
            decay = max(0.5, 1.0 - days_held * 0.03)  # ~3% per day theta estimate
            current_price = round(pos["entry_price"] * decay, 2)
            estimated = True
        else:
            estimated = False

        sell_commission = round(commission * pos["contracts"], 2)
        gross_proceeds = current_price * pos["contracts"] * 100
        net_proceeds = gross_proceeds - sell_commission
        # P&L = net_proceeds - total_cost (both include their respective commissions)
        pnl = net_proceeds - pos["total_cost"]
        total_pnl += pnl
        total_commissions += sell_commission
        portfolio["cash"] += net_proceeds

        est_tag = " (estimated)" if estimated else ""
        pnl_color = "+" if pnl >= 0 else ""

        print(f"\n  {pos['ticker']} {pos['type'].upper()} ${pos['strike']} "
              f"exp {pos['expiration']}")
        print(f"    Entry: ${pos['entry_price']:.2f}  →  Exit: ${current_price:.2f}{est_tag}")
        commission_tag = f"  (commission: ${sell_commission:.2f})" if sell_commission > 0 else ""
        print(f"    {pos['contracts']} contract(s)  P&L: ${pnl_color}{pnl:,.2f}{commission_tag}")

        closed = {**pos,
                  "close_date": datetime.now().isoformat(),
                  "close_price": current_price,
                  "sell_commission": sell_commission,
                  "pnl": round(pnl, 2),
                  "estimated": estimated}
        portfolio["closed_trades"].append(closed)

    # Record weekly P&L
    portfolio["pnl_history"].append({
        "date": datetime.now().isoformat(),
        "weekly_pnl": round(total_pnl, 2),
        "weekly_commissions": round(total_commissions, 2),
        "cash_after": round(portfolio["cash"], 2),
        "n_positions_closed": len(portfolio["positions"]),
    })

    portfolio["positions"] = []
    save_portfolio(portfolio)

    # Cumulative P&L is the sum of all realized trade P&L. Equivalent to
    # cash - initial_capital here (since all positions just closed) but the
    # closed_trades formulation is robust regardless of portfolio state.
    cumulative_pnl = sum(t.get("pnl", 0) for t in portfolio["closed_trades"])
    cumulative_pct = cumulative_pnl / portfolio["initial_capital"] * 100

    print(f"\n{'='*60}")
    print(f"  WEEKLY SUMMARY")
    print(f"  This week P&L:   ${total_pnl:+,.2f}")
    if total_commissions > 0:
        print(f"  Commissions:     ${total_commissions:.2f}  (this week, sell-side only)")
    print(f"  Cash balance:    ${portfolio['cash']:,.2f}")
    print(f"  Cumulative P&L:  ${cumulative_pnl:+,.2f} ({cumulative_pct:+.1f}%)")
    print(f"  Total trades:    {len(portfolio['closed_trades'])}")
    print(f"\n  Next: Run 'python simulation.py predict' for next week's picks")
    print(f"{'='*60}\n")


def cmd_status(args):
    """Show current portfolio status."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found. Run 'python simulation.py setup' first.")
        return

    from tickers import get_sector_names
    names = get_sector_names()

    print(f"\n{'='*60}")
    print(f"  PORTFOLIO STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    print(f"\n  Initial capital:  ${portfolio['initial_capital']:,.2f}")
    print(f"  Cash balance:     ${portfolio['cash']:,.2f}")

    if portfolio["positions"]:
        print(f"\n  OPEN POSITIONS ({len(portfolio['positions'])}):")
        total_invested = 0
        total_current = 0

        for pos in portfolio["positions"]:
            current_price = get_current_option_price(
                pos.get("contract_symbol", ""), pos["ticker"],
                pos["type"], pos["strike"], pos["expiration"])

            current_val = (current_price or pos["entry_price"]) * pos["contracts"] * 100
            cost = pos["total_cost"]
            unrealized = current_val - cost
            total_invested += cost
            total_current += current_val

            price_str = f"${current_price:.2f}" if current_price else "N/A"
            days_held = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days

            print(f"\n    {pos['ticker']} {pos['type'].upper()} ${pos['strike']} "
                  f"exp {pos['expiration']} ({pos['dte'] - days_held}d left)")
            print(f"      {pos['contracts']} contract(s) @ ${pos['entry_price']:.2f} "
                  f"→ {price_str}")
            print(f"      Cost: ${cost:,.2f}  Current: ${current_val:,.2f}  "
                  f"Unrealized: ${unrealized:+,.2f}")

        print(f"\n    Total invested:   ${total_invested:,.2f}")
        print(f"    Total current:    ${total_current:,.2f}")
        print(f"    Unrealized P&L:   ${total_current - total_invested:+,.2f}")
        print(f"    Portfolio value:   ${portfolio['cash'] + total_current:,.2f}")
    else:
        print(f"\n  No open positions")
        print(f"  Portfolio value:   ${portfolio['cash']:,.2f}")

    # Cumulative P&L = realized (from closed trades) + unrealized (from open positions).
    # Old code computed (cash - initial) + (current - cost), which double-counted
    # the cost of open positions because cash already had the cost subtracted.
    realized_pnl = sum(t.get("pnl", 0) for t in portfolio["closed_trades"])
    unrealized_pnl = 0.0
    if portfolio["positions"]:
        total_current = sum(
            (get_current_option_price(p.get("contract_symbol",""), p["ticker"],
             p["type"], p["strike"], p["expiration"]) or p["entry_price"])
            * p["contracts"] * 100 for p in portfolio["positions"])
        unrealized_pnl = total_current - sum(p["total_cost"] for p in portfolio["positions"])
    cumulative_pnl = realized_pnl + unrealized_pnl

    print(f"\n  Realized P&L:      ${realized_pnl:+,.2f}")
    if portfolio["positions"]:
        print(f"  Unrealized P&L:    ${unrealized_pnl:+,.2f}  (open positions, mark-to-market)")
    print(f"  Cumulative P&L:    ${cumulative_pnl:+,.2f} "
          f"({cumulative_pnl/portfolio['initial_capital']*100:+.1f}%)")
    print(f"  Closed trades:     {len(portfolio['closed_trades'])}")
    print(f"  Weeks traded:      {len(portfolio['pnl_history'])}")

    if portfolio["pnl_history"]:
        pnls = [p["weekly_pnl"] for p in portfolio["pnl_history"]]
        wins = sum(1 for p in pnls if p > 0)
        print(f"  Win rate:          {wins}/{len(pnls)} ({wins/len(pnls)*100:.0f}%)")
        print(f"  Best week:         ${max(pnls):+,.2f}")
        print(f"  Worst week:        ${min(pnls):+,.2f}")

    print(f"{'='*60}\n")


def cmd_history(args):
    """Show full trade history."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found.")
        return

    print(f"\n{'='*60}")
    print(f"  TRADE HISTORY")
    print(f"{'='*60}")

    if not portfolio["closed_trades"]:
        print("\n  No closed trades yet.\n")
        return

    print(f"\n  {'Date':<12s}  {'Ticker':>6s}  {'Type':>5s}  {'Strike':>7s}  "
          f"{'Entry':>6s}  {'Exit':>6s}  {'Qty':>3s}  {'P&L':>10s}")
    print(f"  {'----':<12s}  {'------':>6s}  {'-----':>5s}  {'-------':>7s}  "
          f"{'------':>6s}  {'------':>6s}  {'---':>3s}  {'----------':>10s}")

    for trade in portfolio["closed_trades"]:
        date = trade.get("close_date", "")[:10]
        pnl = trade.get("pnl", 0)
        print(f"  {date:<12s}  {trade['ticker']:>6s}  {trade['type']:>5s}  "
              f"${trade['strike']:>6.2f}  ${trade['entry_price']:>5.2f}  "
              f"${trade.get('close_price', 0):>5.2f}  {trade['contracts']:>3d}  "
              f"${pnl:>+9,.2f}")

    total_pnl = sum(t.get("pnl", 0) for t in portfolio["closed_trades"])
    print(f"\n  Total realized P&L: ${total_pnl:+,.2f}")

    if portfolio["pnl_history"]:
        print(f"\n  Weekly P&L:")
        for week in portfolio["pnl_history"]:
            date = week["date"][:10]
            print(f"    {date}: ${week['weekly_pnl']:+,.2f}  "
                  f"(cash: ${week['cash_after']:,.2f})")

    print(f"{'='*60}\n")


def cmd_week(args):
    """Run a full weekly cycle: close existing + predict + execute."""
    portfolio = load_portfolio()
    if portfolio is None:
        print("No portfolio found. Run 'python simulation.py setup --capital 10000' first.")
        return

    # Close existing positions if any
    if portfolio["positions"]:
        print("\n>>> Closing existing positions...")
        cmd_close(args)

    # Generate new predictions
    print("\n>>> Generating predictions...")
    cmd_predict(args)

    # Execute new trades
    print("\n>>> Executing trades...")
    cmd_execute(args)


def cmd_reset(args):
    """Reset the simulation."""
    if PORTFOLIO_FILE.exists():
        PORTFOLIO_FILE.unlink()
        print("Simulation reset. Run 'python simulation.py setup' to start fresh.")
    else:
        print("No simulation to reset.")


def main():
    parser = argparse.ArgumentParser(
        description="Paper trading simulation for sector rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weekly workflow:
  Friday:  python simulation.py predict
  Monday:  python simulation.py execute  
  Friday:  python simulation.py close
  (repeat)

Or run everything at once:
  python simulation.py week
        """)

    subparsers = parser.add_subparsers(dest="command")

    sp = subparsers.add_parser("setup", help="Initialize paper trading portfolio")
    sp.add_argument("--capital", type=float, default=10000,
                    help="Starting capital in dollars (default: $10,000)")
    sp.add_argument("--otm", type=float, default=1.0,
                    help="Strike distance OTM in dollars, e.g. 0.5 for $0.50 OTM (default: 1.0)")
    sp.add_argument("--commissions", type=float, default=0.0,
                    help="Per-contract commission in dollars, charged on buy AND sell, "
                         "e.g. 0.65 (default: 0.0 = no commissions)")

    subparsers.add_parser("predict", help="Generate this week's sector picks")
    subparsers.add_parser("execute", help="Execute trades (buy options)")
    subparsers.add_parser("close", help="Close all positions")
    subparsers.add_parser("status", help="Show current portfolio status")
    subparsers.add_parser("history", help="Show full trade history")
    subparsers.add_parser("week", help="Full weekly cycle: close + predict + execute")
    subparsers.add_parser("reset", help="Reset the simulation")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "setup": cmd_setup,
        "predict": cmd_predict,
        "execute": cmd_execute,
        "close": cmd_close,
        "status": cmd_status,
        "history": cmd_history,
        "week": cmd_week,
        "reset": cmd_reset,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
