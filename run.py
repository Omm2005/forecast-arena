"""Generate a portfolio risk report.

    python run.py                          # 50 large-cap US stocks, 5y of Yahoo Finance data
    python run.py --tickers AAPL MSFT JPM  # your own ticker list
    python run.py --prices prices.csv      # your own price history (date + one column per asset)
    python run.py --synthetic              # offline synthetic 50-asset universe
"""

import argparse

from riskmc.data import download_prices, load_prices, synthetic_prices, DEFAULT_TICKERS
from riskmc.report import build_report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices", help="CSV of prices instead of downloading")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--years", type=int, default=5, help="years of history to download")
    p.add_argument("--refresh", action="store_true", help="ignore today's cached download")
    p.add_argument("--synthetic", action="store_true", help="use simulated prices (no network)")
    p.add_argument("--assets", type=int, default=50, help="synthetic universe size")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--horizon", type=int, default=10, help="VaR horizon in trading days")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--backtest-days", type=int, default=250)
    p.add_argument("--out", default="reports")
    a = p.parse_args()

    if a.prices:
        prices = load_prices(a.prices)
    elif a.synthetic:
        prices = synthetic_prices(a.assets)
    else:
        prices = download_prices(a.tickers, a.years, a.refresh)
    print(f"{prices.shape[1]} assets, {len(prices)} days "
          f"({prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d})")
    path = build_report(prices, portfolio_value=a.nav, horizon_days=a.horizon,
                        n_paths=a.paths, backtest_days=a.backtest_days, out_dir=a.out)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
