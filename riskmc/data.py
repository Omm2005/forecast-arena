"""Market data: load prices from CSV or generate a realistic synthetic universe."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = ["Tech", "Financials", "Energy", "Healthcare", "Industrials"]

# 50 large-cap US stocks, 5 per sector, all with long trading histories.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",   # Tech / communication
    "JPM", "BAC", "GS", "MS", "WFC",           # Financials
    "XOM", "CVX", "COP", "SLB", "EOG",         # Energy
    "JNJ", "UNH", "PFE", "MRK", "ABBV",        # Healthcare
    "CAT", "HON", "GE", "UPS", "DE",           # Industrials
    "AMZN", "HD", "MCD", "NKE", "SBUX",        # Consumer discretionary
    "PG", "KO", "PEP", "WMT", "COST",          # Consumer staples
    "NEE", "DUK", "SO", "D", "AEP",            # Utilities
    "AMT", "PLD", "CCI", "SPG", "O",           # Real estate
    "LIN", "SHW", "APD", "NEM", "FCX",         # Materials
]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    years: int = 5,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily adjusted closes from Yahoo Finance, cached to data/ once per day."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"prices_{len(tickers)}_{years}y_{date.today():%Y%m%d}.csv"
    if cache.exists() and not refresh:
        return load_prices(str(cache))

    import yfinance as yf

    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(axis=1, thresh=int(len(raw) * 0.95)).ffill().dropna()
    if prices.shape[1] < 2:
        raise RuntimeError("Price download failed; check your connection or use --synthetic")
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        print(f"Skipped tickers with incomplete data: {', '.join(missing)}")
    prices.to_csv(cache)
    return prices


def load_prices(path: str) -> pd.DataFrame:
    """Load a wide CSV of prices: first column is the date, one column per asset."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return prices.dropna(axis=1, how="any")


def synthetic_prices(
    n_assets: int = 50,
    n_days: int = 1260,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate daily prices from a market + sector factor model.

    Correlations come from shared exposure to a market factor and a sector
    factor, which gives the block structure seen in real equity universes.
    Volatility follows a simple regime switch so the history contains calm
    and stressed periods (useful for backtesting VaR).
    """
    rng = np.random.default_rng(seed)
    sector_of = np.arange(n_assets) % len(SECTORS)

    beta_mkt = rng.uniform(0.6, 1.4, n_assets)
    beta_sec = rng.uniform(0.3, 0.9, n_assets)
    idio_vol = rng.uniform(0.008, 0.018, n_assets)
    drift = rng.normal(0.0003, 0.0002, n_assets)

    # Two-state volatility regime: calm (x1) and stressed (x2.5).
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        stay = 0.995 if regime[t - 1] == 0 else 0.97
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    vol_mult = np.where(regime == 1, 2.5, 1.0)

    mkt = rng.normal(0, 0.009, n_days) * vol_mult
    sec = rng.normal(0, 0.006, (n_days, len(SECTORS))) * vol_mult[:, None]
    eps = rng.normal(0, 1, (n_days, n_assets)) * idio_vol * vol_mult[:, None]

    returns = drift + mkt[:, None] * beta_mkt + sec[:, sector_of] * beta_sec + eps
    start = rng.uniform(20, 300, n_assets)
    prices = start * np.exp(np.cumsum(returns, axis=0))

    dates = pd.bdate_range(end="2026-10-16", periods=n_days)
    cols = [f"{SECTORS[s][:4].upper()}{i:02d}" for i, s in enumerate(sector_of)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
