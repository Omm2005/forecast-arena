"""Rolling VaR backtest with Kupiec and Christoffersen tests of model performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .montecarlo import estimate_parameters, simulate_portfolio


@dataclass
class BacktestResult:
    frame: pd.DataFrame          # date, forecast VaR, realized P&L, breach flag
    confidence: float
    kupiec_stat: float
    kupiec_p: float
    christoffersen_stat: float
    christoffersen_p: float

    @property
    def n_obs(self) -> int:
        return len(self.frame)

    @property
    def n_breaches(self) -> int:
        return int(self.frame["breach"].sum())

    @property
    def expected_breaches(self) -> float:
        return self.n_obs * (1 - self.confidence)

    @property
    def traffic_light(self) -> str:
        """Basel traffic-light zone, scaled from the standard 250-day / 99% rule."""
        p = 1 - self.confidence
        from scipy.stats import binom
        cdf = binom.cdf(self.n_breaches, self.n_obs, p)
        if cdf < 0.95:
            return "GREEN"
        return "YELLOW" if cdf < 0.9999 else "RED"


def _log_lik(p: float, n_hit: int, n_miss: int) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return n_hit * np.log(p) + n_miss * np.log(1 - p)


def kupiec_pof(breaches: np.ndarray, confidence: float) -> tuple[float, float]:
    """Proportion-of-failures LR test: is the breach rate consistent with 1 - confidence?"""
    n, x = len(breaches), int(breaches.sum())
    p = 1 - confidence
    lr = -2 * (_log_lik(p, x, n - x) - _log_lik(x / n, x, n - x))
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """LR test that breaches don't cluster (today's breach independent of yesterday's)."""
    b = breaches.astype(int)
    prev, curr = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (curr == 0)); n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0)); n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr = -2 * (_log_lik(pi, n01 + n11, n00 + n10)
               - _log_lik(pi0, n01, n00) - _log_lik(pi1, n11, n10))
    return float(lr), float(1 - chi2.cdf(lr, 1))


def rolling_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float = 10_000_000,
    window: int = 500,
    confidence: float = 0.99,
    n_paths: int = 10_000,
    step: int = 1,
    ewma_lambda: float | None = 0.97,
    seed: int = 0,
) -> BacktestResult:
    """Each day: fit on the trailing window, forecast 1-day VaR, compare with realized P&L."""
    weights = np.asarray(weights) / np.sum(weights)
    rows = []
    for i in range(window, len(returns), step):
        mu, cov = estimate_parameters(returns.iloc[i - window:i], ewma_lambda)
        sim = simulate_portfolio(mu, cov, weights, portfolio_value, horizon_days=1,
                                 n_paths=n_paths, confidence_levels=(confidence,),
                                 seed=seed + i)
        realized = portfolio_value * (np.exp(returns.iloc[i].to_numpy()) @ weights - 1)
        var = sim.var[confidence]
        rows.append({"date": returns.index[i], "var": var, "pnl": realized,
                     "breach": realized < -var})

    frame = pd.DataFrame(rows).set_index("date")
    breaches = frame["breach"].to_numpy()
    k_stat, k_p = kupiec_pof(breaches, confidence)
    c_stat, c_p = christoffersen_independence(breaches)
    return BacktestResult(frame, confidence, k_stat, k_p, c_stat, c_p)
