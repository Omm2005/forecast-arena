"""Correlated Monte Carlo simulation of portfolio P&L and VaR / CVaR."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    pnl: np.ndarray                      # simulated portfolio P&L, one per path
    portfolio_value: float
    horizon_days: int
    confidence_levels: tuple[float, ...]
    var: dict[float, float] = field(default_factory=dict)
    cvar: dict[float, float] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "confidence": f"{c:.0%}",
                "VaR": self.var[c],
                "CVaR (ES)": self.cvar[c],
                "VaR % of NAV": self.var[c] / self.portfolio_value,
            }
            for c in self.confidence_levels
        ]
        return pd.DataFrame(rows).set_index("confidence")


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues so a (possibly shocked) covariance is factorable."""
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    return (vecs * np.clip(vals, eps, None)) @ vecs.T


def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_psd(cov))


def var_cvar(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """VaR and expected shortfall, both reported as positive loss amounts."""
    cutoff = np.quantile(pnl, 1 - confidence)
    tail = pnl[pnl <= cutoff]
    return float(-cutoff), float(-tail.mean())


def simulate_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float = 10_000_000,
    horizon_days: int = 10,
    n_paths: int = 100_000,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    seed: int | None = 42,
    chunk_size: int = 20_000,
) -> SimulationResult:
    """Simulate correlated daily log-return paths and value the portfolio at horizon.

    Each path draws `horizon_days` vectors of independent normals z ~ N(0, I)
    and maps them to correlated shocks via the Cholesky factor L (L L^T = cov),
    so r_t = mu + L z_t. Asset prices compound along the path; only the final
    portfolio value is kept. Paths are generated in chunks to bound memory.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    L = cholesky_factor(np.asarray(cov))
    n_assets = len(mu)
    rng = np.random.default_rng(seed)

    pnl = np.empty(n_paths)
    for start in range(0, n_paths, chunk_size):
        n = min(chunk_size, n_paths - start)
        z = rng.standard_normal((n, horizon_days, n_assets))
        cum_log_ret = (z @ L.T + mu).sum(axis=1)          # (n, n_assets)
        gross = np.exp(cum_log_ret) @ weights               # portfolio growth factor
        pnl[start:start + n] = portfolio_value * (gross - 1)

    result = SimulationResult(pnl, portfolio_value, horizon_days, confidence_levels)
    for c in confidence_levels:
        result.var[c], result.cvar[c] = var_cvar(pnl, c)
    return result


def estimate_parameters(returns: pd.DataFrame, ewma_lambda: float | None = None):
    """Mean vector and covariance of daily log returns (optionally EWMA-weighted)."""
    x = returns.to_numpy()
    if ewma_lambda is None:
        return x.mean(axis=0), np.cov(x, rowvar=False)
    w = ewma_lambda ** np.arange(len(x))[::-1]
    w /= w.sum()
    mu = w @ x
    centered = x - mu
    return mu, (centered * w[:, None]).T @ centered


def component_var(cov: np.ndarray, weights: np.ndarray, portfolio_value: float,
                  confidence: float, horizon_days: int) -> np.ndarray:
    """Parametric (delta-normal) Euler allocation: each asset's contribution to VaR.

    Contributions sum to the total parametric VaR, which makes it easy to see
    which positions drive risk.
    """
    from scipy.stats import norm

    weights = np.asarray(weights) / np.sum(weights)
    sigma_p = np.sqrt(weights @ cov @ weights)
    z = norm.ppf(confidence)
    marginal = cov @ weights / sigma_p
    return portfolio_value * z * np.sqrt(horizon_days) * weights * marginal
