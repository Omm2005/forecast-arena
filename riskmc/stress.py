"""Stress scenarios: transform (mu, cov) and re-run the simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    vol_multiplier: float = 1.0
    correlation_floor: float | None = None   # push all pairwise correlations up to at least this
    daily_drift_shock: float = 0.0           # added to every asset's daily mean log return


SCENARIOS = [
    Scenario("Baseline", "Historical estimates, no shock"),
    Scenario("Vol spike", "Volatility doubles across all assets", vol_multiplier=2.0),
    Scenario("Correlation breakdown", "Diversification fails: correlations jump to >= 0.8",
             correlation_floor=0.8),
    Scenario("2008-style crash", "Vol x3, correlations >= 0.7, -1% daily drift",
             vol_multiplier=3.0, correlation_floor=0.7, daily_drift_shock=-0.01),
    Scenario("Rate shock", "Vol x1.5 with -0.3% daily drift", vol_multiplier=1.5,
             daily_drift_shock=-0.003),
]


def apply_scenario(mu: np.ndarray, cov: np.ndarray, s: Scenario):
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    if s.correlation_floor is not None:
        corr = np.maximum(corr, s.correlation_floor)
        np.fill_diagonal(corr, 1.0)
    vol = vol * s.vol_multiplier
    return mu + s.daily_drift_shock, corr * np.outer(vol, vol)
