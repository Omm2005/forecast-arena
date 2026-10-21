import numpy as np
from scipy.stats import norm

from riskmc.backtest import kupiec_pof
from riskmc.montecarlo import cholesky_factor, simulate_portfolio
from riskmc.stress import SCENARIOS, apply_scenario


def test_single_asset_var_matches_closed_form():
    # One asset, lognormal returns: VaR has an exact answer.
    sigma, nav, h = 0.02, 1_000_000, 10
    sim = simulate_portfolio(np.zeros(1), np.array([[sigma**2]]), np.ones(1), nav, h,
                             n_paths=200_000, seed=1)
    expected = nav * (1 - np.exp(norm.ppf(0.01) * sigma * np.sqrt(h)))
    assert abs(sim.var[0.99] - expected) / expected < 0.02


def test_cvar_exceeds_var():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 20))
    cov = (a @ a.T) * 1e-5
    sim = simulate_portfolio(np.zeros(20), cov, np.ones(20), n_paths=50_000)
    for c in sim.confidence_levels:
        assert sim.cvar[c] > sim.var[c] > 0
    assert sim.var[0.99] > sim.var[0.95]


def test_simulated_correlation_recovered():
    cov = np.array([[1.0, 0.8], [0.8, 1.0]]) * 1e-4
    L = cholesky_factor(cov)
    z = np.random.default_rng(3).standard_normal((100_000, 2)) @ L.T
    assert abs(np.corrcoef(z.T)[0, 1] - 0.8) < 0.01


def test_stress_scenarios_increase_risk():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(10, 10))
    cov = (a @ a.T + 10 * np.eye(10)) * 1e-5
    mu = np.zeros(10)
    base = simulate_portfolio(mu, cov, np.ones(10), n_paths=20_000, seed=9).var[0.99]
    for s in SCENARIOS[1:]:
        s_mu, s_cov = apply_scenario(mu, cov, s)
        assert simulate_portfolio(s_mu, s_cov, np.ones(10), n_paths=20_000, seed=9).var[0.99] > base


def test_kupiec_accepts_correct_rate_and_rejects_bad():
    rng = np.random.default_rng(11)
    good = rng.random(1000) < 0.01
    bad = rng.random(1000) < 0.05
    assert kupiec_pof(good, 0.99)[1] > 0.05
    assert kupiec_pof(bad, 0.99)[1] < 0.01


def test_monte_carlo_minute_ranges_are_calibrated():
    import pandas as pd
    from riskmc.forecasters import MonteCarloForecaster
    from riskmc.minute import QCOLS, score

    rng = np.random.default_rng(4)
    n = 900
    r = rng.standard_t(6, n) * np.sqrt(4 / 6) * 0.0005
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="1min", tz="America/New_York")
    closes = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    idxs = list(range(400, n - 1))
    q = MonteCarloForecaster(seed=1).replay(closes, idxs)
    rows = pd.DataFrame(q, columns=QCOLS)
    rows["last"], rows["actual"] = closes.to_numpy()[idxs], closes.to_numpy()[[i + 1 for i in idxs]]
    s = score(rows)
    assert abs(s["coverage80"] - 0.8) < 0.05


def test_fit_scale_recovers_spread():
    from riskmc.minute import LEVELS, fit_scale
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    n = 4000
    last = np.full(n, 100.0)
    actual = last + rng.normal(0, 2.0, n)
    q = last[:, None] + norm.ppf(LEVELS)[None, :] * 1.0     # forecasts half as wide as reality
    assert abs(fit_scale(q, actual, last) - 2.0) < 0.15
