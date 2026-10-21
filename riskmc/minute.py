"""Next-minute Monte Carlo forecaster and the scoring shared by every model.

Model (filtered historical simulation), using only information up to the current bar:
    r_next = phi * r_last + sigma * z
- sigma: EWMA volatility (lambda = 0.97), reacts quickly to bursts
- phi:   shrunk rolling AR(1) coefficient, captures short-term momentum / bid-ask bounce
- z:     drawn from this ticker's own recent standardized forecast errors, so the
         band width and tail shape calibrate themselves per asset
The forecast distribution comes from simulating many next-minute prices.

Every model in the app reports the same nine quantiles (10th..90th percentile),
which is what TimesFM produces natively, so the comparison is like for like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP = pd.Timedelta(minutes=5)   # returns across longer gaps (overnight, lunch) are skipped
LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)             # 0.1, 0.2, ..., 0.9
QCOLS = [f"q{int(round(l * 100))}" for l in LEVELS]          # q10 ... q90


def minute_returns(closes: pd.Series) -> pd.Series:
    """Log returns between consecutive bars; NaN where the bars are separated by a gap."""
    r = np.log(closes / closes.shift(1))
    gap = closes.index.to_series().diff() > MAX_GAP
    return r.mask(gap.to_numpy())


def model_frame(closes: pd.Series, lookback: int = 120, lam: float = 0.97,
                shrink: float = 0.5) -> pd.DataFrame:
    """Per-bar model state. Row t only uses returns up to and including bar t."""
    r = minute_returns(closes)
    sigma = np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False, ignore_na=True,
                                 min_periods=20).mean()).clip(lower=1e-6)
    r_prev = r.shift(1)
    num = (r * r_prev).rolling(lookback, min_periods=30).sum()
    den = (r_prev ** 2).rolling(lookback, min_periods=30).sum()
    phi = (shrink * num / den).clip(-0.3, 0.3).fillna(0.0)
    mean_next = phi * r.fillna(0.0)                       # expected return of bar t+1
    z = (r - mean_next.shift(1)) / sigma.shift(1)         # realized surprise at bar t
    return pd.DataFrame({"r": r, "sigma": sigma, "phi": phi, "mean_next": mean_next, "z": z})


def mc_quantiles(closes: pd.Series, frame: pd.DataFrame, i: int, calib: int = 300,
                 n_paths: int = 10_000, rng: np.random.Generator | None = None) -> np.ndarray | None:
    """Monte Carlo quantiles (LEVELS) of the close of bar i+1, using bars up to i."""
    rng = rng or np.random.default_rng()
    zs = frame["z"].iloc[:i + 1].dropna().to_numpy()[-calib:]
    sigma, mean = frame["sigma"].iloc[i], frame["mean_next"].iloc[i]
    if len(zs) < 60 or np.isnan(sigma):
        return None
    sim_r = mean + sigma * rng.choice(zs, n_paths)        # Monte Carlo over past surprises
    return float(closes.iloc[i]) * np.exp(np.quantile(sim_r, LEVELS))


def rescale(q: np.ndarray, scale: float) -> np.ndarray:
    """Widen (scale > 1) or narrow a quantile forecast around its median."""
    q = np.asarray(q, dtype=float)
    med = q[..., [len(LEVELS) // 2]]
    return med + scale * (q - med)


def pinball_bps(q: np.ndarray, actual: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Average quantile (pinball) loss over LEVELS, in basis points of the last price.

    A proper scoring rule: it rewards ranges that are both correctly placed and tight,
    so it is the single number used to rank models.
    """
    q, actual, last = np.atleast_2d(q), np.asarray(actual)[:, None], np.asarray(last)[:, None]
    diff = actual - q
    loss = np.maximum(LEVELS * diff, (LEVELS - 1) * diff)
    return (loss / last * 1e4).mean(axis=1)


def fit_scale(q: np.ndarray, actual: np.ndarray, last: np.ndarray) -> float:
    """Spread multiplier that minimizes pinball loss on a calibration set."""
    grid = np.round(np.arange(0.4, 3.01, 0.05), 2)
    losses = [pinball_bps(rescale(q, s), actual, last).mean() for s in grid]
    return float(grid[int(np.argmin(losses))])


def score(rows: pd.DataFrame) -> dict:
    """Scores for a set of forecasts with columns q10..q90, last, actual."""
    if rows is None or rows.empty:
        return {"n": 0}
    q = rows[QCOLS].to_numpy()
    last, actual = rows["last"].to_numpy(), rows["actual"].to_numpy()
    med = rows["q50"].to_numpy()
    moved = (actual != last) & (med != last)
    direction = (np.sign(actual - last) == np.sign(med - last))[moved]
    err_model = np.abs(actual - med) / last * 1e4
    err_naive = np.abs(actual - last) / last * 1e4
    return {
        "n": int(len(rows)),
        "coverage80": float(((actual >= rows["q10"]) & (actual <= rows["q90"])).mean()),
        "width80_bps": float(((rows["q90"] - rows["q10"]) / last * 1e4).mean()),
        "pinball_bps": float(pinball_bps(q, actual, last).mean()),
        "direction_acc": float(direction.mean()) if len(direction) else None,
        "mae_bps": float(err_model.mean()),
        "naive_mae_bps": float(err_naive.mean()),
    }
