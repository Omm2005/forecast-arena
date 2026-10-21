"""Automated risk report: runs every analysis and writes a self-contained HTML file."""

from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import rolling_backtest
from .montecarlo import component_var, estimate_parameters, simulate_portfolio
from .stress import SCENARIOS, apply_scenario


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _loss_histogram(sim) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(sim.pnl / 1e3, bins=200, color="#4c72b0", alpha=0.85)
    for c, color in zip(sim.confidence_levels, ["#dd8452", "#c44e52"]):
        ax.axvline(-sim.var[c] / 1e3, color=color, lw=2, label=f"{c:.0%} VaR")
    ax.set_xlabel(f"{sim.horizon_days}-day P&L ($K)")
    ax.set_ylabel("Paths")
    ax.set_title(f"Simulated P&L distribution ({len(sim.pnl):,} paths)")
    ax.legend()
    return _fig_to_b64(fig)


def _correlation_heatmap(returns: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(returns.corr().to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Asset correlation matrix")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _fig_to_b64(fig)


def _backtest_chart(bt) -> str:
    f = bt.frame
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(f.index, f["pnl"] / 1e3, lw=0.6, color="#555", label="Realized P&L")
    ax.plot(f.index, -f["var"] / 1e3, lw=1.4, color="#c44e52", label=f"-{bt.confidence:.0%} VaR")
    hits = f[f["breach"]]
    ax.scatter(hits.index, hits["pnl"] / 1e3, color="#c44e52", s=14, zorder=3, label="Breach")
    ax.set_ylabel("$K")
    ax.set_title("1-day VaR backtest")
    ax.legend(loc="lower left", fontsize=8)
    return _fig_to_b64(fig)


def build_report(
    prices: pd.DataFrame,
    weights: np.ndarray | None = None,
    portfolio_value: float = 10_000_000,
    horizon_days: int = 10,
    n_paths: int = 100_000,
    backtest_days: int = 250,
    out_dir: str | Path = "reports",
) -> Path:
    t0 = time.perf_counter()
    returns = np.log(prices / prices.shift(1)).dropna()
    n_assets = returns.shape[1]
    weights = np.full(n_assets, 1 / n_assets) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()

    mu, cov = estimate_parameters(returns, ewma_lambda=0.97)

    base = simulate_portfolio(mu, cov, weights, portfolio_value, horizon_days, n_paths)

    stress_rows = []
    for s in SCENARIOS:
        s_mu, s_cov = apply_scenario(mu, cov, s)
        r = simulate_portfolio(s_mu, s_cov, weights, portfolio_value, horizon_days, n_paths)
        stress_rows.append({"Scenario": s.name, "Description": s.description,
                            "95% VaR": r.var[0.95], "99% VaR": r.var[0.99],
                            "99% CVaR": r.cvar[0.99]})
    stress = pd.DataFrame(stress_rows)

    contrib = pd.Series(component_var(cov, weights, portfolio_value, 0.99, horizon_days),
                        index=returns.columns).sort_values(ascending=False)

    window = min(500, len(returns) - backtest_days)
    bt = rolling_backtest(returns.iloc[-(window + backtest_days):], weights,
                          portfolio_value, window=window, confidence=0.99)
    runtime = time.perf_counter() - t0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "generated": stamp, "assets": n_assets, "paths": n_paths,
        "horizon_days": horizon_days, "portfolio_value": portfolio_value,
        "var": {f"{c:.0%}": base.var[c] for c in base.confidence_levels},
        "cvar": {f"{c:.0%}": base.cvar[c] for c in base.confidence_levels},
        "backtest": {"days": bt.n_obs, "breaches": bt.n_breaches,
                     "expected": bt.expected_breaches, "kupiec_p": bt.kupiec_p,
                     "christoffersen_p": bt.christoffersen_p, "zone": bt.traffic_light},
        "runtime_seconds": runtime,
    }
    (out_dir / f"risk_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    stress.to_csv(out_dir / f"stress_{stamp}.csv", index=False)

    fmt = {c: _money for c in ["95% VaR", "99% VaR", "99% CVaR"]}
    top = contrib.head(10).to_frame("99% component VaR")
    top["Share"] = top["99% component VaR"] / contrib.sum()
    zone_color = {"GREEN": "#2a9d4b", "YELLOW": "#d4a017", "RED": "#c44e52"}[bt.traffic_light]

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Portfolio Risk Report</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:960px;margin:32px auto;padding:0 16px;color:#222}}
h1{{margin-bottom:4px}} .muted{{color:#777}} table{{border-collapse:collapse;margin:12px 0}}
td,th{{border-bottom:1px solid #ddd;padding:6px 12px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}} .card{{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:180px}}
.card b{{font-size:22px;display:block}} img{{max-width:100%}}
</style></head><body>
<h1>Portfolio Risk Report</h1>
<p class="muted">{stamp} · {n_assets} assets · {n_paths:,} Monte Carlo paths · {horizon_days}-day horizon · NAV {_money(portfolio_value)} · built in {runtime:.1f}s</p>
<div class="cards">
 <div class="card">95% VaR<b>{_money(base.var[0.95])}</b>{base.var[0.95]/portfolio_value:.2%} of NAV</div>
 <div class="card">99% VaR<b>{_money(base.var[0.99])}</b>{base.var[0.99]/portfolio_value:.2%} of NAV</div>
 <div class="card">99% CVaR<b>{_money(base.cvar[0.99])}</b>avg loss beyond VaR</div>
 <div class="card">Backtest<b style="color:{zone_color}">{bt.traffic_light}</b>{bt.n_breaches} breaches / {bt.expected_breaches:.1f} expected</div>
</div>
<h2>Loss distribution</h2><img src="data:image/png;base64,{_loss_histogram(base)}">
<h2>Stress scenarios</h2>{stress.to_html(index=False, formatters=fmt, border=0)}
<h2>Risk contributors (top 10)</h2>
{top.to_html(formatters={"99% component VaR": _money, "Share": "{:.1%}".format}, border=0)}
<h2>Model performance</h2>
<p>Rolling {window}-day EWMA fit, 1-day 99% VaR, {bt.n_obs} out-of-sample days.
Kupiec POF p = {bt.kupiec_p:.3f} · Christoffersen independence p = {bt.christoffersen_p:.3f}
(p &lt; 0.05 means the model is rejected).</p>
<img src="data:image/png;base64,{_backtest_chart(bt)}">
<h2>Correlation structure</h2><img src="data:image/png;base64,{_correlation_heatmap(returns)}">
</body></html>"""
    path = out_dir / f"risk_report_{stamp}.html"
    path.write_text(html)
    return path
