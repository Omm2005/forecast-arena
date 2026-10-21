# Forecast Arena

**A classic Monte Carlo simulation vs. two open-source AI foundation models, competing live to forecast the next minute of real market prices.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omm2005/forecast-arena/blob/main/colab/Forecast_Arena.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/models-Monte%20Carlo%20%7C%20Chronos--2%20%7C%20TimesFM%202.5-orange)

Pick any stock, currency or crypto from 15 markets. Every minute, three models each predict a range for the next
1-minute price. The forecasts are locked in before the price exists, then scored the moment it arrives.

| | Model | How it forecasts |
|---|---|---|
| 🎲 | **Monte Carlo** | EWMA volatility + momentum, 10,000 simulated paths drawn from the ticker's own recent forecast errors |
| 🟠 | **Chronos-2** (Amazon) | 120M-parameter time-series foundation model, LoRA fine-tuned on minute prices |
| 🟢 | **TimesFM 2.5** (Google) | 200M-parameter time-series foundation model, LoRA fine-tuned on minute prices |

The repo also includes a [portfolio Value-at-Risk engine](#portfolio-var-engine) (100,000 correlated Monte Carlo paths across 50 assets).

---

## Results

Scored on the last 2 trading sessions (7,200 forecasts across 48 tickers in 15 markets), which no model saw during training.
Pinball (quantile) loss in basis points, lower is better:

| Model | Out of the box (price input) | Return input | Fine-tuned (LoRA) |
|---|---|---|---|
| **Chronos-2** | 2.093 | 1.986 | **1.979** 🏆 |
| Monte Carlo | 1.990 | 1.990 | 1.990 |
| TimesFM 2.5 | 2.055 | 2.015 | 1.998 |

- **Reading prices, both AI models lose to Monte Carlo.** They rescale each window by the price level, so the next minute's move
  is a rounding error and they collapse to "next price ≈ last price".
- **Switching their input to 1-minute returns fixes most of it**: Chronos-2 improves 5.1% and overtakes Monte Carlo. The
  input and context length (returns, 512 minutes) were chosen on a held-out calibration session, never the test days.
- **LoRA fine-tuning adds a little more** (Chronos-2 −0.4%, TimesFM −0.8%), for a total gain of 5.4% and 2.8%.
- Margins are small because 1-minute prices are mostly noise; direction accuracy stays at a coin flip (50–51%) for every model.

The trained LoRA adapters and settings are in [`models/`](models/), and the app loads them automatically.

## How a round works

```
 1. Aim                       2. Wait                     3. Score
 ──────────────────────       ─────────────────────       ──────────────────────────────
 Every player locks in its    Next 1-minute price         🎯 Bullseye (inner zone)  2 pts
 best guess before the        arrives from Yahoo          Hit (inside the range)    1 pt
 price exists                 Finance                     Miss                      0 pts
```

- **Same range for everyone.** Each round, all players get the same range width: the average of their own
  10th–90th percentile ranges. Each player's range is centered on its own best guess (median), so the game only rewards aim.
- **Bullseye zone** = the middle 20% of the range. A well-calibrated model lands about 20% bullseyes and 80% hits-or-better,
  so roughly 1 point per round is par.
- **Warm-up** replays the last 2 hours the moment you pick a ticker, so the scoreboard isn't empty while the live match starts.
- **Pro stats** still judge each model's *own* range: pinball (quantile) loss, which rewards ranges that are both correct and
  narrow, plus 80% coverage and direction accuracy.

## Quick start

### 1. Train the AI models (Google Colab, free GPU)

Click **Open in Colab** above, choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** (20–30 minutes). The notebook:

1. Downloads ~4 weeks of 1-minute prices for ~45 tickers across 15 markets
2. Scores all three models out of the box (baseline)
3. Fine-tunes Chronos-2 and TimesFM with LoRA adapters
4. Tunes every model's range width on a held-out session
5. Re-scores on the last 2 trading sessions, which no model saw in training
6. Downloads `trained_models.zip` and opens the live arena inside Colab

```
data per ticker:  [ fine-tune (older weeks) ][ calibrate: 1 session ][ test: last 2 sessions ]
```

### 2. Run the arena locally with your trained models

```bash
git clone https://github.com/Omm2005/forecast-arena.git && cd forecast-arena
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-ai.txt   # includes PyTorch (~1 GB)
unzip trained_models.zip -d models/                                # adapters + range widths
.venv/bin/python app.py                                            # → http://127.0.0.1:5050
```

The base models download from Hugging Face on first run; the LoRA adapters in `models/` are applied on top. It runs on CPU.
No `models/` folder? You get the untrained base models. `python app.py --no-ai` runs Monte Carlo alone.

## Markets

United States · Canada · Brazil · United Kingdom · Germany · France · Switzerland · India · Singapore ·
Hong Kong · South Korea · Japan · Australia · FX (24/5) · Crypto (24/7). Open markets are listed first, and any
Yahoo Finance symbol works.

Data is free 1-minute bars from Yahoo Finance (no API key). Some exchanges (Tokyo, Hong Kong) are delayed ~15 minutes
by Yahoo; forecasts are still locked in before each bar is published.

---

## Portfolio VaR engine

Monte Carlo Value-at-Risk for multi-asset portfolios on 5 years of daily prices.

- **Correlated simulation**: Cholesky-factored covariance turns independent normals into correlated shocks; 100,000 multi-day paths across 50 assets in seconds.
- **Risk metrics**: VaR and CVaR (Expected Shortfall) at 95% and 99%, plus Euler component VaR showing which positions drive risk.
- **Stress tests**: vol spike, correlation breakdown, 2008-style crash, rate shock.
- **Model validation**: rolling out-of-sample VaR backtest with Kupiec and Christoffersen tests and a Basel traffic-light zone.
- **Automated reporting**: one command writes a self-contained HTML report plus JSON/CSV.

```bash
.venv/bin/python run.py                          # 50 large-cap US stocks
.venv/bin/python run.py --tickers AAPL MSFT JPM  # your own universe
.venv/bin/python run.py --prices prices.csv      # your own price history
.venv/bin/python run.py --synthetic              # offline, simulated market
```

## Project layout

```
app.py                  Arena web server (Flask)
web/index.html          Arena UI
riskmc/minute.py        Monte Carlo minute forecaster + shared scoring (pinball loss, coverage)
riskmc/forecasters.py   Monte Carlo, Chronos-2, TimesFM behind one interface (+ LoRA fine-tuning)
riskmc/training.py      Data download, train / calibrate / test split, evaluation
riskmc/markets.py       Exchanges, trading hours, tickers
colab/                  Training notebook and the script that builds it from the source
run.py, riskmc/*.py     Portfolio VaR engine (montecarlo, stress, backtest, report)
tests/                  pytest suite
```

## Tests

```bash
.venv/bin/python -m pytest
```
