# Forecast Arena

Two parts: a Monte Carlo Value-at-Risk engine for multi-asset portfolios, and a live arena where a Monte Carlo
model competes with two open-source AI forecasting models, **Chronos-2** (Amazon) and **TimesFM 2.5** (Google),
at predicting the next 1-minute price.

## 1. Portfolio VaR engine

- **Correlated simulation**: Cholesky-factored covariance turns independent normals into correlated daily shocks; 100,000 multi-day price paths across 50 assets in a few seconds (chunked to bound memory).
- **Risk metrics**: VaR and CVaR / Expected Shortfall at 95% and 99%, plus Euler component VaR showing which positions drive risk.
- **Stress testing**: vol spike, correlation breakdown, 2008-style crash, rate shock (shocks applied to the covariance, then re-simulated).
- **Model validation**: rolling out-of-sample 1-day VaR backtest with Kupiec POF and Christoffersen independence tests and a Basel traffic-light zone.
- **Automated reporting**: one command produces a self-contained HTML report plus JSON/CSV outputs.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                         # 50 large-cap US stocks, 5y of daily Yahoo Finance data
.venv/bin/python run.py --tickers AAPL MSFT JPM # custom universe
.venv/bin/python run.py --prices prices.csv     # your own data (date column + one column per asset)
.venv/bin/python run.py --synthetic             # offline, simulated universe
.venv/bin/python -m pytest
```

Downloads are cached in `data/` once per day (`--refresh` to re-fetch).

## 2. Minute forecast arena: Monte Carlo vs Chronos-2 vs TimesFM

Every model predicts the 10th–90th percentile range of the next 1-minute close for stocks, FX and crypto across
15 markets. In the web app, each minute all three lock in a forecast before the price exists and are scored when it arrives.

| Model | How it forecasts | Training |
|---|---|---|
| Monte Carlo | EWMA volatility + shrunk AR(1) momentum, 10,000 simulated paths drawn from the ticker's own recent forecast errors | Refits every minute |
| Chronos-2 | Amazon's 120M-parameter time-series foundation model | LoRA fine-tuned on ~4 weeks of minute prices |
| TimesFM 2.5 | Google's 200M-parameter time-series foundation model | LoRA fine-tuned on the same data |

Every model also gets one range-width multiplier fitted on a held-out session. Models are ranked by **pinball
(quantile) loss**, which rewards ranges that catch the price *and* stay narrow, alongside 8-in-10 range coverage
(target 80%) and direction accuracy.

### Train (Google Colab, free T4 GPU)

Open `colab/Minute_Forecast_Lab.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**, then **Run all**
(about 20–30 minutes). It downloads the data, scores the models before and after training on the last 2 sessions
(which no model trains on), opens the live app inside Colab, and downloads `trained_models.zip`.

The notebook is generated from the project source: `python colab/build_notebook.py`.

### Run the trained models locally

```bash
.venv/bin/pip install -r requirements-ai.txt     # PyTorch, Transformers, PEFT, Chronos (~1 GB)
unzip trained_models.zip -d models/              # models/report.json, models/chronos2-lora/, models/timesfm-lora/
.venv/bin/python app.py                          # open http://127.0.0.1:5050
```

The app downloads the base models from Hugging Face once, applies the LoRA adapters and range widths from `models/`,
and runs on CPU. Without `models/` it uses the untrained base models; `python app.py --no-ai` runs Monte Carlo alone.

Data is free 1-minute bars from Yahoo Finance (no API key). Some exchanges (e.g. Tokyo, Hong Kong) are delayed
~15 minutes by Yahoo; forecasts are still locked in before each bar is published.
