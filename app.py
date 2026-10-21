"""Forecast Arena: Monte Carlo vs Chronos-2 vs TimesFM, scored live on real minute prices.

    python app.py              # local, Monte Carlo only (AI models load if installed)
    In Colab: see colab/Forecast_Arena.ipynb, which calls serve(models, report)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, send_from_directory

from riskmc.markets import catalog, market_for, status
from riskmc.minute import MAX_GAP, QCOLS, score

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "web"))

CACHE_SECONDS = 5
REPLAY_BARS = 120
MODELS: dict = {}
REPORT: dict | None = None

_lock = threading.Lock()
_model_lock = threading.Lock()                               # one model call at a time (shared GPU)
_bars: dict[str, tuple[float, pd.Series, dict]] = {}       # symbol -> (fetched_at, closes, meta)
_replays: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}
_live: dict[str, dict[pd.Timestamp, dict]] = {}           # symbol -> base bar -> {made_at, last, q: {model: array}}
_feed: dict[str, dict] = {}                               # symbol -> newest bar seen + feed latency


def get_bars(symbol: str) -> tuple[pd.Series, dict]:
    """Completed 1-minute closes for the last few sessions (cached briefly)."""
    with _lock:
        hit = _bars.get(symbol)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1], hit[2]

    t = yf.Ticker(symbol)
    hist = t.history(period="5d", interval="1m", auto_adjust=True)
    if hist.empty:
        raise LookupError(f"No minute data for {symbol}")
    closes = hist["Close"].dropna()
    closes = closes[closes.index.second == 0]                          # drop Yahoo's partial "live" row
    now = pd.Timestamp.now(tz=closes.index.tz)
    closes = closes[closes.index + pd.Timedelta(minutes=1) <= now]     # drop the bar still forming
    meta = t.history_metadata or {}
    with _lock:
        _bars[symbol] = (time.time(), closes, meta)
    return closes, meta


def available_models() -> dict:
    return {k: m for k, m in MODELS.items() if m.available}


def get_replay(symbol: str, closes: pd.Series) -> pd.DataFrame:
    """Every model's forecast for each of the last REPLAY_BARS bars, using only earlier bars."""
    last = closes.index[-1]
    with _lock:
        hit = _replays.get(symbol)
    if hit and hit[0] == last:
        return hit[1]
    t = closes.index
    idxs = [i for i in range(max(600, len(closes) - REPLAY_BARS - 1), len(closes) - 1)
            if t[i + 1] - t[i] <= MAX_GAP]
    frames = []
    if idxs:
        for key, m in available_models().items():
            with _model_lock:
                q = m.replay(closes, idxs)
            df = pd.DataFrame(q, columns=QCOLS)
            df["model"], df["time"] = key, t[[i + 1 for i in idxs]]
            df["last"], df["actual"] = closes.to_numpy()[idxs], closes.to_numpy()[[i + 1 for i in idxs]]
            frames.append(df.dropna(subset=QCOLS))
    rp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    with _lock:
        _replays[symbol] = (last, rp)
    return rp


def update_live(symbol: str, closes: pd.Series) -> tuple[dict, list[dict]]:
    """Lock in every model's forecast the first time a bar is seen; score it when the next bar lands."""
    base = closes.index[-1]
    with _lock:
        have = base in _live.get(symbol, {})
    if not have:
        q = {}
        for key, m in available_models().items():
            with _model_lock:
                row = m.quantiles([closes])[0]
            if np.all(np.isfinite(row)):
                q[key] = row
        with _lock:
            _live.setdefault(symbol, {})[base] = {
                "made_at": datetime.now(timezone.utc), "last": float(closes.iloc[-1]), "q": q}

    results = []
    with _lock:
        preds = dict(_live[symbol])
    for b, p in sorted(preds.items()):
        later = closes[closes.index > b]
        if later.empty:
            continue
        nxt = later.index[0]
        results.append({"base": b, "time": nxt, "last": p["last"], "actual": float(later.iloc[0]),
                        "gap": nxt - b > MAX_GAP, "q": p["q"]})
    return preds[base], results


def feed_latency(symbol: str, closes: pd.Series) -> float:
    """Seconds between a bar closing and Yahoo publishing it, learned as new bars appear."""
    newest = closes.index[-1]
    since_close = (pd.Timestamp.now(tz=newest.tz) - newest).total_seconds() - 60
    with _lock:
        feed = _feed.get(symbol)
        if feed is None:
            # First look: a fresh feed is assumed ~20s behind; a stale one is at least this far behind.
            feed = _feed[symbol] = {"bar": newest, "latency": since_close if since_close > 120 else 20.0}
        elif newest > feed["bar"]:
            if newest - feed["bar"] <= pd.Timedelta(minutes=2):      # we watched it arrive
                feed["latency"] = 0.5 * feed["latency"] + 0.5 * max(since_close, 0.0)
            feed["bar"] = newest
        return feed["latency"]


def _hhmm(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _qdict(q) -> dict:
    return {c: float(v) for c, v in zip(QCOLS, q)}


def _inside(q, actual) -> bool:
    return bool(q[0] <= actual <= q[-1])


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/markets")
def markets():
    return jsonify({"markets": catalog(), "utc": datetime.now(timezone.utc).isoformat()})


@app.get("/api/forecast/<symbol>")
def forecast_api(symbol: str):
    symbol = symbol.upper().strip()
    try:
        closes, meta = get_bars(symbol)
    except Exception as e:
        return jsonify({"error": f"Couldn't load {symbol}: {e}"}), 404
    if len(closes) < 700:
        return jsonify({"error": f"Not enough minute history for {symbol} yet (need about 2 sessions)"}), 404

    replay = get_replay(symbol, closes)
    pred, results = update_live(symbol, closes)
    keys = list(available_models())

    m = market_for(symbol)
    latency = feed_latency(symbol, closes)
    target = closes.index[-1] + pd.Timedelta(minutes=1)
    expected_in = (target + pd.Timedelta(minutes=1, seconds=latency)
                   - pd.Timestamp.now(tz=target.tz)).total_seconds()
    age = (pd.Timestamp.now(tz=closes.index.tz) - closes.index[-1]).total_seconds() - 60

    # chart: last hour of actual prices with each model's replayed 8-in-10 range
    by_model = {k: replay[replay["model"] == k].set_index("time") for k in keys} if not replay.empty else {}
    points = []
    for t, c in closes.iloc[-60:].items():
        p = {"t": _hhmm(t), "actual": float(c), "ranges": {}}
        for k, df in by_model.items():
            if t in df.index:
                p["ranges"][k] = [float(df.at[t, "q10"]), float(df.at[t, "q90"])]
        points.append(p)

    # live: every scored forecast this session, per model
    scored = [r for r in results if not r["gap"]]
    live_scores = {}
    for k in keys:
        rows = pd.DataFrame([{**_qdict(r["q"][k]), "last": r["last"], "actual": r["actual"]}
                             for r in scored if k in r["q"]])
        live_scores[k] = score(rows)
    last_result = None
    if scored:
        r = scored[-1]
        last_result = {"target": _hhmm(r["time"]), "actual": r["actual"], "models": {
            k: {"q10": float(r["q"][k][0]), "q90": float(r["q"][k][-1]), "inside": _inside(r["q"][k], r["actual"])}
            for k in keys if k in r["q"]}}
    streak = {k: [{"t": _hhmm(r["time"]), "inside": _inside(r["q"][k], r["actual"])}
                  for r in scored[-40:] if k in r["q"]] for k in keys}
    log = [{"target": _hhmm(r["time"]), "actual": r["actual"], "gap": bool(r["gap"]),
            "models": {k: [float(r["q"][k][0]), float(r["q"][k][-1])] for k in keys if k in r["q"]}}
           for r in reversed(results[-30:])]

    day_open = closes[closes.index.normalize() == closes.index[-1].normalize()].iloc[0]
    return jsonify(_clean({
        "symbol": symbol,
        "name": meta.get("longName") or meta.get("shortName") or symbol,
        "exchange": m["exchange"] if m else meta.get("exchangeName", ""),
        "currency": meta.get("currency", ""),
        "tz": str(closes.index.tz),
        "market": status(m) if m else None,
        "last_price": float(closes.iloc[-1]),
        "last_bar": _hhmm(closes.index[-1]),
        "session_change": float(closes.iloc[-1] / day_open - 1),
        "data_age_sec": max(0.0, age),
        "models": [MODELS[k].info() for k in MODELS],
        "report": REPORT,
        "forecast": {"target": _hhmm(target), "last": pred["last"],
                     "models": {k: _qdict(q) for k, q in pred["q"].items()}},
        "points": points,
        "replay": {k: score(replay[replay["model"] == k]) if not replay.empty else {"n": 0} for k in keys},
        "live": {
            "made_at_local": pred["made_at"].astimezone(closes.index.tz).strftime("%H:%M:%S"),
            "made_at": pred["made_at"].isoformat(),
            "expected_in_sec": expected_in,
            "latency_sec": latency,
            "last_result": last_result,
            "streak": streak,
            "scores": live_scores,
            "log": log,
        },
    }))


def serve(models: dict, report: dict | None = None, port: int = 5050, host: str = "127.0.0.1",
          background: bool = False):
    """Run the app with the given forecasters (the Colab notebook runs it in the background)."""
    global MODELS, REPORT
    MODELS, REPORT = models, report
    run = lambda: app.run(host=host, port=port, threaded=True, use_reloader=False)
    if background:
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th
    run()


if __name__ == "__main__":
    import argparse

    from riskmc.forecasters import build

    p = argparse.ArgumentParser()
    p.add_argument("--no-ai", action="store_true", help="Monte Carlo only")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    a = p.parse_args()
    report_path = Path(a.models_dir) / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    chronos_dir, timesfm_dir = Path(a.models_dir) / "chronos2-lora", Path(a.models_dir) / "timesfm-lora"
    models = build(load_ai=not a.no_ai,
                   chronos_path=str(chronos_dir) if chronos_dir.exists() else None,
                   timesfm_adapter=str(timesfm_dir) if timesfm_dir.exists() else None,
                   scales=(report or {}).get("scales") if report else None)
    for m in models.values():
        state = "not loaded" if not m.available else "trained" if m.info()["fine_tuned"] else "base model"
        print(f"  {m.name:<12} {state}, range width x{m.scale:.2f}")
    print("Forecast Arena → http://127.0.0.1:5050")
    serve(models, report)
