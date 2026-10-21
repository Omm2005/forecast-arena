"""Download minute history, train each forecaster, and score them on unseen days.

Time is split three ways per ticker, so nothing is scored on data it learned from:
    [ fine-tune ............ ][ calibrate: 1 session ][ test: last 2 sessions ]
- fine-tune: Chronos-2 is fine-tuned on these prices
- calibrate: each model's range width (spread scale) is fitted here
- test:      all three models are scored here, out of sample
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .markets import MARKETS
from .minute import LEVELS, MAX_GAP, QCOLS, fit_scale, pinball_bps, rescale, returns_bps, score

TRAIN_TICKERS = [s for m in MARKETS for s, _ in m["tickers"]]


def download_minutes(symbol: str, days: int = 28, retries: int = 3) -> pd.Series:
    """Completed 1-minute closes for the last `days` days (Yahoo serves 7 days per request)."""
    import yfinance as yf

    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    parts = []
    for k in range(int(np.ceil(days / 7))):
        e, s = end - pd.Timedelta(days=7 * k), end - pd.Timedelta(days=7 * (k + 1))
        for attempt in range(retries):
            try:
                h = yf.Ticker(symbol).history(start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
                                              interval="1m", auto_adjust=True)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            continue
        if not h.empty:
            parts.append(h["Close"])
    if not parts:
        return pd.Series(dtype=float)
    closes = pd.concat(parts).sort_index()
    closes = closes[~closes.index.duplicated()].dropna()
    closes = closes[closes.index.second == 0]
    return closes[closes.index + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz=closes.index.tz)]


def download_all(tickers=TRAIN_TICKERS, days: int = 28, min_bars: int = 2_000) -> dict[str, pd.Series]:
    data = {}
    for i, t in enumerate(tickers, 1):
        s = download_minutes(t, days)
        status = f"{len(s):,} bars" if len(s) >= min_bars else f"skipped ({len(s)} bars)"
        print(f"[{i}/{len(tickers)}] {t:<12} {status}")
        if len(s) >= min_bars:
            data[t] = s
    return data


def split_points(closes: pd.Series, test_sessions: int = 2, calib_sessions: int = 1):
    """Timestamps where the calibration and test windows start (by trading session)."""
    sessions = pd.Index(closes.index.normalize().unique())
    if len(sessions) < test_sessions + calib_sessions + 3:
        return None
    test_start = sessions[-test_sessions]
    calib_start = sessions[-(test_sessions + calib_sessions)]
    return calib_start, test_start


def eval_indices(closes: pd.Series, start, end, n: int, min_history: int = 600) -> list[int]:
    """Up to n evenly spaced bars i whose next bar (the target) falls in [start, end)."""
    t = closes.index
    ok = [i for i in range(min_history, len(closes) - 1)
          if start <= t[i + 1] < end and t[i + 1] - t[i] <= MAX_GAP]
    if len(ok) > n:
        ok = [ok[k] for k in np.linspace(0, len(ok) - 1, n).astype(int)]
    return ok


def collect(models: dict, data: dict[str, pd.Series], window: str, n_per_ticker: int,
            verbose: bool = True) -> pd.DataFrame:
    """Forecast every sampled point in the given window with every model."""
    rows = []
    for sym, closes in data.items():
        sp = split_points(closes)
        if sp is None:
            continue
        calib_start, test_start = sp
        start, end = (calib_start, test_start) if window == "calib" else (test_start, closes.index[-1] + MAX_GAP)
        idxs = eval_indices(closes, start, end, n_per_ticker)
        if not idxs:
            continue
        last = closes.to_numpy()[idxs]
        actual = closes.to_numpy()[[i + 1 for i in idxs]]
        for key, m in models.items():
            if not m.available:
                continue
            q = m.replay(closes, idxs)
            df = pd.DataFrame(q, columns=QCOLS)
            df["model"], df["symbol"], df["last"], df["actual"] = key, sym, last, actual
            df["time"] = closes.index[[i + 1 for i in idxs]]
            rows.append(df.dropna(subset=QCOLS))
        if verbose:
            print(f"  {window}: {sym:<12} {len(idxs)} points")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibrate(models: dict, calib_rows: pd.DataFrame) -> dict[str, float]:
    """Fit each model's spread scale on the calibration window (raw outputs, scale = 1)."""
    scales = {}
    for key, m in models.items():
        r = calib_rows[calib_rows["model"] == key]
        if r.empty:
            continue
        m.scale = fit_scale(r[QCOLS].to_numpy(), r["actual"].to_numpy(), r["last"].to_numpy())
        scales[key] = m.scale
        print(f"  {m.name:<12} spread scale = {m.scale:.2f}")
    return scales


def summarize(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, m in models.items():
        r = rows[rows["model"] == key]
        if r.empty:
            continue
        s = score(r)
        out.append({"model": m.name, "points": s["n"], "in 8-in-10 range": s["coverage80"],
                    "avg range width (bps)": s["width80_bps"], "pinball loss (bps)": s["pinball_bps"],
                    "direction right": s["direction_acc"]})
    return pd.DataFrame(out).set_index("model").sort_values("pinball loss (bps)")


def save_report(path: str | Path, models: dict, zero_shot: pd.DataFrame, trained: pd.DataFrame,
                scales: dict, data: dict, extra: dict | None = None) -> dict:
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "tickers": len(data),
        "bars": int(sum(len(s) for s in data.values())),
        "scales": scales,
        "zero_shot": zero_shot.reset_index().to_dict(orient="records"),
        "configs": {k: {"input_mode": m.input_mode, "context": m.context}
                    for k, m in models.items() if hasattr(m, "input_mode")},
        "trained": trained.reset_index().to_dict(orient="records"),
        **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str))
    return report


def validation_windows(data: dict[str, pd.Series], per_ticker: int = 20, context: int = 512,
                       mode: str = "prices", seed: int = 0) -> list[np.ndarray]:
    """Held-out windows from the calibration session: `context` values followed by the target value.

    `mode` matches the model's input: prices, or 1-minute returns in bps. Used to watch validation
    loss while fine-tuning; none of these targets are in the training series.
    """
    rng = np.random.default_rng(seed)
    out = []
    for closes in data.values():
        sp = split_points(closes)
        if sp is None:
            continue
        if mode == "returns":
            r = returns_bps(closes)
            ok = np.flatnonzero((r.index >= sp[0]) & (r.index < sp[1]))
            ok = ok[ok >= context]
            for j in rng.choice(ok, size=min(per_ticker, len(ok)), replace=False):
                out.append(r.to_numpy(dtype=np.float32)[j - context:j + 1])
            continue
        t = closes.index
        ok = [i for i in range(context, len(closes))
              if sp[0] <= t[i] < sp[1] and t[i] - t[i - 1] <= MAX_GAP]
        for i in rng.choice(ok, size=min(per_ticker, len(ok)), replace=False):
            out.append(closes.to_numpy(dtype=np.float32)[i - context:i + 1])
    return out


def _pretrain(closes: pd.Series) -> pd.Series | None:
    sp = split_points(closes)
    return None if sp is None else closes[closes.index < sp[0]]


def finetune_series(data: dict[str, pd.Series], split_sessions: bool = False,
                    min_len: int = 65, mode: str = "prices") -> list[np.ndarray]:
    """Per-ticker histories that end before the calibration window.

    mode "returns": 1-minute returns in bps with gap returns dropped, one series per ticker.
    mode "prices" with `split_sessions`: prices cut at overnight / lunch gaps, so no training target
    is a next-morning opening price (a jump no 1-minute model should be fitted to).
    """
    out = []
    for closes in data.values():
        c = _pretrain(closes)
        if c is None:
            continue
        if mode == "returns":
            out.append(returns_bps(closes)[lambda r: r.index <= c.index[-1]].to_numpy(dtype=np.float32))
            continue
        if not split_sessions:
            out.append(c.to_numpy(dtype=np.float32))
            continue
        session_id = (c.index.to_series().diff() > MAX_GAP).cumsum().to_numpy()
        for k in np.unique(session_id):
            seg = c.to_numpy(dtype=np.float32)[session_id == k]
            if len(seg) >= min_len:
                out.append(seg)
    return out


def finetune_targets(data: dict[str, pd.Series], context: int = 512) -> list[np.ndarray]:
    """For each series from finetune_series(data), the positions that are valid prediction targets:
    a full `context` of history before them and no gap between the target and the bar before it."""
    out = []
    for closes in data.values():
        c = _pretrain(closes)
        if c is None:
            continue
        ok = np.ones(len(c), dtype=bool)
        ok[:context] = False
        ok[1:] &= (c.index[1:] - c.index[:-1]) <= MAX_GAP
        out.append(np.flatnonzero(ok))
    return out


def select_config(key: str, model, data: dict[str, pd.Series], modes=("prices", "returns"),
                  contexts=(128, 512), n_per_ticker: int = 60) -> pd.DataFrame:
    """Try each input mode and context length on the calibration session; keep the best.

    Configurations are compared by pinball loss after each gets its best range width, so the choice
    reflects where the ranges sit and how they adapt, not just their starting width. The test
    sessions are never used here.
    """
    rows = []
    for mode in modes:
        for ctx in contexts:
            model.input_mode, model.context, model.scale = mode, ctx, 1.0
            r = collect({key: model}, data, "calib", n_per_ticker, verbose=False)
            q, actual, last = r[QCOLS].to_numpy(), r["actual"].to_numpy(), r["last"].to_numpy()
            sc = fit_scale(q, actual, last)
            rows.append({"input": mode, "context": ctx, "pinball (bps)": float(pinball_bps(rescale(q, sc), actual, last).mean()),
                         "raw coverage 80%": float(((actual >= r["q10"]) & (actual <= r["q90"])).mean())})
    table = pd.DataFrame(rows).sort_values("pinball (bps)").reset_index(drop=True)
    model.input_mode, model.context = table.loc[0, "input"], int(table.loc[0, "context"])
    print(f"  {model.name}: using {model.input_mode} input, context {model.context}")
    return table
