"""The three competing next-minute forecasters behind one interface.

Each forecaster maps a list of price histories to a (batch, 9) array of price
quantiles at LEVELS (10th..90th percentile) for the next bar. A learned spread
`scale` (see minute.fit_scale) is applied on top of every model's raw output.

- Monte Carlo: filtered historical simulation (riskmc.minute)
- Chronos-2:   Amazon's time-series foundation model (chronos-forecasting, Apache 2.0)
- TimesFM 2.5: Google's time-series foundation model (Hugging Face Transformers port, Apache 2.0)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from .minute import LEVELS, mc_quantiles, model_frame, rescale

NAN_ROW = np.full(len(LEVELS), np.nan)


class Forecaster:
    key = "base"
    name = "Base"
    blurb = ""

    def __init__(self):
        self.scale = 1.0
        self.available = True
        self.error = None

    def raw_quantiles(self, histories: list[pd.Series]) -> np.ndarray:
        raise NotImplementedError

    def quantiles(self, histories: list[pd.Series]) -> np.ndarray:
        return rescale(self.raw_quantiles(histories), self.scale)

    def replay(self, closes: pd.Series, idxs: list[int]) -> np.ndarray:
        """Quantiles for bar i+1 for each i in idxs, using only bars up to i."""
        return self.quantiles([closes.iloc[:i + 1] for i in idxs])

    def info(self) -> dict:
        return {"key": self.key, "name": self.name, "blurb": self.blurb, "available": self.available,
                "error": self.error, "scale": self.scale, "fine_tuned": getattr(self, "fine_tuned", False)}


class MonteCarloForecaster(Forecaster):
    key, name = "mc", "Monte Carlo"
    blurb = "Volatility + momentum model, 10,000 simulated paths from the stock's own recent surprises"

    def __init__(self, n_paths: int = 10_000, seed: int | None = None):
        super().__init__()
        self.n_paths, self.rng = n_paths, np.random.default_rng(seed)

    def raw_quantiles(self, histories):
        out = []
        for h in histories:
            q = mc_quantiles(h, model_frame(h), len(h) - 1, n_paths=self.n_paths, rng=self.rng)
            out.append(NAN_ROW if q is None else q)
        return np.array(out)

    def replay(self, closes, idxs):
        frame = model_frame(closes)                       # row i only depends on bars <= i
        out = []
        for i in idxs:
            q = mc_quantiles(closes, frame, i, n_paths=self.n_paths, rng=self.rng)
            out.append(NAN_ROW if q is None else q)
        return rescale(np.array(out), self.scale)


def _batches(items, size):
    for k in range(0, len(items), size):
        yield items[k:k + size]


class Chronos2Forecaster(Forecaster):
    key, name = "chronos2", "Chronos-2"
    blurb = "Amazon's 120M-parameter time-series foundation model"

    def __init__(self, path: str = "amazon/chronos-2", device: str = "cpu", context: int = 512,
                 batch_size: int = 128):
        """`path` is a Hugging Face model id or a fine-tuned checkpoint folder (LoRA adapters load directly)."""
        super().__init__()
        from chronos import BaseChronosPipeline

        self.path, self.device, self.context, self.batch_size = path, device, context, batch_size
        self.pipe = BaseChronosPipeline.from_pretrained(path, device_map=device)
        self.fine_tuned = path != "amazon/chronos-2"

    def _predict(self, arrays: list[np.ndarray]) -> np.ndarray:
        import torch

        out = self.pipe.predict_quantiles([torch.tensor(a) for a in arrays], prediction_length=1,
                                          quantile_levels=LEVELS.tolist())
        q = out[0] if isinstance(out, tuple) else out
        if isinstance(q, (list, tuple)):                  # Chronos-2: one tensor per series
            return np.stack([np.asarray(x.float().cpu()).reshape(-1)[:len(LEVELS)] for x in q])
        return np.asarray(q.float().cpu()).reshape(len(arrays), -1)[:, :len(LEVELS)]

    def raw_quantiles(self, histories):
        arrays = [h.to_numpy(dtype=np.float32)[-self.context:] for h in histories]
        return np.concatenate([self._predict(b) for b in _batches(arrays, self.batch_size)])

    def fine_tune(self, series: list[np.ndarray], out_dir: str, num_steps: int = 600,
                  learning_rate: float = 1e-5, batch_size: int = 64, mode: str = "lora") -> None:
        """Fine-tune on minute price histories with chronos-forecasting's own `fit` (LoRA by default)."""
        if not hasattr(self.pipe, "fit"):
            raise RuntimeError("This chronos-forecasting version has no fit(); upgrade with "
                               "pip install -U chronos-forecasting")
        wanted = {"inputs": [s.astype(np.float32) for s in series], "prediction_length": 1,
                  "finetune_mode": mode, "num_steps": num_steps, "learning_rate": learning_rate, "batch_size": batch_size,
                  "context_length": self.context, "output_dir": out_dir}
        params = inspect.signature(self.pipe.fit).parameters
        kwargs = {k: v for k, v in wanted.items() if k in params}
        print("Chronos-2 fit() with", {k: v for k, v in kwargs.items() if k != "inputs"})
        self.pipe = self.pipe.fit(**kwargs)                # fit() saves to out_dir/finetuned-ckpt itself
        self.path, self.fine_tuned = str(Path(out_dir) / "finetuned-ckpt"), True


class TimesFMForecaster(Forecaster):
    key, name = "timesfm", "TimesFM 2.5"
    blurb = "Google's 200M-parameter time-series foundation model"

    def __init__(self, path: str = "google/timesfm-2.5-200m-transformers", device: str = "cpu",
                 context: int = 512, batch_size: int = 128, adapter: str | None = None):
        """`adapter` is a folder with LoRA weights saved by `fine_tune`, applied on top of the base model."""
        super().__init__()
        from transformers import TimesFm2_5ModelForPrediction

        self.device, self.context, self.batch_size = device, context, batch_size
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(path, torch_dtype="float32").to(device)
        self.fine_tuned = adapter is not None
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).to(device)
        self.model.eval()

    def raw_quantiles(self, histories):
        import torch

        arrays = [h.to_numpy(dtype=np.float32)[-self.context:] for h in histories]
        out = []
        with torch.no_grad():
            for b in _batches(arrays, self.batch_size):
                res = self.model(past_values=[torch.tensor(a, device=self.device) for a in b],
                                 forecast_context_len=self.context)
                # full_predictions: (batch, horizon, 10) = mean, then the 10th..90th percentiles
                out.append(res.full_predictions[:, 0, 1:1 + len(LEVELS)].float().cpu().numpy())
        return np.concatenate(out)

    def fine_tune(self, series: list[np.ndarray], out_dir: str, num_steps: int = 600,
                  learning_rate: float = 1e-4, batch_size: int = 32, lora_r: int = 4, seed: int = 0) -> list[float]:
        """LoRA fine-tuning, following TimesFM's own example (timesfm-forecasting/examples/finetuning).

        Each step samples random (512 context -> next price) windows from the training
        series; the model's built-in loss (MSE + quantile loss in normalized space) is minimized.
        """
        import torch
        from peft import LoraConfig, get_peft_model

        rng = np.random.default_rng(seed)
        series = [s for s in series if len(s) > self.context + 1]
        model = get_peft_model(self.model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules="all-linear",
                                                      lora_dropout=0.05, bias="none"))
        model.print_trainable_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)
        losses = []
        model.train()
        for step in range(1, num_steps + 1):
            ctx, tgt = [], []
            for _ in range(batch_size):
                s = series[rng.integers(len(series))]
                start = rng.integers(0, len(s) - self.context - 1)
                ctx.append(torch.tensor(s[start:start + self.context], device=self.device))
                tgt.append(float(s[start + self.context]))
            loss = model(past_values=ctx, future_values=torch.tensor(tgt, device=self.device)[:, None],
                         forecast_context_len=self.context).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            losses.append(loss.item())
            if step % 50 == 0 or step == num_steps:
                print(f"  TimesFM step {step}/{num_steps}  loss {np.mean(losses[-50:]):.4f}")
        model.eval()
        model.save_pretrained(out_dir)
        self.model, self.fine_tuned = model, True
        return losses


def build(load_ai: bool = True, device: str | None = None, chronos_path: str | None = None,
          timesfm_adapter: str | None = None, scales: dict | None = None) -> dict[str, Forecaster]:
    """All forecasters that can be loaded here. A model that fails to load stays listed as unavailable."""
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    models: dict[str, Forecaster] = {"mc": MonteCarloForecaster()}
    specs = [("chronos2", lambda: Chronos2Forecaster(chronos_path or "amazon/chronos-2", device)),
             ("timesfm", lambda: TimesFMForecaster(device=device, adapter=timesfm_adapter))]
    for key, make in specs:
        if not load_ai:
            continue
        try:
            models[key] = make()
        except Exception as e:                            # keep the app usable without this model
            stub = Forecaster()
            stub.key = key
            stub.name = {"chronos2": Chronos2Forecaster.name, "timesfm": TimesFMForecaster.name}[key]
            stub.available, stub.error = False, f"{type(e).__name__}: {e}"
            models[key] = stub
            print(f"Could not load {stub.name}: {stub.error}")
    for key, s in (scales or {}).items():
        if key in models:
            models[key].scale = float(s)
    return models
