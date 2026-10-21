"""Build colab/Forecast_Arena.ipynb from the project's source files.

    python colab/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["riskmc/markets.py", "riskmc/minute.py", "riskmc/forecasters.py", "riskmc/training.py",
         "app.py", "web/index.html"]


DOWNLOAD_CELL = """
import shutil
from pathlib import Path
pkg = Path("/content/trained_models")
shutil.rmtree(pkg, ignore_errors=True); pkg.mkdir()
shutil.copy("/content/models/report.json", pkg / "report.json")
if Path("/content/models/chronos2-lora/finetuned-ckpt").exists():
    shutil.copytree("/content/models/chronos2-lora/finetuned-ckpt", pkg / "chronos2-lora")
if Path("/content/models/timesfm-lora").exists():
    shutil.copytree("/content/models/timesfm-lora", pkg / "timesfm-lora")
shutil.make_archive("/content/trained_models", "zip", pkg)
!du -sh /content/trained_models.zip
from google.colab import files
files.download("/content/trained_models.zip")
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n")}


cells = [
    md("""
# Forecast Arena: Monte Carlo vs Chronos-2 vs TimesFM

Three models predict where a stock's **next 1-minute price** will land, as a range (10th–90th percentile):

| Model | What it is |
|---|---|
| **Monte Carlo** | Volatility + momentum model, 10,000 simulated paths from the stock's own recent surprises |
| **Chronos-2** | Amazon's 120M-parameter time-series foundation model, fine-tuned here with LoRA |
| **TimesFM 2.5** | Google's 200M-parameter time-series foundation model, fine-tuned here with LoRA |

The notebook downloads ~4 weeks of 1-minute prices for ~45 stocks, FX pairs and crypto across 15 markets, trains the AI models,
scores all three on the last 2 trading sessions (which no model sees during training), then opens a live app that
compares them minute by minute.

**Before you start:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → **Run all** (about 20–30 minutes).
"""),
    md("## 1. Install and write the project files"),
    code("""
!pip install -q -U yfinance flask peft "transformers>=5" "chronos-forecasting>=2.2"
# Colab ships an old torchao that the latest peft refuses to load alongside; this project doesn't use it
!pip uninstall -y -q torchao
import os
os.makedirs("/content/lab/riskmc", exist_ok=True); os.makedirs("/content/lab/web", exist_ok=True)
open("/content/lab/riskmc/__init__.py", "w").close()
os.chdir("/content/lab")
"""),
]
for f in FILES:
    cells.append(code(f"%%writefile /content/lab/{f}\n" + (ROOT / f).read_text(), hidden=True))

cells += [
    code("""
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, "/content/lab")
from riskmc import training as T
from riskmc.forecasters import build
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: switch the runtime to T4 GPU for speed)")
"""),
    md("""
## 2. Download training data
About 4 weeks of 1-minute prices per ticker (Yahoo Finance keeps 1-minute bars for ~30 days). Each ticker's history is split by time:

`[ fine-tune (older weeks) ][ calibrate: 1 session ][ test: last 2 sessions ]`
"""),
    code("""
DATA = T.download_all(T.TRAIN_TICKERS, days=28)
print(f"\\n{len(DATA)} tickers, {sum(len(s) for s in DATA.values()):,} one-minute prices")
"""),
    md("## 3. Load the models (out of the box)"),
    code("""
MODELS = build(device=DEVICE)
for m in MODELS.values():
    print(f"{m.name:<12} {'loaded' if m.available else 'FAILED: ' + str(m.error)}")
"""),
    md("## 4. Score the untrained models on the test sessions\nThis is the baseline the training has to beat."),
    code("""
N_TEST = 150   # forecasts per ticker
ZERO_SHOT = T.summarize(MODELS, T.collect(MODELS, DATA, "test", N_TEST))
ZERO_SHOT.style.format(precision=3)
"""),
    md("""
## 5. Fine-tune Chronos-2
Uses chronos-forecasting's built-in `fit()` with LoRA adapters on the older weeks of every ticker.

**Reading the table:** *Validation Loss* is measured on held-out minutes from the calibration session (never trained on),
starting before step 1. It should fall below its starting value; the checkpoint with the lowest validation loss is kept.
One-minute prices are mostly noise, so expect small gains (a few percent), not a steep curve.
"""),
    code("""
FT_SERIES = T.finetune_series(DATA)                          # full histories (TimesFM)
FT_SESSIONS = T.finetune_series(DATA, split_sessions=True)   # cut at overnight gaps (Chronos-2)
FT_TARGETS = T.finetune_targets(DATA)                        # gap-free target positions (TimesFM)
VAL_WINDOWS = T.validation_windows(DATA)
print(f"{sum(len(s) for s in FT_SERIES):,} training prices · {len(FT_SESSIONS)} trading sessions · "
      f"{len(VAL_WINDOWS)} held-out validation windows")
if MODELS["chronos2"].available:
    MODELS["chronos2"].fine_tune(FT_SESSIONS, out_dir="/content/models/chronos2-lora", validation=VAL_WINDOWS,
                                 num_steps=600, learning_rate=1e-4)
"""),
    md("""
## 6. Fine-tune TimesFM 2.5
LoRA fine-tuning following TimesFM's own example: random 512-minute windows, predict the next minute,
minimizing the model's built-in quantile loss. Targets right after an overnight or lunch gap are skipped: an opening
jump isn't a 1-minute move, and a few of them can swamp training. Validation loss on the same held-out windows is checked every 25 steps
(and before training); the best adapter is kept.
"""),
    code("""
if MODELS["timesfm"].available:
    HIST = pd.DataFrame(MODELS["timesfm"].fine_tune(FT_SERIES, out_dir="/content/models/timesfm-lora",
                                                    validation=VAL_WINDOWS, targets=FT_TARGETS,
                                                    num_steps=600, learning_rate=5e-5, batch_size=64))
    ax = HIST.set_index("step")[["train", "val"]].rename(columns={"train": "training", "val": "validation (held out)"}).plot(
        marker="o", ms=3, figsize=(7, 3), title="TimesFM fine-tuning loss")
    ax.axhline(HIST["val"].iloc[0], ls="--", lw=1, color="gray", label="before training"); ax.legend();
"""),
    md("""
## 7. Tune every model's range width
One held-out session is used to fit a single spread multiplier per model (Monte Carlo included),
so all three get the same calibration treatment.
"""),
    code("""
for m in MODELS.values():
    m.scale = 1.0
SCALES = T.calibrate(MODELS, T.collect(MODELS, DATA, "calib", N_TEST))
"""),
    md("## 8. Score the trained models on the test sessions"),
    code("""
TEST_ROWS = T.collect(MODELS, DATA, "test", N_TEST)
TRAINED = T.summarize(MODELS, TEST_ROWS)
print("Before training"); display(ZERO_SHOT.style.format(precision=3))
print("After training");  display(TRAINED.style.format(precision=3))
"""),
    code("""
import matplotlib.pyplot as plt
from riskmc.minute import score
by_market = (TEST_ROWS.assign(market=TEST_ROWS.symbol.map({s: m["name"] for m in T.MARKETS for s, _ in m["tickers"]}))
             .groupby(["market", "model"]).apply(lambda r: score(r)["pinball_bps"]).unstack())
by_market = by_market.rename(columns={k: m.name for k, m in MODELS.items()})
ax = by_market.plot.barh(figsize=(8, 7), color=["#eb6834", "#2a78d6", "#1baf7a"][:by_market.shape[1]])
ax.set_xlabel("Pinball loss, bps (lower is better)"); ax.set_title("Test sessions, by market"); plt.tight_layout()
"""),
    code("""
REPORT = T.save_report("/content/models/report.json", MODELS, ZERO_SHOT, TRAINED, SCALES, DATA,
                       extra={"device": DEVICE})
print("Saved /content/models/report.json")
"""),
    md("""
## 9. Download the trained models
Saves `trained_models.zip` to your computer: both LoRA adapters (a few MB each) and the range widths.
Unzip it into the project's `models/` folder and run `python app.py` locally (see the README).
"""),
    code(DOWNLOAD_CELL),
    md("""
## 10. Live comparison app
Opens the app in a new tab. Pick a ticker from a market that is open now; every minute all three models
lock in a forecast, and each is checked when the real price arrives.
"""),
    code("""
import app
app.serve(MODELS, REPORT, port=5050, background=True)
from google.colab import output
output.serve_kernel_port_as_window(5050, anchor_text="Open Forecast Arena")
"""),
    code("""
# Or show it inside the notebook:
output.serve_kernel_port_as_iframe(5050, height=1400)
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "colab" / "Forecast_Arena.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({len(cells)} cells)")
