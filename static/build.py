"""Build the static dashboard.

The site is a shell plus one JSON payload per view. Rendering every combination
into a single HTML file would produce ~8 MB the browser must parse before first
paint; instead the forecast view is inlined and everything else is fetched on
demand and cached. Each payload carries its own HTML and its Plotly figures.

Run:  python -m static.build --out site
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from btcmodels.config import HORIZONS
from btcmodels.engine import Engine
from btcmodels.registry import MODEL_ORDER
from dashboard import layout as L

from .render_html import HtmlRenderer, figure_to_json

log = logging.getLogger("build")

THEMES = ("dark", "light")
GREEKS = ("delta", "gamma", "theta", "vega", "vanna", "charm")


def panel_key(tab: str, theme: str, horizon: str = "", model: str = "") -> str:
    return "-".join(p for p in (tab, theme, horizon, model) if p)


def render_panel(component) -> dict:
    renderer = HtmlRenderer()
    markup = renderer.render(component)
    return {
        "html": markup,
        "figures": {k: json.loads(figure_to_json(v)) for k, v in renderer.figures.items()},
    }


def build_panels(snapshot, out: Path) -> dict:
    """Every view the site can show, keyed for the client to fetch."""
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    inline: dict[str, dict] = {}

    def emit(key: str, payload: dict, keep_inline: bool = False) -> None:
        (data_dir / f"{key}.json").write_text(json.dumps(payload, separators=(",", ":")))
        manifest[key] = f"data/{key}.json"
        if keep_inline:
            inline[key] = payload

    available = [k for k in MODEL_ORDER if k in snapshot.bundle.forecasts]
    option_models = [k for k in MODEL_ORDER
                     if k in snapshot.option_view.get("per_model", {})
                     and "error" not in snapshot.option_view["per_model"][k]]

    total = len(THEMES) * (len(HORIZONS) * (1 + 2 * len(available)) + len(option_models) + 1)
    done = 0
    start = time.time()

    for theme in THEMES:
        emit(panel_key("masthead", theme),
             render_panel(L.render_masthead(snapshot, theme, "1w", "ready")), keep_inline=True)

        for horizon in HORIZONS:
            emit(panel_key("forecast", theme, horizon),
                 render_panel(L.render_forecast(snapshot, theme, horizon)),
                 keep_inline=(horizon == "1w"))
            done += 1

            for model in available:
                emit(panel_key("models", theme, horizon, model),
                     render_panel(L.render_models(snapshot, theme, horizon, model)))
                emit(panel_key("backtest", theme, horizon, model),
                     render_panel(L.render_backtest(snapshot, theme, horizon, model)))
                done += 2

            log.info("  %s/%s panels (%d/%d, %.0fs)", theme, horizon, done, total,
                     time.time() - start)

        for model in option_models:
            # All six greek figures ship in one payload; the page swaps between
            # them rather than refetching a near-identical view six times.
            payload = render_panel(L.render_options(snapshot, theme, model, "theta"))
            for greek in GREEKS:
                if greek == "theta":
                    continue
                extra = render_panel(L.render_options(snapshot, theme, model, greek))
                for fid, fig in extra["figures"].items():
                    payload["figures"][f"{fid}::{greek}"] = fig
            emit(panel_key("options", theme, model), payload)
            done += 1

        emit(panel_key("method", theme), render_panel(L.render_method(theme)))
        done += 1
        log.info("  %s complete (%d/%d, %.0fs)", theme, done, total, time.time() - start)

    return {"manifest": manifest, "inline": inline,
            "models": available, "option_models": option_models,
            "horizons": list(HORIZONS), "greeks": list(GREEKS)}


def build(out: Path, snapshot=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out.mkdir(parents=True, exist_ok=True)

    if snapshot is None:
        log.info("fitting models and pulling the option chain…")
        snapshot = Engine().refresh()
    log.info("snapshot ready: %d models, errors=%s",
             len(snapshot.bundle.forecasts), snapshot.errors or "none")

    log.info("rendering panels…")
    built = build_panels(snapshot, out)

    css = Path("assets/style.css").read_text()
    extra_css = Path(__file__).with_name("static.css").read_text()
    script = Path(__file__).with_name("site.js").read_text()

    generated = dt.datetime.now(dt.UTC)
    boot = {
        "manifest": built["manifest"],
        "inline": built["inline"],
        "models": built["models"],
        "optionModels": built["option_models"],
        "modelNames": {k: snapshot.bundle.forecasts[k]["1w"].model_name
                       for k in built["models"]},
        "horizons": built["horizons"],
        "greeks": built["greeks"],
        "generatedAt": generated.isoformat(),
        "hasBacktest": bool(snapshot.backtest.get("available")),
    }

    html = TEMPLATE.format(
        css=css + "\n" + extra_css,
        boot=json.dumps(boot, separators=(",", ":")),
        script=script,
        generated=generated.strftime("%d %b %Y %H:%M UTC"),
    )
    (out / "index.html").write_text(html)
    (out / ".nojekyll").touch()

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("built %s — %d files, %.1f MB", out, sum(1 for _ in out.rglob("*")), size / 1e6)
    return out


TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Model Dashboard</title>
<meta name="description" content="Eleven stochastic, machine-learning and hybrid models forecasting Bitcoin, with walk-forward backtests and live options analytics.">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}</style>
</head>
<body>
<div class="app">
  <div class="masthead-wrap">
    <div id="masthead"></div>
    <div class="controls shell-controls">
      <div class="radio-group" id="horizon-toggle">
        <label><input type="radio" name="horizon" value="24h"> Next 24 hours</label>
        <label><input type="radio" name="horizon" value="1w" checked> Next 7 days</label>
      </div>
      <select class="picker" id="model-picker" hidden></select>
      <select class="picker" id="greek-picker" hidden></select>
      <button class="btn" id="theme-button">Light</button>
    </div>
  </div>
  <nav class="tab-bar" id="tabs">
    <button class="tab tab--selected" data-tab="forecast">Forecasts</button>
    <button class="tab" data-tab="models">Model detail</button>
    <button class="tab" data-tab="backtest">Backtest</button>
    <button class="tab" data-tab="options">Options</button>
    <button class="tab" data-tab="method">Method</button>
  </nav>
  <div id="content"><div class="panel loading-note">Loading…</div></div>
  <footer class="site-footer">
    Rebuilt {generated} by GitHub Actions · research tool, not financial advice ·
    options can lose 100% of the premium paid
  </footer>
</div>
<script>window.__BOOT__ = {boot};</script>
<script>{script}</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
