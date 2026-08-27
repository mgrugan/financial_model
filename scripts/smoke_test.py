#!/usr/bin/env python3
"""End-to-end check: build a real snapshot, render every tab, serve the app.

Exercises the whole stack against live data rather than fixtures, because the
failure modes worth catching here -- a rate-limited API, an empty option chain,
a model that will not converge -- only appear against the real thing.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.ERROR)

FAILURES: list[str] = []
PASSES: list[str] = []


def check(name: str, fn):
    start = time.time()
    try:
        result = fn()
        PASSES.append(f"PASS  {name}  ({time.time() - start:.1f}s)  {result or ''}")
        return result
    except Exception as exc:
        import traceback
        FAILURES.append(f"FAIL  {name}: {type(exc).__name__}: {exc}\n"
                        + "".join(traceback.format_tb(exc.__traceback__)[-3:]))
        return None


def main() -> int:
    from btcmodels.config import HORIZONS
    from btcmodels.engine import Engine
    from dashboard import layout as L

    print("building snapshot (fits 11 models + pulls live option chain)...", flush=True)
    engine = Engine()
    snapshot = check("engine.refresh", lambda: engine.refresh())
    if snapshot is None:
        print("\n".join(FAILURES))
        return 1

    print(f"  spot ${snapshot.context.spot:,.0f}, "
          f"{len(snapshot.bundle.forecasts)} models, "
          f"errors={snapshot.errors or 'none'}", flush=True)

    check("all 11 models forecast",
          lambda: _assert(len(snapshot.bundle.forecasts) == 11,
                          f"only {len(snapshot.bundle.forecasts)} models produced output")
          or f"{len(snapshot.bundle.forecasts)}/11")
    check("no model errors",
          lambda: _assert(not snapshot.errors, f"errors: {snapshot.errors}") or "clean")

    for horizon in HORIZONS:
        for f in snapshot.bundle.by_horizon(horizon):
            check(f"forecast sane: {f.model_key}/{horizon}",
                  lambda f=f: _sane(f))

    for theme in ("dark", "light"):
        for horizon in HORIZONS:
            check(f"render forecast tab [{theme}/{horizon}]",
                  lambda t=theme, h=horizon: _len(L.render_forecast(snapshot, t, h)))
            check(f"render backtest tab [{theme}/{horizon}]",
                  lambda t=theme, h=horizon: _len(L.render_backtest(snapshot, t, h, "gbm")))
        check(f"render method tab [{theme}]", lambda t=theme: _len(L.render_method(t)))
        check(f"render masthead [{theme}]",
              lambda t=theme: _len(L.render_masthead(snapshot, t, "1w", "ready")))

    # every model on the detail and options tabs, both themes
    from btcmodels.registry import MODEL_ORDER
    for key in MODEL_ORDER:
        check(f"render model detail: {key}",
              lambda k=key: _len(L.render_models(snapshot, "dark", "1w", k)))
        if key in snapshot.option_view.get("per_model", {}):
            check(f"render options tab: {key}",
                  lambda k=key: _len(L.render_options(snapshot, "dark", k, "theta")))

    for greek in ("delta", "gamma", "theta", "vega", "vanna", "charm"):
        check(f"render options greek: {greek}",
              lambda g=greek: _len(L.render_options(snapshot, "light", "gbm", g)))

    print()
    for line in PASSES:
        print(line)
    if FAILURES:
        print("\n" + "=" * 70)
        for line in FAILURES:
            print(line)
        print(f"\n{len(FAILURES)} FAILED, {len(PASSES)} passed")
        return 1
    print(f"\nALL {len(PASSES)} CHECKS PASSED")
    return 0


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)
    return None


def _sane(forecast) -> str:
    import numpy as np

    _assert(0.0 < forecast.p_up < 1.0, f"p_up out of range: {forecast.p_up}")
    _assert(forecast.terminal_prices.size > 1000, "too few paths")
    _assert(np.all(forecast.terminal_prices > 0), "non-positive prices")
    _assert(np.isfinite(forecast.expected_return), "non-finite expected return")
    _assert(1.0 < forecast.annualised_vol < 400.0,
            f"implausible vol {forecast.annualised_vol:.1f}%")
    q = forecast.quantile_prices()
    _assert(q["q05"] < q["q50"] < q["q95"], "quantiles out of order")
    return f"P(up)={forecast.p_up:.3f} vol={forecast.annualised_vol:.0f}%"


def _len(component) -> str:
    from dash.development.base_component import Component

    _assert(component is not None, "returned None")
    _assert(isinstance(component, Component), f"not a Dash component: {type(component)}")
    text = str(component)
    _assert(len(text) > 200, f"suspiciously small render ({len(text)} chars)")
    return f"{len(text):,} chars"


if __name__ == "__main__":
    raise SystemExit(main())
