"""Application engine: one refresh cycle, one immutable snapshot, one lock.

The dashboard never computes anything itself.  A background thread rebuilds a
:class:`Snapshot` on a schedule and swaps it in atomically, so every callback
reads a consistent, already-computed view and page loads stay fast even though
a full refresh trains eleven models.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import data, deribit, options
from . import backtest as backtest_module
from .backtest import BacktestResult, summary_frame
from .base import MarketContext, build_context
from .config import (
    BACKTEST_DAYS,
    BACKTEST_ON_START,
    BACKTEST_REFIT_EVERY,
    CACHE_DIR,
    HORIZONS,
    N_PATHS,
    OPTIONS_MAX_DTE,
    OPTIONS_MIN_DTE,
    RANDOM_SEED,
    REFRESH_INTERVAL_SECONDS,
)
from .features import build_features
from .registry import ForecastBundle, MODEL_ORDER, build_all, run_all

log = logging.getLogger(__name__)

BACKTEST_CACHE = CACHE_DIR / "backtest.pkl"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    generated_at: dt.datetime
    market: dict[str, Any]
    context: MarketContext
    bundle: ForecastBundle
    option_view: dict[str, Any]
    backtest: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)

    def feature_importance(self, model_key: str, horizon_days: int, top: int = 12):
        """Top drivers for a learned model, or None for the stochastic ones."""
        model = self.models.get(model_key)
        if model is None or not hasattr(model, "feature_importance"):
            return None
        try:
            return model.feature_importance(horizon_days, top=top)
        except Exception:
            return None

    @property
    def age_seconds(self) -> float:
        return (dt.datetime.now(dt.UTC) - self.generated_at).total_seconds()

    def backtest_results(self) -> dict[str, dict[str, BacktestResult]]:
        return self.backtest.get("results", {})

    def refresh_backtest(self) -> None:
        """Re-read the cached artefact without rebuilding the whole snapshot."""
        self.backtest = load_backtest()

    def backtest_summary(self) -> pd.DataFrame:
        results = self.backtest_results()
        return summary_frame(results) if results else pd.DataFrame()

    def reliability(self, model_key: str, horizon_key: str) -> dict[str, Any] | None:
        """Backtest evidence for one model at one horizon, for the live card."""
        result = self.backtest_results().get(model_key, {}).get(horizon_key)
        if result is None:
            return None
        m = result.metrics
        return {
            "accuracy_pct": m["accuracy_pct"],
            "always_up_accuracy_pct": m["always_up_accuracy_pct"],
            "edge_pp": m["edge_vs_always_up_pp"],
            "brier_skill": m["brier_skill"],
            "auc": m["auc"],
            "n": m["n"],
            "sharpe": result.strategy.get("sharpe", float("nan")),
            "has_skill": bool(m["brier_skill"] > 0 and m["auc"] > 0.5),
        }


# ---------------------------------------------------------------------------
# Option view
# ---------------------------------------------------------------------------
def build_option_view(bundle: ForecastBundle, horizon_key: str = "1w") -> dict[str, Any]:
    """Live chain, greeks, and every model's read on it."""
    view: dict[str, Any] = {"available": False, "error": None}
    try:
        snap = deribit.chain_snapshot(max_dte=OPTIONS_MAX_DTE, min_dte=OPTIONS_MIN_DTE)
    except Exception as exc:
        log.error("deribit chain unavailable: %s", exc)
        view["error"] = f"Deribit unavailable: {exc}"
        return view

    chain, rate, index = snap["chain"], snap["rate"], snap["index_price"]
    if chain.empty:
        view["error"] = "No option contracts inside the horizon window."
        return view

    enriched = options.enrich_chain(chain, rate, index)
    if enriched.empty:
        view["error"] = "No contracts survived the tradeable-delta screen."
        return view

    expiries = sorted(enriched["expiry"].unique())
    # The headline expiry is the one closest to seven days out.
    target = min(expiries, key=lambda e: abs(
        float(enriched.loc[enriched["expiry"] == e, "dte"].iloc[0]) - 7.0))

    per_model: dict[str, Any] = {}
    for key in MODEL_ORDER:
        forecast = bundle.forecasts.get(key, {}).get(horizon_key)
        if forecast is None:
            continue
        try:
            valued = options.value_chain_under_model(enriched, forecast, rate, index)
            per_model[key] = {
                "model_name": forecast.model_name,
                "family": forecast.family,
                "p_up": forecast.p_up,
                "direction": forecast.direction,
                "valued": valued,
                "suggestions": options.suggest_trades(valued, forecast, expiry=target),
                "spread": options.vertical_spread(valued, forecast, expiry=target),
                "vol_context": options.volatility_context(enriched, forecast, target),
                "quantiles": forecast.quantile_prices(),
            }
        except Exception as exc:
            log.exception("option valuation failed for %s", key)
            per_model[key] = {"error": str(exc), "model_name": forecast.model_name}

    view.update({
        "available": True,
        "chain": enriched,
        "rate": rate,
        "index_price": index,
        "expiries": expiries,
        "target_expiry": target,
        "target_dte": float(enriched.loc[enriched["expiry"] == target, "dte"].iloc[0]),
        "atm_iv_pct": options.atm_implied_vol(enriched, target),
        "per_model": per_model,
        "fetched_at": snap["fetched_at"],
        "n_contracts": int(len(enriched)),
        "horizon_key": horizon_key,
    })
    return view


def load_backtest() -> dict[str, Any]:
    """Read the cached walk-forward results, if a run has completed."""
    if not BACKTEST_CACHE.exists():
        return {"available": False, "reason": "No backtest has been run yet."}
    try:
        with open(BACKTEST_CACHE, "rb") as fh:
            payload = pickle.load(fh)
        payload["available"] = True
        payload["age_hours"] = (time.time() - payload.get("generated_at", 0)) / 3600.0
        return payload
    except Exception as exc:
        log.error("backtest cache unreadable: %s", exc)
        return {"available": False, "reason": f"Cache unreadable: {exc}"}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, refresh_interval: int = REFRESH_INTERVAL_SECONDS) -> None:
        self.refresh_interval = refresh_interval
        self._lock = threading.RLock()
        self._snapshot: Snapshot | None = None
        self._models = build_all()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.status: str = "starting"
        self.last_error: str | None = None

    # -- access -----------------------------------------------------------
    @property
    def snapshot(self) -> Snapshot | None:
        with self._lock:
            return self._snapshot

    def wait_for_snapshot(self, timeout: float = 240.0) -> Snapshot | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.snapshot
            if snap is not None:
                return snap
            time.sleep(0.5)
        return self.snapshot

    # -- refresh ----------------------------------------------------------
    def refresh(self, force_data: bool = False) -> Snapshot:
        start = time.time()
        self.status = "refreshing"
        errors: dict[str, str] = {}

        daily = data.load_daily(force=force_data)
        completed = daily[~daily.get("is_partial", pd.Series(False, index=daily.index)).astype(bool)]
        features = build_features(completed)
        context = build_context(daily, features=features)

        bundle = run_all(context, HORIZONS, N_PATHS, RANDOM_SEED, models=self._models)
        errors.update(bundle.errors)

        try:
            market = data.market_snapshot()
        except Exception as exc:
            market = {"price": context.spot, "as_of": context.as_of}
            errors["market"] = str(exc)

        option_view = build_option_view(bundle, horizon_key="1w")
        if option_view.get("error"):
            errors["options"] = option_view["error"]

        snapshot = Snapshot(
            generated_at=dt.datetime.now(dt.UTC),
            market=market,
            context=context,
            bundle=bundle,
            option_view=option_view,
            backtest=load_backtest(),
            models=dict(self._models),
            duration_seconds=time.time() - start,
            errors=errors,
        )
        with self._lock:
            self._snapshot = snapshot
        self.status = "ready"
        self.last_error = None
        log.info("refresh complete in %.1fs", snapshot.duration_seconds)
        return snapshot

    # -- background loop --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as exc:
                log.exception("refresh failed")
                self.status = "error"
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.refresh_interval)

    # -- optional first-run backtest ---------------------------------------
    def _run_backtest_once(self) -> None:
        """Generate the walk-forward artefact if none exists.

        Without this a fresh deployment shows an empty backtest tab until the
        scheduled job first runs.  It uses its own model instances rather than
        the live ones: the backtester deliberately refits and freezes models as
        it walks forward, which would corrupt the state the dashboard is serving.
        """
        if BACKTEST_CACHE.exists():
            return
        log.info("no cached backtest; starting one in the background")
        try:
            results = backtest_module.run_walk_forward(
                data.load_daily(), HORIZONS,
                backtest_days=BACKTEST_DAYS, refit_every=BACKTEST_REFIT_EVERY,
                progress=lambda msg, frac: log.info("backtest %s", msg),
            )
            payload = {
                "results": results,
                "generated_at": time.time(),
                "elapsed_seconds": 0.0,
                "backtest_days": BACKTEST_DAYS,
                "refit_every": BACKTEST_REFIT_EVERY,
                "data_end": str(data.load_daily().index[-1].date()),
            }
            BACKTEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(BACKTEST_CACHE, "wb") as fh:
                pickle.dump(payload, fh)
            log.info("background backtest complete")
            # Fold the new artefact into the current snapshot immediately.
            with self._lock:
                if self._snapshot is not None:
                    self._snapshot.backtest = load_backtest()
        except Exception:
            log.exception("background backtest failed")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="btc-refresh", daemon=True)
        self._thread.start()
        if BACKTEST_ON_START:
            threading.Thread(target=self._run_backtest_once,
                             name="btc-backtest", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


_ENGINE: Engine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine(start: bool = True) -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()
            if start:
                _ENGINE.start()
        return _ENGINE
