"""Model registry: the single place that knows every model in the stack."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import BaseModel, MarketContext, ModelForecast, stable_hash
from .hybrid import NeuralDriftDiffusionModel
from .ml import GradientBoostingModel, LogisticModel, MLPModel, RandomForestModel
from .nn import LSTMModel
from .stochastic import (
    GARCHModel,
    GBMModel,
    HestonModel,
    MertonJumpModel,
    RegimeSwitchingModel,
)

log = logging.getLogger(__name__)

FAMILY_ORDER = ["Stochastic", "Machine Learning", "Neural Network", "Hybrid"]

FAMILY_BLURB = {
    "Stochastic": (
        "Continuous-time price processes calibrated by maximum likelihood or "
        "method of moments. They describe the *distribution* of outcomes well "
        "and say very little about direction -- which is the honest state of "
        "affairs for a near-efficient market."
    ),
    "Machine Learning": (
        "Supervised classifiers over 47 engineered features, trained with "
        "recency weighting and calibrated on pooled out-of-sample folds."
    ),
    "Neural Network": (
        "Networks that learn their own feature interactions: a dense net over "
        "the daily snapshot, and a recurrent net over the 24-day path."
    ),
    "Hybrid": (
        "A diffusion whose drift and volatility are machine-learned functions of "
        "market state -- the point where the two families above actually meet."
    ),
}

MODEL_BUILDERS: dict[str, Callable[[], BaseModel]] = {
    "gbm": GBMModel,
    "merton": MertonJumpModel,
    "garch": GARCHModel,
    "heston": HestonModel,
    "regime": RegimeSwitchingModel,
    "xgboost": GradientBoostingModel,
    "random_forest": RandomForestModel,
    "elasticnet": LogisticModel,
    "mlp": MLPModel,
    "lstm": LSTMModel,
    "hybrid": NeuralDriftDiffusionModel,
}

MODEL_ORDER = list(MODEL_BUILDERS.keys())


def build_all() -> dict[str, BaseModel]:
    return {key: builder() for key, builder in MODEL_BUILDERS.items()}


def model_catalogue() -> list[dict[str, Any]]:
    """Static metadata for every model, for the dashboard's reference section."""
    entries = []
    for key, builder in MODEL_BUILDERS.items():
        model = builder()
        entries.append({
            "key": key,
            "name": model.name,
            "family": model.family,
            "method": model.method,
            "description": model.description,
        })
    entries.sort(key=lambda e: (FAMILY_ORDER.index(e["family"]), MODEL_ORDER.index(e["key"])))
    return entries


@dataclass
class ForecastBundle:
    """Every model's view of every horizon, plus the consensus across them."""

    context: MarketContext
    forecasts: dict[str, dict[str, ModelForecast]]      # model_key -> horizon_key -> forecast
    consensus: dict[str, dict[str, Any]]                # horizon_key -> summary
    timings: dict[str, float]
    errors: dict[str, str]

    def by_horizon(self, horizon_key: str) -> list[ModelForecast]:
        out = []
        for key in MODEL_ORDER:
            forecast = self.forecasts.get(key, {}).get(horizon_key)
            if forecast is not None:
                out.append(forecast)
        return out


def _consensus(forecasts: list[ModelForecast],
               weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Aggregate the models into one view.

    Probabilities are averaged in **log-odds** space, not in probability space:
    averaging probabilities directly drags every model toward 0.5 and makes a
    unanimous panel look less certain than each of its members.
    """
    if not forecasts:
        return {}

    keys = [f.model_key for f in forecasts]
    w = np.array([max((weights or {}).get(k, 1.0), 0.0) for k in keys], dtype="float64")
    if w.sum() <= 0:
        w = np.ones(len(keys))
    w = w / w.sum()

    probs = np.array([f.p_up for f in forecasts])
    logodds = np.log(np.clip(probs, 1e-4, 1 - 1e-4) / np.clip(1 - probs, 1e-4, 1 - 1e-4))
    p_consensus = float(1.0 / (1.0 + np.exp(-float(np.sum(w * logodds)))))

    n_up = int(np.sum(probs > 0.5))
    pooled = np.concatenate([
        np.random.default_rng(stable_hash(f.model_key)).choice(
            f.terminal_prices, size=min(8000, f.terminal_prices.size), replace=False)
        for f in forecasts
    ])

    return {
        "p_up": p_consensus,
        "p_up_mean": float(np.average(probs, weights=w)),
        "direction": "UP" if p_consensus >= 0.5 else "DOWN",
        "n_models": len(forecasts),
        "n_up": n_up,
        "n_down": len(forecasts) - n_up,
        "agreement": float(max(n_up, len(forecasts) - n_up) / len(forecasts)),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "prob_std": float(probs.std(ddof=1)) if len(probs) > 1 else 0.0,
        "median_price": float(np.median(pooled)),
        "expected_price": float(np.mean(pooled)),
        "q05": float(np.quantile(pooled, 0.05)),
        "q25": float(np.quantile(pooled, 0.25)),
        "q75": float(np.quantile(pooled, 0.75)),
        "q95": float(np.quantile(pooled, 0.95)),
        "pooled_samples": pooled,
        "weights": dict(zip(keys, w.tolist())),
    }


def run_all(context: MarketContext, horizons: dict[str, int], n_paths: int,
            seed: int, models: dict[str, BaseModel] | None = None,
            weights: dict[str, float] | None = None,
            live_spot: float | None = None) -> ForecastBundle:
    """Fit and forecast every model. One model failing never sinks the rest.

    ``live_spot`` re-anchors every forecast onto the current price.  Models are
    fitted on close-to-close returns and so forecast from the last completed
    daily bar; leaving them there would make the intraday move since that close
    read as a directional signal everywhere downstream, most damagingly in the
    option valuations, which are compared against a live book.
    """
    models = models or build_all()
    forecasts: dict[str, dict[str, ModelForecast]] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}

    for key in MODEL_ORDER:
        model = models.get(key)
        if model is None:
            continue
        start = time.time()
        try:
            model.fit(context)
            per_horizon = {}
            for horizon_key, horizon_days in horizons.items():
                rng = np.random.default_rng(seed + 977 * horizon_days + stable_hash(key) % 10_000)
                forecast = model.forecast(
                    context, horizon_key, horizon_days, n_paths, rng)
                if live_spot:
                    forecast = forecast.rebased(live_spot)
                per_horizon[horizon_key] = forecast
            forecasts[key] = per_horizon
        except Exception as exc:
            log.exception("model %s failed", key)
            errors[key] = f"{type(exc).__name__}: {exc}"
        timings[key] = time.time() - start

    consensus = {
        horizon_key: _consensus(
            [forecasts[k][horizon_key] for k in MODEL_ORDER
             if k in forecasts and horizon_key in forecasts[k]],
            weights,
        )
        for horizon_key in horizons
    }
    return ForecastBundle(context=context, forecasts=forecasts, consensus=consensus,
                          timings=timings, errors=errors)
