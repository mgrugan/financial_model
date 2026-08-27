"""Walk-forward backtesting.

The protocol is the one that makes a crypto forecasting result believable
rather than decorative:

* **Strictly out of sample.**  On test day ``t`` a model has seen nothing after
  ``t``.  Features are causal; labels for ``(t, t+h]`` are never in any
  training set that produced the prediction for ``t``.
* **Periodic re-estimation, daily state.**  Parameters are re-estimated on a
  fixed refit schedule -- as a desk would -- while fast-moving state (GARCH
  conditional variance, the regime filter, today's features) updates every day.
* **Purged calibration.**  Each refit re-derives its probability calibration
  from purged folds *inside* its own training window.
* **Costed, non-overlapping trading.**  A 7-day signal evaluated daily produces
  seven overlapping bets on the same move, which flatters Sharpe by roughly
  sqrt(7).  The equity curve is therefore built on non-overlapping blocks, net
  of transaction costs.

Everything is scored against the right null: the "always up" base rate, not 50%.
Bitcoin rose on 53% of days in this sample, so a model that is 52% accurate has
negative skill, and only skill scores make that visible.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .base import BaseModel, build_context
from .config import (
    BACKTEST_COST_BPS,
    BACKTEST_DAYS,
    BACKTEST_MIN_TRAIN,
    BACKTEST_REFIT_EVERY,
    DAYS_PER_YEAR,
)
from .features import build_features
from .registry import MODEL_ORDER, build_all

log = logging.getLogger(__name__)

BACKTEST_PATHS = 6_000          # enough for +/-0.006 on P(up)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype="float64")
    ranks[order] = np.arange(1, len(score) + 1, dtype="float64")
    sorted_score = score[order]
    start = 0
    for i in range(1, len(sorted_score) + 1):
        if i == len(sorted_score) or sorted_score[i] != sorted_score[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y > 0].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0)) * 100.0


def _calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 6) -> list[dict[str, Any]]:
    """Observed frequency against predicted probability, in equal-count bins."""
    if len(p) < n_bins * 5:
        return []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        mask = (p > edges[i]) & (p <= edges[i + 1])
        if mask.sum() < 3:
            continue
        rows.append({
            "bin": i + 1,
            "n": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
            "lo": float(edges[i]),
            "hi": float(edges[i + 1]),
        })
    return rows


def _strategy(p: np.ndarray, fwd_logret: np.ndarray, horizon: int,
              cost_bps: float) -> dict[str, Any]:
    """Long/short on the sign of the edge, on non-overlapping blocks, net of cost."""
    idx = np.arange(0, len(p), horizon)          # non-overlapping
    p_b, r_b = p[idx], fwd_logret[idx]
    if len(p_b) < 4:
        return {}

    position = np.where(p_b >= 0.5, 1.0, -1.0)
    simple = np.expm1(r_b)
    gross = position * simple

    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    net = gross - turnover * (cost_bps / 10_000.0)

    equity = np.cumprod(1.0 + net)
    bh_equity = np.cumprod(1.0 + simple)
    periods_per_year = DAYS_PER_YEAR / horizon
    sd = float(np.std(net, ddof=1))

    years = len(net) / periods_per_year
    return {
        "n_trades": int(len(net)),
        "total_return_pct": float(equity[-1] - 1.0) * 100.0,
        "annualised_return_pct": float(equity[-1] ** (1.0 / max(years, 1e-9)) - 1.0) * 100.0,
        "sharpe": float(np.mean(net) / sd * np.sqrt(periods_per_year)) if sd > 1e-12 else 0.0,
        "hit_rate_pct": float(np.mean(net > 0)) * 100.0,
        "max_drawdown_pct": _max_drawdown(equity),
        "buy_hold_return_pct": float(bh_equity[-1] - 1.0) * 100.0,
        "buy_hold_sharpe": float(np.mean(simple) / np.std(simple, ddof=1)
                                 * np.sqrt(periods_per_year)) if np.std(simple, ddof=1) > 1e-12 else 0.0,
        "buy_hold_max_drawdown_pct": _max_drawdown(bh_equity),
        "pct_long": float(np.mean(position > 0)) * 100.0,
        "equity_curve": equity.tolist(),
        "buy_hold_curve": bh_equity.tolist(),
        "cost_bps": cost_bps,
    }


@dataclass
class BacktestResult:
    model_key: str
    model_name: str
    family: str
    horizon_key: str
    horizon_days: int
    dates: pd.DatetimeIndex
    p_up: np.ndarray
    y_true: np.ndarray
    fwd_logret: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    strategy: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model": self.model_name,
            "family": self.family,
            "horizon": self.horizon_key,
            **self.metrics,
            "sharpe": self.strategy.get("sharpe", float("nan")),
            "total_return_pct": self.strategy.get("total_return_pct", float("nan")),
            "max_drawdown_pct": self.strategy.get("max_drawdown_pct", float("nan")),
            "buy_hold_return_pct": self.strategy.get("buy_hold_return_pct", float("nan")),
        }


def score(model_key: str, model_name: str, family: str, horizon_key: str, horizon_days: int,
          dates: pd.DatetimeIndex, p_up: np.ndarray, y_true: np.ndarray,
          fwd_logret: np.ndarray, cost_bps: float = BACKTEST_COST_BPS) -> BacktestResult:
    p = np.clip(np.asarray(p_up, dtype="float64"), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype="float64")

    base_rate = float(y.mean())
    predicted_up = (p >= 0.5).astype("float64")
    accuracy = float(np.mean(predicted_up == y))
    always_up = max(base_rate, 1 - base_rate)

    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base_rate - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    logloss_base = float(-(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate)))

    # Accuracy restricted to the model's most confident third of days.
    edge = np.abs(p - 0.5)
    if len(edge) >= 30:
        cut = float(np.quantile(edge, 2 / 3))
        conf_mask = edge >= cut
        conf_accuracy = float(np.mean(predicted_up[conf_mask] == y[conf_mask]))
        conf_n = int(conf_mask.sum())
    else:
        conf_accuracy, conf_n = float("nan"), 0

    metrics = {
        "n": int(len(p)),
        "accuracy_pct": accuracy * 100.0,
        "base_rate_pct": base_rate * 100.0,
        "always_up_accuracy_pct": always_up * 100.0,
        "edge_vs_always_up_pp": (accuracy - always_up) * 100.0,
        "brier": brier,
        "brier_skill": 1.0 - brier / brier_base if brier_base > 0 else float("nan"),
        "log_loss": logloss,
        "log_loss_skill": 1.0 - logloss / logloss_base if logloss_base > 0 else float("nan"),
        "auc": _auc(y, p),
        "mean_p_up": float(p.mean()),
        "p_up_std": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
        "confident_third_accuracy_pct": conf_accuracy * 100.0,
        "confident_third_n": conf_n,
        "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()),
    }
    return BacktestResult(
        model_key=model_key, model_name=model_name, family=family,
        horizon_key=horizon_key, horizon_days=horizon_days,
        dates=dates, p_up=p, y_true=y, fwd_logret=np.asarray(fwd_logret),
        metrics=metrics,
        calibration=_calibration_table(p, y),
        strategy=_strategy(p, np.asarray(fwd_logret), horizon_days, cost_bps),
    )


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
def run_walk_forward(
    daily: pd.DataFrame,
    horizons: dict[str, int],
    models: dict[str, BaseModel] | None = None,
    backtest_days: int = BACKTEST_DAYS,
    min_train: int = BACKTEST_MIN_TRAIN,
    refit_every: int = BACKTEST_REFIT_EVERY,
    n_paths: int = BACKTEST_PATHS,
    cost_bps: float = BACKTEST_COST_BPS,
    seed: int = 7,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, dict[str, BacktestResult]]:
    """Run every model through the walk-forward protocol."""
    frame = daily[~daily.get("is_partial", pd.Series(False, index=daily.index)).astype(bool)]
    features = build_features(frame)
    close = frame["close"]
    log_close = np.log(close.to_numpy())
    n = len(frame)
    max_h = max(horizons.values())

    usable = features.dropna()
    first_valid = frame.index.get_loc(usable.index[0])
    start = max(first_valid + min_train, n - backtest_days)
    end = n - max_h                     # need a realised label at the longest horizon
    if start >= end:
        raise ValueError(f"not enough history: start={start} end={end}")

    models = models or build_all()
    records: dict[str, dict[str, list]] = {
        key: {hk: [] for hk in horizons} for key in models
    }

    blocks = list(range(start, end, refit_every))
    total_steps = len(blocks)
    t_begin = time.time()

    for block_no, i0 in enumerate(blocks):
        block_end = min(i0 + refit_every, end)
        as_of = frame.index[i0 - 1]

        # ---- re-estimate every model on data available at the block start ----
        train_daily = frame.iloc[:i0]
        try:
            train_ctx = build_context(train_daily, features=features)
        except Exception as exc:
            log.error("context build failed at %s: %s", as_of, exc)
            continue
        garch_params = train_ctx.garch_params or None

        live: dict[str, BaseModel] = {}
        for key, model in models.items():
            try:
                if hasattr(model, "frozen"):
                    model.frozen = False
                model.fitted_ = False
                model.fit(train_ctx)
                if hasattr(model, "frozen"):
                    model.frozen = True
                live[key] = model
            except Exception as exc:
                log.warning("refit failed for %s at %s: %s", key, as_of, exc)

        # ---- step through the block one day at a time ------------------------
        for j in range(i0, block_end):
            day_daily = frame.iloc[:j + 1]
            try:
                ctx = build_context(day_daily, features=features, garch_params=garch_params)
            except Exception as exc:
                log.warning("daily context failed at %s: %s", frame.index[j], exc)
                continue

            for horizon_key, h in horizons.items():
                if j + h >= n:
                    continue
                fwd = float(log_close[j + h] - log_close[j])
                y = 1.0 if fwd > 0 else 0.0
                for key, model in live.items():
                    rng = np.random.default_rng(seed + j * 31 + h * 7 + abs(hash(key)) % 1000)
                    try:
                        p, _ = model.predict_proba_up(ctx, h, n_paths, rng)
                    except Exception as exc:
                        log.debug("predict failed %s %s: %s", key, frame.index[j], exc)
                        continue
                    records[key][horizon_key].append((frame.index[j], float(p), y, fwd))

        if progress:
            elapsed = time.time() - t_begin
            done = (block_no + 1) / total_steps
            progress(
                f"walk-forward {block_no + 1}/{total_steps} "
                f"(through {frame.index[block_end - 1].date()}), "
                f"{elapsed:.0f}s elapsed, ~{elapsed / done - elapsed:.0f}s left",
                done,
            )

    # ---- score ------------------------------------------------------------
    results: dict[str, dict[str, BacktestResult]] = {}
    for key in MODEL_ORDER:
        if key not in records:
            continue
        model = models[key]
        per_horizon = {}
        for horizon_key, h in horizons.items():
            rows = records[key][horizon_key]
            if len(rows) < 30:
                continue
            dates = pd.DatetimeIndex([r[0] for r in rows])
            per_horizon[horizon_key] = score(
                key, model.name, model.family, horizon_key, h,
                dates,
                np.array([r[1] for r in rows]),
                np.array([r[2] for r in rows]),
                np.array([r[3] for r in rows]),
                cost_bps=cost_bps,
            )
        if per_horizon:
            results[key] = per_horizon
    return results


def summary_frame(results: dict[str, dict[str, BacktestResult]]) -> pd.DataFrame:
    rows = [r.summary_row() for per in results.values() for r in per.values()]
    return pd.DataFrame(rows)
