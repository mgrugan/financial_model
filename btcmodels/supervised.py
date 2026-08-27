"""Shared machinery for the supervised (learned) models.

Three concerns are handled here so no individual model has to repeat them:

1. **Leak-free calibration.**  Raw classifier scores on financial data are badly
   over-confident.  Probabilities are rescaled with Platt scaling fitted on a
   *chronologically held-out* tail of the training set, with the overlapping
   labels between the two blocks purged.
2. **Recency weighting.**  Bitcoin's 2015 microstructure is not the 2026 one, so
   observations are exponentially down-weighted by age.
3. **Turning a probability into a price distribution.**  A classifier only says
   "up with probability p".  The horizon distribution is built by taking the
   fat-tailed, GARCH-scaled innovation sample from the shared context and
   shifting it until exactly ``p`` of its mass is positive.  The learned model
   supplies the location; the stochastic backbone supplies the shape.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel, MarketContext, bootstrap_innovations, tilt_to_probability
from .features import build_features, supervised_frame

log = logging.getLogger(__name__)

DEFAULT_HALFLIFE_DAYS = 730.0


def recency_weights(index: pd.Index, halflife_days: float = DEFAULT_HALFLIFE_DAYS) -> np.ndarray:
    """Exponentially decaying observation weights, newest = 1.0."""
    age = (index[-1] - index).days.to_numpy().astype("float64")
    return np.power(0.5, age / halflife_days)


class PlattCalibrator:
    """Logistic recalibration of a score in log-odds space.

    ``p_calibrated = sigmoid(a * logit(p_raw) + b)``.  Two parameters only, which
    is what you want when the calibration block is a few hundred observations;
    isotonic regression would be more flexible and would overfit here.

    Two constraints make this safe on financial data:

    * **The slope is confined to [0, MAX_SLOPE].**  A negative slope would mean
      "this model was anti-predictive on the held-out block, so invert it".
      Inverting a weak model on the evidence of one contiguous block is a
      near-perfect recipe for overfitting.  The honest response to a model with
      no demonstrated edge is ``a -> 0``, which collapses its output to the base
      rate, and that is what the bound enforces.
    * **The slope is ridge-penalised toward 1.**  With little evidence the
      calibration stays close to the identity; it takes a lot of contrary
      evidence to flatten a model out.

    Together these also stop the Newton iteration from running away when the
    scores separate the calibration labels perfectly (the unpenalised MLE is
    then infinite).
    """

    MAX_SLOPE = 3.0

    def __init__(self, shrink: float = 1.0, ridge_strength: float = 0.01) -> None:
        self.a = 1.0
        self.b = 0.0
        self.shrink = shrink
        self.ridge_strength = ridge_strength
        self.base_rate = 0.5
        self.fitted = False

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p_raw: np.ndarray, y: np.ndarray,
            weights: np.ndarray | None = None) -> "PlattCalibrator":
        p_raw = np.asarray(p_raw, dtype="float64")
        y = np.asarray(y, dtype="float64")
        self.base_rate = float(np.average(y, weights=weights)) if y.size else 0.5
        if y.size < 40 or len(np.unique(y)) < 2:
            self.a, self.b, self.fitted = 1.0, 0.0, False
            return self

        x = self._logit(p_raw)
        w = np.ones_like(y) if weights is None else np.asarray(weights, dtype="float64")
        lam = self.ridge_strength * float(w.sum())      # prior strength on the slope
        a, b = 1.0, 0.0

        # Newton-Raphson on the penalised weighted binomial log-likelihood,
        # with the slope projected back into [0, MAX_SLOPE] after every step.
        for _ in range(100):
            z = a * x + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            resid = w * (y - p)
            grad = np.array([np.dot(resid, x) - 2.0 * lam * (a - 1.0), resid.sum()])
            v = w * p * (1 - p)
            hess = np.array([[np.dot(v, x * x) + 2.0 * lam, np.dot(v, x)],
                             [np.dot(v, x), v.sum()]])
            hess += np.eye(2) * 1e-8
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -2.0, 2.0)              # damped, keeps it stable
            a_new = float(np.clip(a + step[0], 0.0, self.MAX_SLOPE))
            b_new = float(np.clip(b + step[1], -6.0, 6.0))
            converged = abs(a_new - a) < 1e-9 and abs(b_new - b) < 1e-9
            a, b = a_new, b_new
            if converged:
                break

        if np.isfinite(a) and np.isfinite(b):
            self.a, self.b, self.fitted = float(a), float(b), True
        return self

    def transform(self, p_raw: np.ndarray | float) -> np.ndarray:
        p = np.atleast_1d(np.asarray(p_raw, dtype="float64"))
        if self.fitted:
            z = self.a * self._logit(p) + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        if self.shrink < 1.0:
            p = 0.5 + self.shrink * (p - 0.5)
        return np.clip(p, 0.001, 0.999)

    def transform_scalar(self, p_raw: float) -> float:
        return float(self.transform(p_raw)[0])


def chronological_split(n: int, calib_frac: float, purge: int) -> tuple[slice, slice]:
    """Train / calibration split in time order with overlapping labels purged."""
    n_calib = max(int(n * calib_frac), 60)
    n_calib = min(n_calib, max(n // 3, 60))
    cut = n - n_calib
    train_end = max(cut - purge, 10)
    return slice(0, train_end), slice(cut, n)


def purged_folds(n: int, n_folds: int, purge: int,
                 coverage: float = 0.5) -> list[tuple[slice, slice]]:
    """Expanding-window folds over the most recent ``coverage`` of the sample.

    Each fold trains on everything before its test block, minus a ``purge`` gap
    so that no training label's forward window overlaps the test block.  Pooling
    the out-of-sample scores from several folds gives the calibrator a few
    thousand honest predictions spanning several market regimes, instead of the
    few hundred from one contiguous recent window -- which is the difference
    between measuring a model's reliability and measuring one regime's luck.
    """
    start = int(n * (1.0 - coverage))
    block = max((n - start) // max(n_folds, 1), 60)
    folds: list[tuple[slice, slice]] = []
    for i in range(n_folds):
        test_start = start + i * block
        test_end = min(test_start + block, n)
        if test_end - test_start < 40:
            break
        train_end = max(test_start - purge, 10)
        if train_end < 200:
            continue
        folds.append((slice(0, train_end), slice(test_start, test_end)))
    return folds


class SupervisedModel(BaseModel):
    """Base class for models that learn P(up) from a feature matrix."""

    family = "Machine Learning"
    calib_frac = 0.18
    calib_folds = 3           # purged walk-forward folds used to fit the calibrator
    halflife_days = DEFAULT_HALFLIFE_DAYS
    max_train_rows: int | None = None
    block_bootstrap = 2       # block length when sampling innovations

    def __init__(self) -> None:
        super().__init__()
        self.estimators_: dict[int, Any] = {}
        self.calibrators_: dict[int, PlattCalibrator] = {}
        self.feature_names_: list[str] = []
        self.horizon_info_: dict[int, dict[str, Any]] = {}
        self._context_stamp: Any = None
        # When frozen, a new context does not trigger re-estimation.  The
        # backtester uses this to hold parameters fixed between refit dates
        # while features and market state still advance day by day.
        self.frozen = False

    # -- to be provided by concrete models ---------------------------------
    def _make_estimator(self, horizon_days: int) -> Any:
        raise NotImplementedError

    def _fit_estimator(self, estimator: Any, X: np.ndarray, y: np.ndarray,
                       weights: np.ndarray) -> Any:
        raise NotImplementedError

    def _raw_proba(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _extra_info(self, estimator: Any, horizon_days: int) -> dict[str, Any]:
        return {}

    def _dispersion(self, estimator: Any, X: np.ndarray) -> float | None:
        """Spread of member predictions, used as a model-uncertainty proxy."""
        return None

    # -- shared training ----------------------------------------------------
    def _prepare(self, context: MarketContext, horizon_days: int):
        X, y, fwd = supervised_frame(context.daily, horizon_days,
                                     features=getattr(context, "features", None))
        if self.max_train_rows:
            X, y, fwd = X.iloc[-self.max_train_rows:], y.iloc[-self.max_train_rows:], fwd.iloc[-self.max_train_rows:]
        return X, y, fwd

    def _fit_calibrator(self, Xv: np.ndarray, yv: np.ndarray, weights: np.ndarray,
                        horizon_days: int) -> tuple[PlattCalibrator, int]:
        """Fit Platt scaling on pooled out-of-sample scores."""
        calibrator = PlattCalibrator()
        if self.calib_folds > 1:
            folds = purged_folds(len(Xv), self.calib_folds, horizon_days)
        else:
            train_slice, calib_slice = chronological_split(len(Xv), self.calib_frac, horizon_days)
            folds = [(train_slice, calib_slice)]

        scores, labels, fold_weights = [], [], []
        for train_slice, test_slice in folds:
            if train_slice.stop < 200 or test_slice.stop - test_slice.start < 40:
                continue
            try:
                warm = self._fit_estimator(self._make_estimator(horizon_days),
                                           Xv[train_slice], yv[train_slice], weights[train_slice])
                scores.append(self._raw_proba(warm, Xv[test_slice]))
                labels.append(yv[test_slice])
                fold_weights.append(weights[test_slice])
            except Exception as exc:  # pragma: no cover
                log.warning("%s: calibration fold failed (%s)", self.key, exc)

        if not scores:
            return calibrator, 0
        pooled_scores = np.concatenate(scores)
        pooled_labels = np.concatenate(labels)
        pooled_weights = np.concatenate(fold_weights)
        calibrator.fit(pooled_scores, pooled_labels, pooled_weights)
        return calibrator, len(pooled_scores)

    def ensure_fit(self, context: MarketContext, horizon_days: int) -> None:
        if self.frozen and horizon_days in self.estimators_:
            return
        stamp = (context.as_of, len(context.daily))
        if stamp != self._context_stamp:
            self.estimators_.clear()
            self.calibrators_.clear()
            self.horizon_info_.clear()
            self._context_stamp = stamp
        if horizon_days in self.estimators_:
            return

        X, y, _ = self._prepare(context, horizon_days)
        self.feature_names_ = list(X.columns)
        Xv, yv = X.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
        weights = recency_weights(X.index, self.halflife_days)

        calibrator, n_calib = self._fit_calibrator(Xv, yv, weights, horizon_days)

        # Refit on everything now that the calibration map is known.
        estimator = self._fit_estimator(self._make_estimator(horizon_days), Xv, yv, weights)

        self.estimators_[horizon_days] = estimator
        self.calibrators_[horizon_days] = calibrator
        self.horizon_info_[horizon_days] = {
            "n_train": int(len(Xv)),
            "n_calibration": int(n_calib),
            "train_start": str(X.index[0].date()),
            "train_end": str(X.index[-1].date()),
            "base_rate_up": float(np.mean(yv)),
            "calibration_slope": calibrator.a,
            "calibration_intercept": calibrator.b,
            "calibrated": calibrator.fitted,
            **self._extra_info(estimator, horizon_days),
        }
        self.fitted_ = True

    def fit(self, context: MarketContext) -> "SupervisedModel":
        for horizon in (1, 7):
            try:
                self.ensure_fit(context, horizon)
            except Exception as exc:  # pragma: no cover
                log.error("%s: fit failed for h=%d (%s)", self.key, horizon, exc)
        return self

    # -- inference ----------------------------------------------------------
    def _feature_frame(self, context: MarketContext) -> pd.DataFrame:
        cached = getattr(context, "features", None)
        if cached is not None:
            return cached
        return build_features(context.daily)

    def _live_row(self, context: MarketContext) -> np.ndarray:
        features = self._feature_frame(context).dropna()
        if features.empty:
            raise ValueError(f"{self.key}: no usable live feature row")
        row = features.iloc[[-1]]
        if self.feature_names_:
            row = row[self.feature_names_]
        return row.to_numpy(dtype="float64")

    def probability_up(self, context: MarketContext, horizon_days: int,
                       with_dispersion: bool = True) -> tuple[float, float]:
        self.ensure_fit(context, horizon_days)
        estimator = self.estimators_[horizon_days]
        X = self._live_row(context)
        raw = float(self._raw_proba(estimator, X)[0])
        p = self.calibrators_[horizon_days].transform_scalar(raw)
        # Ensemble dispersion means querying every member separately -- 600 trees
        # for the forest.  The backtester scores probabilities only, so it skips
        # it; the live dashboard, which shows the uncertainty, does not.
        spread = self._dispersion(estimator, X) if with_dispersion else None
        if spread is None:
            n = self.horizon_info_.get(horizon_days, {}).get("n_calibration", 200)
            spread = float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))
        self.horizon_info_.setdefault(horizon_days, {})["raw_proba"] = raw
        return p, float(spread)

    def simulate(self, context, horizon_days, n_paths, rng):
        p_up, spread = self.probability_up(context, horizon_days)
        innovations = bootstrap_innovations(context, horizon_days, n_paths, rng,
                                            block=self.block_bootstrap)
        tilted = tilt_to_probability(innovations, p_up)
        terminal = context.spot * np.exp(tilted)
        info = dict(self.horizon_info_.get(horizon_days, {}))
        info.update({
            "p_up_model": p_up,
            "p_up_stderr": spread,
            "implied_drift_pct": float(np.mean(tilted) * 100.0),
        })
        return terminal, info

    def predict_proba_up(self, context, horizon_days, n_paths, rng):
        return self.probability_up(context, horizon_days, with_dispersion=False)

    def feature_importance(self, horizon_days: int, top: int = 12) -> pd.Series | None:
        estimator = self.estimators_.get(horizon_days)
        if estimator is None or not self.feature_names_:
            return None
        values = None
        for attr in ("feature_importances_", "coef_"):
            raw = getattr(estimator, attr, None)
            if raw is not None:
                values = np.abs(np.ravel(raw))
                break
        if values is None and hasattr(estimator, "named_steps"):
            final = list(estimator.named_steps.values())[-1]
            for attr in ("feature_importances_", "coef_"):
                raw = getattr(final, attr, None)
                if raw is not None:
                    values = np.abs(np.ravel(raw))
                    break
        if values is None or len(values) != len(self.feature_names_):
            return None
        series = pd.Series(values, index=self.feature_names_)
        total = series.sum()
        if total > 0:
            series = series / total
        return series.sort_values(ascending=False).head(top)
