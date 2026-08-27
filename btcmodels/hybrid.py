"""Neural-drift diffusion: the model where the two families actually meet.

A classical diffusion assumes ``d log S = mu*dt + sigma*dW`` with *constant*
``mu`` and ``sigma``.  Both assumptions are the weak point: the drift is
statistically invisible over any sample you can collect, and the volatility is
demonstrably not constant.

This model keeps the diffusion structure and replaces the two constants with
learned, state-dependent functions -- the discrete-time cousin of a neural SDE:

    d log S = mu(x_t)*dt + sigma(x_t)*dW,  dW drawn from empirical fat tails

* ``mu(x_t)`` is a gradient-boosted regression on the forward return.
* ``sigma(x_t)`` is a gradient-boosted regression on forward realised
  volatility, blended with the GARCH forecast.
* the shock is a block bootstrap of standardised GARCH residuals, so skew and
  kurtosis come from the data rather than from a normal assumption.

Both learned pieces are passed through an **out-of-sample shrinkage
regression**.  Pooled predictions from purged walk-forward folds are regressed
on what actually happened; the fitted slope is the fraction of the raw
prediction that survived out of sample, and only that fraction is used.  A
model with no genuine forecasting power gets a slope near zero and collapses
gracefully back to a plain calibrated diffusion, which is the behaviour you
want when the alternative is a confident wrong drift.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel, MarketContext, bootstrap_innovations, prob_stderr
from .config import DAYS_PER_YEAR
from .features import build_features, supervised_frame
from .supervised import purged_folds, recency_weights

log = logging.getLogger(__name__)


def _shrinkage_regression(pred: np.ndarray, actual: np.ndarray,
                          weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Weighted OLS of ``actual`` on ``pred``.

    The slope is the classic optimal-shrinkage factor: how much of the raw
    forecast's variation was actually realised out of sample.  Slope <= 0 means
    the forecast carried no usable information and should be discarded.
    """
    if pred.size < 30 or np.std(pred) < 1e-12:
        return 0.0, float(np.mean(actual)) if actual.size else 0.0, 0.0
    w = np.ones_like(pred) if weights is None else weights
    w = w / w.sum()
    pm = float(np.sum(w * pred))
    am = float(np.sum(w * actual))
    cov = float(np.sum(w * (pred - pm) * (actual - am)))
    var = float(np.sum(w * (pred - pm) ** 2))
    slope = cov / var if var > 1e-18 else 0.0
    intercept = am - slope * pm
    denom = float(np.sum(w * (actual - am) ** 2))
    r2 = (slope * cov / denom) if denom > 1e-18 else 0.0
    return float(slope), float(intercept), float(r2)


class NeuralDriftDiffusionModel(BaseModel):
    key = "hybrid"
    name = "ML-Drift Diffusion (Hybrid)"
    family = "Hybrid"
    method = ("Boosted drift + boosted volatility inside a bootstrapped diffusion, "
              "both shrunk by out-of-sample regression slope")
    description = (
        "A Brownian diffusion whose drift and volatility are machine-learned "
        "functions of the current market state instead of fixed constants, with "
        "shocks bootstrapped from the empirical residual distribution so the "
        "tails are real rather than Gaussian. Each learned component is scaled "
        "by the slope of an out-of-sample regression of outcomes on forecasts, "
        "so a component with no demonstrated skill is automatically shrunk "
        "toward zero and the model degrades to a well-calibrated stochastic-"
        "volatility diffusion rather than to a confident wrong answer."
    )

    halflife_days = 730.0
    n_folds = 3
    drift_cap_sigma = 0.75      # drift is capped at this many horizon sigmas

    def __init__(self) -> None:
        super().__init__()
        self.drift_models_: dict[int, Any] = {}
        self.vol_models_: dict[int, Any] = {}
        self.drift_shrink_: dict[int, tuple[float, float, float]] = {}
        self.vol_shrink_: dict[int, tuple[float, float, float]] = {}
        self.horizon_info_: dict[int, dict[str, Any]] = {}
        self.feature_names_: list[str] = []
        self._vol_bias_: dict[int, float] = {}
        self._context_stamp: Any = None
        self.frozen = False

    def _vol_bias(self, context: MarketContext, horizon: int) -> float:
        """Jensen correction between the regression target and log-volatility.

        The learned volatility is trained on ``log sqrt(mean(r^2))`` over the
        forward window, which is *not* ``log sigma``: for a one-day horizon it is
        ``log sigma + E[log|z|]``, and ``E[log|z|]`` is around -0.64 for a normal
        shock and considerably more negative for the fat-tailed residuals here.
        Exponentiating without removing that term understates volatility by
        roughly a factor of two. The offset is estimated by simulating the
        statistic from the empirical standardised residuals.
        """
        if horizon in self._vol_bias_:
            return self._vol_bias_[horizon]
        rng = np.random.default_rng(4242)
        draws = rng.choice(context.std_resid, size=(20_000, horizon), replace=True)
        stat = np.log(np.sqrt(np.maximum((draws**2).mean(axis=1), 1e-12)))
        bias = float(np.mean(stat))
        self._vol_bias_[horizon] = bias
        return bias

    # -- estimators ---------------------------------------------------------
    @staticmethod
    def _make_regressor(seed: int = 0):
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=15.0,
            reg_lambda=4.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        )

    @staticmethod
    def _vol_target(daily: pd.DataFrame, horizon: int) -> pd.Series:
        """log of forward realised volatility (per day) over the horizon."""
        logret = np.log(daily["close"]).diff()
        # mean squared return over days t+1 .. t+horizon
        fwd_sq = (logret**2).rolling(horizon).mean().shift(-horizon)
        return np.log(np.sqrt(fwd_sq.clip(lower=1e-10)))

    # -- fitting ------------------------------------------------------------
    def _prepare(self, context: MarketContext, horizon: int):
        X, _, fwd = supervised_frame(context.daily, horizon,
                                     features=getattr(context, "features", None))
        frame = context.daily
        if "is_partial" in frame.columns:
            frame = frame[~frame["is_partial"].astype(bool)]
        vol_target = self._vol_target(frame, horizon).reindex(X.index)
        keep = vol_target.notna() & fwd.notna()
        return X[keep], fwd[keep], vol_target[keep]

    def _fit_horizon(self, context: MarketContext, horizon: int) -> None:
        X, fwd, vol_y = self._prepare(context, horizon)
        self.feature_names_ = list(X.columns)
        Xv = X.to_numpy(dtype="float64")
        drift_y = fwd.to_numpy(dtype="float64")
        volv = vol_y.to_numpy(dtype="float64")
        weights = recency_weights(X.index, self.halflife_days)

        # --- out-of-sample shrinkage from purged walk-forward folds ---------
        folds = purged_folds(len(Xv), self.n_folds, horizon)
        dp, da, vp, va, fw = [], [], [], [], []
        for train_slice, test_slice in folds:
            try:
                dm = self._make_regressor(1).fit(Xv[train_slice], drift_y[train_slice],
                                                 sample_weight=weights[train_slice])
                vm = self._make_regressor(2).fit(Xv[train_slice], volv[train_slice],
                                                 sample_weight=weights[train_slice])
                dp.append(dm.predict(Xv[test_slice]))
                da.append(drift_y[test_slice])
                vp.append(vm.predict(Xv[test_slice]))
                va.append(volv[test_slice])
                fw.append(weights[test_slice])
            except Exception as exc:  # pragma: no cover
                log.warning("hybrid: fold failed (%s)", exc)

        if dp:
            pooled_w = np.concatenate(fw)
            drift_shrink = _shrinkage_regression(np.concatenate(dp), np.concatenate(da), pooled_w)
            vol_shrink = _shrinkage_regression(np.concatenate(vp), np.concatenate(va), pooled_w)
            n_oos = len(pooled_w)
        else:
            drift_shrink = (0.0, float(np.mean(drift_y)), 0.0)
            vol_shrink = (0.0, float(np.mean(volv)), 0.0)
            n_oos = 0

        # A negative slope means the forecast was worse than useless; clamp to
        # zero rather than betting on the inverse of a noisy signal.
        drift_shrink = (float(np.clip(drift_shrink[0], 0.0, 1.5)), drift_shrink[1], drift_shrink[2])
        vol_shrink = (float(np.clip(vol_shrink[0], 0.0, 1.5)), vol_shrink[1], vol_shrink[2])

        self.drift_models_[horizon] = self._make_regressor(1).fit(
            Xv, drift_y, sample_weight=weights)
        self.vol_models_[horizon] = self._make_regressor(2).fit(
            Xv, volv, sample_weight=weights)
        self.drift_shrink_[horizon] = drift_shrink
        self.vol_shrink_[horizon] = vol_shrink
        self.horizon_info_[horizon] = {
            "n_train": int(len(Xv)),
            "n_oos_shrinkage": int(n_oos),
            "drift_oos_slope": drift_shrink[0],
            "drift_oos_r2": drift_shrink[2],
            "vol_oos_slope": vol_shrink[0],
            "vol_oos_r2": vol_shrink[2],
            "train_end": str(X.index[-1].date()),
        }

    def ensure_fit(self, context: MarketContext, horizon: int) -> None:
        if self.frozen and horizon in self.drift_models_:
            return
        stamp = (context.as_of, len(context.daily))
        if stamp != self._context_stamp:
            self.drift_models_.clear()
            self.vol_models_.clear()
            self.horizon_info_.clear()
            self._vol_bias_.clear()
            self._context_stamp = stamp
        if horizon not in self.drift_models_:
            self._fit_horizon(context, horizon)
            self.fitted_ = True

    def fit(self, context: MarketContext) -> "NeuralDriftDiffusionModel":
        for horizon in (1, 7):
            try:
                self.ensure_fit(context, horizon)
            except Exception as exc:  # pragma: no cover
                log.error("hybrid: fit failed for h=%d (%s)", horizon, exc)
        return self

    # -- inference ----------------------------------------------------------
    def _live_row(self, context: MarketContext) -> np.ndarray:
        cached = getattr(context, "features", None)
        features = (build_features(context.daily) if cached is None else cached).dropna()
        row = features.iloc[[-1]]
        if self.feature_names_:
            row = row[self.feature_names_]
        return row.to_numpy(dtype="float64")

    def _state_params(self, context: MarketContext, horizon: int) -> dict[str, float]:
        self.ensure_fit(context, horizon)
        X = self._live_row(context)

        d_slope, d_intercept, _ = self.drift_shrink_[horizon]
        raw_drift = float(self.drift_models_[horizon].predict(X)[0])
        drift = d_slope * raw_drift + d_intercept

        v_slope, v_intercept, _ = self.vol_shrink_[horizon]
        raw_logvol = float(self.vol_models_[horizon].predict(X)[0])
        bias = self._vol_bias(context, horizon)
        ml_sigma_day = float(np.exp(v_slope * raw_logvol + v_intercept - bias))

        # Blend the learned volatility with the GARCH forecast: they use
        # different information (cross-sectional features vs the variance
        # recursion) and averaging in log space is the stable way to combine.
        garch_sigma_day = context.horizon_sigma(horizon) / np.sqrt(horizon)
        sigma_day = float(np.exp(0.5 * np.log(max(ml_sigma_day, 1e-8))
                                 + 0.5 * np.log(max(garch_sigma_day, 1e-8))))
        sigma_h = sigma_day * np.sqrt(horizon)

        # Cap the drift so an extreme feature row cannot produce an absurd
        # directional bet: at most `drift_cap_sigma` horizon standard deviations.
        cap = self.drift_cap_sigma * sigma_h
        drift = float(np.clip(drift, -cap, cap))

        return {
            "drift": drift,
            "raw_drift": raw_drift,
            "sigma_h": sigma_h,
            "sigma_day": sigma_day,
            "ml_sigma_annual_pct": ml_sigma_day * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "garch_sigma_annual_pct": garch_sigma_day * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "blended_sigma_annual_pct": sigma_day * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "drift_cap_hit": abs(drift) >= cap - 1e-12,
            "vol_jensen_bias": bias,
        }

    def simulate(self, context, horizon_days, n_paths, rng):
        params = self._state_params(context, horizon_days)
        shocks = bootstrap_innovations(context, horizon_days, n_paths, rng,
                                       sigma_override=params["sigma_h"], block=2)
        terminal = context.spot * np.exp(params["drift"] + shocks)
        info = dict(self.horizon_info_.get(horizon_days, {}))
        info.update({
            "drift_pct": params["drift"] * 100.0,
            "raw_drift_pct": params["raw_drift"] * 100.0,
            "sigma_horizon": params["sigma_h"],
            "ml_sigma_annual_pct": params["ml_sigma_annual_pct"],
            "garch_sigma_annual_pct": params["garch_sigma_annual_pct"],
            "blended_sigma_annual_pct": params["blended_sigma_annual_pct"],
            "drift_cap_hit": params["drift_cap_hit"],
            "vol_jensen_bias": params["vol_jensen_bias"],
            "drift_in_sigmas": params["drift"] / max(params["sigma_h"], 1e-12),
        })
        return terminal, info

    def predict_proba_up(self, context, horizon_days, n_paths, rng):
        terminal, _ = self.simulate(context, horizon_days, min(n_paths, 20_000), rng)
        p = float(np.mean(terminal > context.spot))
        return p, prob_stderr(p, min(n_paths, 20_000))

    def feature_importance(self, horizon_days: int, top: int = 12) -> pd.Series | None:
        model = self.drift_models_.get(horizon_days)
        if model is None or not self.feature_names_:
            return None
        values = np.abs(np.ravel(model.feature_importances_))
        if len(values) != len(self.feature_names_):
            return None
        series = pd.Series(values, index=self.feature_names_)
        if series.sum() > 0:
            series = series / series.sum()
        return series.sort_values(ascending=False).head(top)
