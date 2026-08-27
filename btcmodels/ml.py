"""Machine-learning direction models.

Three learners with deliberately different inductive biases, so that agreement
between them is informative rather than tautological:

* **Gradient boosting** -- sequential residual fitting, sharp non-linear splits.
* **Random forest** -- bagged decorrelated trees, high variance reduction.
* **Elastic-net logistic regression** -- a linear, heavily regularised control
  that shows how much of the signal is genuinely non-linear.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .supervised import SupervisedModel

log = logging.getLogger(__name__)

N_JOBS = int(os.environ.get("BTC_N_JOBS", 2))


class SeedEnsemble:
    """Average several identically-specified members trained on different seeds.

    Two things come out of this: a modest accuracy gain from variance
    averaging, and -- more useful here -- the spread across members, which is a
    genuine measure of how much of the prediction is model noise.
    """

    def __init__(self, members: list[Any]) -> None:
        self.members = members

    def member_probas(self, X: np.ndarray) -> np.ndarray:
        return np.stack([m.predict_proba(X)[:, 1] for m in self.members], axis=0)

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        return self.member_probas(X).mean(axis=0)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        mats = [np.ravel(getattr(m, "feature_importances_")) for m in self.members
                if hasattr(m, "feature_importances_")]
        if mats:
            return np.mean(mats, axis=0)
        finals = [list(m.named_steps.values())[-1] for m in self.members
                  if hasattr(m, "named_steps")]
        mats = [np.abs(np.ravel(f.coef_)) for f in finals if hasattr(f, "coef_")]
        return np.mean(mats, axis=0) if mats else None


# ---------------------------------------------------------------------------
# 6. Gradient boosting
# ---------------------------------------------------------------------------
class GradientBoostingModel(SupervisedModel):
    key = "xgboost"
    name = "Gradient-Boosted Trees (XGBoost)"
    family = "Machine Learning"
    method = "Depth-3 boosted trees, recency-weighted, Platt-calibrated, 3-seed ensemble"
    description = (
        "Boosting fits each tree to the errors of the ensemble so far, which lets "
        "it pick up conditional structure a diffusion cannot express -- for "
        "instance 'momentum only pays when realised volatility is compressed'. "
        "Trees are kept shallow (depth 3) and strongly regularised because the "
        "signal-to-noise ratio in daily crypto direction is very low, and an "
        "unconstrained booster memorises noise almost immediately."
    )

    def __init__(self, n_seeds: int = 3) -> None:
        super().__init__()
        self.n_seeds = n_seeds

    def _make_estimator(self, horizon_days: int):
        import xgboost as xgb

        return [
            xgb.XGBClassifier(
                n_estimators=260 if horizon_days == 1 else 200,
                max_depth=3,
                learning_rate=0.028,
                subsample=0.8,
                colsample_bytree=0.65,
                min_child_weight=12.0,
                reg_lambda=3.0,
                reg_alpha=0.4,
                gamma=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=N_JOBS,
                random_state=100 + 7 * seed,
                verbosity=0,
            )
            for seed in range(self.n_seeds)
        ]

    def _fit_estimator(self, estimator, X, y, weights):
        for member in estimator:
            member.fit(X, y, sample_weight=weights)
        return SeedEnsemble(estimator)

    def _raw_proba(self, estimator, X):
        return estimator.predict_proba_up(X)

    def _dispersion(self, estimator, X):
        members = estimator.member_probas(X)
        return float(np.std(members[:, 0], ddof=1)) if members.shape[0] > 1 else None

    def _extra_info(self, estimator, horizon_days):
        return {"n_seeds": self.n_seeds, "max_depth": 3, "learning_rate": 0.028}


# ---------------------------------------------------------------------------
# 7. Random forest
# ---------------------------------------------------------------------------
class RandomForestModel(SupervisedModel):
    key = "random_forest"
    name = "Random Forest"
    family = "Machine Learning"
    method = "600 bagged trees, min-leaf regularised, per-tree vote dispersion"
    description = (
        "Bagged trees on bootstrapped samples and random feature subsets. Where "
        "boosting chases the residual and can lock onto a spurious pattern, the "
        "forest averages many weak independent views, so it is biased toward the "
        "unconditional base rate and rarely produces extreme probabilities. Its "
        "per-tree vote spread doubles as an honest uncertainty estimate."
    )

    def _make_estimator(self, horizon_days: int):
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=25,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=N_JOBS,
            random_state=17,
            class_weight=None,
        )

    def _fit_estimator(self, estimator, X, y, weights):
        estimator.fit(X, y, sample_weight=weights)
        return estimator

    def _raw_proba(self, estimator, X):
        return estimator.predict_proba(X)[:, 1]

    def _dispersion(self, estimator, X):
        votes = np.array([t.predict_proba(X)[0, 1] for t in estimator.estimators_])
        # Standard error of the forest mean, not the spread of individual trees.
        return float(np.std(votes, ddof=1) / np.sqrt(len(votes)))

    def _extra_info(self, estimator, horizon_days):
        return {"n_trees": estimator.n_estimators, "max_depth": estimator.max_depth,
                "min_samples_leaf": estimator.min_samples_leaf}


# ---------------------------------------------------------------------------
# 8. Elastic-net logistic regression
# ---------------------------------------------------------------------------
class LogisticModel(SupervisedModel):
    key = "elasticnet"
    name = "Elastic-Net Logistic Regression"
    family = "Machine Learning"
    method = "Standardised features, L1/L2 mixed penalty, SAGA solver"
    description = (
        "The linear control. If the tree models cannot beat a regularised "
        "logistic regression on the same features, their extra capacity is "
        "fitting noise rather than structure. The elastic-net penalty drives most "
        "of the 47 coefficients to zero, so what survives is a readable short "
        "list of what actually moves the odds."
    )

    def _make_estimator(self, horizon_days: int):
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                l1_ratio=0.4, C=0.05, solver="saga",
                max_iter=4000, tol=1e-4, random_state=3)),
        ])

    def _fit_estimator(self, estimator, X, y, weights):
        estimator.fit(X, y, clf__sample_weight=weights)
        return estimator

    def _raw_proba(self, estimator, X):
        return estimator.predict_proba(X)[:, 1]

    def _extra_info(self, estimator, horizon_days):
        coef = np.ravel(estimator.named_steps["clf"].coef_)
        return {"n_nonzero_coefs": int(np.sum(np.abs(coef) > 1e-8)),
                "n_features": int(coef.size), "l1_ratio": 0.4, "C": 0.05}


# ---------------------------------------------------------------------------
# 9. Feed-forward neural network
# ---------------------------------------------------------------------------
class MLPModel(SupervisedModel):
    key = "mlp"
    name = "Feed-Forward Neural Network (MLP)"
    family = "Neural Network"
    method = "2 hidden layers (32-12), tanh, Adam, chronological early stopping, 5-seed ensemble"
    description = (
        "A dense network over the same feature matrix. Unlike the trees it "
        "produces a smooth decision surface, so it interpolates between feature "
        "regimes instead of stepping between them, and it can represent smooth "
        "interactions among many features at once. Five seeds are averaged "
        "because a single small network on noisy financial data depends heavily "
        "on its initialisation."
    )

    # An earlier version claimed sklearn's MLP had no sample_weight hook and
    # substituted a rectangular window. That was wrong -- MLPClassifier.fit and
    # .partial_fit both accept sample_weight in the installed version -- and it
    # meant the recency weights computed upstream were silently discarded for
    # this model while its own Platt calibrator still used them, so estimator
    # and calibration were trained under different weightings. The exponential
    # kernel is also the better estimator on its own merits: a rectangular
    # window weights a seven-year-old row exactly like yesterday's and then
    # drops it off a cliff.
    max_train_rows = 2555          # kept only as a compute cap
    val_fraction = 0.12
    max_epochs = 220
    patience = 25

    def __init__(self, n_seeds: int = 5) -> None:
        super().__init__()
        self.n_seeds = n_seeds

    def _make_estimator(self, horizon_days: int):
        return [
            MLPClassifier(
                hidden_layer_sizes=(32, 12),
                activation="tanh",
                solver="adam",
                alpha=2e-2,                      # strong L2; the signal is faint
                learning_rate_init=1.5e-3,
                batch_size=128,
                max_iter=1,                      # stepped manually below
                warm_start=True,
                shuffle=True,
                random_state=200 + 13 * seed,
            )
            for seed in range(self.n_seeds)
        ]

    def _train_member(self, net: MLPClassifier, Xtr, ytr, Xva, yva,
                      wtr=None, wva=None) -> tuple[MLPClassifier, int]:
        """Adam with early stopping on a *chronological* validation tail.

        sklearn's own ``early_stopping`` carves out its validation set with a
        shuffled split, which on overlapping time-series windows is optimistic
        enough to be useless as a stopping signal.
        """
        best_loss, best_state, best_epoch = np.inf, None, 0
        classes = np.array([0.0, 1.0])
        for epoch in range(1, self.max_epochs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.partial_fit(Xtr, ytr, classes=classes, sample_weight=wtr)
            p = np.clip(net.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
            per_row = -(yva * np.log(p) + (1 - yva) * np.log(1 - p))
            loss = float(np.average(per_row, weights=wva) if wva is not None
                         else np.mean(per_row))
            if loss < best_loss - 1e-5:
                best_loss, best_epoch = loss, epoch
                best_state = ([w.copy() for w in net.coefs_],
                              [b.copy() for b in net.intercepts_])
            elif epoch - best_epoch >= self.patience:
                break
        if best_state is not None:
            net.coefs_, net.intercepts_ = best_state
        return net, best_epoch

    def _fit_estimator(self, estimator, X, y, weights):
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        n_val = int(np.clip(len(Xs) * self.val_fraction, 100, len(Xs) // 4))
        cut = len(Xs) - n_val
        Xtr, ytr, Xva, yva = Xs[:cut], y[:cut], Xs[cut:], y[cut:]
        wtr, wva = weights[:cut], weights[cut:]

        trained, epochs = [], []
        for net in estimator:
            net, best_epoch = self._train_member(net, Xtr, ytr, Xva, yva, wtr, wva)
            trained.append(net)
            epochs.append(best_epoch)
        ensemble = SeedEnsemble(trained)
        ensemble.scaler = scaler
        ensemble.epochs = epochs
        return ensemble

    def _raw_proba(self, estimator, X):
        return estimator.predict_proba_up(estimator.scaler.transform(X))

    def _dispersion(self, estimator, X):
        members = estimator.member_probas(estimator.scaler.transform(X))
        return float(np.std(members[:, 0], ddof=1) / np.sqrt(members.shape[0]))

    def _extra_info(self, estimator, horizon_days):
        return {"n_seeds": self.n_seeds, "hidden_layers": "32-12",
                "mean_best_epoch": float(np.mean(estimator.epochs)),
                "l2_alpha": 2e-2, "train_window_rows": self.max_train_rows}
