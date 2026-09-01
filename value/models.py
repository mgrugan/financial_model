"""Cross-sectional models predicting 6-month relative returns from fundamentals.

Three choices decide whether this measures anything real.

**Rank-transform everything, within each period.** Both the features and the
target are converted to within-period percentiles. Raw ratios cannot be pooled
across 2012 and 2022 -- the level of the whole cross-section moves with rates
and sentiment, so a model trained on levels mostly learns which half-years were
expensive, which is useless for ranking the stocks inside a half-year. Ranking
within the period asks the only question a stock-picker can act on: is this
name cheap *relative to its peers right now*.

**Expanding window by period, never a shuffled split.** A random split would put
December 2018 in training and June 2018 in test, and since the two share most of
the same companies at nearly the same fundamentals, the model would be scored on
rows it had effectively already seen.

**Score on the period spread, not on pooled R-squared.** A pooled R-squared over
13,000 stock-periods is dominated by the market factor and would look impressive
for a model that has learned nothing but "everything went up in 2020".
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# market_cap is deliberately absent. In this universe -- today's index
# membership applied backwards -- being small in 2012 implies having grown
# enough to still be a member in 2026, so size "predicts" returns with a 100%
# hit rate across 28 half-years. A model given that column learns the selection
# rule rather than anything about value, and reports a t-statistic near 8 for
# doing so. pb and pe are dropped for the same reason in raw form: they are
# monotone functions of the yields already present, so they add nothing except
# another route to the same artifact.
FEATURES = [
    "ebitda_ev_yield", "fcf_ev_yield", "earnings_yield", "book_yield",
    "graham_ratio", "graham_score", "current_ratio", "debt_to_equity",
    "ncav_yield", "ev_sales",
]

MIN_TRAIN_PERIODS = 8


def _rank_within(frame: pd.DataFrame, cols: list[str], by: str = "date") -> pd.DataFrame:
    """Within-period percentile ranks; missing values sit at the median (0.5)."""
    out = pd.DataFrame(index=frame.index)
    for col in cols:
        if col not in frame:
            continue
        r = frame.groupby(by)[col].rank(pct=True)
        out[col] = r.fillna(0.5)
    return out


def build_design(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    work = panel.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "ncav_yield" not in work:
            work["ncav_yield"] = work["ncav"] / work["market_cap"]
    X = _rank_within(work, FEATURES)
    y = work.groupby("date")["fwd_return"].rank(pct=True)
    return X, y, work["date"]


def model_builders() -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    return {
        "ridge": lambda: Ridge(alpha=10.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=40,
            random_state=7, n_jobs=1),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, random_state=7),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(24, 12), alpha=1.0, max_iter=600,
            early_stopping=False, random_state=7),
    }


def walk_forward(panel: pd.DataFrame, model_key: str,
                 min_train: int = MIN_TRAIN_PERIODS,
                 size_neutral: bool = False) -> dict[str, Any]:
    """Train on every earlier period, predict the next one, roll forward.

    ``size_neutral`` residualises the prediction on the within-period size rank
    before bucketing. Dropping market capitalisation from the feature list is
    not enough on its own: every price-scaled ratio in the design matrix has
    market cap in its denominator, so the model can rebuild the size bet -- and
    with it the survivorship artifact -- out of the value features themselves.
    The single-factor tests are neutralised the same way, so the two are
    directly comparable.
    """
    X, y, dates = build_design(panel)
    order = sorted(panel["date"].unique())
    build = model_builders()[model_key]

    rows = []
    for i in range(min_train, len(order)):
        test_date = order[i]
        train_mask = dates.isin(order[:i])
        test_mask = dates == test_date
        keep = y.notna()
        if (train_mask & keep).sum() < 500 or (test_mask & keep).sum() < 40:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build()
            model.fit(X[train_mask & keep], y[train_mask & keep])
            pred = model.predict(X[test_mask & keep])

        actual = panel.loc[test_mask & keep, "fwd_return"]
        pred_s = pd.Series(pred, index=actual.index)
        if size_neutral:
            from .factors import _residualise_on_size
            pred_s = _residualise_on_size(pred_s, panel.loc[test_mask & keep])
            good = pred_s.notna()
            pred_s, actual = pred_s[good], actual[good]
            if len(pred_s) < 40:
                continue
        buckets = pd.qcut(pred_s.rank(method="first"), 5, labels=False) + 1
        means = actual.groupby(buckets).mean()
        if 1 not in means.index or 5 not in means.index:
            continue
        ic = stats.spearmanr(pred_s, actual).statistic
        rows.append({
            "date": test_date,
            "n": int(len(actual)),
            "spread": float(means[5] - means[1]),
            "top": float(means[5]),
            "universe": float(actual.mean()),
            "ic": float(ic) if np.isfinite(ic) else np.nan,
        })

    if len(rows) < 6:
        return {"model": model_key, "status": "too few test periods", "n_periods": len(rows)}

    frame = pd.DataFrame(rows)
    spread = frame["spread"].to_numpy()
    excess = (frame["top"] - frame["universe"]).to_numpy()
    n = len(spread)

    def _t(a: np.ndarray) -> tuple[float, float]:
        t = float(np.mean(a) / (np.std(a, ddof=1) / np.sqrt(len(a))))
        return t, float(2.0 * stats.t.sf(abs(t), df=len(a) - 1))

    t_spread, p_spread = _t(spread)
    t_excess, p_excess = _t(excess)
    ic = frame["ic"].to_numpy()
    t_ic, p_ic = _t(ic)

    return {
        "model": model_key,
        "size_neutral": size_neutral,
        "status": "ok",
        "n_periods": n,
        "first_test": frame["date"].iloc[0],
        "mean_spread": float(np.mean(spread)),
        "t_spread": t_spread,
        "p_spread": p_spread,
        "hit_rate": float(np.mean(spread > 0)),
        "mean_excess_long_only": float(np.mean(excess)),
        "t_excess": t_excess,
        "p_excess": p_excess,
        "mean_ic": float(np.mean(ic)),
        "t_ic": t_ic,
        "p_ic": p_ic,
        "periods": frame.to_dict("records"),
    }
