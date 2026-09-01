"""Stage-1 screen: is there *any* directional signal in this ticker?

The design problem here is not modelling, it is multiple testing. Run one
honest test per ticker at the 5% level across 600 tickers and two horizons and
roughly 60 of those 1,200 tests come back "significant" with no signal present
anywhere. Reading those 60 as discoveries is the single most common way a
cross-sectional study like this fools its author. So:

* every ticker gets the **same** pre-specified model and protocol -- no
  per-ticker tuning, no picking the best of several models, because a maximum
  over choices has a much wider null distribution than any single choice;
* the standard error is computed on **effective** sample size, ``n/h``, since
  overlapping h-bar labels are not h independent observations;
* p-values go through **Benjamini-Hochberg** across the entire family of
  ``2 x n_tickers`` tests, and only the FDR-adjusted q-value is allowed to turn
  a row green;
* the same pipeline is run over synthetic random walks as a placebo arm, so the
  page can show what the screen does when there is provably nothing to find.

The model is plain L2 logistic regression. That is deliberate: it is the
cheapest member of the stack, and stage 1 is a filter, not a verdict. Anything
it flags goes to stage 2 for the full eleven-model treatment.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from btcmodels.backtest import _auc, _auc_stderr
from btcmodels.base import stable_hash
from btcmodels.features import build_features, make_labels

from .data import EQUITY_HORIZONS, TRADING_DAYS_PER_YEAR

log = logging.getLogger(__name__)

# Walk-forward geometry, in trading bars.
MIN_TRAIN = 756          # 3 years of warm-up before the first prediction
TEST_BARS = 1512         # ~6 years of out-of-sample predictions
REFIT_EVERY = 63         # refit quarterly

COST_BPS = 20.0          # round-trip cost assumption for a small cap

FDR_ALPHA = 0.10         # Benjamini-Hochberg level for the green light


# ---------------------------------------------------------------------------
# Multiple-testing machinery
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up q-values. NaN p-values pass through as NaN and are excluded."""
    p = np.asarray(pvals, dtype="float64")
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q

    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    # Step-up: enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = out
    return q


def block_bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, horizon: int,
                           n_boot: int = 2000, seed: int = 11,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the AUC from a stationary block bootstrap.

    The parametric Hanley-McNeil interval assumes independent observations and
    only corrects for label overlap through the effective count. Resampling in
    blocks whose mean length is the horizon keeps the actual serial dependence
    of both the predictions and the returns, which is the honest check for a
    survivor.
    """
    n = len(y)
    if n < 50:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    mean_block = max(int(horizon) * 5, 10)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype="int64")
        filled = 0
        while filled < n:
            start = rng.integers(0, n)
            length = min(int(rng.geometric(1.0 / mean_block)), n - filled)
            take = (start + np.arange(length)) % n
            idx[filled:filled + length] = take
            filled += length
        yb, pb = y[idx], p[idx]
        aucs[b] = _auc(yb, pb) if 0 < yb.sum() < n else np.nan
    aucs = aucs[np.isfinite(aucs)]
    if aucs.size < 100:
        return float("nan"), float("nan")
    return (float(np.quantile(aucs, alpha / 2.0)),
            float(np.quantile(aucs, 1.0 - alpha / 2.0)))


# ---------------------------------------------------------------------------
# The walk-forward itself
# ---------------------------------------------------------------------------
def _fit_predict_block(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                       seed: int) -> np.ndarray | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_tr)) < 2 or len(y_tr) < 100:
        return None
    scaler = StandardScaler().fit(X_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=0.05, max_iter=400, solver="lbfgs",
                                 random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        return clf.predict_proba(scaler.transform(X_te))[:, 1]


def _strategy_stats(p: np.ndarray, fwd: np.ndarray, horizon: int,
                    cost_bps: float = COST_BPS) -> dict[str, float]:
    """Long/flat on p>0.5, on non-overlapping blocks so returns are additive."""
    step = max(int(horizon), 1)
    sel = np.arange(0, len(p), step)
    if len(sel) < 8:
        return {"strategy_cagr": float("nan"), "strategy_sharpe": float("nan"),
                "buyhold_cagr": float("nan"), "n_trades": 0}
    ps, fs = p[sel], fwd[sel]
    pos = (ps >= 0.5).astype("float64")
    turns = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * fs - turns * (cost_bps / 10_000.0)

    periods_per_year = TRADING_DAYS_PER_YEAR / step
    years = len(sel) / periods_per_year
    strat = float(np.exp(net.sum()) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    hold = float(np.exp(fs.sum()) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    sd = float(np.std(net, ddof=1))
    sharpe = float(np.mean(net) / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    return {"strategy_cagr": strat * 100.0, "strategy_sharpe": sharpe,
            "buyhold_cagr": hold * 100.0, "n_trades": int(turns.sum())}


def screen_ticker(ticker: str, frame: pd.DataFrame,
                  horizons: dict[str, int] | None = None,
                  min_train: int = MIN_TRAIN, test_bars: int = TEST_BARS,
                  refit_every: int = REFIT_EVERY,
                  keep_predictions: bool = False) -> list[dict[str, Any]]:
    """One expanding-window walk-forward per horizon. Returns one row each."""
    horizons = horizons or EQUITY_HORIZONS
    features = build_features(frame)
    close = frame["close"]
    log_close = np.log(close.to_numpy())
    n = len(frame)

    usable = features.dropna()
    if usable.empty:
        return []
    first_valid = int(frame.index.get_indexer([usable.index[0]])[0])

    rows: list[dict[str, Any]] = []
    for horizon_key, h in horizons.items():
        start = max(first_valid + min_train, n - test_bars)
        end = n - h
        if start >= end - 60:
            rows.append({"ticker": ticker, "horizon": horizon_key, "horizon_bars": h,
                         "status": "insufficient history", "n": 0})
            continue

        labels = make_labels(frame, h)
        Xf = features.to_numpy(dtype="float64")
        yf = labels["y_up"].to_numpy(dtype="float64")
        fwdf = labels["fwd_logret"].to_numpy(dtype="float64")
        row_ok = np.isfinite(Xf).all(axis=1)

        preds, ys, fwds, dates = [], [], [], []
        for i0 in range(start, end, refit_every):
            block_end = min(i0 + refit_every, end)
            # Purge: a training row at t has a label reaching t+h, so only rows
            # with t <= i0-h are known at the block boundary.
            tr = np.arange(0, max(i0 - h + 1, 0))
            tr = tr[row_ok[tr] & np.isfinite(yf[tr])]
            te = np.arange(i0, block_end)
            te = te[row_ok[te] & np.isfinite(fwdf[te])]
            if len(tr) < 100 or len(te) == 0:
                continue
            p = _fit_predict_block(Xf[tr], yf[tr], Xf[te], seed=stable_hash(ticker) % 10_000)
            if p is None:
                continue
            preds.append(p)
            ys.append((log_close[te + h] - log_close[te] > 0).astype("float64"))
            fwds.append(log_close[te + h] - log_close[te])
            dates.append(frame.index[te])

        if not preds:
            rows.append({"ticker": ticker, "horizon": horizon_key, "horizon_bars": h,
                         "status": "no usable blocks", "n": 0})
            continue

        p = np.clip(np.concatenate(preds), 1e-6, 1 - 1e-6)
        y = np.concatenate(ys)
        fwd = np.concatenate(fwds)

        auc = _auc(y, p)
        n_eff = len(p) / h
        se = _auc_stderr(auc, float(y.sum()) / h, float((1 - y).sum()) / h)
        if np.isfinite(auc) and np.isfinite(se) and se > 0:
            z = (auc - 0.5) / se
            from scipy.stats import norm
            pval = float(2.0 * norm.sf(abs(z)))
        else:
            z, pval = float("nan"), float("nan")

        row = {
            "ticker": ticker,
            "horizon": horizon_key,
            "horizon_bars": h,
            "status": "ok",
            "n": int(len(p)),
            "n_eff": float(n_eff),
            "base_rate": float(y.mean()),
            "accuracy": float(np.mean((p >= 0.5) == (y > 0))),
            "auc": float(auc),
            "auc_se": float(se),
            "auc_z": float(z),
            "p_value": pval,
            "start": str(pd.DatetimeIndex(np.concatenate(dates))[0].date()),
            "end": str(pd.DatetimeIndex(np.concatenate(dates))[-1].date()),
            **_strategy_stats(p, fwd, h),
        }
        if keep_predictions:
            row["_p"], row["_y"] = p, y
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Placebo arm
# ---------------------------------------------------------------------------
def synthetic_frame(seed: int, n_bars: int = 3000, ann_vol: float = 0.35,
                    drift: float = 0.08) -> pd.DataFrame:
    """A random walk dressed as an OHLCV bar series.

    There is by construction no predictable structure here, so any 'edge' the
    screen reports on these is a false positive. Running a few hundred of them
    is how we measure the screen's own error rate instead of asserting it.
    """
    rng = np.random.default_rng(seed)
    sigma = ann_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift / TRADING_DAYS_PER_YEAR - 0.5 * sigma**2
    rets = rng.normal(mu, sigma, n_bars)
    close = 40.0 * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0.0, sigma * 0.7, (n_bars, 2)))
    frame = pd.DataFrame({
        "open": close * np.exp(rng.normal(0.0, sigma * 0.3, n_bars)),
        "high": close * np.exp(intraday[:, 0]),
        "low": close * np.exp(-intraday[:, 1]),
        "close": close,
        "volume": rng.lognormal(13.0, 0.6, n_bars),
    }, index=pd.date_range("2014-01-02", periods=n_bars, freq="B", tz="UTC"))
    frame["is_partial"] = False
    return frame


def correlated_universe(seed: int, n_names: int, rho: float = 0.278,
                        n_bars: int = 1512 + 800, ann_vol: float = 0.40,
                        drift: float = 0.08) -> list[pd.DataFrame]:
    """A cohort of random walks sharing one market factor.

    The plain placebo arm answers "what does the screen do on *independent*
    noise?". It cannot answer the question that matters for any statistic
    aggregated across the universe -- such as the mean AUC -- because the 532
    real small caps are not independent: their daily returns correlate about
    0.28 on average, so they move together and their AUC estimates move together
    too. Under that dependence the variance of a cross-sectional mean does not
    shrink like ``1/N``; it converges to ``rho * sigma^2``, which for this
    universe is roughly seventeen times larger than the independent formula
    says. Treating 1,064 correlated tests as 1,064 independent ones is exactly
    how a 0.503 mean AUC gets mistaken for a market-wide edge.

    Each cohort here is one draw from that null: a common factor plus
    idiosyncratic noise, calibrated to the measured correlation and volatility.
    """
    rng = np.random.default_rng(10_000 + seed)
    sigma = ann_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift / TRADING_DAYS_PER_YEAR - 0.5 * sigma**2

    factor = rng.normal(0.0, 1.0, n_bars)
    idio = rng.normal(0.0, 1.0, (n_names, n_bars))
    shocks = np.sqrt(rho) * factor[None, :] + np.sqrt(1.0 - rho) * idio

    index = pd.date_range("2012-01-02", periods=n_bars, freq="B", tz="UTC")
    frames = []
    for i in range(n_names):
        rets = mu + sigma * shocks[i]
        close = 40.0 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0.0, sigma * 0.7, (n_bars, 2)))
        frame = pd.DataFrame({
            "open": close * np.exp(rng.normal(0.0, sigma * 0.3, n_bars)),
            "high": close * np.exp(intraday[:, 0]),
            "low": close * np.exp(-intraday[:, 1]),
            "close": close,
            "volume": rng.lognormal(13.0, 0.6, n_bars),
        }, index=index)
        frame["is_partial"] = False
        frames.append(frame)
    return frames
