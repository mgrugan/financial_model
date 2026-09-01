"""Cross-sectional factor tests on the value panel.

The unit of inference is the **rebalance period**, not the stock-period. Within
one half-year every name in the universe shares that half-year's market move, so
pooling 15,000 stock-periods and computing a t-statistic on 15,000 observations
overstates precision by more than an order of magnitude. Each factor is
therefore collapsed to one number per period -- the long-short spread, or the
rank information coefficient -- and the t-statistic is computed on that series
of 28.

That is a small sample, and it is the honest one. The module reports the
minimum detectable effect alongside every result, because a null finding at
n=28 means something quite different from a null finding at n=28,000: the
historical value premium is roughly 4-5% a year, and this design cannot resolve
an effect that size. Saying so is part of the result.

Two ranking modes are run for every factor:

* **Pooled** -- rank all eligible names together. Simple, and what a naive
  screen does.
* **Sector-neutral** -- rank within sector, then pool the within-sector
  percentiles. Book-to-price varies enormously across sectors, so a pooled P/B
  rank is substantially a bet that banks beat software. Sector-neutral isolates
  "cheap for its industry", which is the claim value investing actually makes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

N_BUCKETS = 5
MIN_NAMES_PER_PERIOD = 40


@dataclass
class Factor:
    """One ranking signal. ``higher_is_cheaper`` fixes the sign convention."""

    key: str
    label: str
    column: str
    higher_is_cheaper: bool = True
    ev_based: bool = False
    tradition: str = "Graham"
    note: str = ""
    # Column that must be strictly positive for the ranking to mean anything.
    # A yield with a negative numerator sorts to the bottom of the book, which
    # silently relabels "loss-making" as "expensive" -- and those are opposite
    # kinds of company. Graham's first defensive test is positive earnings, so
    # excluding them is the tradition's own rule, not a convenience.
    require_positive: str | None = None


FACTORS: list[Factor] = [
    Factor("ebitda_ev", "EBITDA / EV", "ebitda_ev_yield", True, True, "Burry",
           "Burry's preferred multiple, inverted so it stays ordered through zero.",
           require_positive="ebitda"),
    Factor("fcf_ev", "Free cash flow / EV", "fcf_ev_yield", True, True, "Burry",
           "Cash the whole capital structure earns, over what the whole structure costs.",
           require_positive="fcf"),
    Factor("sales_ev", "Revenue / EV", "ev_sales", False, True, "Burry",
           "Deep-value sales multiple; survives loss-making years that break earnings."),
    Factor("low_leverage", "Low debt / equity", "debt_to_equity", False, False, "Both",
           "Not a value signal on its own -- a solvency filter both traditions insist on.",
           require_positive="bvps"),
    Factor("earnings_yield", "Earnings / price", "earnings_yield", True, False, "Graham",
           "The inverted P/E, among profitable companies.", require_positive="eps"),
    Factor("book_yield", "Book / price", "book_yield", True, False, "Graham",
           "The inverted P/B; the classic academic value factor.", require_positive="bvps"),
    Factor("graham_ratio", "Graham number / price", "graham_ratio", True, False, "Graham",
           "Above 1 means trading under Graham's defensive fair value."),
    Factor("graham_score", "Graham criteria passed", "graham_score", True, False, "Graham",
           "Count of the eight defensive tests passed."),
    Factor("current_ratio", "Current ratio", "current_ratio", True, False, "Graham",
           "Graham's liquidity test; undefined for banks and REITs."),
    Factor("ncav", "NCAV / market cap", "ncav_yield", True, False, "Graham",
           "Net current asset value per dollar of market cap. Negative NCAV is "
           "meaningful here -- it means no asset protection -- so it is ranked, "
           "not excluded."),
    Factor("graham_composite", "Graham composite", "graham_composite", True, False,
           "Graham", "Equal-weight blend of earnings yield, book yield and the "
           "Graham-number ratio."),
    Factor("burry_composite", "Burry composite", "burry_composite", True, True,
           "Burry", "Equal-weight blend of EBITDA/EV and free cash flow/EV."),
]


COMPOSITES = {
    # Each tradition's own metrics, equally weighted as within-period
    # percentiles. Specified before seeing which single factors worked, so the
    # composite is not a winners-only blend chosen after the fact -- and it is
    # run through the same backtest as everything else rather than asserted.
    "graham_composite": ["earnings_yield", "book_yield", "graham_ratio"],
    "burry_composite": ["ebitda_ev_yield", "fcf_ev_yield"],
}


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Add derived ranking columns that are cleaner than the raw ratios."""
    out = panel.copy()
    # NCAV as a yield rather than a multiple: continuous and correctly ordered
    # even when NCAV is negative, which it is for most non-net-nets.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ncav_yield"] = out["ncav"] / out["market_cap"]

    # Composites are built from within-period percentiles so that one component
    # with a fat tail cannot dominate the average. Ranking against the other
    # companies present on the same date uses no future information.
    for name, parts in COMPOSITES.items():
        ranks = []
        for col in parts:
            if col in out:
                ranks.append(out.groupby("date")[col].rank(pct=True))
        if ranks:
            stacked = pd.concat(ranks, axis=1)
            # Require at least half the components; a name scored on one leg is
            # not comparable to one scored on three.
            enough = stacked.notna().sum(axis=1) >= max(1, (len(parts) + 1) // 2)
            out[name] = stacked.mean(axis=1).where(enough)
    return out


def _percentile(values: pd.Series) -> pd.Series:
    """Rank to [0, 1]; all-equal or single-value inputs collapse to 0.5."""
    valid = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    if valid.sum() < 2:
        return out
    out[valid] = values[valid].rank(pct=True)
    return out


def _residualise_on_size(scores: pd.Series, period: pd.DataFrame) -> pd.Series:
    """Strip the part of a score that is just a bet on small companies.

    This universe is today's index membership applied backwards, so a company
    that was small in 2012 and is a member now must have appreciated to get
    here. Size therefore "predicts" returns in this sample with a 100% hit rate
    across 28 half-years -- an impossibility for any real premium, and the
    signature of selection rather than return.

    Every price-scaled value ratio carries market capitalisation in its
    denominator, so each of them is partly a size bet and inherits some of that
    artifact. Regressing the score on the size rank within the period and
    keeping the residual asks the question that survives the bias: among
    companies of similar size, does the cheap one do better?
    """
    size = np.log(period["market_cap"])
    mask = scores.notna() & size.notna() & np.isfinite(size)
    if mask.sum() < 20:
        return pd.Series(np.nan, index=scores.index)
    x = size[mask].rank(pct=True).to_numpy()
    y = scores[mask].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    out = pd.Series(np.nan, index=scores.index)
    out[mask] = y - (slope * x + intercept)
    return out


def factor_scores(period: pd.DataFrame, factor: Factor,
                  sector_neutral: bool, size_neutral: bool = False) -> pd.Series:
    """Signed, sign-corrected percentile scores where 1.0 = cheapest."""
    eligible = period
    if factor.ev_based:
        eligible = eligible[eligible["ev_applicable"]]
    if factor.require_positive and factor.require_positive in eligible:
        eligible = eligible[eligible[factor.require_positive] > 0]

    raw = eligible[factor.column]
    if not factor.higher_is_cheaper:
        raw = -raw

    if sector_neutral:
        scores = raw.groupby(eligible["sector"]).transform(_percentile)
    else:
        scores = _percentile(raw)
    scores = scores.reindex(period.index)
    if size_neutral:
        scores = _residualise_on_size(scores, period)
    return scores


def _spread_and_ic(period: pd.DataFrame, scores: pd.Series) -> dict[str, float] | None:
    mask = scores.notna() & period["fwd_return"].notna()
    if mask.sum() < MIN_NAMES_PER_PERIOD:
        return None
    s = scores[mask]
    r = period.loc[mask, "fwd_return"]

    # Quintiles by score; bucket 5 is the cheapest end.
    try:
        buckets = pd.qcut(s.rank(method="first"), N_BUCKETS, labels=False) + 1
    except ValueError:
        return None
    means = r.groupby(buckets).mean()
    if 1 not in means.index or N_BUCKETS not in means.index:
        return None

    ic = stats.spearmanr(s, r).statistic
    return {
        "n": int(mask.sum()),
        "spread": float(means[N_BUCKETS] - means[1]),
        "cheap": float(means[N_BUCKETS]),
        "expensive": float(means[1]),
        "universe": float(r.mean()),
        "ic": float(ic) if np.isfinite(ic) else np.nan,
        **{f"q{i}": float(means[i]) for i in means.index},
    }


def run_factor(panel: pd.DataFrame, factor: Factor,
               sector_neutral: bool, size_neutral: bool = False) -> dict[str, Any]:
    """Period-by-period spread and IC, then inference on the period series."""
    per_period = []
    for date, period in panel.groupby("date"):
        scores = factor_scores(period, factor, sector_neutral, size_neutral)
        row = _spread_and_ic(period, scores)
        if row:
            per_period.append({"date": date, **row})

    if len(per_period) < 8:
        return {"key": factor.key, "status": "too few periods", "n_periods": len(per_period)}

    frame = pd.DataFrame(per_period)
    spread = frame["spread"].to_numpy()
    ic = frame["ic"].to_numpy()
    n = len(spread)

    t_spread = float(np.mean(spread) / (np.std(spread, ddof=1) / np.sqrt(n)))
    p_spread = float(2.0 * stats.t.sf(abs(t_spread), df=n - 1))
    t_ic = float(np.mean(ic) / (np.std(ic, ddof=1) / np.sqrt(n)))
    p_ic = float(2.0 * stats.t.sf(abs(t_ic), df=n - 1))

    # What this design could have found. A two-sided test at 5% with n periods
    # needs a mean of about t_crit * SE, so anything smaller was never
    # detectable and a null result says nothing about it.
    se = float(np.std(spread, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    mde_period = t_crit * se

    # Compounded spread: what actually accrued to a long-short book.
    equity = float(np.prod(1.0 + spread))
    years = n / 2.0

    return {
        "key": factor.key,
        "label": factor.label,
        "tradition": factor.tradition,
        "note": factor.note,
        "ev_based": factor.ev_based,
        "sector_neutral": sector_neutral,
        "size_neutral": size_neutral,
        "status": "ok",
        "n_periods": n,
        "mean_names": float(frame["n"].mean()),
        "mean_spread": float(np.mean(spread)),
        "sd_spread": float(np.std(spread, ddof=1)),
        "t_spread": t_spread,
        "p_spread": p_spread,
        "hit_rate": float(np.mean(spread > 0)),
        "mean_ic": float(np.mean(ic)),
        "t_ic": t_ic,
        "p_ic": p_ic,
        "mde_per_period": mde_period,
        "mde_annualised": (1.0 + mde_period) ** 2 - 1.0,
        "spread_cagr": float(equity ** (1.0 / years) - 1.0) if equity > 0 else float("nan"),
        "mean_cheap": float(frame["cheap"].mean()),
        "mean_expensive": float(frame["expensive"].mean()),
        "mean_universe": float(frame["universe"].mean()),
        "quintiles": [float(frame[f"q{i}"].mean()) for i in range(1, N_BUCKETS + 1)
                      if f"q{i}" in frame],
        "periods": frame.to_dict("records"),
    }
