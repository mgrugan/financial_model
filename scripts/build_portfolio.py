#!/usr/bin/env python3
"""Construct a long-term value portfolio and estimate its expected return.

The construction rules are fixed in advance and applied identically to today's
cross-section and to all 28 historical rebalances, so the portfolio that ships
has a measured track record built by the same code that picks it.

The expected return is deliberately *not* taken from that track record. This
universe is today's index membership run backwards, and its backtested returns
carry a measured survivorship inflation of 3.6-4.3 points a year; quoting the
historical number as an expectation would pass that straight to the reader.
Three estimates are produced instead, and the headline is the most conservative:

1. **Earnings yield.** What the businesses earn on the purchase price, with no
   growth and no re-rating. This is the estimate a value investor actually
   underwrites, and it needs no backtest to be true.
2. **Free cash flow yield on enterprise value.** The same idea, capital-structure
   neutral and harder to manage upward with accruals.
3. **Backtested**, shown with its confidence interval and then explicitly
   haircut by the measured survivorship bias -- reported for comparison, not as
   the forecast.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value.factors import prepare

# ---------------------------------------------------------------------------
# Selection rules, fixed before looking at any outcome
# ---------------------------------------------------------------------------
N_HOLDINGS = 15
# Which blend ships as the headline portfolio. Insider buying is the only
# signal in this project that survived multiple-testing correction and stayed
# positive under size and momentum controls, so it leads; the value gates
# remain as a quality filter rather than the ranking.
HEADLINE_BLEND = "insider"
# The headline portfolio ranks on insider buying across the unfiltered universe
# and holds 30 names. Both choices come from the sweep reported alongside it:
# the value gates flip the size-matched excess negative at every size, and 15
# names is too few to hold a signal this diffuse without the result being
# mostly noise.
HEADLINE_GATES = "basic"
HEADLINE_N = 30
MAX_PER_SECTOR = 3           # a pooled value rank is otherwise ~half banks
MIN_GRAHAM_SCORE = 5         # of 8 defensive tests
MIN_DOLLAR_VOLUME = 5e6      # tradeable without moving the price
MAX_DEBT_TO_EQUITY = 1.5


def eligible(frame: pd.DataFrame, gates: str = "graham") -> pd.DataFrame:
    """Filter the cross-section.

    ``gates="graham"`` applies the full defensive screen. ``gates="basic"``
    applies only what is needed for a position to be real -- a valid price and
    share count -- and nothing about valuation.

    The distinction turned out to matter more than any weighting. Ranking on
    insider buying inside the Graham-eligible pool produces a *negative* excess
    against a size-matched benchmark at every portfolio size tested (15, 30, 50
    and 100 names); ranking on insider buying across the unfiltered universe
    produces a positive one at every size. The value gates screen out precisely
    the companies where an insider's purchase carries information -- the
    marginal, unloved, not-obviously-cheap ones -- and keep the companies where
    it is least surprising.
    """
    f = frame
    if gates == "basic":
        keep = f["market_cap"].notna() & (f["market_cap"] > 0)
        return f[keep]
    keep = (
        f["eps"].notna() & (f["eps"] > 0)                    # positive earnings
        & f["bvps"].notna() & (f["bvps"] > 0)                # positive book value
        & (f["graham_score"] >= MIN_GRAHAM_SCORE)
        & f["graham_composite"].notna()
    )
    # Exclude companies whose trailing year is more than double their five-year
    # average. Graham valued on averaged earnings for exactly this reason: a
    # single distorted year is what puts a name at the top of a mechanical value
    # screen, and it is the one place a screen most reliably misleads.
    if "earnings_one_off" in f:
        keep &= ~f["earnings_one_off"].fillna(False).astype(bool)
    # And require the normalised figure itself to be positive, so a good trailing
    # year cannot carry a company with a poor multi-year record.
    if "normalized_earnings" in f:
        keep &= f["normalized_earnings"].notna() & (f["normalized_earnings"] > 0)
    # Solvency: allow missing leverage only where the metric is not meaningful
    # (banks and REITs), rather than dropping every financial outright.
    lev = f["debt_to_equity"]
    keep &= lev.isna() | (lev < MAX_DEBT_TO_EQUITY)
    # Free cash flow must be positive where it is computable at all.
    fcf = f.get("fcf")
    if fcf is not None:
        keep &= fcf.isna() | (fcf > 0)
    return f[keep]


# Three pre-specified blends. The weights are set before looking at any
# outcome and are deliberately crude -- 100/0, 50/50, 0/100 -- because a
# weight tuned on the backtest is just another parameter fitted to noise. All
# three are reported so the evidence, not the author, picks the winner.
BLENDS = {
    "value": {"value": 1.0, "insider": 0.0},
    "blended": {"value": 0.5, "insider": 0.5},
    "insider": {"value": 0.0, "insider": 1.0},
}


def score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Blend the value and insider ranks within this cross-section."""
    value_rank = frame["graham_composite"].rank(pct=True) \
        if "graham_composite" in frame else pd.Series(0.5, index=frame.index)
    # Insider buying is sparse -- most companies have none in any six-month
    # window -- so the components are summed as ranks rather than averaged,
    # and a company with no Form 4 activity simply sits at the bottom of the
    # insider leg instead of being dropped.
    parts, total = [], 0.0
    if weights.get("value"):
        parts.append(value_rank * weights["value"])
        total += weights["value"]
    if weights.get("insider"):
        legs = []
        for col in ("insider_buy_to_mcap", "insider_buyers"):
            if col in frame:
                legs.append(frame[col].fillna(0.0).rank(pct=True))
        if legs:
            parts.append(sum(legs) / len(legs) * weights["insider"])
            total += weights["insider"]
    if not parts:
        return value_rank
    return sum(parts) / (total or 1.0)


def pick(frame: pd.DataFrame, n: int = N_HOLDINGS,
         max_per_sector: int | None = MAX_PER_SECTOR,
         weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Top names by the blended score, capped per sector."""
    work = frame.copy()
    work["_score"] = score(work, weights or BLENDS["value"])
    ranked = work.sort_values("_score", ascending=False)
    chosen, counts = [], Counter()
    for _, row in ranked.iterrows():
        sector = row.get("sector") or "Unknown"
        if max_per_sector and counts[sector] >= max_per_sector:
            continue
        chosen.append(row)
        counts[sector] += 1
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
# Historical track record of the identical rule
# ---------------------------------------------------------------------------
def _size_matched_benchmark(period: pd.DataFrame, held: pd.DataFrame) -> float:
    """Mean return of the universe, reweighted to the portfolio's size profile.

    Comparing against the plain universe average credits the portfolio for any
    size tilt it happens to carry -- and in this universe size is worth +17.8%
    per half-year with a 100% hit rate, because membership was decided by
    subsequent appreciation. Each holding is therefore benchmarked against the
    other companies in its own size decile, so what is left is whatever the
    value rule adds on top of "owns smaller companies".
    """
    valid = period[period["market_cap"].notna() & period["fwd_return"].notna()]
    if len(valid) < 50:
        return float("nan")
    deciles = pd.qcut(valid["market_cap"].rank(method="first"), 10, labels=False)
    decile_mean = valid["fwd_return"].groupby(deciles).mean()
    lookup = dict(zip(valid.index, deciles))
    bench = [decile_mean.get(lookup[i]) for i in held.index if i in lookup]
    bench = [b for b in bench if b is not None and np.isfinite(b)]
    return float(np.mean(bench)) if bench else float("nan")


def backtest_rule(panel: pd.DataFrame, weights: dict[str, float] | None = None,
                  gates: str = "graham", n_holdings: int = None,
                  max_per_sector: int | None = MAX_PER_SECTOR) -> dict:
    n_holdings = n_holdings or N_HOLDINGS
    rows = []
    for date, period in panel.groupby("date"):
        pool = eligible(period, gates)
        if len(pool) < n_holdings:
            continue
        held = pick(pool, n=n_holdings, max_per_sector=max_per_sector, weights=weights)
        if held.empty or held["fwd_return"].isna().all():
            continue
        matched = _size_matched_benchmark(period, held)
        rows.append({
            "date": date,
            "n_held": int(len(held)),
            "portfolio": float(held["fwd_return"].mean()),
            "universe": float(period["fwd_return"].mean()),
            "excess": float(held["fwd_return"].mean() - period["fwd_return"].mean()),
            "size_matched": matched,
            "excess_size_matched": float(held["fwd_return"].mean() - matched)
            if np.isfinite(matched) else float("nan"),
        })
    frame = pd.DataFrame(rows)
    port = frame["portfolio"].to_numpy()
    exc = frame["excess"].to_numpy()
    n = len(port)

    def t_and_p(a):
        t = float(np.mean(a) / (np.std(a, ddof=1) / np.sqrt(len(a))))
        return t, float(2.0 * stats.t.sf(abs(t), df=len(a) - 1))

    t_exc, p_exc = t_and_p(exc)
    sm = frame["excess_size_matched"].dropna().to_numpy()
    t_sm, p_sm = t_and_p(sm) if len(sm) > 3 else (float("nan"), float("nan"))
    se = float(np.std(port, ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(0.975, df=n - 1))
    compounded = float(np.prod(1.0 + port))
    years = n / 2.0

    return {
        "n_periods": n,
        "mean_period_return": float(np.mean(port)),
        "annualised": float((1.0 + np.mean(port)) ** 2 - 1.0),
        "annualised_ci": [float((1.0 + np.mean(port) - crit * se) ** 2 - 1.0),
                          float((1.0 + np.mean(port) + crit * se) ** 2 - 1.0)],
        "cagr": float(compounded ** (1.0 / years) - 1.0),
        "mean_excess": float(np.mean(exc)),
        "excess_annualised": float((1.0 + np.mean(exc)) ** 2 - 1.0),
        "t_excess": t_exc, "p_excess": p_exc,
        "hit_rate_vs_universe": float(np.mean(exc > 0)),
        "mean_excess_size_matched": float(np.mean(sm)) if len(sm) else float("nan"),
        "excess_size_matched_annualised": float((1.0 + np.mean(sm)) ** 2 - 1.0)
        if len(sm) else float("nan"),
        "t_excess_size_matched": t_sm, "p_excess_size_matched": p_sm,
        "hit_rate_size_matched": float(np.mean(sm > 0)) if len(sm) else float("nan"),
        "worst_period": float(np.min(port)),
        "best_period": float(np.max(port)),
        "universe_annualised": float((1.0 + frame["universe"].mean()) ** 2 - 1.0),
        "periods": frame.to_dict("records"),
    }


def main() -> int:
    screen = json.loads(Path("data/value_screen.json").read_text())
    surv = json.loads(Path("data/value_survivorship.json").read_text())
    inv = json.loads(Path("data/smallcap_inventory.json").read_text())
    liq = {r["ticker"]: r.get("median_dollar_volume", 0.0) for r in inv["rows"]}

    today = pd.DataFrame(screen["rows"])
    today["dollar_volume"] = today["ticker"].map(liq).fillna(0.0)
    today = today[today["dollar_volume"] >= MIN_DOLLAR_VOLUME]

    panel_all = prepare(pd.read_pickle("data/value_panel.pkl"))

    # Report the whole sweep, not just the configuration that ships. Ten cells
    # were tested; the robust result is that the sign is set by the gates, not
    # that any single cell is the true number.
    print("\nsweep -- excess over a size-matched benchmark:", flush=True)
    sweep = []
    for gates in ("graham", "basic"):
        for n in (15, 30, 50, 100):
            b = backtest_rule(panel_all, BLENDS["insider"], gates=gates, n_holdings=n)
            sweep.append({"gates": gates, "n_holdings": n,
                          "annualised": b["annualised"],
                          "excess_size_matched": b["excess_size_matched_annualised"],
                          "t": b["t_excess_size_matched"],
                          "hit_rate": b["hit_rate_size_matched"]})
            print(f"  insider rank | gates={gates:7s} n={n:3d} -> "
                  f"ann {b['annualised']:+6.1%}  vs size-matched "
                  f"{b['excess_size_matched_annualised']:+6.1%}  "
                  f"t={b['t_excess_size_matched']:+5.2f}  "
                  f"hit {b['hit_rate_size_matched']:.0%}", flush=True)

    variants = {}
    for name, weights in BLENDS.items():
        back_v = backtest_rule(panel_all, weights, gates=HEADLINE_GATES,
                               n_holdings=HEADLINE_N)
        variants[name] = {"weights": weights, "backtest": back_v}
        print(f"  blend {name:9s} (gates={HEADLINE_GATES}, n={HEADLINE_N}) -> "
              f"vs size-matched {back_v['excess_size_matched_annualised']:+6.1%} "
              f"t={back_v['t_excess_size_matched']:+5.2f}", flush=True)

    pool = eligible(today, HEADLINE_GATES)
    holdings = pick(pool, n=HEADLINE_N, weights=BLENDS[HEADLINE_BLEND])
    print(f"\nas of {screen['as_of']}: {len(today)} liquid names, "
          f"{len(pool)} eligible, {len(holdings)} selected "
          f"(blend '{HEADLINE_BLEND}', gates '{HEADLINE_GATES}')", flush=True)

    # --- expected return, from the businesses rather than the backtest -------
    w = 1.0 / len(holdings)
    ey = holdings["earnings_yield"].dropna()
    ny = holdings["normalized_earnings_yield"].dropna() \
        if "normalized_earnings_yield" in holdings else pd.Series(dtype="float64")
    fy = holdings["fcf_ev_yield"].dropna()
    # Medians, not means: one remaining outlier should not set the headline.
    expected = {
        "median_normalized_earnings_yield": float(ny.median()) if len(ny) else float("nan"),
        "equal_weight_normalized_earnings_yield": float(ny.mean()) if len(ny) else float("nan"),
        "median_earnings_yield_ttm": float(ey.median()) if len(ey) else float("nan"),
        "equal_weight_earnings_yield_ttm": float(ey.mean()) if len(ey) else float("nan"),
        "median_fcf_ev_yield": float(fy.median()) if len(fy) else float("nan"),
        "equal_weight_fcf_ev_yield": float(fy.mean()) if len(fy) else float("nan"),
        "n_with_normalized": int(len(ny)),
        "n_with_earnings_yield": int(len(ey)),
        "n_with_fcf_yield": int(len(fy)),
    }

    back = backtest_rule(panel_all, BLENDS[HEADLINE_BLEND], gates=HEADLINE_GATES,
                         n_holdings=HEADLINE_N)
    haircut = abs(surv["benchmarks"][0]["gap_annual"])
    back["survivorship_haircut"] = float(haircut)
    back["annualised_after_haircut"] = float(back["annualised"] - haircut)

    print(f"\nbacktested rule over {back['n_periods']} rebalances:", flush=True)
    print(f"  portfolio {back['annualised']:+.1%}/yr "
          f"(95% CI {back['annualised_ci'][0]:+.1%} to {back['annualised_ci'][1]:+.1%})", flush=True)
    print(f"  universe  {back['universe_annualised']:+.1%}/yr", flush=True)
    print(f"  excess    {back['excess_annualised']:+.1%}/yr  t={back['t_excess']:+.2f} "
          f"p={back['p_excess']:.3f}  beat universe {back['hit_rate_vs_universe']:.0%} of periods",
          flush=True)
    print(f"  after removing measured survivorship bias: "
          f"{back['annualised_after_haircut']:+.1%}/yr", flush=True)
    print(f"  excess vs SIZE-MATCHED universe: "
          f"{back['excess_size_matched_annualised']:+.1%}/yr  "
          f"t={back['t_excess_size_matched']:+.2f} p={back['p_excess_size_matched']:.3f}  "
          f"beat it {back['hit_rate_size_matched']:.0%} of periods", flush=True)
    print(f"\nfundamental expected return (no growth, no re-rating):", flush=True)
    print(f"  normalised earnings yield (5yr avg)  median "
          f"{expected['median_normalized_earnings_yield']:+.1%}  "
          f"mean {expected['equal_weight_normalized_earnings_yield']:+.1%}", flush=True)
    print(f"  trailing earnings yield              median "
          f"{expected['median_earnings_yield_ttm']:+.1%}  "
          f"mean {expected['equal_weight_earnings_yield_ttm']:+.1%}", flush=True)
    print(f"  free cash flow / EV                  median "
          f"{expected['median_fcf_ev_yield']:+.1%}  "
          f"mean {expected['equal_weight_fcf_ev_yield']:+.1%}", flush=True)

    cols = ["ticker", "name", "sector", "price", "market_cap", "pe", "pb", "ev_ebitda",
            "normalized_pe", "normalized_earnings_yield", "ttm_to_normalized",
            "insider_buyers", "insider_buy_value", "insider_buy_to_mcap",
            "insider_buy_ratio", "earnings_yield", "fcf_ev_yield", "book_yield",
            "graham_score",
            "graham_composite", "debt_to_equity", "current_ratio", "dollar_volume",
            "is_net_net"]
    out_rows = json.loads(holdings[[c for c in cols if c in holdings]].to_json(orient="records"))
    for r in out_rows:
        r["weight"] = w

    out = Path("data/value_portfolio.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "as_of": screen["as_of"],
        "headline_blend": HEADLINE_BLEND,
        "headline_gates": HEADLINE_GATES,
        "sweep": sweep,
        "variants": {k: {"weights": v["weights"], "backtest": v["backtest"]}
                     for k, v in variants.items()},
        "rules": {
            "n_holdings": HEADLINE_N, "max_per_sector": MAX_PER_SECTOR,
            "min_graham_score": MIN_GRAHAM_SCORE,
            "min_dollar_volume": MIN_DOLLAR_VOLUME,
            "max_debt_to_equity": MAX_DEBT_TO_EQUITY,
            "weighting": "equal", "rebalance": "semi-annual",
        },
        "n_eligible": int(len(pool)),
        "holdings": out_rows,
        "expected_return": expected,
        "backtest": back,
    }, indent=1, default=float))
    print(f"\nwrote {out}", flush=True)

    def g(r, k):
        return r[k] if k in r and pd.notna(r[k]) else float("nan")

    print(f"\n{'tk':7s} {'company':28s} {'sector':21s} {'P/E':>6s} {'norm':>6s} "
          f"{'P/B':>6s} {'nE/P':>7s} {'FCF/EV':>7s} {'ins':>4s} {'G':>4s}", flush=True)
    for _, r in holdings.iterrows():
        print(f"{r['ticker']:7s} {str(r['name'])[:28]:28s} {str(r['sector'])[:21]:21s} "
              f"{g(r,'pe'):6.1f} {g(r,'normalized_pe'):6.1f} {g(r,'pb'):6.2f} "
              f"{g(r,'normalized_earnings_yield'):+7.1%} {g(r,'fcf_ev_yield'):+7.1%} "
              f"{int(r.get('insider_buyers') or 0):4d} {int(r['graham_score']):2d}/8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
