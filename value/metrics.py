"""Graham and Burry valuation metrics from a point-in-time snapshot.

Two traditions, deliberately kept separate rather than blended into one score.

**Graham** is a balance-sheet-first, rules-based discipline: pay less than the
assets are worth, demand a margin of safety, and refuse anything without a
record of positive earnings and a survivable capital structure. His criteria are
absolute thresholds, so they are implemented as pass/fail tests and counted.

**Burry** works from enterprise value. His stated preference is EV/EBITDA over
P/E because it is indifferent to capital structure and to the tax and
depreciation choices that make P/E incomparable across companies -- a leveraged
company and an unleveraged one with identical operations have very different
P/Es and nearly identical EV/EBITDAs. His metrics are relative, so they are
implemented as continuous values to be ranked cross-sectionally.

Sign conventions matter here and are easy to get wrong. A negative EBITDA makes
EV/EBITDA negative, which would sort as "extremely cheap" in a naive ranking
when it actually means the company is burning money. Every ratio with a
signable denominator returns None rather than a negative number, and the
screens treat None as failing rather than as missing-at-random.
"""

from __future__ import annotations

import math
from typing import Any


# Plausibility band for a listed small/mid-cap. Used only to reject impossible
# share counts, never to filter on valuation.
MIN_SHARES = 1_000_000
MIN_MARKET_CAP = 20e6
MAX_MARKET_CAP = 200e9


def _get(snap: dict[str, float], key: str) -> float | None:
    v = snap.get(key)
    if v is None or not math.isfinite(v):
        return None
    return float(v)


def _ratio(num: float | None, den: float | None,
           require_positive_den: bool = True) -> float | None:
    if num is None or den is None:
        return None
    if den == 0 or (require_positive_den and den <= 0):
        return None
    val = num / den
    return val if math.isfinite(val) else None


# ---------------------------------------------------------------------------
# Derived aggregates
# ---------------------------------------------------------------------------
def total_debt(snap: dict[str, float]) -> float | None:
    long_ = _get(snap, "debt_long")
    short = _get(snap, "debt_short")
    if long_ is None and short is None:
        return None
    return (long_ or 0.0) + (short or 0.0)


def net_cash(snap: dict[str, float]) -> float:
    return (_get(snap, "cash") or 0.0) + (_get(snap, "short_term_investments") or 0.0)


def ebitda(snap: dict[str, float]) -> float | None:
    """Operating income plus depreciation and amortisation.

    Not reconstructed from net income: that route picks up interest, tax and
    every one-off below the operating line, which is the whole thing EV/EBITDA
    exists to strip out.
    """
    op = _get(snap, "operating_income")
    if op is None:
        return None
    return op + (_get(snap, "dep_amort") or 0.0)


def free_cash_flow(snap: dict[str, float]) -> float | None:
    ocf = _get(snap, "operating_cash_flow")
    if ocf is None:
        return None
    # capex is reported as a positive outflow in the investing section.
    return ocf - abs(_get(snap, "capex") or 0.0)


def enterprise_value(snap: dict[str, float], market_cap: float) -> float | None:
    debt = total_debt(snap)
    if debt is None or not math.isfinite(market_cap) or market_cap <= 0:
        return None
    ev = market_cap + debt - net_cash(snap)
    # A negative EV means the company holds more net cash than its whole market
    # value. That is a real and interesting situation, but it breaks every
    # EV multiple, so it is flagged rather than ranked.
    return ev if ev > 0 else None


# ---------------------------------------------------------------------------
# The metric set
# ---------------------------------------------------------------------------
def compute(snap: dict[str, float], price: float, shares: float | None) -> dict[str, Any]:
    """Every metric for one company at one point in time.

    ``price`` must be the **unadjusted** close, and ``shares`` the share count
    the company itself reported by that date, so their product is the market
    capitalisation an investor would actually have observed.
    """
    out: dict[str, Any] = {}
    if shares is None or not math.isfinite(shares) or shares <= 0:
        return out
    if not math.isfinite(price) or price <= 0:
        return out

    # Scale sanity. Some filers tag share counts in thousands, and where the
    # cover-page count is missing entirely the fallback is a weighted-average
    # figure that can carry the same defect: Solaris Energy reports 25,829
    # diluted shares against $902m of equity, implying a book value of $34,900 a
    # share and a price-to-book of 0.00. Left unguarded that row does not merely
    # add noise -- it sorts straight to the top of a value screen, because every
    # cheapness ratio is understated by the same three orders of magnitude.
    # A listed S&P 600 member has neither fewer than a million shares nor a
    # market capitalisation outside this band, so a violation means the share
    # count is wrong and the company is dropped rather than guessed at.
    if shares < MIN_SHARES:
        return out
    market_cap = price * shares
    if not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
        return out
    out["market_cap"] = market_cap
    out["shares"] = shares
    out["price"] = price

    equity = _get(snap, "equity")
    net_income = _get(snap, "net_income")
    revenue = _get(snap, "revenue")
    ebit = _get(snap, "operating_income")
    ebitda_v = ebitda(snap)
    fcf = free_cash_flow(snap)
    debt = total_debt(snap)
    ev = enterprise_value(snap, market_cap)

    out["enterprise_value"] = ev
    out["ebitda"] = ebitda_v
    out["total_debt"] = debt
    out["net_cash"] = net_cash(snap)

    # ---- Burry: enterprise-value multiples --------------------------------
    out["ev_ebitda"] = _ratio(ev, ebitda_v)
    out["ev_ebit"] = _ratio(ev, ebit)
    out["ev_sales"] = _ratio(ev, revenue)
    # Yields are inverted multiples. They are the better ranking variable: a
    # yield stays continuous and correctly ordered through zero earnings, where
    # a multiple blows up to infinity and then reappears as a large negative.
    out["ebitda_ev_yield"] = _ratio(ebitda_v, ev, require_positive_den=True)
    out["fcf_ev_yield"] = _ratio(fcf, ev, require_positive_den=True)
    out["fcf"] = fcf

    # ---- Graham: price against book and earnings --------------------------
    out["pe"] = _ratio(market_cap, net_income)
    out["pb"] = _ratio(market_cap, equity)
    out["earnings_yield"] = _ratio(net_income, market_cap)
    out["book_yield"] = _ratio(equity, market_cap)
    out["current_ratio"] = _ratio(_get(snap, "assets_current"),
                                  _get(snap, "liabilities_current"))
    out["debt_to_equity"] = _ratio(debt, equity)

    eps = _ratio(net_income, shares, require_positive_den=True)
    bvps = _ratio(equity, shares, require_positive_den=True)
    out["eps"] = eps
    out["bvps"] = bvps

    # Graham Number: the price at which a defensive investor's P/E x P/B budget
    # of 22.5 is exactly spent. Only defined when both inputs are positive.
    if eps is not None and bvps is not None and eps > 0 and bvps > 0:
        gn = math.sqrt(22.5 * eps * bvps)
        out["graham_number"] = gn
        out["graham_ratio"] = gn / price          # >1 means trading below it
    else:
        out["graham_number"] = None
        out["graham_ratio"] = None

    # Net current asset value: current assets less *all* liabilities, which is
    # the liquidation-flavoured figure Graham used, not working capital.
    ac = _get(snap, "assets_current")
    liabilities = _get(snap, "liabilities")
    if ac is not None and liabilities is not None:
        ncav = ac - liabilities
        out["ncav"] = ncav
        out["ncav_per_share"] = ncav / shares
        out["net_net_ratio"] = _ratio(market_cap, ncav)
        out["is_net_net"] = bool(ncav > 0 and market_cap < (2.0 / 3.0) * ncav)
    else:
        out["ncav"] = out["ncav_per_share"] = out["net_net_ratio"] = None
        out["is_net_net"] = False

    return out


# ---------------------------------------------------------------------------
# Graham's defensive criteria, as pass/fail
# ---------------------------------------------------------------------------
GRAHAM_CRITERIA = [
    ("earnings_positive",  "Positive trailing earnings"),
    ("pe_under_15",        "P/E below 15"),
    ("pb_under_15",        "P/B below 1.5"),
    ("graham_combined",    "P/E x P/B below 22.5"),
    ("current_ratio_2",    "Current ratio above 2"),
    ("debt_under_equity",  "Debt below book equity"),
    ("earnings_stable",    "Positive earnings every year for 5 years"),
    ("earnings_growth",    "Earnings higher than 5 years ago"),
]


def graham_scorecard(metrics: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    """Which of Graham's defensive tests this company passes, and how many.

    ``history`` carries the multi-year facts the point-in-time snapshot cannot:
    whether earnings were positive in every one of the last five fiscal years,
    and whether they have grown. Both are computed only from filings visible at
    the as-of date.
    """
    pe, pb = metrics.get("pe"), metrics.get("pb")
    checks = {
        "earnings_positive": bool(metrics.get("eps") is not None and metrics["eps"] > 0),
        "pe_under_15": bool(pe is not None and 0 < pe < 15),
        "pb_under_15": bool(pb is not None and 0 < pb < 1.5),
        "graham_combined": bool(pe is not None and pb is not None
                                and pe > 0 and pb > 0 and pe * pb < 22.5),
        "current_ratio_2": bool(metrics.get("current_ratio") is not None
                                and metrics["current_ratio"] > 2.0),
        "debt_under_equity": bool(metrics.get("debt_to_equity") is not None
                                  and 0 <= metrics["debt_to_equity"] < 1.0),
        "earnings_stable": bool(history.get("earnings_stable")),
        "earnings_growth": bool(history.get("earnings_growth")),
    }
    return {
        "checks": checks,
        "graham_score": int(sum(checks.values())),
        "graham_max": len(GRAHAM_CRITERIA),
    }
