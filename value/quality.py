"""Qualitative due diligence, computed from filings rather than asserted.

A screen tells you a stock is cheap. It does not tell you whether the business
behind it is growing or shrinking, whether the margin is widening or being
competed away, whether the share count is quietly doubling, or whether the
balance sheet is drifting the wrong way. Those are the questions a human asks
before holding something for five years, and most of them are answerable from
the same filings the valuation came from.

Nothing here is a narrative generated about a company. Every field is a
measured trajectory over the last five fiscal years -- a growth rate, a trend
slope, a count of years satisfying a condition -- plus the SIC industry the
company itself reports. The risk flags are thresholds on those numbers, so a
flag always points at a figure the reader can check.

The deliberate omission is anything requiring judgement about the future:
competitive position, management quality, moat. Those matter more than
everything here and cannot be computed, so the output is framed as the input to
that judgement rather than a substitute for it.
"""

from __future__ import annotations

import datetime as dt
import logging
import statistics
from typing import Any

import numpy as np

from .edgar import Fact

log = logging.getLogger(__name__)

ANNUAL = (330, 400)
LOOKBACK_YEARS = 5


def _annual_series(facts: list[Fact], as_of: str, duration: bool,
                   years: int = LOOKBACK_YEARS) -> list[tuple[str, float]]:
    """Fiscal-year values visible at ``as_of``, oldest first, first-filed wins."""
    seen: dict[str, Fact] = {}
    for f in sorted(facts or [], key=lambda x: x.filed):
        if f.filed > as_of or f.end > f.filed:
            continue
        if duration and not (ANNUAL[0] <= f.days <= ANNUAL[1]):
            continue
        if not duration and f.start:
            continue
        seen.setdefault(f.end, f)
    if duration:
        rows = sorted(seen.items())[-(years + 1):]
        return [(end, fact.val) for end, fact in rows]

    # Balance-sheet items are filed every quarter, so "the last six readings"
    # spans eighteen months, not six years -- a five-year growth rate computed
    # from them would silently be an eighteen-month one. Keep one reading per
    # fiscal year (the latest in each) before taking the last few years.
    per_year: dict[str, tuple[str, Fact]] = {}
    for end, fact in sorted(seen.items()):
        per_year[end[:4]] = (end, fact)
    rows = [v for _, v in sorted(per_year.items())][-(years + 1):]
    return [(end, fact.val) for end, fact in rows]


def _cagr(series: list[tuple[str, float]]) -> float | None:
    if len(series) < 3:
        return None
    first, last = series[0][1], series[-1][1]
    if first <= 0 or last <= 0:
        return None
    span = (dt.date.fromisoformat(series[-1][0])
            - dt.date.fromisoformat(series[0][0])).days / 365.25
    if span < 1:
        return None
    return float((last / first) ** (1.0 / span) - 1.0)


def _trend(values: list[float]) -> float | None:
    """Ordinary least-squares slope per year, in the units of the input."""
    if len(values) < 3:
        return None
    x = np.arange(len(values), dtype="float64")
    y = np.asarray(values, dtype="float64")
    if not np.isfinite(y).all():
        return None
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def profile(facts: dict[str, list[Fact]], as_of: str) -> dict[str, Any]:
    """Five-year business trajectory and the risk flags that follow from it."""
    rev = _annual_series(facts.get("revenue", []), as_of, True)
    op = _annual_series(facts.get("operating_income", []), as_of, True)
    ni = _annual_series(facts.get("net_income", []), as_of, True)
    eq = _annual_series(facts.get("equity", []), as_of, False)
    sh = _annual_series(facts.get("shares", []), as_of, False)
    ocf = _annual_series(facts.get("operating_cash_flow", []), as_of, True)
    capex = _annual_series(facts.get("capex", []), as_of, True)
    debt_l = dict(_annual_series(facts.get("debt_long", []), as_of, False))
    debt_s = dict(_annual_series(facts.get("debt_short", []), as_of, False))

    out: dict[str, Any] = {"years_of_data": len(rev)}

    out["revenue_cagr_5y"] = _cagr(rev)
    out["earnings_cagr_5y"] = _cagr(ni)

    # Operating margin by year, and whether it is widening or narrowing.
    rev_by_end = dict(rev)
    margins = [(end, val / rev_by_end[end])
               for end, val in op if end in rev_by_end and rev_by_end[end] > 0]
    out["operating_margin_latest"] = margins[-1][1] if margins else None
    out["operating_margin_trend"] = _trend([m for _, m in margins])
    out["operating_margin_mean"] = (float(np.mean([m for _, m in margins]))
                                    if margins else None)

    # Return on equity, averaged, using each year's own book value.
    eq_by_end = dict(eq)
    roes = [val / eq_by_end[end] for end, val in ni
            if end in eq_by_end and eq_by_end[end] > 0]
    out["roe_mean_5y"] = float(np.mean(roes)) if roes else None

    # Dilution: a rising share count transfers value away from holders even
    # when the business itself is fine.
    out["share_count_cagr_5y"] = _cagr(sh)

    # Earnings consistency: how many of the last five years were profitable,
    # and how volatile the profit was relative to its own mean.
    ni_vals = [v for _, v in ni][-LOOKBACK_YEARS:]
    out["profitable_years"] = int(sum(1 for v in ni_vals if v > 0))
    out["years_counted"] = len(ni_vals)
    if len(ni_vals) >= 3 and abs(float(np.mean(ni_vals))) > 0:
        out["earnings_volatility"] = float(
            statistics.pstdev(ni_vals) / abs(float(np.mean(ni_vals))))
    else:
        out["earnings_volatility"] = None

    # Free cash flow: how many years it was positive.
    capex_by_end = dict(capex)
    fcfs = [val - abs(capex_by_end.get(end, 0.0)) for end, val in ocf]
    out["fcf_positive_years"] = int(sum(1 for v in fcfs[-LOOKBACK_YEARS:] if v > 0))
    out["fcf_years_counted"] = len(fcfs[-LOOKBACK_YEARS:])

    # Leverage direction: debt over equity, year by year.
    lev = []
    for end, equity in eq:
        if equity <= 0:
            continue
        total = debt_l.get(end, 0.0) + debt_s.get(end, 0.0)
        if total or end in debt_l or end in debt_s:
            lev.append(total / equity)
    out["leverage_latest"] = lev[-1] if lev else None
    out["leverage_trend"] = _trend(lev)

    out["flags"] = _flags(out)
    return out


FLAG_TEXT = {
    "revenue_shrinking": "Revenue has declined over five years",
    "margin_compressing": "Operating margin is trending down",
    "heavy_dilution": "Share count growing more than 3% a year",
    "leverage_rising": "Debt-to-equity trending up",
    "loss_years": "Lost money in at least two of the last five years",
    "cash_burn": "Negative free cash flow in most years",
    "earnings_erratic": "Earnings swing more than their own average",
    "thin_history": "Fewer than four years of comparable filings",
}


def _flags(p: dict[str, Any]) -> list[str]:
    """Threshold checks. Each one points at a number the reader can verify."""
    out = []
    if (p.get("revenue_cagr_5y") or 0) < 0:
        out.append("revenue_shrinking")
    if (p.get("operating_margin_trend") or 0) < -0.005:          # >0.5pp/yr
        out.append("margin_compressing")
    if (p.get("share_count_cagr_5y") or 0) > 0.03:
        out.append("heavy_dilution")
    if (p.get("leverage_trend") or 0) > 0.05:
        out.append("leverage_rising")
    counted = p.get("years_counted") or 0
    if counted and (counted - (p.get("profitable_years") or 0)) >= 2:
        out.append("loss_years")
    fcf_n = p.get("fcf_years_counted") or 0
    if fcf_n and (p.get("fcf_positive_years") or 0) < fcf_n / 2:
        out.append("cash_burn")
    if (p.get("earnings_volatility") or 0) > 1.0:
        out.append("earnings_erratic")
    if (p.get("years_of_data") or 0) < 4:
        out.append("thin_history")
    return out
