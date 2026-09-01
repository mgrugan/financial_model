"""Assemble a point-in-time fundamental snapshot for a given as-of date.

Everything here obeys one rule: a number may be used on date T only if the
filing that first disclosed it was filed on or before T.

The subtlety is trailing-twelve-month flows. XBRL duration facts are a mix of
quarterly, half-year, nine-month and full-year spans, and many filers never tag
a standalone Q4 -- it exists only inside the annual figure. Summing "the last
four quarters" therefore silently drops Q4 for those companies and understates
a year of earnings by roughly a quarter, which in a value screen means
systematically mispricing the exact names whose earnings matter most. The
standard construction is used instead:

    TTM = latest year-to-date + prior full year - prior year's same year-to-date

which telescopes correctly whatever interim span the filer actually reports.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable

from .edgar import DURATION_FIELDS, INSTANT_FIELDS, Fact

ANNUAL_DAYS = (330, 400)
INTERIM_SPANS = ((75, 105), (160, 200), (250, 290))     # Q, H1, 9M


def _visible(facts: Iterable[Fact], as_of: str) -> list[Fact]:
    """Facts public by ``as_of``, excluding forward-looking ones.

    Two conditions, not one. ``filed <= as_of`` is the obvious point-in-time
    rule. The second, ``end <= filed``, rejects a period that ends *after* the
    filing that reports it -- which is not a historical fact at all. Two things
    produce these: companies emerging from Chapter 11 file projected financials
    (VTOL filed 2020 Q3 and Q4 figures in June 2020), and mortgage REITs tag
    debt maturity dates as period ends (PMT carries "periods" ending in 2030).
    Both are rare -- about 0.003% of facts -- but a single one hijacks a whole
    snapshot, because the balance-sheet reading is chosen by latest end date.
    """
    return [f for f in facts if f.filed <= as_of and f.end <= f.filed]


def _latest_instant(facts: list[Fact], as_of: str) -> tuple[float, str] | None:
    """Most recent balance-sheet reading that was public by ``as_of``."""
    visible = _visible(facts, as_of)
    if not visible:
        return None
    best = max(visible, key=lambda f: (f.end, f.filed))
    return best.val, best.end


def _span(f: Fact) -> str | None:
    d = f.days
    if ANNUAL_DAYS[0] <= d <= ANNUAL_DAYS[1]:
        return "FY"
    for lo, hi in INTERIM_SPANS:
        if lo <= d <= hi:
            return f"I{lo}"
    return None


def _ttm(facts: list[Fact], as_of: str) -> tuple[float, str] | None:
    """Trailing twelve months as of ``as_of``, or None if it cannot be built."""
    visible = [f for f in _visible(facts, as_of) if _span(f)]
    if not visible:
        return None

    annuals = {f.end: f for f in visible if _span(f) == "FY"}
    latest_end = max(f.end for f in visible)

    # A freshly filed 10-K is already a clean twelve months.
    if latest_end in annuals:
        return annuals[latest_end].val, latest_end

    # Otherwise telescope: YTD_now + FY_prior - YTD_prior_year.
    ytd = [f for f in visible if f.end == latest_end and _span(f) != "FY"]
    if ytd and annuals:
        cur = max(ytd, key=lambda f: f.days)
        cur_end = dt.date.fromisoformat(cur.end)
        prior_fy = [f for f in annuals.values()
                    if dt.date.fromisoformat(f.end) < cur_end
                    and (cur_end - dt.date.fromisoformat(f.end)).days <= 370]
        if prior_fy:
            fy = max(prior_fy, key=lambda f: f.end)
            # The same year-to-date span, one year earlier. Day arithmetic, not
            # a year replacement: a period ending 29 February has no counterpart
            # date in the prior year and replace() raises on it.
            target = cur_end - dt.timedelta(days=365)
            same_ytd = [
                f for f in visible
                if _span(f) == _span(cur)
                and abs((dt.date.fromisoformat(f.end) - target).days) <= 20
            ]
            if same_ytd:
                prev = min(same_ytd,
                           key=lambda f: abs((dt.date.fromisoformat(f.end) - target).days))
                return cur.val + fy.val - prev.val, latest_end

    # Last resort: the most recent complete year on file, even if stale.
    if annuals:
        end = max(annuals)
        return annuals[end].val, end
    return None


def snapshot(facts: dict[str, list[Fact]], as_of: str,
             max_staleness_days: int = 400) -> dict[str, float] | None:
    """Every canonical field as of ``as_of``, or None if the company is dark.

    ``max_staleness_days`` drops companies whose newest visible balance sheet
    predates the as-of date by more than a year. Without it a company that
    stopped filing keeps appearing in the screen at its last known -- and by
    then meaningless -- book value.
    """
    out: dict[str, float] = {}
    dates: list[str] = []

    for field in INSTANT_FIELDS:
        got = _latest_instant(facts.get(field, []), as_of)
        if got is not None:
            out[field], end = got
            # The cover-page share count is dated at filing, not at period end,
            # so counting it would make a company that has stopped reporting
            # financials look current and slip past the staleness filter.
            if field != "shares":
                dates.append(end)

    for field in DURATION_FIELDS:
        got = _ttm(facts.get(field, []), as_of)
        if got is not None:
            out[field], end = got
            dates.append(end)

    if not dates:
        return None
    newest = max(dates)
    age = (dt.date.fromisoformat(as_of) - dt.date.fromisoformat(newest)).days
    if age > max_staleness_days:
        return None

    # Total liabilities are not always tagged directly, but the balance sheet
    # identity supplies them exactly: Assets = Liabilities + Equity. Recovering
    # them this way rather than leaving a hole keeps the NCAV test available for
    # the ~80 filers who only tag LiabilitiesAndStockholdersEquity.
    if "liabilities" not in out and "assets" in out and "equity" in out:
        out["liabilities"] = out["assets"] - out["equity"]

    out["_as_of"] = as_of
    out["_latest_period"] = newest
    out["_staleness_days"] = float(age)
    return out
