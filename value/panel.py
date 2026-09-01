"""Build the point-in-time value panel: metrics at T, realised return over T+6m.

Design decisions that determine whether the result means anything:

**Rebalance grid.** Semi-annual, on the last trading day of June and December.
A fixed calendar grid rather than each company's own fiscal calendar, because
staggering rebalances by fiscal year-end would mix market regimes into the
cross-section: companies with October year-ends would be measured over
different six-month windows than companies with December ones, and the
difference between them would show up as a spurious factor.

**The sample size that matters is 28, not 15,000.** Every stock in one
rebalance shares that half-year's market move, so ~15,000 stock-periods carry
roughly 28 independent observations. All inference here is run on the
period-level spread series, never on the pooled stock-periods -- the same
mistake, and the same correction, as the technical study.

**Sector handling.** EV/EBITDA is not defined in any useful sense for a bank:
debt is raw material rather than financing, so enterprise value is not a
takeover price. It is nearly as bad for a REIT, where EBITDA ignores exactly the
depreciation that real-estate accounting exists to argue about. Financials and
Real Estate are therefore excluded from EV-based factors, though they stay in
the book- and earnings-based ones where P/B is the standard measure.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

from .edgar import Fact
from .metrics import compute, graham_scorecard
from .pit import snapshot

log = logging.getLogger(__name__)

# EV is not a meaningful takeover price for these business models.
EV_EXCLUDED_SECTORS = {"Financials", "Real Estate"}

HORIZON_MONTHS = 6


def rebalance_dates(start: str = "2012-06-30", end: str = "2025-12-31") -> list[str]:
    """Last calendar day of each June and December in the range."""
    out = []
    year = int(start[:4])
    while True:
        for month, day in ((6, 30), (12, 31)):
            d = f"{year}-{month:02d}-{day:02d}"
            if start <= d <= end:
                out.append(d)
        year += 1
        if year > int(end[:4]):
            break
    return sorted(out)


def _price_on(frame: pd.DataFrame, date: str, column: str) -> float | None:
    """Last available close at or before ``date`` (never after)."""
    stamp = pd.Timestamp(date, tz="UTC")
    sub = frame.loc[frame.index <= stamp, column]
    if sub.empty:
        return None
    # Guard against a long halt: a price three months stale is not a price.
    if (stamp - sub.index[-1]).days > 15:
        return None
    val = float(sub.iloc[-1])
    return val if np.isfinite(val) and val > 0 else None


def _earnings_history(facts: dict[str, list[Fact]], as_of: str) -> dict[str, Any]:
    """Graham's five-year earnings record, using only filings visible at as_of."""
    annuals = [f for f in facts.get("net_income", [])
               if f.filed <= as_of and 330 <= f.days <= 400]
    if not annuals:
        return {"earnings_stable": False, "earnings_growth": False, "n_years": 0}

    # One value per fiscal year end, earliest filing wins.
    by_end: dict[str, Fact] = {}
    for f in sorted(annuals, key=lambda x: x.filed):
        by_end.setdefault(f.end, f)
    years = sorted(by_end)[-6:]
    vals = [by_end[y].val for y in years]

    stable = len(vals) >= 5 and all(v > 0 for v in vals[-5:])
    growth = len(vals) >= 5 and vals[-1] > vals[0] and vals[0] != 0
    return {"earnings_stable": bool(stable), "earnings_growth": bool(growth),
            "n_years": len(vals), "earnings_5y": vals}


def build_panel(prices: dict[str, pd.DataFrame],
                facts: dict[str, dict[str, list[Fact]]],
                sectors: dict[str, str],
                names: dict[str, str] | None = None,
                dates: list[str] | None = None,
                progress: Any = None) -> pd.DataFrame:
    """One row per (ticker, rebalance date) with metrics and forward return."""
    dates = dates or rebalance_dates()
    names = names or {}
    rows: list[dict[str, Any]] = []

    for date in dates:
        fwd_date = (dt.date.fromisoformat(date)
                    + dt.timedelta(days=int(HORIZON_MONTHS * 30.44))).isoformat()
        n_ok = 0
        for ticker, frame in prices.items():
            company_facts = facts.get(ticker)
            if company_facts is None:
                continue

            raw = _price_on(frame, date, "raw_close")
            adj = _price_on(frame, date, "close")
            adj_fwd = _price_on(frame, fwd_date, "close")
            if raw is None or adj is None or adj_fwd is None:
                continue

            snap = snapshot(company_facts, date)
            if snap is None:
                continue

            shares = snap.get("shares") or snap.get("shares_diluted")
            metrics = compute(snap, raw, shares)
            if not metrics:
                continue

            history = _earnings_history(company_facts, date)
            card = graham_scorecard(metrics, history)

            sector = sectors.get(ticker, "")
            row: dict[str, Any] = {
                "date": date,
                "ticker": ticker,
                "name": names.get(ticker, ""),
                "sector": sector,
                "ev_applicable": sector not in EV_EXCLUDED_SECTORS,
                "fwd_return": adj_fwd / adj - 1.0,
                "staleness_days": snap.get("_staleness_days"),
                "latest_period": snap.get("_latest_period"),
                "graham_score": card["graham_score"],
                "n_earnings_years": history["n_years"],
                **{k: v for k, v in metrics.items() if not k.startswith("_")},
                **{f"chk_{k}": v for k, v in card["checks"].items()},
            }
            rows.append(row)
            n_ok += 1
        if progress:
            progress(f"  {date}: {n_ok} companies")

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
