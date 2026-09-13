#!/usr/bin/env python3
"""Due diligence on congressional trading: is any of it a signal?

Four questions, in the order that matters:

1. **Are these independent decisions?** A senator, their spouse and a joint
   account buying the same stock on the same day is one decision reported three
   times. Counting them separately inflates conviction.
2. **Are they stock picks at all?** One filer here reports buying 256 different
   stocks on a single day. That is a portfolio being moved, not a view on a
   company, and it dominates the raw counts.
3. **Does the disclosure arrive in time to act on?** Returns are measured from
   the filing date, never the trade date, because the trade date is not
   observable to anyone outside the filer's household.
4. **Is the information where their job would put it?** Trades inside a
   legislator's committee jurisdiction are separated from trades outside it.

The committee test carries a caveat that cannot be engineered away: the public
roster gives *current* assignments, so a trade is matched against the committees
its author sits on today rather than on the day of the trade, and members who
have since left office drop out entirely.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from congress.committees import build_index, match
from smallcaps.data import fetch_history
from value.panel import _price_on

HOLD_DAYS = 183
_PX: dict[str, pd.DataFrame | None] = {}


def _frame(ticker: str):
    if ticker not in _PX:
        f = fetch_history(ticker)
        _PX[ticker] = f if (f is not None and not f.empty) else None
    return _PX[ticker]


def excess(ticker: str, start: str, bench) -> float | None:
    f = _frame(ticker)
    if f is None or not start:
        return None
    end = (dt.date.fromisoformat(start) + dt.timedelta(days=HOLD_DAYS)).isoformat()
    a, b = _price_on(f, start, "close"), _price_on(f, end, "close")
    sa, sb = _price_on(bench, start, "close"), _price_on(bench, end, "close")
    if not (a and b and sa and sb):
        return None
    return (b / a - 1.0) - (sb / sa - 1.0)


def study(sub: pd.DataFrame, bench, label: str) -> dict | None:
    """Event study clustered by filing date, which is the unit of decision."""
    vals = [(excess(r.ticker, r.filed, bench), r.filed) for r in sub.itertuples()]
    vals = [(v, d) for v, d in vals if v is not None]
    if len(vals) < 8:
        return {"label": label, "n": len(vals), "status": "too few"}
    arr = np.array([v for v, _ in vals])
    n_dates = len({d for _, d in vals})
    t_naive = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))
    t_clustered = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(n_dates)))
    return {"label": label, "status": "ok", "n": int(len(arr)),
            "n_dates": int(n_dates), "mean_excess": float(arr.mean()),
            "t_naive": t_naive, "t_clustered": t_clustered,
            "hit_rate": float(np.mean(arr > 0))}


def main() -> int:
    bench = fetch_history("SPY", years=16)
    uni = {e["ticker"]: e for e in
           json.loads(Path("data/smallmid_universe.json").read_text())["constituents"]}
    inv = {r["ticker"]: r for r in
           json.loads(Path("data/smallcap_inventory.json").read_text())["rows"]}

    def sector_of(t: str):
        if t in uni and uni[t].get("sector"):
            return uni[t]["sector"]
        if t in inv and inv[t].get("sector"):
            return inv[t]["sector"]
        return None

    out: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    idx = build_index()

    # ---------------- Senate ----------------
    sen = pd.DataFrame(json.loads(Path("data/senate_transactions.json").read_text())
                       ["transactions"])
    raw_buys = sen[sen.is_purchase & sen.asset_type.eq("Stock")]
    hh = raw_buys.drop_duplicates(subset=["senator", "ticker", "trans_date"]).copy()
    hh["basket"] = hh.groupby(["senator", "trans_date"])["ticker"].transform("nunique")
    hh["sector"] = hh.ticker.map(sector_of)

    rec = hh.senator.map(lambda s: match(idx, s, chamber="sen"))
    hh["jurisdiction"] = rec.map(lambda m: set(m["sectors"]) if m else set())
    hh["in_jurisdiction"] = [
        (s in j) if (s and j) else None for s, j in zip(hh.sector, hh.jurisdiction)]

    big = hh[hh.basket >= 5]
    out["senate"] = {
        "raw_purchase_rows": int(len(raw_buys)),
        "household_decisions": int(len(hh)),
        "household_inflation": float(len(raw_buys) / max(len(hh), 1)),
        "owner_breakdown": raw_buys.owner.value_counts().to_dict(),
        "share_from_bulk_days": float(len(big) / max(len(hh), 1)),
        "largest_single_day_basket": int(hh.basket.max()),
        "n_filers": int(hh.senator.nunique()),
        "matched_to_committee": int(rec.notna().sum()),
        "events": [
            study(hh, bench, "All household purchases"),
            study(hh[hh.basket >= 5], bench, "Bulk account days (5+ names at once)"),
            study(hh[hh.basket <= 2], bench, "Discretionary (1-2 names that day)"),
            study(hh[hh.basket == 1], bench, "Single-name conviction buys"),
        ],
        "committee": [
            study(hh[hh.in_jurisdiction == True], bench, "In committee jurisdiction"),
            study(hh[hh.in_jurisdiction == False], bench, "Outside jurisdiction"),
        ],
    }

    # ---------------- House ----------------
    hp = Path("data/house_transactions.json")
    if hp.exists():
        hr = pd.DataFrame(json.loads(hp.read_text())["transactions"])
        if not hr.empty:
            hb = hr[hr.is_purchase].drop_duplicates(
                subset=["member", "ticker", "trans_date"]).copy()
            hb["basket"] = hb.groupby(["member", "trans_date"])["ticker"].transform("nunique")
            hb["sector"] = hb.ticker.map(sector_of)
            hrec = hb.member.map(lambda m: match(idx, m, chamber="rep"))
            hb["jurisdiction"] = hrec.map(lambda m: set(m["sectors"]) if m else set())
            hb["in_jurisdiction"] = [
                (s in j) if (s and j) else None
                for s, j in zip(hb.sector, hb.jurisdiction)]
            out["house"] = {
                "raw_purchase_rows": int(len(hr[hr.is_purchase])),
                "household_decisions": int(len(hb)),
                "n_filers": int(hb.member.nunique()),
                "share_via_named_manager": float(hr.managed_by.notna().mean()),
                "largest_single_day_basket": int(hb.basket.max()),
                "matched_to_committee": int(hrec.notna().sum()),
                "events": [
                    study(hb, bench, "All House purchases"),
                    study(hb[hb.basket >= 5], bench, "Bulk account days"),
                    study(hb[hb.basket <= 2], bench, "Discretionary"),
                    study(hb[hb.basket == 1], bench, "Single-name conviction buys"),
                ],
                "committee": [
                    study(hb[hb.in_jurisdiction == True], bench, "In committee jurisdiction"),
                    study(hb[hb.in_jurisdiction == False], bench, "Outside jurisdiction"),
                ],
            }
            # Both chambers pooled, discretionary only: the best-powered cut.
            both = pd.concat([
                hh[["ticker", "filed", "basket"]].assign(chamber="Senate"),
                hb[["ticker", "filed", "basket"]].assign(chamber="House")])
            out["combined"] = {
                "n": int(len(both)),
                "events": [study(both, bench, "Both chambers, all purchases"),
                           study(both[both.basket <= 2], bench,
                                 "Both chambers, discretionary")],
            }

    path = Path("data/congress_diligence.json")
    path.write_text(json.dumps(out, indent=1, default=float))

    for chamber in ("senate", "house", "combined"):
        if chamber not in out:
            continue
        print(f"\n=== {chamber.upper()} ===", flush=True)
        block = out[chamber]
        for k, v in block.items():
            if k in ("events", "committee"):
                continue
            print(f"  {k}: {v}", flush=True)
        for group in ("events", "committee"):
            for e in block.get(group, []):
                if e and e.get("status") == "ok":
                    print(f"    {e['label']:44s} n={e['n']:5d} dates={e['n_dates']:4d} "
                          f"excess {e['mean_excess']:+6.2%} t={e['t_clustered']:+5.2f} "
                          f"hit {e['hit_rate']:3.0%}", flush=True)
                elif e:
                    print(f"    {e['label']:44s} n={e['n']} — too few", flush=True)
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
