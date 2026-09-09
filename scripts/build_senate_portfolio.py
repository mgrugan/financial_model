#!/usr/bin/env python3
"""Senator-purchase portfolio for small and mid caps, and an event study.

Two things are produced. The portfolio is what the user asked for: the small and
mid caps senators have disclosed buying most recently. The event study is what
makes it honest: what actually happened to those names over the six months after
the purchase became *public*, which is the earliest anyone outside the Senate
could have acted.

The gap between trade date and filing date is the whole problem. Senators have
45 days to disclose, and the realised lag in this sample runs to a median of a
month with a long tail beyond a year. Any return measured from the trade date is
unobtainable; only the return from the filing date is real.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.data import fetch_history
from value.panel import _price_on

HOLD_DAYS = 183
RECENT_MONTHS = 12


def main() -> int:
    payload = json.loads(Path("data/senate_transactions.json").read_text())
    frame = pd.DataFrame(payload["transactions"])
    universe = {e["ticker"]: e for e in
                json.loads(Path("data/smallmid_universe.json").read_text())["constituents"]}

    frame["in_universe"] = frame["ticker"].isin(universe)
    buys = frame[frame["is_purchase"] & frame["in_universe"]
                 & frame["asset_type"].eq("Stock")].copy()
    buys["index"] = buys["ticker"].map(lambda t: universe[t]["index"])
    buys["sector"] = buys["ticker"].map(lambda t: universe[t].get("sector", ""))
    buys["name"] = buys["ticker"].map(lambda t: universe[t].get("name", ""))
    print(f"{len(buys)} senator purchases of small/mid-cap stock, "
          f"{buys['ticker'].nunique()} tickers, {buys['senator'].nunique()} senators",
          flush=True)

    # --- event study: return measured from the DISCLOSURE date ------------
    closes = {}
    for ticker in buys["ticker"].unique():
        f = fetch_history(ticker)
        if f is not None and not f.empty:
            closes[ticker] = f[["close"]]

    rows = []
    for _, r in buys.iterrows():
        frame_px = closes.get(r["ticker"])
        if frame_px is None or not r["filed"]:
            continue
        end = (dt.date.fromisoformat(r["filed"]) + dt.timedelta(days=HOLD_DAYS)).isoformat()
        a = _price_on(frame_px, r["filed"], "close")
        b = _price_on(frame_px, end, "close")
        if a is None or b is None:
            continue
        spy = closes.get("SPY")
        rows.append({
            "ticker": r["ticker"], "senator": r["senator"], "index": r["index"],
            "trans_date": r["trans_date"], "filed": r["filed"],
            "lag_days": (dt.date.fromisoformat(r["filed"])
                         - dt.date.fromisoformat(r["trans_date"])).days
            if r["trans_date"] else None,
            "fwd_6m_from_disclosure": b / a - 1.0,
        })
    events = pd.DataFrame(rows)

    study = {}
    if len(events) >= 10:
        bench = fetch_history("IJR", years=16)
        ex = []
        for _, e in events.iterrows():
            end = (dt.date.fromisoformat(e["filed"]) + dt.timedelta(days=HOLD_DAYS)).isoformat()
            a = _price_on(bench, e["filed"], "close")
            b = _price_on(bench, end, "close")
            if a and b:
                ex.append(e["fwd_6m_from_disclosure"] - (b / a - 1.0))
        ex = np.array(ex)
        r6 = events["fwd_6m_from_disclosure"].to_numpy()
        # Events cluster: a handful of senators, often several names on one day.
        # The independent count is nearer the number of distinct filing dates
        # than the number of transactions, so both are reported.
        n_dates = events["filed"].nunique()
        t_naive = float(np.mean(ex) / (np.std(ex, ddof=1) / np.sqrt(len(ex)))) if len(ex) > 3 else float("nan")
        t_clustered = float(np.mean(ex) / (np.std(ex, ddof=1) / np.sqrt(n_dates))) if len(ex) > 3 else float("nan")
        study = {
            "n_events": int(len(events)), "n_distinct_filing_dates": int(n_dates),
            "n_distinct_senators": int(events["senator"].nunique()),
            "mean_6m_return": float(np.mean(r6)), "median_6m_return": float(np.median(r6)),
            "mean_excess_vs_ijr": float(np.mean(ex)) if len(ex) else float("nan"),
            "t_naive": t_naive, "t_clustered_by_date": t_clustered,
            "hit_rate": float(np.mean(ex > 0)) if len(ex) else float("nan"),
            "median_lag_days": float(events["lag_days"].median()),
            "p90_lag_days": float(events["lag_days"].quantile(0.9)),
        }
        print(f"\nevent study, 6 months from the DISCLOSURE date:", flush=True)
        print(f"  {study['n_events']} events on {study['n_distinct_filing_dates']} distinct "
              f"filing dates, {study['n_distinct_senators']} senators", flush=True)
        print(f"  mean return {study['mean_6m_return']:+.2%}  "
              f"excess vs IJR {study['mean_excess_vs_ijr']:+.2%}", flush=True)
        print(f"  t treating events as independent: {study['t_naive']:+.2f}", flush=True)
        print(f"  t clustered by filing date:       {study['t_clustered_by_date']:+.2f}", flush=True)
        print(f"  beat the index {study['hit_rate']:.0%} of the time", flush=True)
        print(f"  disclosure lag: median {study['median_lag_days']:.0f}d, "
              f"p90 {study['p90_lag_days']:.0f}d", flush=True)

    # --- the portfolio: most recently disclosed purchases -----------------
    cutoff = (dt.date.today() - dt.timedelta(days=RECENT_MONTHS * 31)).isoformat()
    recent = buys[buys["filed"] >= cutoff]
    holdings = (recent.groupby("ticker")
                .agg(name=("name", "first"), sector=("sector", "first"),
                     index=("index", "first"), n_buys=("ticker", "size"),
                     senators=("senator", "nunique"),
                     senator_names=("senator", lambda s: sorted(set(s))),
                     last_trade=("trans_date", "max"), last_filed=("filed", "max"),
                     est_usd=("amount_mid", "sum"))
                .sort_values(["senators", "last_filed"], ascending=[False, False])
                .reset_index())
    if len(holdings):
        holdings["weight"] = 1.0 / len(holdings)

    out = Path("data/senate_portfolio.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "as_of": dt.date.today().isoformat(),
        "window_months": RECENT_MONTHS,
        "n_holdings": int(len(holdings)),
        "holdings": json.loads(holdings.to_json(orient="records")) if len(holdings) else [],
        "event_study": study,
        "coverage": {
            "reports_searched": payload["n_reports"],
            "reports_unparseable": payload["n_unparseable_reports"],
            "all_transactions": payload["n_transactions"],
            "smallmid_purchases": int(len(buys)),
        },
    }, indent=1, default=float))
    print(f"\nwrote {out}: {len(holdings)} holdings disclosed in the last "
          f"{RECENT_MONTHS} months", flush=True)
    if len(holdings):
        print(holdings[["ticker", "name", "index", "senators", "n_buys",
                        "last_trade", "last_filed"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
