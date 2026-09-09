#!/usr/bin/env python3
"""Pull SEC Form 345 insider transactions for the universe into the cache."""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from insider.form345 import load_history, validate_prices
from insider.recent import fetch_recent
from value.edgar import cik_map

logging.basicConfig(level=logging.ERROR, format="%(message)s")


def main() -> int:
    inv = json.loads(Path("data/smallcap_inventory.json").read_text())
    tickers = [r["ticker"] for r in inv["rows"] if r["ok"]]
    mapping = cik_map()
    ciks = {mapping[t.upper()] for t in tickers if t.upper() in mapping}
    print(f"{len(tickers)} tickers -> {len(ciks)} distinct CIKs", flush=True)

    t0 = time.time()
    frame = load_history(ciks, start="2011-01-01",
                         progress=lambda m: print(m, flush=True))
    print(f"\n{len(frame):,} insider transactions in {time.time()-t0:.0f}s", flush=True)
    if frame.empty:
        return 1

    # The quarterly datasets lag by a quarter or more, which would leave most of
    # a six-month insider window empty. Top up from EDGAR itself.
    latest = str(frame["filed"].max())
    since = (dt.date.fromisoformat(latest) + dt.timedelta(days=1)).isoformat()
    if since < dt.date.today().isoformat():
        print(f"\nquarterly data ends {latest}; fetching Form 4s filed since {since}",
              flush=True)
        extra = fetch_recent(sorted(ciks), since,
                             progress=lambda m: print(m, flush=True))
        if not extra.empty:
            frame = pd.concat([frame, extra], ignore_index=True)
            frame = frame.drop_duplicates(
                subset=["cik", "filed", "trans_date", "shares", "price", "owner_cik"])
            print(f"  added {len(extra):,} recent transactions "
                  f"({extra['is_buy'].sum()} buys)", flush=True)

    # Cross-check reported prices against the market before anything downstream
    # weights by dollars.
    from smallcaps.data import fetch_history
    closes = {}
    for t in {str(x) for x in frame["ticker"].dropna().unique()}:
        f = fetch_history(t)
        if f is not None and not f.empty and "raw_close" in f:
            closes[t] = f["raw_close"]
    frame, stats = validate_prices(frame, closes)
    print(f"  price check: {stats['implausible']} implausible prices removed, "
          f"{stats['checked']:,} cross-checked against the market, {stats['dropped']} "
          f"dropped as filer errors, {stats['unverifiable']:,} unverifiable", flush=True)

    out = Path("data/insider_transactions.pkl")
    frame.to_pickle(out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    buys = frame[frame["is_buy"]]
    sells = frame[frame["is_sell"]]
    print(f"\n  open-market buys : {len(buys):>8,}  ${buys['value'].sum()/1e9:>7.2f}bn", flush=True)
    print(f"  open-market sells: {len(sells):>8,}  ${sells['value'].sum()/1e9:>7.2f}bn", flush=True)
    print(f"  span             : {frame['filed'].min()} .. {frame['filed'].max()}", flush=True)
    print(f"  companies with >=1 buy: {buys['cik'].nunique()} of {len(ciks)}", flush=True)
    lag = (frame["filed"] > frame["trans_date"]).mean()
    print(f"  filed after trade date: {lag:.1%} (rest same day)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
