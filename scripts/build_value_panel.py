#!/usr/bin/env python3
"""Assemble the point-in-time value panel and cache it."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.data import fetch_history
from value.edgar import cik_map, fetch_company
from value.panel import build_panel, rebalance_dates

logging.basicConfig(level=logging.ERROR, format="%(message)s")


def main() -> int:
    inv = json.loads(Path("data/smallcap_inventory.json").read_text())
    rows = [r for r in inv["rows"] if r["ok"]]
    sectors = {r["ticker"]: r.get("sector", "") for r in rows}
    names = {r["ticker"]: r.get("name", "") for r in rows}

    mapping = cik_map()
    prices, facts = {}, {}
    for r in rows:
        t = r["ticker"]
        frame = fetch_history(t)
        if frame is None or frame.empty or "raw_close" not in frame:
            continue
        cik = mapping.get(t.upper())
        data = fetch_company(t, cik) if cik else None
        if data is None:
            continue
        prices[t], facts[t] = frame, data
    print(f"{len(prices)} companies with both prices and filings", flush=True)

    dates = rebalance_dates()
    print(f"{len(dates)} semi-annual rebalances: {dates[0]} .. {dates[-1]}", flush=True)

    t0 = time.time()
    panel = build_panel(prices, facts, sectors, names, dates,
                        progress=lambda m: print(m, flush=True))
    print(f"\npanel: {len(panel)} rows in {time.time()-t0:.0f}s", flush=True)

    out = Path("data/value_panel.pkl")
    panel.to_pickle(out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    # Coverage of each metric, since a factor can only be tested where it exists.
    print("\nnon-null coverage by metric:", flush=True)
    for col in ["ebitda_ev_yield", "fcf_ev_yield", "ev_sales", "earnings_yield",
                "book_yield", "graham_ratio", "current_ratio", "debt_to_equity",
                "ncav", "graham_score"]:
        if col in panel:
            print(f"  {panel[col].notna().mean():6.1%}  {col}", flush=True)
    print(f"\nnet-nets ever: {int(panel['is_net_net'].sum())} stock-periods", flush=True)
    print(f"median staleness: {panel['staleness_days'].median():.0f} days", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
