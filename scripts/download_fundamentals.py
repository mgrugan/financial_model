#!/usr/bin/env python3
"""Pull SEC EDGAR companyfacts for the small-cap universe into the local cache."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value.edgar import cik_map, fetch_many

logging.basicConfig(level=logging.ERROR, format="%(message)s")


def main() -> int:
    inventory = json.loads(Path("data/smallcap_inventory.json").read_text())
    tickers = [r["ticker"] for r in inventory["rows"] if r["ok"]]
    mapping = cik_map(refresh=True)
    matched = [t for t in tickers if t.upper() in mapping]
    print(f"{len(tickers)} tradeable tickers, {len(matched)} matched to a CIK", flush=True)
    missing = sorted(set(tickers) - set(matched))
    if missing:
        print(f"  no CIK for: {', '.join(missing[:20])}", flush=True)

    t0 = time.time()
    facts = fetch_many(matched, workers=5, progress=lambda m: print(m, flush=True))
    print(f"fetched {len(facts)}/{len(matched)} in {time.time()-t0:.0f}s", flush=True)

    # Coverage report: a field present for a company is one with any fact at all.
    from collections import Counter
    cov = Counter()
    for data in facts.values():
        for field, rows in data.items():
            if rows:
                cov[field] += 1
    print("\nfield coverage across companies:", flush=True)
    for field, n in sorted(cov.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}/{len(facts)}  {field}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
