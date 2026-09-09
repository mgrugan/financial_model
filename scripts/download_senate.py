#!/usr/bin/env python3
"""Pull Senate Periodic Transaction Reports and extract stock transactions."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from congress.senate import SenateEFD

logging.basicConfig(level=logging.ERROR, format="%(message)s")

START = "01/01/2023"          # eFD search takes US-formatted dates


def main() -> int:
    efd = SenateEFD()
    print(f"searching Senate eFD for PTRs filed since {START}", flush=True)
    reports = efd.reports(START, progress=lambda m: print(m, flush=True))
    print(f"{len(reports)} reports found", flush=True)

    rows, paper, t0 = [], 0, time.time()
    for i, report in enumerate(reports, 1):
        got = efd.transactions(report)
        if not got:
            paper += 1
        rows.extend(got)
        if i % 50 == 0:
            print(f"  parsed {i}/{len(reports)} ({len(rows)} transactions, "
                  f"{paper} unparseable, {time.time()-t0:.0f}s)", flush=True)

    out = Path("data/senate_transactions.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_start": START,
        "n_reports": len(reports),
        "n_unparseable_reports": paper,
        "n_transactions": len(rows),
        "transactions": rows,
    }, indent=1))
    print(f"\nwrote {out}: {len(rows)} transactions from "
          f"{len(reports) - paper}/{len(reports)} machine-readable reports", flush=True)

    if rows:
        import pandas as pd
        f = pd.DataFrame(rows)
        buys = f[f["is_purchase"]]
        print(f"  purchases {len(buys):,} | sales {int(f['is_sale'].sum()):,}", flush=True)
        print(f"  distinct senators {f['senator'].nunique()} | "
              f"distinct tickers {f['ticker'].nunique()}", flush=True)
        lag = (pd.to_datetime(f["filed"]) - pd.to_datetime(f["trans_date"])).dt.days
        lag = lag[lag.between(0, 400)]
        print(f"  disclosure lag: median {lag.median():.0f} days, "
              f"p90 {lag.quantile(0.9):.0f}, max {lag.max():.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
