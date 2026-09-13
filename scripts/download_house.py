#!/usr/bin/env python3
"""Download and parse House of Representatives PTRs."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from congress.house import fetch_ptrs

logging.basicConfig(level=logging.ERROR, format="%(message)s")
YEARS = (2023, 2024, 2025, 2026)


def main() -> int:
    t0 = time.time()
    rows = fetch_ptrs(YEARS, progress=lambda m: print(m, flush=True))
    print(f"\n{len(rows):,} House transactions in {time.time()-t0:.0f}s", flush=True)

    out = Path("data/house_transactions.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "years": list(YEARS), "n_transactions": len(rows), "transactions": rows,
    }, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    import pandas as pd
    f = pd.DataFrame(rows)
    if f.empty:
        return 1
    print(f"  purchases {int(f['is_purchase'].sum()):,} | sales {int(f['is_sale'].sum()):,}",
          flush=True)
    print(f"  members {f['member'].nunique()} | tickers {f['ticker'].nunique()}", flush=True)
    print(f"  reported via a named manager: "
          f"{f['managed_by'].notna().mean():.0%}", flush=True)
    lag = (pd.to_datetime(f["filing_date"]) - pd.to_datetime(f["trans_date"])).dt.days
    lag = lag[lag.between(0, 500)]
    print(f"  disclosure lag: median {lag.median():.0f}d  p90 {lag.quantile(.9):.0f}d",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
