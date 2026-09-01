"""Download the whole small-cap universe into the local cache."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.data import download_many, liquidity_stats, passes_filters
from smallcaps.universe import load_universe

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main() -> int:
    entries = load_universe()
    tickers = [e["ticker"] for e in entries]
    print(f"universe: {len(tickers)} tickers", flush=True)

    t0 = time.time()
    frames = download_many(tickers, years=12, workers=6,
                           progress=lambda m: print(m, flush=True))
    print(f"downloaded {len(frames)}/{len(tickers)} in {time.time()-t0:.0f}s", flush=True)

    rows, kept = [], 0
    for entry in entries:
        ticker = entry["ticker"]
        frame = frames.get(ticker)
        if frame is None:
            rows.append({**entry, "ok": False, "reason": "no data from Yahoo"})
            continue
        stats = liquidity_stats(frame)
        stats["bad_ticks"] = int(frame.attrs.get("bad_ticks", 0))
        ok, why = passes_filters(stats)
        kept += ok
        rows.append({**entry, **stats, "ok": bool(ok), "reason": why})

    out = Path("data/smallcap_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(rows), "n_tradeable": kept, "rows": rows}, indent=1))
    print(f"tradeable after filters: {kept}/{len(rows)} -> {out}", flush=True)

    from collections import Counter
    for reason, n in Counter(r["reason"] for r in rows if not r["ok"]).most_common():
        print(f"  dropped {n:4d}  {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
