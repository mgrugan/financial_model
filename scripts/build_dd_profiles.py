#!/usr/bin/env python3
"""Qualitative due-diligence profile for every company in the universe.

Pairs each company's self-reported industry with a five-year trajectory of its
own filings: is revenue growing, is the margin widening, is the share count
being inflated, is leverage drifting up, how many of the last five years were
profitable and cash-generative. Risk flags are thresholds on those numbers, so
every flag points at a figure on the same row.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcmodels.config import CACHE_DIR
from value.edgar import SEC_UA, cik_map, fetch_company
from value.quality import FLAG_TEXT, profile

logging.basicConfig(level=logging.ERROR, format="%(message)s")

META_CACHE = Path(CACHE_DIR) / "company_meta.json"
_LOCK = threading.Lock()


def company_meta(ciks: dict[str, str]) -> dict[str, dict]:
    """Name, industry and exchange from each company's EDGAR submission index."""
    cached = {}
    if META_CACHE.exists():
        try:
            cached = json.loads(META_CACHE.read_text())
        except Exception:
            cached = {}
    todo = [(t, c) for t, c in ciks.items() if t not in cached]
    if not todo:
        return cached

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
    done = 0

    def work(item):
        nonlocal done
        ticker, cik = item
        try:
            r = session.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=45)
            time.sleep(0.12)
            if r.status_code == 200:
                d = r.json()
                with _LOCK:
                    cached[ticker] = {
                        "legal_name": d.get("name"),
                        "industry": d.get("sicDescription"),
                        "sic": d.get("sic"),
                        "exchange": (d.get("exchanges") or [None])[0],
                        "state": d.get("stateOfIncorporation"),
                        "employees": None,
                        "fiscal_year_end": d.get("fiscalYearEnd"),
                    }
        except Exception:
            pass
        with _LOCK:
            done += 1
            if done % 100 == 0:
                print(f"  metadata {done}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(work, todo))
    META_CACHE.write_text(json.dumps(cached))
    return cached


def main() -> int:
    screen = json.loads(Path("data/value_screen.json").read_text())
    as_of = screen["as_of"]
    rows = screen["rows"]
    mapping = cik_map()
    ciks = {r["ticker"]: mapping[r["ticker"].upper()]
            for r in rows if r["ticker"].upper() in mapping}
    print(f"{len(ciks)} companies with a CIK; as of {as_of}", flush=True)

    meta = company_meta(ciks)
    out = {}
    for ticker, cik in ciks.items():
        facts = fetch_company(ticker, cik)
        if not facts:
            continue
        prof = profile(facts, as_of)
        prof.update({k: v for k, v in (meta.get(ticker) or {}).items()})
        out[ticker] = prof

    path = Path("data/dd_profiles.json")
    path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "as_of": as_of,
        "flag_text": FLAG_TEXT,
        "profiles": out,
    }, indent=1, default=float))
    print(f"wrote {path}: {len(out)} profiles", flush=True)

    import collections
    flags = collections.Counter(f for p in out.values() for f in p["flags"])
    print("\nrisk flags across the universe:", flush=True)
    for f, n in flags.most_common():
        print(f"  {n:4d}  {FLAG_TEXT.get(f, f)}", flush=True)
    import numpy as np
    for key, label in (("revenue_cagr_5y", "revenue CAGR"),
                       ("operating_margin_latest", "operating margin"),
                       ("roe_mean_5y", "ROE (5y mean)"),
                       ("share_count_cagr_5y", "share count CAGR")):
        vals = [p[key] for p in out.values() if p.get(key) is not None]
        if vals:
            print(f"  median {label:22s} {np.median(vals):+.1%}  "
                  f"(n={len(vals)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
