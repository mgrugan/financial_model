#!/usr/bin/env python3
"""Today's Graham/Burry screen across the small-cap universe.

Uses the same point-in-time machinery as the backtest, with the as-of date set
to now, so what the screen shows is built exactly the way the tested signal was
built. Every ranking here has a measured track record in data/value_factors.json
rather than being asserted.
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

from smallcaps.data import fetch_history
from value.edgar import cik_map, fetch_company
from value.factors import COMPOSITES
from value.metrics import GRAHAM_CRITERIA, compute, graham_scorecard
from value.panel import EV_EXCLUDED_SECTORS, _earnings_history, _price_on
from value.pit import snapshot


def main() -> int:
    inv = json.loads(Path("data/smallcap_inventory.json").read_text())
    rows_meta = [r for r in inv["rows"] if r["ok"]]
    mapping = cik_map()
    as_of = dt.date.today().isoformat()

    rows = []
    for meta in rows_meta:
        ticker = meta["ticker"]
        frame = fetch_history(ticker)
        cik = mapping.get(ticker.upper())
        facts = fetch_company(ticker, cik) if cik else None
        if frame is None or frame.empty or facts is None:
            continue
        raw = _price_on(frame, as_of, "raw_close")
        if raw is None:
            continue
        snap = snapshot(facts, as_of)
        if snap is None:
            continue
        shares = snap.get("shares") or snap.get("shares_diluted")
        metrics = compute(snap, raw, shares)
        if not metrics:
            continue
        card = graham_scorecard(metrics, _earnings_history(facts, as_of))
        sector = meta.get("sector", "")
        rows.append({
            "ticker": ticker,
            "name": meta.get("name", ""),
            "sector": sector,
            "ev_applicable": sector not in EV_EXCLUDED_SECTORS,
            "staleness_days": snap.get("_staleness_days"),
            "latest_period": snap.get("_latest_period"),
            "graham_score": card["graham_score"],
            **{k: v for k, v in metrics.items()},
            **{f"chk_{k}": v for k, v in card["checks"].items()},
        })

    frame = pd.DataFrame(rows)
    print(f"screened {len(frame)} companies as of {as_of}", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        frame["ncav_yield"] = frame["ncav"] / frame["market_cap"]

    # Percentile ranks across today's cross-section, matching the backtest.
    for col in ["earnings_yield", "book_yield", "graham_ratio",
                "ebitda_ev_yield", "fcf_ev_yield"]:
        gate = {"earnings_yield": "eps", "book_yield": "bvps",
                "ebitda_ev_yield": "ebitda", "fcf_ev_yield": "fcf"}.get(col)
        series = frame[col]
        if gate and gate in frame:
            series = series.where(frame[gate] > 0)
        if col in ("ebitda_ev_yield", "fcf_ev_yield"):
            series = series.where(frame["ev_applicable"])
        frame[f"pct_{col}"] = series.rank(pct=True)

    for name, parts in COMPOSITES.items():
        cols = [f"pct_{c}" for c in parts if f"pct_{c}" in frame]
        stacked = frame[cols]
        enough = stacked.notna().sum(axis=1) >= max(1, (len(parts) + 1) // 2)
        frame[name] = stacked.mean(axis=1).where(enough)

    out = Path("data/value_screen.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "as_of": as_of,
        "n": int(len(frame)),
        "criteria": [{"key": k, "label": v} for k, v in GRAHAM_CRITERIA],
        "rows": json.loads(frame.to_json(orient="records")),
    }, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    top = frame.dropna(subset=["graham_composite"]).nlargest(15, "graham_composite")
    print(f"\ncheapest 15 by Graham composite (as of {as_of}):", flush=True)
    print(f"{'tk':6s} {'company':28s} {'P/E':>7s} {'P/B':>6s} {'EV/EBITDA':>10s} {'G-score':>8s}",
          flush=True)
    for _, r in top.iterrows():
        pe = r["pe"] if pd.notna(r["pe"]) else float("nan")
        ev = r["ev_ebitda"] if pd.notna(r["ev_ebitda"]) else float("nan")
        print(f"{r['ticker']:6s} {str(r['name'])[:28]:28s} {pe:7.1f} {r['pb']:6.2f} "
              f"{ev:10.1f} {int(r['graham_score']):5d}/8", flush=True)

    nn = frame[frame["is_net_net"].fillna(False)]
    print(f"\nGraham net-nets today (price < 2/3 NCAV): {len(nn)}"
          + (": " + ", ".join(nn["ticker"]) if len(nn) else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
