#!/usr/bin/env python3
"""Stage-1 screen over the whole small-cap universe, plus a placebo arm.

Writes ``data/smallcap_screen.json``: one row per (ticker, horizon) with the
AUC, its effective-sample standard error, a p-value and a Benjamini-Hochberg
q-value computed across the entire family of tests. The placebo rows come from
synthetic random walks pushed through the identical code path, and exist so the
false-positive rate of the screen is measured rather than assumed.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.data import fetch_history
from smallcaps.screen import FDR_ALPHA, benjamini_hochberg, screen_ticker, synthetic_frame

logging.basicConfig(level=logging.ERROR, format="%(message)s")

N_PLACEBO = 250


def _real(ticker: str) -> list[dict]:
    try:
        frame = fetch_history(ticker)
        if frame is None or frame.empty:
            return []
        return screen_ticker(ticker, frame)
    except Exception as exc:
        return [{"ticker": ticker, "horizon": "?", "status": f"error: {type(exc).__name__}: {exc}"}]


def _placebo(seed: int) -> list[dict]:
    try:
        rows = screen_ticker(f"PLACEBO{seed:04d}", synthetic_frame(seed))
        for r in rows:
            r["placebo"] = True
        return rows
    except Exception:
        return []


def _annotate(rows: list[dict], label: str) -> list[dict]:
    ok = [r for r in rows if r.get("status") == "ok"]
    q = benjamini_hochberg(np.array([r["p_value"] for r in ok], dtype="float64"))
    for r, qv in zip(ok, q):
        r["q_value"] = float(qv) if np.isfinite(qv) else None
        r["edge"] = bool(np.isfinite(qv) and qv < FDR_ALPHA)
        r["family"] = label
        r["family_size"] = len(ok)
    for r in rows:
        r.setdefault("q_value", None)
        r.setdefault("edge", False)
        r.setdefault("family", label)
    return rows


def main() -> int:
    inventory = json.loads(Path("data/smallcap_inventory.json").read_text())
    tickers = [r["ticker"] for r in inventory["rows"] if r["ok"]]
    print(f"screening {len(tickers)} tradeable tickers + {N_PLACEBO} placebos", flush=True)

    t0 = time.time()
    real: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, rows in enumerate(pool.map(_real, tickers, chunksize=4), 1):
            real.extend(rows)
            if i % 50 == 0:
                print(f"  real {i}/{len(tickers)}  ({time.time()-t0:.0f}s)", flush=True)

    placebo: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, rows in enumerate(pool.map(_placebo, range(N_PLACEBO), chunksize=4), 1):
            placebo.extend(rows)
            if i % 50 == 0:
                print(f"  placebo {i}/{N_PLACEBO}  ({time.time()-t0:.0f}s)", flush=True)

    real = _annotate(real, "real")
    placebo = _annotate(placebo, "placebo")

    def summarise(rows: list[dict], label: str) -> dict:
        ok = [r for r in rows if r.get("status") == "ok"]
        aucs = np.array([r["auc"] for r in ok], dtype="float64")
        pvals = np.array([r["p_value"] for r in ok], dtype="float64")
        greens = [r for r in ok if r["edge"]]
        raw_hits = int(np.sum(pvals < 0.05))
        print(f"\n=== {label} ===", flush=True)
        print(f"  tests            {len(ok)}", flush=True)
        print(f"  mean AUC         {np.nanmean(aucs):.4f}   sd {np.nanstd(aucs, ddof=1):.4f}", flush=True)
        print(f"  raw p<0.05       {raw_hits}  ({raw_hits/max(len(ok),1):.1%})   "
              f"expected under null ~{0.05*len(ok):.0f}", flush=True)
        print(f"  survive BH q<{FDR_ALPHA:g}  {len(greens)}", flush=True)
        for g in sorted(greens, key=lambda r: r["q_value"])[:15]:
            print(f"     {g['ticker']:8s} {g['horizon']:3s} AUC {g['auc']:.4f} "
                  f"p={g['p_value']:.2e} q={g['q_value']:.4f}", flush=True)
        return {
            "n_tests": len(ok),
            "mean_auc": float(np.nanmean(aucs)),
            "sd_auc": float(np.nanstd(aucs, ddof=1)),
            "raw_hits_p05": raw_hits,
            "n_edge": len(greens),
        }

    summary = {"real": summarise(real, "REAL SMALL CAPS"),
               "placebo": summarise(placebo, "PLACEBO (random walks)")}

    out = Path("data/smallcap_screen.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fdr_alpha": FDR_ALPHA,
        "summary": summary,
        "rows": real,
        "placebo_rows": placebo,
    }, indent=1))
    print(f"\nwrote {out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
