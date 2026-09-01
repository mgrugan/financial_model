#!/usr/bin/env python3
"""Test every Graham/Burry factor on 6-month forward returns.

Reports, for each factor, the long-short quintile spread per rebalance and the
rank information coefficient, with inference done on the 28-period series rather
than the 13,000 stock-periods. Adds a placebo arm of random signals so the
pipeline's own false-positive rate is measured, and a Benjamini-Hochberg
correction across the whole family of factor tests.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.screen import benjamini_hochberg
from value.factors import FACTORS, Factor, prepare, run_factor

N_PLACEBO = 40
FDR_ALPHA = 0.10


def main() -> int:
    panel = prepare(pd.read_pickle("data/value_panel.pkl"))
    print(f"panel: {len(panel)} stock-periods, {panel['date'].nunique()} rebalances, "
          f"{panel['ticker'].nunique()} companies", flush=True)

    t0 = time.time()
    # Three modes. The third is the one that matters: it removes both the sector
    # tilt and the size artifact, so what is left is "cheap for its industry,
    # among companies of similar size".
    MODES = [(False, False), (True, False), (True, True)]
    results = []
    for factor in FACTORS:
        for neutral, size_neutral in MODES:
            res = run_factor(panel, factor, sector_neutral=neutral,
                             size_neutral=size_neutral)
            if res.get("status") == "ok":
                results.append(res)
    print(f"ran {len(results)} factor tests in {time.time()-t0:.0f}s", flush=True)

    # Placebo: signals with no information, through identical code.
    rng = np.random.default_rng(20260901)
    placebo = []
    work = panel.copy()
    for i in range(N_PLACEBO):
        work["_noise"] = rng.normal(size=len(work))
        res = run_factor(work, Factor(f"placebo{i}", "placebo", "_noise",
                                      True, False, "placebo"),
                         sector_neutral=bool(i % 2), size_neutral=bool(i % 3 == 0))
        if res.get("status") == "ok":
            res.pop("periods", None)
            placebo.append(res)
    print(f"ran {len(placebo)} placebo tests", flush=True)

    # Multiple-testing correction across the real family.
    q = benjamini_hochberg(np.array([r["p_spread"] for r in results]))
    for r, qv in zip(results, q):
        r["q_spread"] = float(qv) if np.isfinite(qv) else None
        r["significant"] = bool(np.isfinite(qv) and qv < FDR_ALPHA)
    qp = benjamini_hochberg(np.array([r["p_spread"] for r in placebo]))
    for r, qv in zip(placebo, qp):
        r["significant"] = bool(np.isfinite(qv) and qv < FDR_ALPHA)

    print(f"\n{'factor':26s} {'mode':8s} {'spread':>8s} {'t':>6s} {'p':>7s} "
          f"{'q':>6s} {'IC':>7s} {'hit':>5s} {'MDE/yr':>7s}", flush=True)
    for r in sorted(results, key=lambda x: x["p_spread"]):
        mode = ("sec+size" if r["size_neutral"]
                else "sector" if r["sector_neutral"] else "pooled")
        print(f"{r['label'][:26]:26s} {mode:8s} {r['mean_spread']:+7.2%} "
              f"{r['t_spread']:+6.2f} {r['p_spread']:7.3f} "
              f"{(r['q_spread'] if r['q_spread'] is not None else float('nan')):6.3f} "
              f"{r['mean_ic']:+7.3f} {r['hit_rate']:5.0%} "
              f"{r['mde_annualised']:7.1%}", flush=True)

    n_sig = sum(r["significant"] for r in results)
    n_plac = sum(r["significant"] for r in placebo)
    raw_real = sum(1 for r in results if r["p_spread"] < 0.05)
    raw_plac = sum(1 for r in placebo if r["p_spread"] < 0.05)
    print(f"\nreal:    {raw_real}/{len(results)} raw p<0.05, {n_sig} survive BH q<{FDR_ALPHA}",
          flush=True)
    print(f"placebo: {raw_plac}/{len(placebo)} raw p<0.05, {n_plac} survive BH", flush=True)

    out = Path("data/value_factors.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fdr_alpha": FDR_ALPHA,
        "n_stock_periods": int(len(panel)),
        "n_rebalances": int(panel["date"].nunique()),
        "n_companies": int(panel["ticker"].nunique()),
        "universe_return_per_period": float(panel.groupby("date")["fwd_return"].mean().mean()),
        "results": results,
        "placebo": placebo,
        "summary": {"n_tests": len(results), "raw_hits": raw_real, "n_significant": n_sig,
                    "placebo_tests": len(placebo), "placebo_raw_hits": raw_plac,
                    "placebo_significant": n_plac},
    }, indent=1, default=float))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
