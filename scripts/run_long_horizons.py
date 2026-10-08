#!/usr/bin/env python3
"""Run every factor at 6-month, 1-year and 5-year horizons.

The point of the exercise is the comparison, and the thing that changes across
the three columns is not really the signal -- it is how much evidence there is.
The rebalance grid is semi-annual, so a 1-year forward return overlaps its
neighbour by half and a 5-year return overlaps by nine tenths. Overlapping
observations are not independent draws, and the ordinary t-statistic treats
them as if they were.

Two corrections are therefore applied and both are reported: a Newey-West
standard error with the lag set to the actual overlap, and an explicit count of
independent windows. At five years that count is under three. Nothing can be
concluded from three observations, and the output says so rather than printing
a t-statistic and leaving the reader to assume it means what it usually means.
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
from value.panel import HORIZONS_MONTHS, independent_windows

FDR_ALPHA = 0.10
N_PLACEBO = 24
MIN_INDEPENDENT = 8          # below this, report but refuse to call anything significant


def main() -> int:
    panel = prepare(pd.read_pickle("data/value_panel.pkl"))
    print(f"panel: {len(panel):,} stock-periods, {panel['date'].nunique()} rebalances",
          flush=True)

    out: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "fdr_alpha": FDR_ALPHA, "horizons": {}}

    for hkey, months in HORIZONS_MONTHS.items():
        col = f"fwd_{hkey}"
        sub = panel[panel[col].notna()]
        n_dates = sub["date"].nunique()
        indep = independent_windows(months)
        inferential = indep >= MIN_INDEPENDENT

        print(f"\n=== {hkey} ({months} months) ===", flush=True)
        print(f"  {len(sub):,} stock-periods over {n_dates} rebalances; "
              f"{indep:.1f} independent windows "
              f"-> {'inference OK' if inferential else 'DESCRIPTIVE ONLY'}", flush=True)

        results = []
        for factor in FACTORS:
            for neutral, size_neutral in ((True, False), (True, True)):
                r = run_factor(sub, factor, sector_neutral=neutral,
                               size_neutral=size_neutral, return_col=col,
                               horizon_months=months, rebalance_months=6)
                if r.get("status") == "ok":
                    r.pop("periods", None)
                    results.append(r)

        rng = np.random.default_rng(7)
        work = sub.copy()
        placebo = []
        for i in range(N_PLACEBO):
            work["_noise"] = rng.normal(size=len(work))
            r = run_factor(work, Factor(f"pl{i}", "placebo", "_noise", True, False,
                                        "placebo"),
                           sector_neutral=True, size_neutral=bool(i % 2),
                           return_col=col, horizon_months=months, rebalance_months=6)
            if r.get("status") == "ok":
                r.pop("periods", None)
                placebo.append(r)

        q = benjamini_hochberg(np.array([r["p_spread"] for r in results]))
        for r, qv in zip(results, q):
            r["q_spread"] = float(qv) if np.isfinite(qv) else None
            # Significance is withheld entirely where the horizon has too few
            # independent windows to support it, however small the p-value.
            r["significant"] = bool(inferential and np.isfinite(qv) and qv < FDR_ALPHA)

        n_sig = sum(r["significant"] for r in results)
        raw = sum(1 for r in results if r["p_spread"] < 0.05)
        p_raw = sum(1 for r in placebo if r["p_spread"] < 0.05)
        print(f"  {raw}/{len(results)} raw p<0.05, {n_sig} survive BH "
              f"| placebo {p_raw}/{len(placebo)} raw", flush=True)

        top = sorted([r for r in results if r["size_neutral"]],
                     key=lambda r: -abs(r["t_spread"]))[:6]
        print(f"  {'factor (sector+size neutral)':34s} {'spread':>9s} {'t(naive)':>9s} "
              f"{'t(NW)':>7s} {'hit':>5s}", flush=True)
        for r in top:
            print(f"  {r['label'][:34]:34s} {r['mean_spread']:+8.2%} "
                  f"{r['t_spread_naive']:+9.2f} {r['t_spread_newey_west']:+7.2f} "
                  f"{r['hit_rate']:5.0%}", flush=True)

        out["horizons"][hkey] = {
            "months": months, "n_stock_periods": int(len(sub)),
            "n_rebalances": int(n_dates), "independent_windows": float(indep),
            "inferential": bool(inferential),
            "universe_return": float(sub.groupby("date")[col].mean().mean()),
            "results": results, "placebo": placebo,
            "summary": {"n_tests": len(results), "raw_hits": raw,
                        "n_significant": n_sig, "placebo_raw_hits": p_raw},
        }

    path = Path("data/value_factors_long.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
