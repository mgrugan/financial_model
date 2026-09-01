#!/usr/bin/env python3
"""Run the cross-sectional models over the value panel and record the result."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.screen import benjamini_hochberg
from value.factors import prepare
from value.models import model_builders, walk_forward

FDR_ALPHA = 0.10


def main() -> int:
    panel = prepare(pd.read_pickle("data/value_panel.pkl"))
    print(f"panel: {len(panel)} stock-periods over {panel['date'].nunique()} rebalances",
          flush=True)

    results = []
    for size_neutral in (False, True):
        print(f"\n--- {'size-neutral' if size_neutral else 'raw'} ---", flush=True)
        for key in model_builders():
            t0 = time.time()
            res = walk_forward(panel, key, size_neutral=size_neutral)
            res["seconds"] = time.time() - t0
            results.append(res)
            if res.get("status") == "ok":
                print(f"  {key:18s} spread {res['mean_spread']:+.2%} "
                      f"t={res['t_spread']:+.2f} | long-only "
                      f"{res['mean_excess_long_only']:+.2%} t={res['t_excess']:+.2f} "
                      f"| IC {res['mean_ic']:+.3f} | {res['n_periods']} periods "
                      f"({res['seconds']:.0f}s)", flush=True)
            else:
                print(f"  {key:18s} {res['status']}", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    q = benjamini_hochberg(np.array([r["p_spread"] for r in ok]))
    for r, qv in zip(ok, q):
        r["q_spread"] = float(qv) if np.isfinite(qv) else None
        r["significant"] = bool(np.isfinite(qv) and qv < FDR_ALPHA)

    n_sig = sum(r["significant"] for r in ok)
    print(f"\n{n_sig}/{len(ok)} models survive BH q<{FDR_ALPHA}", flush=True)

    out = Path("data/value_models.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fdr_alpha": FDR_ALPHA,
        "results": results,
    }, indent=1, default=float))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
