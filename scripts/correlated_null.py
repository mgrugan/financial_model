#!/usr/bin/env python3
"""Null distribution of the *cross-sectional* statistics, under real dependence.

Stage 1 answers the per-ticker question and Benjamini-Hochberg handles the
multiplicity. Neither speaks to the aggregate: the real universe's mean AUC came
out at 0.5029, and with 1,064 tests treated as independent that is a four-sigma
result. It is not one. The 532 names correlate ~0.28 pairwise, so the mean's
standard error is governed by that correlation, not by the sample count.

This script measures the null directly rather than deriving it: many independent
cohorts of correlated random walks, each pushed through the identical screen, so
we can see how far a cohort mean AUC wanders when there is nothing to find.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.screen import correlated_universe, screen_ticker

N_COHORTS = 14
NAMES_PER_COHORT = 22


def run_cohort(seed: int) -> dict:
    frames = correlated_universe(seed, NAMES_PER_COHORT)
    aucs = []
    for i, frame in enumerate(frames):
        for row in screen_ticker(f"C{seed:02d}N{i:02d}", frame):
            if row.get("status") == "ok":
                aucs.append(row["auc"])
    if not aucs:
        return {}
    return {"seed": seed, "n": len(aucs),
            "mean_auc": float(np.mean(aucs)), "sd_auc": float(np.std(aucs, ddof=1))}


def main() -> int:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        cohorts = [c for c in pool.map(run_cohort, range(N_COHORTS)) if c]

    means = np.array([c["mean_auc"] for c in cohorts])
    sds = np.array([c["sd_auc"] for c in cohorts])

    screen = json.loads(Path("data/smallcap_screen.json").read_text())
    real = [r for r in screen["rows"] if r.get("status") == "ok"]
    real_mean = float(np.mean([r["auc"] for r in real]))
    real_sd = float(np.std([r["auc"] for r in real], ddof=1))

    null_sd_of_mean = float(np.std(means, ddof=1))
    z = (real_mean - 0.5) / null_sd_of_mean if null_sd_of_mean > 0 else float("nan")
    naive_se = real_sd / np.sqrt(len(real))
    naive_z = (real_mean - 0.5) / naive_se

    out = {
        "n_cohorts": len(cohorts), "names_per_cohort": NAMES_PER_COHORT,
        "rho": 0.278,
        "cohort_means": means.tolist(),
        "null_mean_of_means": float(means.mean()),
        "null_sd_of_cohort_mean": null_sd_of_mean,
        "null_mean_cross_sectional_sd": float(sds.mean()),
        "real_mean_auc": real_mean,
        "real_cross_sectional_sd": real_sd,
        "naive_se": float(naive_se), "naive_z": float(naive_z),
        "dependence_aware_z": float(z),
        "elapsed_s": time.time() - t0,
    }
    Path("data/smallcap_null.json").write_text(json.dumps(out, indent=1))

    print(f"cohorts: {len(cohorts)} x {NAMES_PER_COHORT} correlated random walks")
    print(f"  null cohort mean AUC   {means.mean():.4f}  sd across cohorts {null_sd_of_mean:.4f}")
    print(f"  null cross-sectional sd of AUC   {sds.mean():.4f}")
    print(f"  REAL mean AUC          {real_mean:.4f}   cross-sectional sd {real_sd:.4f}")
    print()
    print(f"  naive z (assumes 1,064 independent tests):  {naive_z:+.2f}  "
          f"<- what an incorrect analysis would report")
    print(f"  dependence-aware z (measured null):         {z:+.2f}")
    print(f"  elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
