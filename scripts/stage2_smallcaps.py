#!/usr/bin/env python3
"""Stage 2: the full eleven-model stack on whatever cleared the stage-1 screen.

Read the framing carefully, because it is easy to overclaim here. Stage 2 runs
on the *same* out-of-sample window stage 1 used, so it is **not** independent
confirmation -- it cannot be, without spending history we do not have. What it
does provide is three things stage 1 cannot:

1. **Cross-model robustness.** A real signal should show up in more than the one
   specification that happened to find it. If only the logistic regression sees
   it and ten other models do not, that is evidence about the finding, not about
   the other ten models.
2. **A distribution-free interval.** The stage-1 p-value leans on Hanley-McNeil
   with an effective-count correction. The stationary block bootstrap here
   resamples in blocks, so it keeps the actual serial dependence of both the
   predictions and the returns.
3. **Sub-period stability.** An edge that lives entirely in the first half of the
   window and vanishes in the second is a regime artefact, not an edge.

Nothing here is allowed to *promote* a ticker. It can only demote one.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from btcmodels.backtest import _auc, run_walk_forward
from smallcaps.data import EQUITY_HORIZONS, fetch_history
from smallcaps.screen import FDR_ALPHA, block_bootstrap_auc_ci, screen_ticker

logging.basicConfig(level=logging.ERROR, format="%(message)s")


def sub_period_aucs(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    half = len(p) // 2
    if half < 40:
        return float("nan"), float("nan")
    return _auc(y[:half], p[:half]), _auc(y[half:], p[half:])


def main() -> int:
    screen = json.loads(Path("data/smallcap_screen.json").read_text())
    survivors = sorted({r["ticker"] for r in screen["rows"] if r.get("edge")})
    out_path = Path("data/smallcap_stage2.json")

    if not survivors:
        out_path.write_text(json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "survivors": [],
            "note": ("No ticker cleared the stage-1 screen, so the full stack was not run. "
                     "This is the compute-saving design working as intended: eleven models "
                     "over 500 names would have been roughly 500x the cost of the screen "
                     "that showed there was nothing to promote."),
            "rows": [],
        }, indent=1))
        print("no survivors — stage 2 skipped", flush=True)
        return 0

    print(f"stage 2 on {len(survivors)} survivor(s): {', '.join(survivors)}", flush=True)
    rows = []
    for ticker in survivors:
        frame = fetch_history(ticker)
        if frame is None:
            continue
        t0 = time.time()

        # Re-run the stage-1 model keeping predictions, for the bootstrap and
        # the sub-period split.
        for r in screen_ticker(ticker, frame, keep_predictions=True):
            if r.get("status") != "ok":
                continue
            p, y = r.pop("_p"), r.pop("_y")
            lo, hi = block_bootstrap_auc_ci(y, p, r["horizon_bars"])
            first, second = sub_period_aucs(p, y)
            rows.append({
                "ticker": ticker, "model": "Logistic (stage-1)", "horizon": r["horizon"],
                "auc": r["auc"], "auc_ci_low": lo, "auc_ci_high": hi,
                "auc_first_half": first, "auc_second_half": second,
                "strategy_sharpe": r.get("strategy_sharpe"),
                "confirms": bool(np.isfinite(lo) and lo > 0.5),
            })

        try:
            results = run_walk_forward(frame, EQUITY_HORIZONS, backtest_days=1512,
                                       min_train=756, refit_every=63)
        except Exception as exc:
            print(f"  {ticker}: full stack failed ({exc})", flush=True)
            continue

        for model_key, per_horizon in results.items():
            for horizon_key, res in per_horizon.items():
                rows.append({
                    "ticker": ticker, "model": res.model_name, "horizon": horizon_key,
                    "auc": res.auc,
                    "auc_ci_low": getattr(res, "auc_ci_low", float("nan")),
                    "auc_ci_high": getattr(res, "auc_ci_high", float("nan")),
                    "strategy_sharpe": res.strategy_sharpe,
                    "confirms": bool(getattr(res, "auc_beats_chance", False)),
                })
        print(f"  {ticker}: done in {time.time()-t0:.0f}s", flush=True)

    confirmed = sum(1 for r in rows if r["confirms"])
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "survivors": survivors,
        "note": (f"{len(survivors)} ticker(s) cleared stage 1 at a {FDR_ALPHA:.0%} "
                 f"false-discovery rate and were re-run through all eleven models on the "
                 f"same out-of-sample window. {confirmed} of {len(rows)} model-horizon cells "
                 f"have a confidence interval excluding 0.5. This is a robustness check, "
                 f"not independent confirmation: it reuses the window stage 1 already saw, "
                 f"so it can demote a finding but never promote one."),
        "rows": rows,
    }, indent=1))
    print(f"wrote {out_path}: {len(rows)} cells, {confirmed} confirming", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
