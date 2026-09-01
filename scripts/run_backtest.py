#!/usr/bin/env python3
"""Run the walk-forward backtest and cache the result to disk.

Kept as a standalone entry point so the expensive run can happen offline (or in
a Render cron job) and the web process only ever loads the cached artefact.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

warnings.filterwarnings("ignore")

from btcmodels import backtest, data                      # noqa: E402
from btcmodels.config import (                            # noqa: E402
    BACKTEST_DAYS,
    BACKTEST_REFIT_EVERY,
    CACHE_DIR,
    HORIZONS,
)

BACKTEST_CACHE = CACHE_DIR / "backtest.pkl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS)
    parser.add_argument("--refit-every", type=int, default=BACKTEST_REFIT_EVERY)
    parser.add_argument("--out", type=Path, default=BACKTEST_CACHE)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", type=Path, default=Path(__file__).resolve().parent.parent
                        / "data" / "backtest.json",
                        help="also write a committable JSON artefact here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    daily = data.load_daily()
    start = time.time()
    results = backtest.run_walk_forward(
        daily, HORIZONS, backtest_days=args.days, refit_every=args.refit_every,
        progress=None if args.quiet else (lambda msg, frac: print(f"  {msg}", flush=True)),
    )
    elapsed = time.time() - start

    payload = {
        "results": results,
        "generated_at": time.time(),
        "elapsed_seconds": elapsed,
        "backtest_days": args.days,
        "refit_every": args.refit_every,
        "data_end": str(daily.index[-1].date()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(payload, fh)

    if args.json:
        import json as _json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_json.dumps({
            **{k: v for k, v in payload.items() if k != "results"},
            "results": {mk: {hk: r.to_dict() for hk, r in per.items()}
                        for mk, per in results.items()},
        }, separators=(",", ":")))
        print(f"json artefact -> {args.json}")

    print(f"\nbacktest complete in {elapsed / 60:.1f} min -> {args.out}")
    print(backtest.summary_frame(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
