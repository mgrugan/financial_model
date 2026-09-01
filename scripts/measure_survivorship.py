#!/usr/bin/env python3
"""Measure the selection bias in a current-constituents universe.

Two independent readings of the same problem:

1. The universe's own return against the real small-cap indices over identical
   windows. Today's index members, run backwards, should beat the index that
   actually existed -- and by how much is the size of the bias.
2. Market capitalisation as a ranking factor. In an unbiased sample size is a
   modest, unreliable premium. Here it should be enormous and near-perfect,
   because being small in 2012 and still a member in 2026 *requires* having
   appreciated. A 100% hit rate is not a premium; it is the selection rule made
   visible.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcaps.data import fetch_history
from value.panel import _price_on

BENCHMARKS = [("IJR", "S&P SmallCap 600 (cap-weighted)"),
              ("IWM", "Russell 2000")]


def main() -> int:
    panel = pd.read_pickle("data/value_panel.pkl")
    dates = sorted(panel["date"].unique())
    universe = panel.groupby("date")["fwd_return"].mean()

    bench_rows = []
    for ticker, label in BENCHMARKS:
        frame = fetch_history(ticker, years=16)
        if frame is None or frame.empty:
            continue
        # A benchmark whose adjusted close never differs from its raw close has
        # had no dividend adjustment applied, so comparing the universe's total
        # return against it would overstate the gap by the dividend yield.
        if bool(np.allclose(frame["close"], frame["raw_close"])):
            print(f"  skipping {ticker}: no dividend adjustment in the feed", flush=True)
            continue
        got = {}
        for d in dates:
            fwd = (dt.date.fromisoformat(d) + dt.timedelta(days=int(6 * 30.44))).isoformat()
            a, b = _price_on(frame, d, "close"), _price_on(frame, fwd, "close")
            if a and b:
                got[d] = b / a - 1.0
        series = pd.Series(got)
        joined = pd.concat([universe.rename("u"), series.rename("b")], axis=1).dropna()
        diff = (joined["u"] - joined["b"]).to_numpy()
        t = float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff))))
        bench_rows.append({
            "ticker": ticker, "label": label, "n_periods": int(len(joined)),
            "universe_annual": float((1 + joined["u"].mean()) ** 2 - 1),
            "benchmark_annual": float((1 + joined["b"].mean()) ** 2 - 1),
            "gap_annual": float((1 + joined["u"].mean()) ** 2
                                / (1 + joined["b"].mean()) ** 2 - 1),
            "win_rate": float((joined["u"] > joined["b"]).mean()),
            "t_stat": t,
        })
        print(f"  {label}: universe {bench_rows[-1]['universe_annual']:+.2%}/yr vs "
              f"{bench_rows[-1]['benchmark_annual']:+.2%}/yr, gap "
              f"{bench_rows[-1]['gap_annual']:+.1%}/yr, won "
              f"{bench_rows[-1]['win_rate']:.0%} of periods", flush=True)

    # Size as a factor.
    rows = []
    for date, period in panel.groupby("date"):
        m = period["market_cap"].notna() & period["fwd_return"].notna()
        if m.sum() < 40:
            continue
        score = (-period.loc[m, "market_cap"]).rank(pct=True)
        buckets = pd.qcut(score.rank(method="first"), 5, labels=False) + 1
        means = period.loc[m, "fwd_return"].groupby(buckets).mean()
        rows.append({"date": date, "spread": float(means[5] - means[1]),
                     **{f"q{i}": float(means[i]) for i in means.index}})
    frame = pd.DataFrame(rows)
    spread = frame["spread"].to_numpy()
    t = float(np.mean(spread) / (np.std(spread, ddof=1) / np.sqrt(len(spread))))

    size = {
        "n_periods": int(len(spread)),
        "mean_spread": float(np.mean(spread)),
        "t_stat": t,
        "p_value": float(2.0 * stats.t.sf(abs(t), df=len(spread) - 1)),
        "hit_rate": float(np.mean(spread > 0)),
        "quintiles": [float(frame[f"q{i}"].mean()) for i in range(1, 6)],
    }
    print(f"\n  size factor: spread {size['mean_spread']:+.2%} per 6m, t={size['t_stat']:+.2f}, "
          f"positive in {size['hit_rate']:.0%} of {size['n_periods']} periods", flush=True)

    out = Path("data/value_survivorship.json")
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": bench_rows,
        "size_factor": size,
    }, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
