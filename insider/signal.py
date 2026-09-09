"""Point-in-time insider-buying metrics.

The literature on insider trading is fairly consistent about which part of the
data carries information, and the metrics here follow it rather than inventing
new ones:

* **Open-market purchases only.** Grants, vesting and option exercises are not
  decisions to buy at the current price.
* **Distinct buyers matters more than dollars.** Several officers buying
  independently in the same window -- "cluster buying" -- is the form of the
  signal that has held up best, while one large purchase by one person is often
  a financing or control transaction wearing a Form 4.
* **Sales are near-useless as a bearish signal.** Insiders sell to diversify,
  to pay tax, and because a window opened. Net buying is reported for context
  but the primary metrics are buy-side only.

Everything is keyed on ``filed``, never on the trade date, so a rebalance on
date T sees only what a reader of EDGAR could have seen on T. The distinction is
small here -- Form 4 is due within two business days -- but it is free to get
right and the same code path is reused where the lag is long.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 180


class InsiderIndex:
    """Transactions grouped by issuer, sorted by filing date for fast slicing."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.by_cik: dict[str, pd.DataFrame] = {}
        if frame is None or frame.empty:
            return
        work = frame.dropna(subset=["filed"]).sort_values("filed")
        for cik, group in work.groupby("cik"):
            self.by_cik[str(cik)] = group.reset_index(drop=True)

    def window(self, cik: str, as_of: str, lookback_days: int) -> pd.DataFrame:
        group = self.by_cik.get(str(cik))
        if group is None or group.empty:
            return group if group is not None else pd.DataFrame()
        start = (dt.date.fromisoformat(as_of)
                 - dt.timedelta(days=lookback_days)).isoformat()
        filed = group["filed"].to_numpy()
        lo = int(np.searchsorted(filed, start, side="left"))
        hi = int(np.searchsorted(filed, as_of, side="right"))
        return group.iloc[lo:hi]

    def metrics(self, cik: str, as_of: str, market_cap: float | None = None,
                lookback_days: int = LOOKBACK_DAYS) -> dict[str, Any]:
        empty = {
            "insider_buyers": 0, "insider_buy_count": 0, "insider_buy_value": 0.0,
            "insider_sell_value": 0.0, "insider_net_value": 0.0,
            "insider_buy_ratio": np.nan, "insider_buy_to_mcap": np.nan,
            "insider_any_buy": False, "insider_cluster_buy": False,
            "insider_has_filings": False,
        }
        win = self.window(cik, as_of, lookback_days)
        if win is None or win.empty:
            return empty

        buys = win[win["is_buy"]]
        sells = win[win["is_sell"]]
        buy_value = float(buys["value"].sum())
        sell_value = float(sells["value"].sum())
        buyers = int(buys["owner_cik"].nunique()) if len(buys) else 0

        total = buy_value + sell_value
        out = {
            "insider_buyers": buyers,
            "insider_buy_count": int(len(buys)),
            "insider_buy_value": buy_value,
            "insider_sell_value": sell_value,
            "insider_net_value": buy_value - sell_value,
            # Share of insider dollar flow that was buying. Undefined when there
            # was no insider activity at all -- which is different from "there
            # was activity and it was all selling", so it stays NaN rather than 0.
            "insider_buy_ratio": (buy_value / total) if total > 0 else np.nan,
            "insider_buy_to_mcap": (buy_value / market_cap)
            if market_cap and np.isfinite(market_cap) and market_cap > 0 else np.nan,
            "insider_any_buy": buyers > 0,
            "insider_cluster_buy": buyers >= 2,
            "insider_has_filings": True,
        }
        return out
