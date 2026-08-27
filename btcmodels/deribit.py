"""Deribit public-API client.

Deribit lists the deepest BTC option book, so it is the reference for live
strikes, mark IVs and the forward curve.  Only unauthenticated endpoints are
used -- no keys, no order placement.

Two conventions matter downstream:

* Option marks are quoted **in BTC** (inverse contracts, 1 BTC per contract),
  so a USD premium is ``mark_price * index_price``.
* Each expiry has its own forward (the ``underlying_price`` on the book
  summary), which is why the analytics use Black-76 on the forward rather than
  Black-Scholes on spot.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import DERIBIT_BASE, FALLBACK_RATE, OPTIONS_CACHE_TTL

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SESSION: requests.Session | None = None
_CACHE: dict[str, tuple[float, Any]] = {}

_EXPIRY_RE = re.compile(r"^BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}


def _session() -> requests.Session:
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            s = requests.Session()
            s.headers.update({"User-Agent": "btc-model-dashboard/1.0", "Accept": "application/json"})
            _SESSION = s
        return _SESSION


def _get(endpoint: str, params: dict | None = None, ttl: int = OPTIONS_CACHE_TTL,
         attempts: int = 3) -> Any:
    key = f"{endpoint}:{sorted((params or {}).items())}"
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = _session().get(DERIBIT_BASE + endpoint, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            result = payload["result"]
            with _LOCK:
                _CACHE[key] = (time.time(), result)
            return result
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 8) * 0.5)
    # Serve a stale entry rather than breaking the dashboard.
    with _LOCK:
        hit = _CACHE.get(key)
    if hit:
        log.error("deribit %s failed (%s); serving stale", endpoint, last)
        return hit[1]
    raise RuntimeError(f"Deribit {endpoint} failed: {last}")


def parse_instrument(name: str) -> dict | None:
    """``BTC-28AUG26-96000-C`` -> expiry / strike / option type."""
    match = _EXPIRY_RE.match(name)
    if not match:
        return None
    raw, strike, kind = match.groups()
    day = int(raw[:-5])
    month = _MONTHS[raw[-5:-2]]
    year = 2000 + int(raw[-2:])
    # Deribit options settle at 08:00 UTC.
    expiry = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return {
        "instrument": name,
        "expiry": expiry,
        "strike": float(strike),
        "option_type": "call" if kind == "C" else "put",
    }


def index_price() -> float:
    return float(_get("get_index_price", {"index_name": "btc_usd"}, ttl=60)["index_price"])


def futures_curve() -> pd.DataFrame:
    """Dated futures with their implied continuously-compounded carry rate."""
    rows = _get("get_book_summary_by_currency", {"currency": "BTC", "kind": "future"}, ttl=120)
    spot = index_price()
    now = dt.datetime.now(dt.UTC)
    records = []
    for row in rows:
        name = row["instrument_name"]
        mark = row.get("mark_price")
        if mark is None:
            continue
        if name.endswith("PERPETUAL"):
            records.append({"instrument": name, "expiry": None, "mark": float(mark),
                            "years": np.nan, "implied_rate": np.nan})
            continue
        parsed = re.match(r"^BTC-(\d{1,2}[A-Z]{3}\d{2})$", name)
        if not parsed:
            continue
        raw = parsed.group(1)
        expiry = dt.datetime(2000 + int(raw[-2:]), _MONTHS[raw[-5:-2]], int(raw[:-5]),
                             8, 0, tzinfo=dt.UTC)
        years = max((expiry - now).total_seconds() / (365.0 * 86_400), 1e-6)
        records.append({
            "instrument": name,
            "expiry": expiry,
            "mark": float(mark),
            "years": years,
            "implied_rate": float(np.log(float(mark) / spot) / years),
        })
    return pd.DataFrame(records).sort_values("years", na_position="last").reset_index(drop=True)


def implied_rate(default: float = FALLBACK_RATE) -> float:
    """Front-of-curve carry rate implied by the futures basis."""
    try:
        curve = futures_curve().dropna(subset=["implied_rate"])
        near = curve[curve["years"] > 2 / 365.0]
        if len(near):
            rate = float(near["implied_rate"].iloc[0])
            if -0.5 < rate < 1.0:
                return rate
    except Exception as exc:
        log.warning("implied rate unavailable (%s); using %.3f", exc, default)
    return default


def option_chain(max_dte: int | None = None, min_dte: int = 0) -> pd.DataFrame:
    """Live BTC option book: strikes, mark IVs, USD marks, open interest."""
    rows = _get("get_book_summary_by_currency", {"currency": "BTC", "kind": "option"})
    spot = index_price()
    now = dt.datetime.now(dt.UTC)

    records = []
    for row in rows:
        parsed = parse_instrument(row["instrument_name"])
        if parsed is None:
            continue
        dte = (parsed["expiry"] - now).total_seconds() / 86_400.0
        if dte < min_dte or (max_dte is not None and dte > max_dte):
            continue
        mark_btc = row.get("mark_price")
        if mark_btc is None:
            continue
        forward = row.get("underlying_price") or spot
        records.append({
            **parsed,
            "dte": dte,
            "years": max(dte / 365.0, 1e-8),
            "forward": float(forward),
            "mark_iv": (float(row["mark_iv"]) / 100.0) if row.get("mark_iv") else np.nan,
            "mark_btc": float(mark_btc),
            "mark_usd": float(mark_btc) * spot,
            "bid_usd": float(row["bid_price"]) * spot if row.get("bid_price") else np.nan,
            "ask_usd": float(row["ask_price"]) * spot if row.get("ask_price") else np.nan,
            "open_interest": float(row.get("open_interest") or 0.0),
            "volume_usd": float(row.get("volume_usd") or 0.0),
            "moneyness": parsed["strike"] / float(forward),
        })

    chain = pd.DataFrame(records)
    if chain.empty:
        return chain
    chain["index_price"] = spot
    return chain.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def chain_snapshot(max_dte: int, min_dte: int = 0) -> dict[str, Any]:
    """Chain plus the context the options page needs around it."""
    chain = option_chain(max_dte=max_dte, min_dte=min_dte)
    return {
        "chain": chain,
        "index_price": index_price(),
        "rate": implied_rate(),
        "fetched_at": dt.datetime.now(dt.UTC),
        "expiries": sorted(chain["expiry"].unique().tolist()) if len(chain) else [],
    }
