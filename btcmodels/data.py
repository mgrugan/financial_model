"""Market data access.

Yahoo Finance is read straight off the public chart endpoint with ``requests``
rather than through ``yfinance``: the library's curl_cffi transport ignores
proxy CA configuration in container environments, and the raw endpoint gives us
retry/host-rotation control that matters when Yahoo starts handing out 429s.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    DAILY_CACHE_TTL,
    HOURLY_CACHE_TTL,
    TICKER,
    YAHOO_HISTORY_START,
    YAHOO_HOSTS,
    YAHOO_USER_AGENT,
)

log = logging.getLogger(__name__)

_SESSION_LOCK = threading.Lock()
_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": YAHOO_USER_AGENT,
                    "Accept": "application/json,text/plain,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            _SESSION = s
        return _SESSION


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta.json"


def _read_cache(key: str, ttl: int) -> pd.DataFrame | None:
    path, meta = _cache_path(key), _meta_path(key)
    if not (path.exists() and meta.exists()):
        return None
    try:
        stamp = json.loads(meta.read_text()).get("fetched_at", 0)
        if time.time() - stamp > ttl:
            return None
        return pd.read_pickle(path)
    except Exception as exc:  # pragma: no cover - cache is best effort
        log.warning("cache read failed for %s: %s", key, exc)
        return None


def _read_stale_cache(key: str) -> pd.DataFrame | None:
    """Last-resort read that ignores the TTL, used when the network is down."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(_cache_path(key))
        _meta_path(key).write_text(json.dumps({"fetched_at": time.time(), "rows": len(df)}))
    except Exception as exc:  # pragma: no cover
        log.warning("cache write failed for %s: %s", key, exc)


def cache_age_seconds(key: str) -> float | None:
    meta = _meta_path(key)
    if not meta.exists():
        return None
    try:
        return time.time() - json.loads(meta.read_text()).get("fetched_at", 0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Yahoo chart endpoint
# ---------------------------------------------------------------------------
def _yahoo_chart(params: dict[str, Any], attempts: int = 6) -> dict:
    """Hit the Yahoo chart endpoint, rotating hosts and backing off on 429."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        host = YAHOO_HOSTS[attempt % len(YAHOO_HOSTS)]
        url = f"https://{host}/v8/finance/chart/{TICKER}"
        try:
            resp = _session().get(url, params=params, timeout=30)
            if resp.status_code == 429:
                raise RuntimeError("rate limited (429)")
            resp.raise_for_status()
            payload = resp.json()
            err = (payload.get("chart") or {}).get("error")
            if err:
                raise RuntimeError(f"yahoo error: {err}")
            result = (payload.get("chart") or {}).get("result")
            if not result:
                raise RuntimeError("empty chart result")
            return result[0]
        except Exception as exc:
            last_error = exc
            sleep = min(2 ** attempt, 16) * 0.5
            log.warning("yahoo attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, attempts, exc, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Yahoo request failed after {attempts} attempts: {last_error}")


def _chart_to_frame(result: dict) -> pd.DataFrame:
    stamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True),
    )
    frame.index.name = "timestamp"
    frame = frame.astype("float64")
    # Yahoo emits nulls for gaps; forward-fill prices, zero-fill volume.
    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].ffill()
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.dropna(subset=["close"])


def load_daily(force: bool = False) -> pd.DataFrame:
    """Full daily OHLCV history for BTC-USD, oldest first."""
    key = f"{TICKER}_1d"
    if not force:
        cached = _read_cache(key, DAILY_CACHE_TTL)
        if cached is not None:
            return cached

    start = int(pd.Timestamp(YAHOO_HISTORY_START, tz="UTC").timestamp())
    end = int(time.time()) + 86_400
    try:
        result = _yahoo_chart(
            {"period1": start, "period2": end, "interval": "1d", "includePrePost": "false"}
        )
        frame = _chart_to_frame(result)
        # The final bar of a 24/7 market is the still-forming session; keep it
        # but mark it so models can decide whether to trust it.
        frame["is_partial"] = False
        meta_time = result.get("meta", {}).get("regularMarketTime")
        if meta_time is not None and len(frame):
            last_day = pd.Timestamp(meta_time, unit="s", tz="UTC").normalize()
            frame.loc[frame.index.normalize() == last_day, "is_partial"] = True
        _write_cache(key, frame)
        return frame
    except Exception as exc:
        stale = _read_stale_cache(key)
        if stale is not None:
            log.error("daily fetch failed (%s); serving stale cache", exc)
            return stale
        raise


def load_hourly(days: int = 720, force: bool = False) -> pd.DataFrame:
    """Hourly OHLCV, used for intraday realised-volatility estimates."""
    key = f"{TICKER}_1h"
    if not force:
        cached = _read_cache(key, HOURLY_CACHE_TTL)
        if cached is not None:
            return cached
    try:
        result = _yahoo_chart({"range": f"{min(days, 730)}d", "interval": "1h"})
        frame = _chart_to_frame(result)
        _write_cache(key, frame)
        return frame
    except Exception as exc:
        stale = _read_stale_cache(key)
        if stale is not None:
            log.error("hourly fetch failed (%s); serving stale cache", exc)
            return stale
        log.error("hourly fetch failed and no cache available: %s", exc)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def spot_price() -> tuple[float, dt.datetime]:
    """Latest traded price and its timestamp, straight from the chart meta."""
    result = _yahoo_chart({"range": "1d", "interval": "1m"})
    meta = result.get("meta", {})
    price = float(meta.get("regularMarketPrice"))
    stamp = dt.datetime.fromtimestamp(int(meta.get("regularMarketTime", time.time())), dt.UTC)
    return price, stamp


def market_snapshot() -> dict[str, Any]:
    """Headline numbers for the dashboard banner."""
    daily = load_daily()
    close = daily["close"]
    last = float(close.iloc[-1])

    def pct_change(periods: int) -> float | None:
        if len(close) <= periods:
            return None
        return float(last / close.iloc[-1 - periods] - 1.0) * 100.0

    logret = np.diff(np.log(close.to_numpy()))
    return {
        "price": last,
        "as_of": close.index[-1].to_pydatetime(),
        "change_24h": pct_change(1),
        "change_7d": pct_change(7),
        "change_30d": pct_change(30),
        "high_52w": float(close.iloc[-365:].max()) if len(close) >= 365 else float(close.max()),
        "low_52w": float(close.iloc[-365:].min()) if len(close) >= 365 else float(close.min()),
        "realised_vol_30d": float(np.std(logret[-30:], ddof=1) * np.sqrt(365) * 100.0),
        "realised_vol_90d": float(np.std(logret[-90:], ddof=1) * np.sqrt(365) * 100.0),
        "history_start": close.index[0].to_pydatetime(),
        "n_bars": int(len(close)),
        "data_age_seconds": cache_age_seconds(f"{TICKER}_1d"),
    }
