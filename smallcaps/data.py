"""Multi-ticker daily history from the Yahoo chart endpoint.

Two things differ from the Bitcoin loader this is modelled on.

**Adjustment.** Equities split and pay dividends; Bitcoin does not. The raw
``close`` series has a -50% step on the day of a 2:1 split, which is not a
return -- it is a units change. Left alone it would be the single largest
"move" in many of these histories and every model would spend its capacity
learning to predict corporate actions it can already see coming. We therefore
take Yahoo's ``adjclose`` as the price series and rescale open/high/low by the
same per-bar ratio, so the range-based features (ATR, Parkinson, Bollinger)
stay dimensionally consistent with it.

**Calendar.** A trading year is ~252 bars, not 365, and a week is 5 bars, not
7. Anything annualised downstream has to use the equity convention.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from btcmodels.config import CACHE_DIR, YAHOO_HOSTS, YAHOO_USER_AGENT

log = logging.getLogger(__name__)

SMALLCAP_CACHE = Path(CACHE_DIR) / "smallcaps"
SMALLCAP_CACHE.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_PER_YEAR = 252.0

# Equity horizons, in trading bars: one session and one trading week.
EQUITY_HORIZONS = {"1d": 1, "1w": 5}

_LOCAL = threading.local()
_THROTTLE = threading.Semaphore(6)      # Yahoo starts 429-ing above ~8 in flight


def _session() -> requests.Session:
    if getattr(_LOCAL, "session", None) is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": YAHOO_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        _LOCAL.session = s
    return _LOCAL.session


def _chart(ticker: str, params: dict[str, Any], attempts: int = 5) -> dict | None:
    last: Exception | None = None
    for attempt in range(attempts):
        host = YAHOO_HOSTS[attempt % len(YAHOO_HOSTS)]
        url = f"https://{host}/v8/finance/chart/{ticker}"
        try:
            with _THROTTLE:
                resp = _session().get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None                      # delisted or renamed; not retryable
            if resp.status_code == 429:
                raise RuntimeError("rate limited (429)")
            resp.raise_for_status()
            payload = resp.json()
            if (payload.get("chart") or {}).get("error"):
                return None
            result = (payload.get("chart") or {}).get("result")
            if not result:
                return None
            return result[0]
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20) * 0.5 * (1.0 + random.random()))
    log.warning("%s: chart failed after %d attempts (%s)", ticker, attempts, last)
    return None


def _to_frame(result: dict) -> pd.DataFrame | None:
    stamps = result.get("timestamp") or []
    if not stamps:
        return None
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True).normalize(),
    )
    frame.index.name = "timestamp"
    frame = frame.astype("float64")

    # Rescale the whole bar onto the split/dividend-adjusted price scale.
    adj_block = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    adj = adj_block.get("adjclose")
    if adj is not None:
        adjclose = pd.Series(np.asarray(adj, dtype="float64"), index=frame.index)
        ratio = (adjclose / frame["close"]).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.ffill().bfill().fillna(1.0)
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col] * ratio

    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].ffill()
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame["is_partial"] = False
    return frame


def _cache_file(ticker: str) -> Path:
    return SMALLCAP_CACHE / f"{ticker.replace('/', '_')}.pkl"


def fetch_history(ticker: str, years: int = 12, ttl: int = 86_400,
                  force: bool = False) -> pd.DataFrame | None:
    """Adjusted daily OHLCV for one ticker, disk-cached."""
    path = _cache_file(ticker)
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return _finalise(pd.read_pickle(path))
        except Exception:
            pass

    result = _chart(ticker, {"range": f"{years}y", "interval": "1d", "events": "div,splits"})
    if result is None:
        return None
    frame = _to_frame(result)
    if frame is None or frame.empty:
        return None
    try:
        frame.to_pickle(path)                 # cache the raw adjusted series
    except Exception as exc:                                  # pragma: no cover
        log.warning("cache write failed for %s: %s", ticker, exc)
    return _finalise(frame)


def _finalise(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Apply history cleaning on read, so the rule is not frozen into the cache."""
    cleaned, info = clean_history(frame)
    if cleaned.empty:
        return None
    cleaned = cleaned.copy()
    cleaned.attrs.update(info)
    return cleaned


# ---------------------------------------------------------------------------
# Bulk download
# ---------------------------------------------------------------------------
def download_many(tickers: Iterable[str], years: int = 12, workers: int = 6,
                  progress: Any = None) -> dict[str, pd.DataFrame]:
    """Fetch many tickers concurrently. Failures are dropped, never raised."""
    tickers = list(tickers)
    out: dict[str, pd.DataFrame] = {}
    done = 0
    lock = threading.Lock()

    def work(ticker: str) -> None:
        nonlocal done
        frame = fetch_history(ticker, years=years)
        with lock:
            done += 1
            if frame is not None and not frame.empty:
                out[ticker] = frame
            if progress and done % 25 == 0:
                progress(f"  downloaded {done}/{len(tickers)} ({len(out)} ok)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, tickers))
    if progress:
        progress(f"  downloaded {done}/{len(tickers)} ({len(out)} ok)")
    return out


# ---------------------------------------------------------------------------
# History cleaning
# ---------------------------------------------------------------------------
def clean_history(frame: pd.DataFrame, min_dollar_volume: float = 200_000.0,
                  window: int = 63) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trim the pre-liquidity era and flag round-trip bad ticks.

    Small caps routinely carry years of near-untraded history before they were
    really listed: zero-volume bars with a frozen price, then a step change when
    actual trading starts. Two of the names in this universe also carry an
    outright decimal typo (a 4.50 -> 0.45 -> 5.00 sequence on 0 volume).

    Neither is a return, and both are *worse* than noise for this study: a bad
    tick that fully reverses the next bar is genuinely predictable, so leaving
    them in manufactures exactly the false edge the screen exists to rule out.
    We therefore start each history where sustained liquidity starts, and report
    any residual spike-reversal so the caller can drop the name rather than
    silently repairing a price we cannot verify.
    """
    info: dict[str, Any] = {"trimmed_bars": 0, "bad_ticks": 0}
    if frame.empty:
        return frame, info

    dollar = (frame["close"] * frame["volume"]).rolling(window, min_periods=window).median()
    live = (dollar >= min_dollar_volume).to_numpy()
    if live.any():
        # The first bar whose trailing window qualifies. We deliberately do NOT
        # step back to the start of that window: for a name that was a
        # non-traded REIT or a pre-listing stub, those earlier bars are exactly
        # the frozen quotes we are trying to discard.
        first = int(np.argmax(live))
        if first > 0:
            info["trimmed_bars"] = first
            frame = frame.iloc[first:]
    else:
        info["trimmed_bars"] = len(frame)
        return frame.iloc[0:0], info

    # Isolated spike that round-trips on the next bar: a print error, not a move.
    close = frame["close"].to_numpy()
    if len(close) > 3:
        r = np.diff(np.log(close))
        spike = np.abs(r[:-1]) > 0.5
        reverses = np.abs(r[:-1] + r[1:]) < 0.25 * np.abs(r[:-1])
        info["bad_ticks"] = int(np.sum(spike & reverses))
    return frame, info


# ---------------------------------------------------------------------------
# Tradeability filters
# ---------------------------------------------------------------------------
def liquidity_stats(frame: pd.DataFrame) -> dict[str, float]:
    close, volume = frame["close"], frame["volume"]
    dollar = (close * volume).tail(252)
    rets = np.diff(np.log(close.to_numpy()))
    zero_share = float(np.mean(np.abs(rets[-252:]) < 1e-12)) if len(rets) >= 252 else float("nan")
    return {
        "n_bars": int(len(frame)),
        "last_price": float(close.iloc[-1]),
        "median_dollar_volume": float(dollar.median()) if len(dollar) else 0.0,
        "ann_vol_pct": float(np.std(rets[-252:], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
        if len(rets) >= 60 else float("nan"),
        "zero_return_share": zero_share,
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
    }


def passes_filters(stats: dict[str, float], min_bars: int = 1500,
                   min_price: float = 3.0,
                   min_dollar_volume: float = 1_000_000.0,
                   max_zero_share: float = 0.20) -> tuple[bool, str]:
    """Screen out names where a backtest result would not mean anything.

    Each rejection reason is returned so the page can show *why* a ticker was
    dropped rather than silently shrinking the universe.
    """
    if stats["n_bars"] < min_bars:
        return False, f"short history ({stats['n_bars']} bars)"
    if stats["last_price"] < min_price:
        return False, f"price below ${min_price:.0f}"
    if stats["median_dollar_volume"] < min_dollar_volume:
        return False, "thin dollar volume"
    if np.isfinite(stats["zero_return_share"]) and stats["zero_return_share"] > max_zero_share:
        return False, "stale/illiquid quotes"
    return True, ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .universe import load_universe

    sample = [e["ticker"] for e in load_universe()[:6]]
    frames = download_many(sample, progress=print)
    for t, f in sorted(frames.items()):
        s = liquidity_stats(f)
        ok, why = passes_filters(s)
        print(f"{t:6s} {s['n_bars']:5d} bars  {s['start']}..{s['end']}  "
              f"${s['last_price']:8.2f}  vol {s['ann_vol_pct']:5.1f}%  "
              f"$vol {s['median_dollar_volume']/1e6:7.2f}M  {'OK' if ok else 'DROP: '+why}")
