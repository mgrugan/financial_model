"""Open-market insider purchases, point-in-time, from the SEC Form 345 datasets.

The signal here is narrow on purpose. A Form 4 reports every kind of change in
an insider's holdings, and most of them carry no information: option grants,
vesting, tax withholding and gifts happen on a schedule the insider did not
choose. Only ``TRANS_CODE == 'P'`` -- an open-market purchase at the prevailing
price -- represents someone electing to put their own money in.

Sales are recorded but deliberately not treated as the mirror image. Insiders
sell for diversification, tax bills, house purchases and expiring windows, so a
sale is weak evidence at best; purchases have no such benign explanation. The
metrics below therefore lead with buying and carry the sell side only as
context.

``FILING_DATE`` is what makes this usable in a backtest: a Form 4 is due within
two business days of the trade, and the dataset records when it was actually
filed, so a rebalance on date T can be restricted to what was public by T.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from btcmodels.config import CACHE_DIR

log = logging.getLogger(__name__)

SEC_UA = "financial-model-research newa6211@gmail.com"
DATASET = ("https://www.sec.gov/files/structureddata/data/"
           "insider-transactions-data-sets/{year}q{q}_form345.zip")

CACHE = Path(CACHE_DIR) / "form345"
CACHE.mkdir(parents=True, exist_ok=True)

# Open-market purchase. The one code that means "an insider chose to buy".
BUY_CODE = "P"
SELL_CODE = "S"

# No constituent of the S&P 400 or 600 trades near this price; anything above it
# is a unit error in the filing, not a share price.
MAX_PLAUSIBLE_PRICE = 10_000.0

# Relationship flags are packed into one string per owner row.
def _is_insider_role(text: object) -> bool:
    # A missing relationship arrives as NaN, and `nan or ""` yields the NaN
    # rather than the empty string because NaN is truthy.
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in ("officer", "director", "tenpercentowner", "ceo", "cfo"))


def _parse_date(value: str) -> str | None:
    """The datasets use 31-MAR-2026; normalise to ISO for lexical comparison."""
    if not value or pd.isna(value):
        return None
    try:
        return dt.datetime.strptime(str(value).strip(), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def quarters(start: str = "2011-01-01", end: str | None = None) -> list[tuple[int, int]]:
    end_date = dt.date.fromisoformat(end) if end else dt.date.today()
    y, q = int(start[:4]), (int(start[5:7]) - 1) // 3 + 1
    out = []
    while (y, q) <= (end_date.year, (end_date.month - 1) // 3 + 1):
        out.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


def _fetch_quarter(year: int, q: int, attempts: int = 3) -> bytes | None:
    url = DATASET.format(year=year, q=q)
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=180)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            log.warning("%dq%d fetch failed (%s)", year, q, exc)
            time.sleep(min(2 ** attempt, 10))
    return None


def load_quarter(year: int, q: int, ciks: set[str] | None = None,
                 force: bool = False) -> pd.DataFrame:
    """Open-market buys and sells for one quarter, optionally filtered by issuer.

    Only the extracted rows are cached. The quarterly archives are 8-14 MB each
    and there are nearly sixty of them; keeping the raw zips would cost most of
    a gigabyte to store data we use a few columns of.
    """
    path = CACHE / f"{year}q{q}.pkl"
    if path.exists() and not force:
        try:
            return pd.read_pickle(path)
        except Exception:
            pass

    blob = _fetch_quarter(year, q)
    if blob is None:
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = set(z.namelist())
        if not {"SUBMISSION.tsv", "NONDERIV_TRANS.tsv"} <= names:
            return pd.DataFrame()

        with z.open("SUBMISSION.tsv") as fh:
            sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=[
                "ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK",
                "ISSUERNAME", "ISSUERTRADINGSYMBOL"], on_bad_lines="skip")
        sub["ISSUERCIK"] = sub["ISSUERCIK"].str.zfill(10)
        if ciks:
            sub = sub[sub["ISSUERCIK"].isin(ciks)]
        if sub.empty:
            pd.DataFrame().to_pickle(path)
            return pd.DataFrame()

        with z.open("NONDERIV_TRANS.tsv") as fh:
            trans = pd.read_csv(fh, sep="\t", dtype=str, usecols=[
                "ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"],
                on_bad_lines="skip")
        trans = trans[trans["TRANS_CODE"].isin([BUY_CODE, SELL_CODE])]

        owners = pd.DataFrame(columns=["ACCESSION_NUMBER", "RPTOWNERCIK",
                                       "RPTOWNER_RELATIONSHIP"])
        if "REPORTINGOWNER.tsv" in names:
            with z.open("REPORTINGOWNER.tsv") as fh:
                owners = pd.read_csv(fh, sep="\t", dtype=str, usecols=[
                    "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"],
                    on_bad_lines="skip")

    merged = trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
    if merged.empty:
        pd.DataFrame().to_pickle(path)
        return pd.DataFrame()

    # One filing can list several reporting owners; keep the first per filing so
    # a jointly-filed purchase is not counted once per signatory.
    owners = owners.drop_duplicates("ACCESSION_NUMBER")
    merged = merged.merge(owners, on="ACCESSION_NUMBER", how="left")

    merged["filed"] = merged["FILING_DATE"].map(_parse_date)
    merged["trans_date"] = merged["TRANS_DATE"].map(_parse_date)
    merged["shares"] = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce")
    merged["price"] = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce")
    merged["value"] = merged["shares"] * merged["price"]
    merged["is_buy"] = merged["TRANS_CODE"].eq(BUY_CODE) & \
        merged["TRANS_ACQUIRED_DISP_CD"].eq("A")
    merged["is_sell"] = merged["TRANS_CODE"].eq(SELL_CODE) & \
        merged["TRANS_ACQUIRED_DISP_CD"].eq("D")
    merged["insider_role"] = merged["RPTOWNER_RELATIONSHIP"].map(_is_insider_role)

    out = merged[[
        "ISSUERCIK", "ISSUERTRADINGSYMBOL", "filed", "trans_date",
        "shares", "price", "value", "is_buy", "is_sell",
        "insider_role", "RPTOWNERCIK", "ACCESSION_NUMBER",
    ]].rename(columns={"ISSUERCIK": "cik", "ISSUERTRADINGSYMBOL": "ticker",
                       "RPTOWNERCIK": "owner_cik"})
    out = out.dropna(subset=["filed", "value"])
    out = out[out["value"] > 0]
    out.to_pickle(path)
    return out


def load_history(ciks: set[str], start: str = "2011-01-01",
                 end: str | None = None, progress=None) -> pd.DataFrame:
    frames = []
    qs = quarters(start, end)
    for i, (year, q) in enumerate(qs, 1):
        frame = load_quarter(year, q, ciks)
        if not frame.empty:
            frames.append(frame)
        if progress and i % 8 == 0:
            progress(f"  form345 {i}/{len(qs)} quarters ({sum(len(f) for f in frames)} rows)")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["cik", "filed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Price validation
# ---------------------------------------------------------------------------
def validate_prices(frame: pd.DataFrame, closes: dict[str, pd.Series],
                    lo_ratio: float = 0.2, hi_ratio: float = 5.0,
                    ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop transactions whose reported price contradicts the market.

    Filers make unit errors. Theravance filed thirteen Form 4s reporting prices
    between $3.5m and $111m *per share* -- the total consideration written into
    the price-per-share box. Those thirteen rows carried $366 trillion of
    notional against $16.6bn for the whole rest of the universe, so any
    dollar-weighted insider metric would have been entirely those rows.

    Capping at an arbitrary price ceiling would catch this particular error and
    miss the next one, so each transaction is checked against the actual close
    on its trade date: a price more than five times, or less than a fifth of,
    what the stock changed hands at that day is a reporting error rather than a
    trade. Rows with no price history to check against are kept, since absence
    of a benchmark is not evidence of a bad print.
    """
    if frame.empty:
        return frame, {"checked": 0, "dropped": 0, "unverifiable": 0, "implausible": 0}

    # A hard ceiling first, because the market check cannot rescue a company
    # that has since been renamed: Theravance became TBPH, so its ticker no
    # longer resolves to any price history and its thirteen bad prints would
    # pass through as "unverifiable". No S&P 400 or 600 constituent has ever
    # traded above $10,000 a share.
    n_before = len(frame)
    frame = frame[(frame["price"] > 0.005) & (frame["price"] < MAX_PLAUSIBLE_PRICE)]
    implausible = n_before - len(frame)

    work = frame.copy()
    ratio = np.full(len(work), np.nan)
    tickers = work["ticker"].to_numpy()
    dates = work["trans_date"].to_numpy()
    prices = work["price"].to_numpy(dtype="float64")

    for ticker, series in closes.items():
        mask = tickers == ticker
        if not mask.any() or series is None or series.empty:
            continue
        idx = series.index
        for pos in np.flatnonzero(mask):
            day = dates[pos]
            if not isinstance(day, str):
                continue
            stamp = pd.Timestamp(day, tz="UTC")
            loc = int(idx.searchsorted(stamp, side="right")) - 1
            if loc < 0:
                continue
            close = float(series.iloc[loc])
            if close > 0 and prices[pos] > 0:
                ratio[pos] = prices[pos] / close

    work["price_vs_market"] = ratio
    verifiable = np.isfinite(ratio)
    bad = verifiable & ((ratio > hi_ratio) | (ratio < lo_ratio))
    stats = {"checked": int(verifiable.sum()), "dropped": int(bad.sum()),
             "unverifiable": int((~verifiable).sum()), "implausible": implausible}
    return work[~bad].reset_index(drop=True), stats
