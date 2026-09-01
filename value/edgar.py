"""Point-in-time fundamentals from SEC EDGAR's XBRL companyfacts API.

Why EDGAR and not the price vendor's fundamentals endpoint: the free tier there
returns four annual periods and, more importantly, returns them *as they stand
today*. A value backtest built on that is not a backtest. Two distinct
look-ahead problems would be baked in:

* **Timing.** A fiscal year ending 31 October is not public knowledge on
  31 October; the 10-K lands weeks later. Ranking stocks on 31 October by a
  number nobody could read yet is a look-ahead that flatters exactly the deep-
  value names this study is about, because the most distressed companies file
  latest.
* **Restatement.** Vendors overwrite history with restated figures. A company
  that later restated earnings downward looks, in the vendor's history, as
  though the market should have seen the bad number all along.

EDGAR solves both. Every fact carries the ``filed`` date of the filing that
first contained it, so a rebalance on date T can be restricted to facts with
``filed <= T``; and where a period has been reported more than once, we keep the
**earliest** filed value, which is what an investor actually saw.

The cost is that XBRL tagging is not uniform. Companies report revenue under at
least five different tags and debt under eight, so each canonical field is a
fallback chain, ordered most-specific first and calibrated against a sample of
this universe rather than guessed.
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from btcmodels.config import CACHE_DIR

log = logging.getLogger(__name__)

# SEC asks for a descriptive agent with contact details and caps traffic at
# 10 requests/second. We stay well inside that.
SEC_UA = "financial-model-research newa6211@gmail.com"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FACTS_CACHE = Path(CACHE_DIR) / "edgar"
FACTS_CACHE.mkdir(parents=True, exist_ok=True)

_THROTTLE = threading.Semaphore(5)
_LOCAL = threading.local()

# ---------------------------------------------------------------------------
# Canonical fields and their tag fallback chains
# ---------------------------------------------------------------------------
# "instant" = balance-sheet item, dated by a single point in time.
# "duration" = flow item, spanning start..end, so it can be summed to a TTM.
INSTANT_FIELDS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    # Non-current portion first: the bare "LongTermDebt" tag is used by some
    # filers for the total including the current portion, which would
    # double-count against debt_short.
    "debt_long": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    ],
    "debt_short": [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "OtherShortTermBorrowings",
    ],
    "shares": [
        "dei:EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ],
}

DURATION_FIELDS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueServicesNet",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "dep_amort": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}


def _session() -> requests.Session:
    if getattr(_LOCAL, "s", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
        _LOCAL.s = s
    return _LOCAL.s


# ---------------------------------------------------------------------------
# Ticker -> CIK
# ---------------------------------------------------------------------------
def cik_map(refresh: bool = False) -> dict[str, str]:
    path = FACTS_CACHE / "cik_map.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    payload = _session().get(TICKER_MAP_URL, timeout=45).json()
    # Yahoo writes class shares with a dash, EDGAR with a dot.
    out = {str(v["ticker"]).upper().replace(".", "-"): str(v["cik_str"]).zfill(10)
           for v in payload.values()}
    path.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------
@dataclass
class Fact:
    """One reported number, with the date it actually became public."""

    end: str
    val: float
    filed: str
    start: str | None = None
    form: str = ""
    tag: str = ""

    @property
    def days(self) -> int:
        if not self.start:
            return 0
        from datetime import date
        a = date.fromisoformat(self.start)
        b = date.fromisoformat(self.end)
        return (b - a).days


def _collect(facts: dict, chain: list[str], duration: bool) -> list[Fact]:
    """Resolve a fallback chain **per period**, highest-priority tag first.

    Taking the first tag with any data at all looks tidier but leaves holes:
    ASC 606 landed in 2018, so a filer's revenue sits under ``SalesRevenueNet``
    before it and ``RevenueFromContractWithCustomer...`` after. Locking onto
    either tag silently deletes half the history -- and it deletes the earlier
    half, which is precisely the part a long backtest needs.

    So each reporting period is filled independently by the first tag in the
    chain that covers it. The chains hold near-synonyms, so the definitional
    break this risks is far smaller than the coverage gap it avoids; which tag
    supplied each number is recorded on the Fact so the switching rate can be
    measured rather than assumed.
    """
    gaap = facts.get("us-gaap", {})
    dei = facts.get("dei", {})
    best: dict[tuple, Fact] = {}
    rank: dict[tuple, int] = {}

    for priority, tag in enumerate(chain):
        block = dei.get(tag.split(":", 1)[1]) if tag.startswith("dei:") else gaap.get(tag)
        if not block:
            continue
        for unit_rows in block.get("units", {}).values():
            for r in unit_rows:
                if r.get("val") is None or not r.get("filed") or not r.get("end"):
                    continue
                if duration and not r.get("start"):
                    continue
                if not duration and r.get("start"):
                    continue
                key = (r.get("start"), r["end"])
                fact = Fact(end=r["end"], val=float(r["val"]), filed=r["filed"],
                            start=r.get("start"), form=r.get("form", ""), tag=tag)
                prior = best.get(key)
                if prior is None or priority < rank[key]:
                    best[key], rank[key] = fact, priority
                elif priority == rank[key] and fact.filed < prior.filed:
                    # As-first-reported: within one tag, the earliest filing of a
                    # period wins, so a later restatement never travels backwards.
                    best[key] = fact
    return sorted(best.values(), key=lambda x: (x.end, x.filed))


def extract(facts_payload: dict) -> dict[str, list[Fact]]:
    facts = facts_payload.get("facts", {})
    out: dict[str, list[Fact]] = {}
    for field, chain in INSTANT_FIELDS.items():
        out[field] = _collect(facts, chain, duration=False)
    for field, chain in DURATION_FIELDS.items():
        out[field] = _collect(facts, chain, duration=True)
    return out


def fetch_company(ticker: str, cik: str, ttl: int = 7 * 86_400,
                  force: bool = False) -> dict[str, list[Fact]] | None:
    """Extracted facts for one company, cached.

    Only the extracted fields are cached, never the raw payload: companyfacts
    responses run 1-10 MB each and this universe is 532 of them.
    """
    path = FACTS_CACHE / f"{ticker}.pkl"
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    try:
        with _THROTTLE:
            resp = _session().get(FACTS_URL.format(cik=cik), timeout=90)
            time.sleep(0.12)                       # stay under 10 req/s
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = extract(resp.json())
    except Exception as exc:
        log.warning("%s: companyfacts failed (%s)", ticker, exc)
        return None
    try:
        with path.open("wb") as fh:
            pickle.dump(data, fh, protocol=4)
    except Exception as exc:                                   # pragma: no cover
        log.warning("%s: cache write failed (%s)", ticker, exc)
    return data


def fetch_many(tickers: Iterable[str], workers: int = 5,
               progress: Any = None) -> dict[str, dict[str, list[Fact]]]:
    tickers = list(tickers)
    mapping = cik_map()
    out: dict[str, dict[str, list[Fact]]] = {}
    lock = threading.Lock()
    done = 0

    def work(ticker: str) -> None:
        nonlocal done
        cik = mapping.get(ticker.upper())
        data = fetch_company(ticker, cik) if cik else None
        with lock:
            done += 1
            if data:
                out[ticker] = data
            if progress and done % 25 == 0:
                progress(f"  edgar {done}/{len(tickers)} ({len(out)} ok)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, tickers))
    if progress:
        progress(f"  edgar {done}/{len(tickers)} ({len(out)} ok)")
    return out
