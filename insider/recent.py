"""Recent Form 4 filings, straight from EDGAR.

The quarterly Form 345 datasets lag: at the time of writing they end at the
close of 2026Q1 while the calendar reads September, so a six-month insider
window would be covered for only its first eighteen days. Since the whole point
of the signal is what insiders have been doing *lately*, the recent tail has to
come from the filings themselves.

This module fills exactly that gap and nothing more. It reads each issuer's
submission index, takes the Form 4s filed since a cutoff, and parses the
non-derivative transaction table out of the XML. The output has the same shape
as the quarterly extract so the two concatenate cleanly.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import pandas as pd
import requests

from .form345 import MAX_PLAUSIBLE_PRICE, SEC_UA, _is_insider_role

log = logging.getLogger(__name__)

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_LOCAL = threading.local()
_THROTTLE = threading.Semaphore(5)


def _session() -> requests.Session:
    if getattr(_LOCAL, "s", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
        _LOCAL.s = s
    return _LOCAL.s


def _get(url: str, attempts: int = 3) -> requests.Response | None:
    for attempt in range(attempts):
        try:
            with _THROTTLE:
                resp = _session().get(url, timeout=45)
                time.sleep(0.12)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except Exception:
            time.sleep(min(2 ** attempt, 8))
    return None


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


def parse_form4(xml_text: str, filed: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    cik = _text(root, "./issuer/issuerCik")
    symbol = _text(root, "./issuer/issuerTradingSymbol")
    owner_cik = _text(root, "./reportingOwner/reportingOwnerId/rptOwnerCik")
    rel = root.find("./reportingOwner/reportingOwnerRelationship")
    role_bits = []
    if rel is not None:
        for tag, label in (("isDirector", "Director"), ("isOfficer", "Officer"),
                           ("isTenPercentOwner", "TenPercentOwner")):
            val = _text(rel, tag)
            if val in ("1", "true"):
                role_bits.append(label)
    role = "|".join(role_bits)

    rows = []
    for tx in root.findall("./nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(tx, "./transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        shares = _text(tx, "./transactionAmounts/transactionShares/value")
        price = _text(tx, "./transactionAmounts/transactionPricePerShare/value")
        ad = _text(tx, "./transactionAmounts/transactionAcquiredDisposedCode/value")
        date = _text(tx, "./transactionDate/value")
        try:
            n, p = float(shares), float(price)
        except (TypeError, ValueError):
            continue
        if n <= 0 or not (0.005 < p < MAX_PLAUSIBLE_PRICE):
            continue
        rows.append({
            "cik": str(cik).zfill(10) if cik else None,
            "ticker": (symbol or "").strip().upper().replace(".", "-"),
            "filed": filed, "trans_date": date,
            "shares": n, "price": p, "value": n * p,
            "is_buy": code == "P" and ad == "A",
            "is_sell": code == "S" and ad == "D",
            "insider_role": _is_insider_role(role),
            "owner_cik": owner_cik,
            "accession_number": None,
        })
    return rows


def fetch_recent(ciks: Iterable[str], since: str, workers: int = 5,
                 progress: Any = None) -> pd.DataFrame:
    """Form 4 transactions filed on or after ``since``, for these issuers."""
    ciks = list(ciks)
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    done = 0

    def work(cik: str) -> None:
        nonlocal done
        local: list[dict[str, Any]] = []
        resp = _get(SUBMISSIONS.format(cik=cik))
        if resp is not None:
            try:
                recent = resp.json().get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accs = recent.get("accessionNumber", [])
                docs = recent.get("primaryDocument", [])
                for form, date, acc, doc in zip(forms, dates, accs, docs):
                    if form != "4" or date < since or not doc:
                        continue
                    # primaryDocument points at the XSL-rendered *view* of the
                    # filing -- "xslF345X06/doc4.xml" is 16 KB of HTML, not the
                    # 3 KB of XML it renders. Dropping the stylesheet directory
                    # gives the machine-readable original sitting beside it.
                    doc_xml = doc.split("/")[-1]
                    if not doc_xml.endswith(".xml"):
                        doc_xml = doc_xml.rsplit(".", 1)[0] + ".xml"
                    url = ARCHIVE.format(cik=str(int(cik)), acc=acc.replace("-", ""),
                                         doc=doc_xml)
                    got = _get(url)
                    if got is None:
                        continue
                    local.extend(parse_form4(got.text, date))
            except Exception as exc:
                log.warning("recent form4 failed for %s: %s", cik, exc)
        with lock:
            done += 1
            rows.extend(local)
            if progress and done % 40 == 0:
                progress(f"  recent form4 {done}/{len(ciks)} ({len(rows)} transactions)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, ciks))
    if progress:
        progress(f"  recent form4 {done}/{len(ciks)} ({len(rows)} transactions)")
    return pd.DataFrame(rows)
