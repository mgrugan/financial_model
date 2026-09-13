"""House of Representatives Periodic Transaction Reports.

The House publishes an annual index of every financial disclosure filing and
then the filings themselves as PDFs, which is a harder road than the Senate's
HTML tables but a more productive one: there are roughly three and a half times
as many House PTRs as Senate ones.

The parser works on extracted text rather than the PDF's table structure. The
tables in these documents merge cells unpredictably -- one row can swallow the
next four -- while the text layout is consistent: an asset name, a ticker in
parentheses, a transaction type, the trade date, the notification date and an
amount band, in that order.

A field worth noting is "Subholding Of", which names the managing institution
when a trade came from a managed account. It is the House's version of the
pattern that dominates the Senate data, where a single filing can report a
whole portfolio being bought on one day.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import requests

from btcmodels.config import CACHE_DIR

log = logging.getLogger(__name__)

UA = "financial-model-research newa6211@gmail.com"
INDEX = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_PDF = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc}.pdf"

CACHE = Path(CACHE_DIR) / "house"
CACHE.mkdir(parents=True, exist_ok=True)

_THROTTLE = threading.Semaphore(4)
_LOCAL = threading.local()

# The transaction core: type, trade date, notification date, amount band. The
# ticker is deliberately NOT part of this pattern. A long asset name wraps, and
# when it does the ticker lands on the *following* line -- after the amount
# rather than before the type -- so matching them together silently drops every
# company whose name is too long for one line, which is most of them.
TXN = re.compile(
    r"(?P<type>P|S \(partial\)|S|E)\s+"
    r"(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<notified>\d{2}/\d{2}/\d{4})\s+"
    r"\$(?P<low>[\d,]+)(?:\s*-\s*\$(?P<high>[\d,]+))?"
)
TICKER = re.compile(r"\(\s*([A-Z][A-Z.\-]{0,6})\s*\)")
OWNER = re.compile(r"\b(SP|DC|JT)\b")
SUBHOLDING = re.compile(r"S\W*\s*O\W*:\s*([^\n]{2,60})")


def _session() -> requests.Session:
    if getattr(_LOCAL, "s", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _LOCAL.s = s
    return _LOCAL.s


def index(year: int) -> list[dict[str, str]]:
    """Every filing the House logged that year; FilingType 'P' is a PTR."""
    path = CACHE / f"index_{year}.zip"
    if path.exists():
        blob = path.read_bytes()
    else:
        resp = _session().get(INDEX.format(year=year), timeout=120)
        resp.raise_for_status()
        blob = resp.content
        path.write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        with z.open(name) as fh:
            return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8"), delimiter="\t"))


def _us_date(text: str) -> str | None:
    try:
        return dt.datetime.strptime(text, "%m/%d/%Y").date().isoformat()
    except (ValueError, TypeError):
        return None


def parse_pdf(blob: bytes) -> tuple[list[dict[str, Any]], str | None]:
    import pdfplumber

    rows: list[dict[str, Any]] = []
    manager: str | None = None
    try:
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        log.debug("pdf parse failed: %s", exc)
        return rows, None

    if not text.strip():
        return rows, None                       # scanned image, nothing to read
    sub = SUBHOLDING.search(text)
    if sub:
        manager = sub.group(1).strip()

    # Work line by line. For each line carrying a transaction, take the ticker
    # from that same line if it is there, otherwise from the next line, which is
    # where a wrapped asset name puts it.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        m = TXN.search(line)
        if not m:
            continue
        # Ignore the free-text "Description" blocks, which restate a basket of
        # trades in prose and would otherwise be counted as transactions.
        if line.lstrip().startswith("D") and "shares sold @" in line:
            continue
        before = line[:m.start()]
        found = TICKER.findall(before)
        if not found and i + 1 < len(lines):
            found = TICKER.findall(lines[i + 1])
        if not found:
            continue
        low = float(m.group("low").replace(",", ""))
        high = float(m.group("high").replace(",", "")) if m.group("high") else low
        kind = m.group("type")
        rows.append({
            "ticker": found[-1].replace(".", "-"),
            "type": kind,
            "is_purchase": kind == "P",
            "is_sale": kind.startswith("S"),
            "trans_date": _us_date(m.group("date")),
            "filed": _us_date(m.group("notified")),
            "amount_low": low, "amount_high": high, "amount_mid": (low + high) / 2.0,
            "managed_by": manager,
        })
    return rows, manager


def fetch_ptrs(years: Iterable[int], workers: int = 4,
               progress: Any = None) -> list[dict[str, Any]]:
    jobs: list[tuple[int, dict[str, str]]] = []
    for year in years:
        for row in index(year):
            if row.get("FilingType") == "P":
                jobs.append((year, row))

    out: list[dict[str, Any]] = []
    lock = threading.Lock()
    done = empty = 0

    def work(job: tuple[int, dict[str, str]]) -> None:
        nonlocal done, empty
        year, row = job
        doc = row["DocID"]
        cached = CACHE / f"{year}_{doc}.pdf"
        blob = None
        if cached.exists():
            blob = cached.read_bytes()
        else:
            try:
                with _THROTTLE:
                    resp = _session().get(PTR_PDF.format(year=year, doc=doc), timeout=90)
                    time.sleep(0.1)
                if resp.status_code == 200:
                    blob = resp.content
                    cached.write_bytes(blob)
            except Exception as exc:
                log.debug("%s: %s", doc, exc)
        rows, _ = parse_pdf(blob) if blob else ([], None)
        member = f"{row.get('First','').strip()} {row.get('Last','').strip()}".strip()
        for r in rows:
            r.update({"member": member, "state_district": row.get("StateDst", ""),
                      "doc_id": doc, "year": year,
                      "filing_date": _us_date(row.get("FilingDate", ""))})
        with lock:
            done += 1
            if not rows:
                empty += 1
            out.extend(rows)
            if progress and done % 100 == 0:
                progress(f"  house PTRs {done}/{len(jobs)} "
                         f"({len(out)} transactions, {empty} unreadable)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, jobs))
    if progress:
        progress(f"  house PTRs {done}/{len(jobs)} ({len(out)} transactions, "
                 f"{empty} unreadable)")
    return out
