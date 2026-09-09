"""Senate Periodic Transaction Reports from the official eFD system.

Senators must disclose trades over $1,000 within 45 days. That deadline is the
single most important fact about this data and it shapes everything downstream:
by the time a purchase is public it can be a month and a half old, so any
strategy built on it is trading on information the filer acted on long before.
The lag is measured rather than assumed -- see ``disclosure_lag`` in the
scripts that consume this.

Two further limits are structural, not fixable by better parsing:

* **Amounts are ranges**, not figures. "$1,001 - $15,000" is as precise as the
  disclosure gets, so position sizes are estimates and the midpoint of a range
  spanning an order of magnitude is a weak one.
* **Some reports are scanned paper filings**, which carry no machine-readable
  table at all. They are counted and skipped rather than silently dropped, so
  the coverage figure stays honest.

Access follows the site's own flow: accept the prohibition agreement, then use
the same JSON search endpoint the page itself calls. Neither efdsearch.senate.gov
nor the House clerk publishes a robots.txt, and these are public disclosures
mandated by statute; requests are still paced well below one per second.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://efdsearch.senate.gov"
HOME = f"{BASE}/search/home/"
SEARCH = f"{BASE}/search/report/data/"
UA = "financial-model-research newa6211@gmail.com"

REPORT_TYPE_PTR = 11
REQUEST_PAUSE = 0.35            # well under 1 req/s

AMOUNT_RE = re.compile(r"\$([\d,]+)(?:\s*-\s*\$([\d,]+))?")


def parse_amount(text: str) -> tuple[float | None, float | None, float | None]:
    """A disclosure band -> (low, high, midpoint). Ranges are all we are given."""
    if not text or not isinstance(text, str):
        return None, None, None
    m = AMOUNT_RE.search(text.replace("–", "-"))
    if not m:
        return None, None, None
    low = float(m.group(1).replace(",", ""))
    high = float(m.group(2).replace(",", "")) if m.group(2) else low
    return low, high, (low + high) / 2.0


class SenateEFD:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self._token: str | None = None

    def _accept(self) -> str:
        resp = self.session.get(HOME, timeout=40)
        resp.raise_for_status()
        token = self.session.cookies.get("csrftoken")
        m = re.search(r"name=['\"]csrfmiddlewaretoken['\"] value=['\"]([^'\"]+)", resp.text)
        token = m.group(1) if m else token
        self.session.post(HOME, timeout=40, headers={"Referer": HOME},
                          data={"prohibition_agreement": "1",
                                "csrfmiddlewaretoken": token})
        self._token = self.session.cookies.get("csrftoken", token)
        return self._token

    @property
    def token(self) -> str:
        if not self._token:
            self._accept()
        return self._token or ""

    def reports(self, start: str, end: str | None = None,
                page_size: int = 100, progress=None) -> list[dict[str, Any]]:
        """Every Periodic Transaction Report filed in the window."""
        token = self.token
        out: list[dict[str, Any]] = []
        offset, total = 0, None
        while True:
            payload = {
                "start": str(offset), "length": str(page_size),
                "report_types": f"[{REPORT_TYPE_PTR}]", "filer_types": "[]",
                "submitted_start_date": f"{start} 00:00:00",
                "submitted_end_date": f"{end} 23:59:59" if end else "",
                "candidate_state": "", "senator_state": "", "office_id": "",
                "first_name": "", "last_name": "",
                "csrfmiddlewaretoken": token,
            }
            resp = self.session.post(
                SEARCH, data=payload, timeout=60,
                headers={"Referer": f"{BASE}/search/", "X-CSRFToken": token})
            resp.raise_for_status()
            data = resp.json()
            total = data.get("recordsTotal", 0) if total is None else total
            rows = data.get("data", [])
            if not rows:
                break
            for row in rows:
                link = re.search(r'href="([^"]+)"', row[3])
                out.append({
                    "first": str(row[0]).strip(),
                    "last": str(row[1]).strip(),
                    "office": re.sub("<[^>]+>", "", str(row[2])).strip(),
                    "title": re.sub("<[^>]+>", "", str(row[3])).strip(),
                    "filed": _us_date(str(row[4]).strip()),
                    "url": BASE + link.group(1) if link else None,
                })
            offset += len(rows)
            if progress:
                progress(f"  senate PTRs {offset}/{total}")
            if offset >= (total or 0):
                break
            time.sleep(REQUEST_PAUSE)
        return out

    def transactions(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse one report's table. Paper filings return an empty list."""
        if not report.get("url"):
            return []
        try:
            resp = self.session.get(report["url"], timeout=45)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("report fetch failed (%s)", exc)
            return []
        time.sleep(REQUEST_PAUSE)

        try:
            tables = pd.read_html(io.StringIO(resp.text))
        except ValueError:
            return []                      # scanned paper filing: no table
        if not tables:
            return []

        frame = tables[0]
        needed = {"Transaction Date", "Ticker", "Type", "Amount"}
        if not needed <= set(frame.columns):
            return []

        rows = []
        for _, r in frame.iterrows():
            ticker = str(r.get("Ticker", "")).strip().upper()
            if ticker in ("--", "", "NAN", "N/A"):
                continue
            ticker = re.sub(r"[^A-Z.\-]", "", ticker).replace(".", "-")
            if not ticker:
                continue
            kind = str(r.get("Type", "")).strip()
            low, high, mid = parse_amount(str(r.get("Amount", "")))
            rows.append({
                "senator": f"{report['first']} {report['last']}".strip(),
                "office": report["office"],
                "filed": report["filed"],
                "trans_date": _us_date(str(r.get("Transaction Date", "")).strip()),
                "ticker": ticker,
                "asset_name": str(r.get("Asset Name", "")).strip(),
                "asset_type": str(r.get("Asset Type", "")).strip(),
                "owner": str(r.get("Owner", "")).strip(),
                "type": kind,
                "is_purchase": kind.lower().startswith("purchase"),
                "is_sale": kind.lower().startswith("sale"),
                "amount_low": low, "amount_high": high, "amount_mid": mid,
                "report_url": report["url"],
            })
        return rows


def _us_date(text: str) -> str | None:
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None
