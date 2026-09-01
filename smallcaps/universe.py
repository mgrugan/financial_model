"""The small-cap universe under test.

The S&P SmallCap 600 is the reference list: ~600 names, all genuinely small
(the index caps admission at roughly $8bn and most members sit well under
$3bn), all US-listed with clean Yahoo history, and published in one place we
can re-fetch.

One property of this list matters enough to state up front, because it shapes
how the results may be read: it is the list of companies that are *in the
index today*. Firms that were dropped after collapsing or being acquired are
absent. That is survivorship bias, and it inflates any statistic about
**returns** -- the average small cap in here did better than the average small
cap that existed. It is close to neutral for the statistic this study actually
reports, the AUC of a directional forecast, because AUC measures ranking skill
within a series and is invariant to that series' mean drift. We report returns
anyway for context, so the caveat is carried onto the page rather than buried.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

WIKI_SP600 = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
WIKI_SP400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "smallcap_universe.json"

# Yahoo writes class shares with a dash where the index lists a dot.
_TICKER_FIX = str.maketrans({".": "-"})


def _clean_ticker(raw: Any) -> str | None:
    text = str(raw).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return None
    text = re.sub(r"\[.*?\]", "", text).strip()      # strip Wikipedia footnotes
    if not re.fullmatch(r"[A-Z.\-]{1,7}", text):
        return None
    return text.translate(_TICKER_FIX)


def _scrape(url: str, index_name: str) -> list[dict[str, str]]:
    import io

    import pandas as pd

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    # Pick the table that actually looks like a constituent list.
    best: list[dict[str, str]] = []
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        sym_col = next((cols[k] for k in cols if "symbol" in k or "ticker" in k), None)
        if sym_col is None:
            continue
        name_col = next((cols[k] for k in cols if "security" in k or "company" in k), sym_col)
        sec_col = next((cols[k] for k in cols if "sector" in k and "sub" not in k), None)

        rows = []
        for _, row in table.iterrows():
            ticker = _clean_ticker(row[sym_col])
            if ticker is None:
                continue
            rows.append({
                "ticker": ticker,
                "name": str(row[name_col]).strip(),
                "sector": str(row[sec_col]).strip() if sec_col is not None else "",
                "index": index_name,
            })
        if len(rows) > len(best):
            best = rows
    return best


def fetch_universe(target: int = 500, include_midcap: bool = False) -> list[dict[str, str]]:
    """Constituents, de-duplicated and ordered. Raises if the list comes up short."""
    entries = _scrape(WIKI_SP600, "S&P 600")
    log.info("S&P 600 gave %d tickers", len(entries))

    if include_midcap or len(entries) < target:
        entries += _scrape(WIKI_SP400, "S&P 400")

    seen: set[str] = set()
    unique = []
    for entry in entries:
        if entry["ticker"] in seen:
            continue
        seen.add(entry["ticker"])
        unique.append(entry)

    if len(unique) < target:
        raise RuntimeError(f"universe too small: {len(unique)} < {target}")
    unique.sort(key=lambda e: e["ticker"])
    return unique


def load_universe(refresh: bool = False, target: int = 500) -> list[dict[str, str]]:
    """Cached manifest, re-fetched on demand."""
    if not refresh and MANIFEST.exists():
        return json.loads(MANIFEST.read_text())["constituents"]

    entries = fetch_universe(target=target)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"source": WIKI_SP600, "n": len(entries), "constituents": entries}, indent=1))
    return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    got = load_universe(refresh=True)
    print(f"{len(got)} tickers -> {MANIFEST}")
    print("first 12:", ", ".join(e["ticker"] for e in got[:12]))
