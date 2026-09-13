"""Committee assignments, and which industries each committee has power over.

The hypothesis worth testing is narrow and stated in advance: if a legislator's
trades carry information, the information should be concentrated where their
job gives them a view -- an Armed Services member buying a defence contractor,
an Energy member buying a driller -- rather than spread evenly across whatever
they happen to own.

The jurisdiction map below is a coarse, hand-built link from committee to GICS
sector. It is deliberately coarse: the point is to separate "this legislator
oversees this industry" from "this legislator has no particular window onto
it", not to model legislative influence in detail. Where a committee's remit is
genuinely economy-wide -- Appropriations, Budget, Finance -- it is mapped to
nothing, because a committee that touches everything cannot discriminate
between trades and would only add noise to the test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from btcmodels.config import CACHE_DIR

log = logging.getLogger(__name__)

UA = "financial-model-research newa6211@gmail.com"
BASE = "https://unitedstates.github.io/congress-legislators/"
CACHE = Path(CACHE_DIR) / "congress"
CACHE.mkdir(parents=True, exist_ok=True)

# Committee (by its thomas_id) -> GICS sectors it has jurisdiction over.
JURISDICTION: dict[str, set[str]] = {
    "SSAS": {"Industrials"},                       # Armed Services
    "HSAS": {"Industrials"},
    "SSCM": {"Communication Services", "Information Technology",
             "Consumer Discretionary"},            # Commerce, Science & Transportation
    "HSIF": {"Communication Services", "Information Technology",
             "Health Care", "Utilities", "Energy"},  # Energy & Commerce
    "SSEG": {"Energy", "Utilities", "Materials"},  # Energy & Natural Resources
    "SSEV": {"Utilities", "Materials", "Industrials"},  # Environment & Public Works
    "SSBK": {"Financials", "Real Estate"},         # Banking, Housing & Urban Affairs
    "HSBA": {"Financials", "Real Estate"},         # Financial Services
    "SSAF": {"Consumer Staples", "Materials"},     # Agriculture
    "HSAG": {"Consumer Staples", "Materials"},
    "SSHR": {"Health Care"},                       # Health, Education, Labor & Pensions
    "HSII": {"Energy", "Materials"},               # Natural Resources
    "SSSB": {"Financials"},                        # Small Business
    "HSSY": {"Information Technology", "Industrials"},   # Science, Space & Technology
    "HSVR": {"Health Care"},                       # Veterans' Affairs
    "SLIN": {"Information Technology"},            # Intelligence
    "HLIG": {"Information Technology"},
    # Economy-wide remits map to nothing on purpose; see the module docstring.
    "SSFI": set(), "SSAP": set(), "HSAP": set(), "HSWM": set(), "SSBU": set(),
}


def _fetch(name: str) -> Any:
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(BASE + name, headers={"User-Agent": UA}, timeout=60)
    resp.raise_for_status()
    path.write_text(resp.text)
    return resp.json()


def load() -> tuple[dict[str, list[dict]], dict[str, dict], list[dict]]:
    return (_fetch("committee-membership-current.json"),
            {c["thomas_id"]: c for c in _fetch("committees-current.json")},
            _fetch("legislators-current.json"))


def _name_keys(first: str, last: str) -> set[str]:
    first, last = (first or "").strip().lower(), (last or "").strip().lower()
    keys = {f"{first} {last}".strip(), last}
    if first:
        keys.add(f"{first[0]} {last}")
    return {k for k in keys if k}


def _collide_safe(index: dict[str, list[dict]], key: str, payload: dict) -> None:
    index.setdefault(key, []).append(payload)


def build_index() -> dict[str, list[dict[str, Any]]]:
    """Legislator -> their committees and the sectors those committees cover.

    Keyed by several spellings of the name because the disclosure systems and
    the legislator database do not agree on them: eFD writes "Thomas H
    Tuberville" where the roster has "Tommy Tuberville".
    """
    membership, committees, legislators = load()
    # Members who have since left office still appear in the disclosure feeds,
    # so the historical roster is needed to resolve them.
    try:
        legislators = legislators + _fetch("legislators-historical.json")
    except Exception as exc:                                   # pragma: no cover
        log.warning("historical roster unavailable: %s", exc)

    by_bioguide: dict[str, dict[str, Any]] = {}
    for thomas_id, members in membership.items():
        info = committees.get(thomas_id, {})
        sectors = JURISDICTION.get(thomas_id)
        for m in members:
            bid = m.get("bioguide")
            if not bid:
                continue
            entry = by_bioguide.setdefault(bid, {"committees": [], "sectors": set()})
            entry["committees"].append(info.get("name", thomas_id))
            if sectors:
                entry["sectors"] |= sectors

    index: dict[str, list[dict[str, Any]]] = {}
    for leg in legislators:
        bid = leg["id"].get("bioguide")
        entry = by_bioguide.get(bid)
        if not entry:
            continue
        name = leg["name"]
        term = (leg.get("terms") or [{}])[-1]
        payload = {
            "bioguide": bid,
            "full_name": f"{name.get('first','')} {name.get('last','')}".strip(),
            "chamber": term.get("type"),          # "sen" or "rep"
            "party": term.get("party"), "state": term.get("state"),
            "committees": sorted(set(entry["committees"])),
            "sectors": sorted(entry["sectors"]),
        }
        keys = _name_keys(name.get("first"), name.get("last"))
        keys |= _name_keys(name.get("official_full", "").split(" ")[0]
                           if name.get("official_full") else "", name.get("last"))
        if name.get("nickname"):
            keys |= _name_keys(name["nickname"], name.get("last"))
        for key in keys:
            _collide_safe(index, key, payload)
    return index


def match(index: dict[str, list[dict[str, Any]]], disclosed_name: str,
          chamber: str | None = None) -> dict[str, Any] | None:
    """Resolve a name as written on a disclosure to a legislator record.

    Chamber-aware and refuses to guess. A bare surname is only accepted when it
    identifies exactly one person in the relevant chamber: "Mullin" alone
    matches both Senator Markwayne Mullin of Oklahoma and Representative Kevin
    Mullin of California, and silently picking either would attribute one real
    person's trades to another.
    """
    raw = (disclosed_name or "").strip().lower()
    if not raw:
        return None

    def pick(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if chamber:
            candidates = [c for c in candidates if c.get("chamber") == chamber]
        # Deduplicate the same person appearing in both rosters.
        unique = {c["bioguide"]: c for c in candidates}
        return next(iter(unique.values())) if len(unique) == 1 else None

    parts = [p for p in raw.replace(",", " ").split() if len(p) > 1 and "." not in p]
    keys = [raw]
    if parts:
        keys.append(f"{parts[0]} {parts[-1]}")
        keys.append(parts[-1])
    for key in keys:
        got = pick(index.get(key, []))
        if got:
            return got
    return None
