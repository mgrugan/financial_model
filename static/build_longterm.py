#!/usr/bin/env python3
"""Long-horizon page: the same models held for 6 months, 1 year and 5 years.

Run:  python -m static.build_longterm --out site
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .build_smallcaps import CHART_CSS, esc, fmt
from .build_value import EXTRA_CSS, SCRIPT, pct

DATA = Path("data")

HORIZON_LABEL = {"6m": "6 months", "1y": "1 year", "5y": "5 years"}


def _mde(n_eff: float) -> float:
    """Smallest AUC a per-ticker test could resolve at this effective count."""
    per_class = n_eff / 2.0
    if per_class <= 0.5:
        return float("inf")
    return 0.5 + 2.8 * math.sqrt(1.0 / (6.0 * per_class))


def render_horizon_portfolio(pf: dict) -> str:
    hz = pf.get("horizons") or {}
    if not hz:
        return ""
    rows = ""
    for key in ("6m", "1y", "5y"):
        b = hz.get(key)
        if not b:
            continue
        trust = b["independent_windows"] >= 8
        rows += (
            f'<tr><td class="tk">{esc(HORIZON_LABEL[key])}</td>'
            f'<td>{b["annualised"]:+.1%}</td>'
            f'<td>{b["universe_annualised"]:+.1%}</td>'
            f'<td class="{"sc-pos" if b["excess_size_matched_annualised"] > 0 else "sc-neg"}">'
            f'{b["excess_size_matched_annualised"]:+.1%}</td>'
            f'<td>{b["t_excess_size_matched_naive"]:+.2f}</td>'
            f'<td>{b["t_excess_size_matched"]:+.2f}</td>'
            f'<td>{b["n_periods"]}</td>'
            f'<td class="{"" if trust else "sc-neg"}">{b["independent_windows"]:.1f}</td>'
            f'<td>{"yes" if trust else "<strong>no</strong>"}</td></tr>')

    one = hz.get("1y", {})
    five = hz.get("5y", {})
    return f"""
  <div class="sc-section">
    <h2>The same rule, held for longer</h2>
    <p class="lede">One portfolio — 30 names ranked on insider buying, rebalanced
    semi-annually — held for each of three periods. Annualised so the columns are
    comparable, and benchmarked against a size-matched slice of the universe rather than
    the universe itself, because size in a current-constituents universe is a selection
    artifact rather than a premium.</p>
    <div class="sc-table-wrap"><table class="sc-table">
      <thead><tr><th>Hold</th><th>Portfolio</th><th>Universe</th>
      <th>vs size-match</th><th>t naive</th><th>t corrected</th>
      <th>Rebal.</th><th>Indep.<br>windows</th><th>Reliable?</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <div class="sc-warn"><strong>One year is the sweet spot, and five years is not a
    result.</strong> The 1-year hold earns
    {one.get('excess_size_matched_annualised', 0):+.1%} a year over size-matched peers at
    t&nbsp;=&nbsp;{one.get('t_excess_size_matched', 0):+.2f} across 14 independent windows,
    and it turns the book over half as often as the 6-month version. The 5-year row shows a
    bigger number — {five.get('excess_size_matched_annualised', 0):+.1%} a year — on
    <strong>2.9 independent windows</strong>. Fourteen years of history simply does not
    contain enough non-overlapping five-year periods to distinguish that from luck, whatever
    t-statistic is printed beside it.</div>
  </div>"""


def render_horizon_factors(long: dict) -> str:
    hz = long.get("horizons", {})
    keys = [k for k in ("6m", "1y", "5y") if k in hz]
    if not keys:
        return ""

    # One row per factor, one pair of columns per horizon, size+sector neutral.
    by_factor: dict[str, dict] = {}
    for hkey in keys:
        for r in hz[hkey]["results"]:
            if not r.get("size_neutral"):
                continue
            by_factor.setdefault(r["key"], {"label": r["label"],
                                            "tradition": r["tradition"]})[hkey] = r

    order = sorted(by_factor, key=lambda k: -(by_factor[k].get("1y", {}).get("t_spread") or 0))
    rows = ""
    for key in order:
        entry = by_factor[key]
        cells = ""
        for hkey in keys:
            r = entry.get(hkey)
            if not r:
                cells += "<td>—</td><td>—</td>"
                continue
            strong = ' class="sc-strong"' if r.get("significant") else ""
            cells += (f'<td>{r["mean_spread"]:+.1%}</td>'
                      f'<td{strong}>{r["t_spread"]:+.2f}</td>')
        rows += (f'<tr><td class="sc-flabel">{esc(entry["label"])}</td>'
                 f'<td class="sc-trad">{esc(entry["tradition"])}</td>{cells}</tr>')

    head = "".join(f'<th colspan="2">{esc(HORIZON_LABEL[k])}</th>' for k in keys)
    sub = "".join("<th>spread</th><th>t</th>" for _ in keys)
    # The FDR family is every test at this horizon, both neutralisation modes.
    # The table below shows only the size-neutral half, so counting the whole
    # family here would promise more bold cells than the table can show.
    sn = {k: [r for r in hz[k]["results"]
              if r.get("status") == "ok" and r.get("size_neutral")] for k in keys}
    tiles = "".join(
        f'<div class="sc-tile"><div class="k">{esc(HORIZON_LABEL[k])}</div>'
        f'<div class="v">{hz[k]["independent_windows"]:.1f}</div>'
        f'<div class="s">independent windows · {hz[k]["n_rebalances"]} rebalances · '
        f'{sum(1 for r in sn[k] if r.get("significant"))} of {len(sn[k])} size-neutral '
        f'tests survive correction</div></div>' for k in keys)
    family = ", ".join(
        f'{HORIZON_LABEL[k]} {hz[k]["summary"]["n_significant"]}/'
        f'{hz[k]["summary"]["n_tests"]}' for k in keys)

    return f"""
  <div class="sc-section">
    <h2>Every factor, at every horizon</h2>
    <p class="lede">Long-short quintile spread per holding period, sector- and
    size-neutral. The t-statistics carry a Newey–West correction for the overlap that longer
    horizons create on a semi-annual grid. Bold marks a result that survives
    Benjamini–Hochberg; nothing at five years is allowed to, because the horizon has too few
    independent windows to support the claim.</p>
    <div class="sc-tiles">{tiles}</div>
    <div class="sc-table-wrap" style="margin-top:16px"><table class="sc-table sc-factors">
      <thead><tr><th rowspan="2">Factor</th><th rowspan="2">Family</th>{head}</tr>
      <tr>{sub}</tr></thead><tbody>{rows}</tbody></table></div>
    <p class="sc-note"><strong>Read the signs, not the sizes.</strong> At one year every
    value, quality and safety factor in this column is <em>negative</em> once size is
    controlled, and the one that clears correction is Graham's own scorecard pointing the
    wrong way: passing more of his defensive tests predicted worse returns, in 85% of
    periods. The tempting reading is "buy junk". The likelier one is that this is
    survivorship wearing a different hat: in a universe defined by who is still in the index
    today, the riskiest companies that survived are the most selected, so risk looks
    rewarded because the risky failures are missing from the sample. Insider buying is the
    only factor positive at both horizons where inference is possible — significantly so
    before size control, positive but inside the noise after it.</p>
    <p class="sc-note">Counting the full family at each horizon, both neutralisation modes
    together: {family} survive Benjamini–Hochberg. Most of those survivors are the
    sector-neutral-only versions of the insider factors, which is why the column shown here
    — the stricter one — has fewer bold cells than the tiles' family counts would
    suggest.</p>
  </div>"""


def render_technical_limit() -> str:
    rows = ""
    for label, h, test in (("1 day", 1, 1512), ("1 week", 5, 1512),
                           ("1 year", 252, 2200), ("5 years", 1260, 2200)):
        n_eff = test / h
        mde = _mde(n_eff)
        impossible = mde > 1.0
        rows += (f'<tr><td>{esc(label)}</td><td>{h:,}</td><td>{n_eff:,.1f}</td>'
                 f'<td class="{"sc-neg" if impossible else ""}">'
                 f'{"impossible" if impossible else f"{mde:.3f}"}</td></tr>')
    return f"""
  <div class="sc-section">
    <h2>Why the per-ticker technical screen cannot answer this</h2>
    <p class="lede">The original technical study tested each company separately: train on its
    own history, predict its own next move, score the AUC. That design needs many
    independent observations <em>inside one company's history</em>, and a fixed-horizon
    label consumes them fast — a 1-year label on fourteen years of data leaves about nine.</p>
    <div class="sc-table-wrap"><table class="sc-table">
      <thead><tr><th>Horizon</th><th>Label length (bars)</th><th>Effective observations</th>
      <th>Smallest AUC resolvable</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p class="sc-note">At one year the smallest edge that design could detect is an AUC of
    <strong>1.047</strong>, and at five years <strong>1.723</strong>. Those are not
    probabilities. No amount of compute fixes it; the information is not in the sample.
    <br><br>So the technical models are extended here the only way that works: the same
    features, ranked <em>across</em> the 528 companies on each rebalance date instead of
    through one company's calendar. The cross-section supplies the observations that the
    calendar cannot, and momentum, reversal, low volatility, drawdown and trend appear in
    the factor table above on exactly the same footing as the fundamentals.</p>
  </div>"""


def render_holdings(pf: dict, dd: dict, screen: dict) -> str:
    profiles = dd.get("profiles", {})
    flag_text = dd.get("flag_text", {})
    by_ticker = {r["ticker"]: r for r in screen["rows"]}

    rows = ""
    for i, h in enumerate(pf.get("holdings", [])):
        t = h["ticker"]
        p = profiles.get(t, {})
        s = by_ticker.get(t, {})
        flags = p.get("flags", [])
        chips = "".join(
            f'<span class="chip chip--none"><span class="glyph" aria-hidden="true">!</span>'
            f'{esc(flag_text.get(f, f))}</span>' for f in flags) or \
            '<span class="chip chip--edge"><span class="glyph" aria-hidden="true">✓</span>'\
            'No flags raised</span>'

        def g(key, spec="+.1%", src=p):
            v = src.get(key)
            return format(v, spec) if isinstance(v, (int, float)) and v == v else "—"

        rows += (
            f'<tr class="sc-head-row" data-row="{i}">'
            f'<td class="tk">{esc(t)}</td>'
            f'<td>{esc(str(h.get("name", ""))[:30])}</td>'
            f'<td class="sc-sector">{esc(p.get("industry") or h.get("sector") or "")[:30]}</td>'
            f'<td>{int(h.get("insider_buyers") or 0)}</td>'
            f'<td>{g("revenue_cagr_5y")}</td>'
            f'<td>{g("operating_margin_latest")}</td>'
            f'<td>{g("roe_mean_5y")}</td>'
            f'<td>{g("share_count_cagr_5y")}</td>'
            f'<td>{p.get("profitable_years", "—")}/{p.get("years_counted", 5)}</td>'
            f'<td><span class="sc-expand" data-row="{i}">details</span></td></tr>'
            f'<tr class="sc-detail" id="d{i}" hidden><td colspan="10">'
            f'<div class="sc-dd">'
            f'<div><span class="k">Five-year trajectory</span>'
            f'<ul><li>Revenue CAGR <strong>{g("revenue_cagr_5y")}</strong>, '
            f'earnings CAGR <strong>{g("earnings_cagr_5y")}</strong></li>'
            f'<li>Operating margin <strong>{g("operating_margin_latest")}</strong> now, '
            f'trend <strong>{g("operating_margin_trend", "+.2%")}</strong> a year</li>'
            f'<li>Mean ROE <strong>{g("roe_mean_5y")}</strong>, earnings volatility '
            f'<strong>{g("earnings_volatility", ".2f")}</strong>× their own mean</li>'
            f'<li>Free cash flow positive in <strong>{p.get("fcf_positive_years", "—")}'
            f'/{p.get("fcf_years_counted", 5)}</strong> years</li>'
            f'<li>Debt/equity <strong>{g("leverage_latest", ".2f")}</strong>, '
            f'trend <strong>{g("leverage_trend", "+.3f")}</strong> a year</li></ul></div>'
            f'<div><span class="k">Valuation &amp; signal</span>'
            f'<ul><li>P/E (5-yr normalised) <strong>{fmt(h.get("normalized_pe"), ".1f")}</strong>, '
            f'P/B <strong>{fmt(h.get("pb"), ".2f")}</strong></li>'
            f'<li>FCF/EV <strong>{pct(h.get("fcf_ev_yield"))}</strong>, '
            f'Graham score <strong>{int(h.get("graham_score") or 0)}/8</strong></li>'
            f'<li>Insider buyers (6m) <strong>{int(h.get("insider_buyers") or 0)}</strong>, '
            f'${(h.get("insider_buy_value") or 0)/1e6:.2f}m bought</li>'
            f'<li>Momentum (6m) <strong>{pct(s.get("tech_mom_126d"))}</strong>, '
            f'drawdown <strong>{pct(s.get("tech_drawdown"))}</strong></li>'
            f'<li>Incorporated {esc(p.get("state") or "—")}, '
            f'{esc(p.get("exchange") or "—")}</li></ul></div>'
            f'<div><span class="k">Risk flags</span><div class="sc-chips">{chips}</div></div>'
            f'</div></td></tr>')

    return f"""
  <div class="sc-section">
    <h2>The holdings, with the business behind each</h2>
    <p class="lede">Thirty names ranked on insider buying. Every column after the buyer
    count is measured from the company's own filings over five fiscal years — not a view
    about the company, a record of what it has done. Open a row for the full trajectory.</p>
    <div class="sc-table-wrap"><table class="sc-table sc-screen" id="sc-holdings">
      <thead><tr><th>Ticker</th><th>Company</th><th>Industry</th><th>Insider buyers</th>
      <th>Revenue CAGR</th><th>Op. margin</th><th>ROE</th><th>Share count</th>
      <th>Profitable</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="sc-note"><strong>What this cannot tell you.</strong> Competitive position,
    management quality, whether the end market is growing or dying, and whether the insider
    who bought knows something — all of which matter more over five years than anything in
    the table. These numbers are the input to that judgement, not a replacement for it. The
    five-year evidence above also does not support holding any of this for five years on
    statistical grounds; the case for a long hold is lower turnover and tax, not a measured
    long-horizon edge.</p>
  </div>"""


EXTRA = """
.sc-head-row td { cursor: default; }
.sc-expand { color: var(--accent); cursor: pointer; font-size: 11.5px; user-select: none; }
.sc-expand:hover { text-decoration: underline; }
.sc-detail td { background: var(--surface-sunken); }
.sc-dd { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(min(260px,100%), 1fr));
         padding: 6px 2px 10px; }
.sc-dd .k { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
            color: var(--text-muted); font-weight: 600; }
.sc-dd ul { margin: 6px 0 0; padding-left: 17px; }
.sc-dd li { margin: 3px 0; font-size: 12.5px; color: var(--text-secondary); line-height: 1.5; }
.sc-dd strong { color: var(--text-primary); }
.sc-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 7px; }
"""

PAGE_SCRIPT = SCRIPT.rstrip()[:-5] + """
  document.querySelectorAll('.sc-expand').forEach(function (el) {
    el.addEventListener('click', function () {
      var row = document.getElementById('d' + el.getAttribute('data-row'));
      if (!row) return;
      row.hidden = !row.hidden;
      el.textContent = row.hidden ? 'details' : 'hide';
    });
  });
})();
"""


PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-Horizon Small Caps</title>
<meta name="description" content="The same small-cap models held for 6 months, 1 year and 5 years, with per-company due diligence from SEC filings.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232a78d6'/%3E%3Cpath d='M6 24l7-9 5 5 8-13' stroke='%23fff' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<style>{css}</style>
</head>
<body>
<div class="sc-page">
  <div class="sc-head">
    <div class="sc-crumbs">
      <a href="index.html">&larr; Bitcoin dashboard</a>
      <a href="smallcaps.html">Technical study</a>
      <a href="value.html">Value study</a>
      <span>·</span><span>Rebuilt {generated}</span>
      <button class="btn" id="theme-button" style="margin-left:auto">Light</button>
    </div>
    <h1>Long-horizon small caps</h1>
    <p>The same models, asked a longer question: hold for six months, a year, or five years.
    Every company also carries a five-year record of its own filings — growth, margin, return
    on equity, dilution, leverage — so the statistical case and the business case sit on the
    same row.</p>
  </div>

  <div class="sc-verdict {verdict_class}">
    <div class="sc-verdict-mark" aria-hidden="true">{mark}</div>
    <div><h2>{headline}</h2><p>{blurb}</p></div>
  </div>

  {portfolio}
  {factors}
  {technical}
  {holdings}

  <footer class="site-footer" style="margin-top:36px">
    Rebuilt {generated} by GitHub Actions · research tool, not financial advice ·
    backtested results are not achievable returns
  </footer>
</div>
<script>{script}</script>
</body>
</html>"""


def build_page(out: Path) -> Path:
    long = json.loads((DATA / "value_factors_long.json").read_text())
    pf = json.loads((DATA / "value_portfolio.json").read_text())
    dd = json.loads((DATA / "dd_profiles.json").read_text())
    screen = json.loads((DATA / "value_screen.json").read_text())

    hz = pf.get("horizons", {})
    one, five = hz.get("1y", {}), hz.get("5y", {})
    headline = ("One year is the horizon the evidence supports — five years is "
                "not measurable here")
    blurb = (
        f"Held for a year, the insider-ranked portfolio beats a size-matched slice of the "
        f"universe by <strong>{one.get('excess_size_matched_annualised', 0):+.1%} a year</strong> "
        f"(t&nbsp;=&nbsp;{one.get('t_excess_size_matched', 0):+.2f}) across "
        f"{one.get('independent_windows', 0):.0f} independent windows — stronger than the "
        f"six-month version and half the turnover. Stretched to five years the excess looks "
        f"larger still at {five.get('excess_size_matched_annualised', 0):+.1%} a year, but it "
        f"rests on <strong>{five.get('independent_windows', 0):.1f} independent windows</strong>. "
        f"Fourteen years of history cannot distinguish that from luck, so it is published as a "
        f"description and not as a result. Separately, the per-ticker technical screen is "
        f"arithmetically incapable of answering at these horizons, so the technical features "
        f"are extended cross-sectionally instead.")

    generated = dt.datetime.now(dt.UTC)
    html_doc = PAGE.format(
        css=Path("assets/style.css").read_text() + "\n"
            + Path("static/smallcaps.css").read_text() + "\n"
            + CHART_CSS + "\n" + EXTRA_CSS + "\n" + EXTRA,
        verdict_class="sc-verdict--some", mark="◑",
        headline=esc(headline), blurb=blurb,
        portfolio=render_horizon_portfolio(pf),
        factors=render_horizon_factors(long),
        technical=render_technical_limit(),
        holdings=render_holdings(pf, dd, screen),
        script=PAGE_SCRIPT,
        generated=generated.strftime("%d %b %Y %H:%M UTC"),
    )
    out.mkdir(parents=True, exist_ok=True)
    target = out / "longterm.html"
    target.write_text(html_doc)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("site"))
    args = ap.parse_args()
    t = build_page(args.out)
    print(f"wrote {t} ({t.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
