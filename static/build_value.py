#!/usr/bin/env python3
"""Render the long-horizon value study to a standalone page.

Run:  python -m static.build_value --out site
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .build_smallcaps import CHART_CSS, esc, fmt

DATA = Path("data")


def pct(v, spec="+.2%", dash="—"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return format(f, spec) if f == f else dash


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_size_evidence(surv: dict) -> str:
    size = surv["size_factor"]
    q = size["quintiles"]
    bars = []
    hi = max(q) or 1.0
    for i, v in enumerate(q, 1):
        w = max(v / hi * 100.0, 0.5)
        label = ["largest", "", "", "", "smallest"][i - 1]
        bars.append(
            f'<div class="sc-qrow"><span class="sc-qlab">Q{i}'
            f'{" · " + label if label else ""}</span>'
            f'<span class="sc-qtrack"><span class="sc-qbar" '
            f'style="width:{w:.1f}%"></span></span>'
            f'<span class="sc-qval">{v:+.1%}</span></div>')

    # A five-column table gets its last columns clipped inside a half-width
    # card, and the last columns are the point. Compact rows instead.
    rows = "".join(
        f'<div class="sc-bench"><div class="sc-bench-h">{esc(b["label"])}</div>'
        f'<div class="sc-bench-g">{b["gap_annual"]:+.1%}<span>/yr ahead</span></div>'
        f'<div class="sc-bench-d">index {b["benchmark_annual"]:+.1%}/yr &middot; '
        f'universe {b["universe_annual"]:+.1%}/yr &middot; '
        f'universe won {b["win_rate"]:.0%} of {b["n_periods"]} periods</div></div>'
        for b in surv["benchmarks"])

    return f"""
  <div class="sc-section">
    <h2>First, the thing that invalidates the obvious answer</h2>
    <p class="lede">This universe is the S&amp;P SmallCap 600 <em>as it stands today</em>,
    run backwards. To be a member now, a company that was small in 2012 must have grown
    enough to still qualify. So "was small" mechanically predicts "went up" — not because
    small companies earn a premium, but because that is the rule by which the list was
    built. Two independent readings show how large the effect is.</p>
    <div class="sc-charts">
      <div class="sc-card">
        <h3>Market cap as a ranking factor</h3>
        <p class="sub">Mean 6-month return by size quintile, {size['n_periods']} rebalances</p>
        <div class="sc-quints">{''.join(bars)}</div>
        <p class="sc-note" style="margin-top:10px">Spread <strong>{size['mean_spread']:+.1%}</strong>
        per six months, t = <strong>{size['t_stat']:+.2f}</strong>, positive in
        <strong>{size['hit_rate']:.0%}</strong> of periods. A real premium does not go
        {size['n_periods']}-for-{size['n_periods']}. This is the selection rule made visible.</p>
      </div>
      <div class="sc-card">
        <h3>The universe against the index that actually existed</h3>
        <p class="sub">Same windows, total return, annualised</p>
        {rows}
        <p class="sc-note" style="margin-top:10px">Today's members beat the real index by
        3–4 points a year and won four periods in five. Part of that is equal- versus
        cap-weighting; the rest is survivorship.</p>
      </div>
    </div>
    <p class="sc-note"><strong>Why this contaminates value specifically.</strong> Every
    price-scaled ratio — earnings/price, book/price, EV/EBITDA — carries market
    capitalisation in its denominator. A cheap-looking stock is disproportionately a small
    one, so each value factor inherits a share of the size artifact. That is why every
    result below is reported twice: once raw, and once with size and sector removed.</p>
  </div>"""


def render_factors(factors: dict) -> str:
    results = [r for r in factors["results"] if r.get("status") == "ok"]
    by_key: dict[str, dict[str, dict]] = {}
    for r in results:
        mode = ("sec+size" if r.get("size_neutral")
                else "sector" if r["sector_neutral"] else "pooled")
        by_key.setdefault(r["key"], {})[mode] = r

    order = sorted(by_key, key=lambda k: -(by_key[k].get("sector", {}).get("t_spread") or -9))
    rows = []
    for key in order:
        modes = by_key[key]
        base = modes.get("sector") or next(iter(modes.values()))
        cells = []
        for mode in ("pooled", "sector", "sec+size"):
            r = modes.get(mode)
            if not r:
                cells.append("<td>—</td><td>—</td>")
                continue
            strong = ' class="sc-strong"' if abs(r["t_spread"]) >= 2.0 else ""
            cells.append(f"<td>{r['mean_spread']:+.2%}</td><td{strong}>{r['t_spread']:+.2f}</td>")
        rows.append(
            f'<tr><td class="sc-flabel">{esc(base["label"])}</td>'
            f'<td class="sc-trad">{esc(base["tradition"])}</td>'
            + "".join(cells)
            + f"<td>{base['mde_annualised']:.1%}</td></tr>")

    s = factors["summary"]
    return f"""
  <div class="sc-section">
    <h2>Every factor, before and after the correction</h2>
    <p class="lede">Long-short quintile spread per six months, with the t-statistic computed
    on the {factors['n_rebalances']}-period series rather than the
    {factors['n_stock_periods']:,} stock-periods. <strong>Pooled</strong> ranks everything
    together; <strong>sector</strong> ranks within industry; <strong>sec+size</strong> also
    removes the size artifact. Read the last pair of columns.</p>
    <div class="sc-table-wrap"><table class="sc-table sc-factors">
      <thead>
        <tr><th rowspan="2">Factor</th><th rowspan="2">Tradition</th>
            <th colspan="2">Pooled</th><th colspan="2">Sector-neutral</th>
            <th colspan="2">Sector + size neutral</th><th rowspan="2">Min. detectable<br>effect/yr</th></tr>
        <tr><th>spread</th><th>t</th><th>spread</th><th>t</th><th>spread</th><th>t</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    <p class="sc-note">Across all {s['n_tests']} tests, {s['raw_hits']} clear a raw 5%
    threshold and <strong>{s['n_significant']}</strong> survive Benjamini–Hochberg at a 10%
    false-discovery rate. The placebo arm — {s['placebo_tests']} random signals through the
    same code — returns {s['placebo_raw_hits']} raw hits and {s['placebo_significant']}
    survivors. The last column is what this design could have detected at all: the
    historical value premium is roughly 4–5% a year, below most of these thresholds, so a
    null here does not disprove value. It fails to find it.</p>
  </div>"""


def render_models(models: dict) -> str:
    ok = [r for r in models["results"] if r.get("status") == "ok"]
    raw = {r["model"]: r for r in ok if not r.get("size_neutral")}
    neut = {r["model"]: r for r in ok if r.get("size_neutral")}
    names = {"ridge": "Ridge regression", "random_forest": "Random forest",
             "gradient_boosting": "Gradient boosting", "mlp": "Neural network (MLP)"}
    rows = []
    for key in ["ridge", "random_forest", "gradient_boosting", "mlp"]:
        a, b = raw.get(key), neut.get(key)
        if not a:
            continue
        rows.append(
            f'<tr><td class="tk">{esc(names.get(key, key))}</td>'
            f"<td>{a['mean_spread']:+.2%}</td>"
            f'<td class="sc-strong">{a["t_spread"]:+.2f}</td>'
            f"<td>{(b['mean_spread'] if b else float('nan')):+.2%}</td>"
            f"<td>{(b['t_spread'] if b else float('nan')):+.2f}</td>"
            f"<td>{(b['mean_ic'] if b else float('nan')):+.3f}</td></tr>")
    return f"""
  <div class="sc-section">
    <h2>The machine-learning models tell the same story twice</h2>
    <p class="lede">Four cross-sectional models trained on the fundamentals, expanding
    window, predicting each rebalance from every earlier one. Market capitalisation was
    already removed from the feature list — but the models rebuild the size bet out of the
    price-scaled ratios, so the prediction itself has to be neutralised too.</p>
    <div class="sc-table-wrap"><table class="sc-table">
      <thead><tr><th rowspan="2">Model</th><th colspan="2">As trained</th>
      <th colspan="3">Size-neutralised</th></tr>
      <tr><th>spread</th><th>t</th><th>spread</th><th>t</th><th>mean IC</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    <p class="sc-note">Left: t-statistics of 2.6 to 4.3, which would be a serious finding.
    Right: the same models, same data, with the size component removed from their
    predictions — spreads near zero and information coefficients that are slightly
    <em>negative</em>. Nothing was learned about value that was not really about size.</p>
  </div>"""


def render_screen(screen: dict, factors: dict) -> str:
    rows = screen["rows"]
    ranked = [r for r in rows if r.get("graham_composite") is not None]
    ranked.sort(key=lambda r: -r["graham_composite"])

    crit = screen["criteria"]
    head = "".join(f'<th title="{esc(c["label"])}">{esc(c["label"][:14])}</th>' for c in crit)

    body = []
    for r in ranked[:60]:
        checks = "".join(
            '<td class="sc-chk">' + ("✓" if r.get(f"chk_{c['key']}") else "·") + "</td>"
            for c in crit)
        body.append(
            f'<tr><td class="tk">{esc(r["ticker"])}</td>'
            f'<td>{esc(str(r["name"])[:34])}</td>'
            f'<td class="sc-sector">{esc(r["sector"])}</td>'
            f'<td>{fmt(r.get("pe"), ".1f")}</td>'
            f'<td>{fmt(r.get("pb"), ".2f")}</td>'
            f'<td>{fmt(r.get("ev_ebitda"), ".1f") if r.get("ev_applicable") else "n/a"}</td>'
            f'<td>{pct(r.get("fcf_ev_yield"), "+.1%") if r.get("ev_applicable") else "n/a"}</td>'
            f'<td><strong>{int(r["graham_score"])}</strong>/8</td>'
            + checks + "</tr>")

    net_nets = [r["ticker"] for r in rows if r.get("is_net_net")]

    # A pooled value rank is substantially a sector bet: banks carry structurally
    # low price-to-book, so an unneutralised screen fills with them. Saying how
    # much beats leaving the reader to notice it.
    from collections import Counter
    top_sectors = Counter(r["sector"] for r in ranked[:60]).most_common(3)
    all_share = Counter(r["sector"] for r in rows)
    conc = " &middot; ".join(
        f"{esc(sec)} {n}/60 (universe {all_share[sec] / max(len(rows), 1):.0%})"
        for sec, n in top_sectors)
    return f"""
  <div class="sc-section">
    <h2>What is cheap today</h2>
    <p class="lede">The 60 cheapest of {screen['n']} companies by the Graham composite
    (earnings yield, book yield and the Graham-number ratio, equally weighted as
    cross-sectional percentiles), as of {esc(screen['as_of'])}. Built by the same
    point-in-time code as the backtest.</p>
    <div class="sc-warn"><strong>Read this as a description, not a recommendation.</strong>
    Everything above says this ranking has no demonstrated ability to predict six-month
    returns once size is controlled for. These are the companies that are statistically
    cheap on Graham's and Burry's measures right now. That is a starting point for
    reading annual reports, which is what both men actually did — not a signal.</div>
    <div class="sc-table-wrap"><table class="sc-table sc-screen">
      <thead><tr><th>Ticker</th><th>Company</th><th>Sector</th><th>P/E</th><th>P/B</th>
      <th>EV/EBITDA</th><th>FCF/EV</th><th>Graham</th>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table></div>
    <p class="sc-note">EV/EBITDA is blank for banks, insurers and REITs: enterprise value
    is not a takeover price when debt is the raw material rather than the financing.
    <br><strong>Sector concentration:</strong> {conc}. A pooled value rank is largely a
    sector bet — banks and REITs carry structurally low price-to-book — which is exactly
    why the factor table above reports a sector-neutral column.
    <br>Graham net-nets today (market cap under two-thirds of net current asset value):
    <strong>{esc(', '.join(net_nets)) if net_nets else 'none'}</strong> — in Graham's own
    era these were common; in this universe they have essentially disappeared.</p>
  </div>"""


METHOD = """
<h3>Where the fundamentals come from</h3>
<p>SEC EDGAR's XBRL <code>companyfacts</code> API, not a price vendor's fundamentals
endpoint. Every fact carries the date of the filing that first disclosed it, so a
rebalance on date T uses only what was filed by T, and where a period was reported more
than once the <em>earliest</em> filing wins. Both halves matter: a fiscal year ending
31 October is not public on 31 October, and the companies that file latest are
disproportionately the distressed ones a value screen is most interested in.</p>

<h3>Why the sample size is 28</h3>
<p>There are {n_stock:,} stock-periods but only {n_periods} rebalances, and within one
half-year every company shares that half-year's market move. Pooling the stock-periods
would overstate precision by more than an order of magnitude. Every t-statistic on this
page is computed on the period-level spread series.</p>

<h3>Decisions that changed the answer</h3>
<ul>
<li><strong>Unadjusted prices for market cap.</strong> An adjusted close is scaled
backwards through every later split and dividend, so pairing it with a point-in-time
share count misstates historical market cap — by 27% for one name in 2011, and worst for
the companies with the most corporate actions.</li>
<li><strong>Loss-makers are excluded from earnings rankings, not ranked as expensive.</strong>
A negative earnings yield sorts to the bottom of the book, which silently relabels
"loss-making" as "expensive". They are opposite kinds of company: the bottom quintile was
83% loss-making and returned +14.4%, well above the profitable middle. Graham's first
defensive test is positive earnings, so gating on it is the tradition's own rule.</li>
<li><strong>Share counts are sanity-checked.</strong> One company tags 25,829 diluted
shares against $902m of equity — a $34,900 book value per share and a price-to-book of
0.00, which sorts straight to the top of a value screen because every ratio is
understated by the same three orders of magnitude.</li>
<li><strong>Projected financials are rejected.</strong> Companies emerging from Chapter 11
file forecast figures, and mortgage REITs tag debt maturities as period ends — one
carries "periods" ending in 2030. Any fact whose period ends after the filing that
reports it is discarded.</li>
</ul>

<h3>What would still be wrong</h3>
<ul>
<li><strong>Survivorship is not fully repairable here.</strong> Size-neutralising is a
conservative correction, and it may over-correct: value and size are genuinely correlated
in the real world, so removing the size component removes some real value signal along
with the artifact. The honest position is that this dataset cannot cleanly separate the
two, because value ratios are price-scaled and price is exactly what the selection
operated on.</li>
<li><strong>Power.</strong> The minimum detectable effect is 5–10% a year depending on the
factor. The historical value premium is roughly 4–5%. Even a clean dataset this size
could not settle the question.</li>
<li><strong>Costs and capacity.</strong> Nothing here charges spread or impact. A
semi-annual long-short book in small caps pays both, and the short leg in particular is
expensive to borrow in exactly the names a value screen wants to short.</li>
<li><strong>Point-in-time membership.</strong> Index constituents are as of today applied
backwards; a company only joined the 600 after it had already performed.</li>
</ul>
"""


PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Small-Cap Value Study</title>
<meta name="description" content="Point-in-time Graham and Burry value factors across 528 small caps, tested on 6-month forward returns with survivorship bias measured rather than assumed.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231b7f4d'/%3E%3Cpath d='M8 21h4v5H8zm6-6h4v11h-4zm6-8h4v19h-4z' fill='%23fff'/%3E%3C/svg%3E">
<style>{css}</style>
</head>
<body>
<div class="sc-page">
  <div class="sc-head">
    <div class="sc-crumbs">
      <a href="index.html">&larr; Bitcoin dashboard</a>
      <a href="smallcaps.html">&larr; Small-cap technical study</a>
      <span>·</span><span>Rebuilt {generated}</span>
      <button class="btn" id="theme-button" style="margin-left:auto">Light</button>
    </div>
    <h1>Small-cap value study</h1>
    <p>Graham and Burry, tested properly. {n_companies} small caps, {n_periods} semi-annual
    rebalances from {first_date} to {last_date}, fundamentals taken from SEC filings as they
    were actually filed. The question is whether cheap companies outperformed over the
    following six months — and the answer turns almost entirely on one correction most
    screens never make.</p>
  </div>

  <div class="sc-verdict {verdict_class}">
    <div class="sc-verdict-mark" aria-hidden="true">{mark}</div>
    <div><h2>{headline}</h2><p>{blurb}</p></div>
  </div>

  <div class="sc-tiles">{tiles}</div>

  {size_evidence}
  {factors}
  {models}
  {screen}

  <div class="sc-section">
    <h2>Method, and what would still be wrong</h2>
    <div class="sc-prose">{method}</div>
  </div>

  <footer class="site-footer" style="margin-top:36px">
    Rebuilt {generated} by GitHub Actions · research tool, not financial advice ·
    backtested results are not achievable returns
  </footer>
</div>
<script>{script}</script>
</body>
</html>"""


SCRIPT = """
(function () {
  var root = document.documentElement, stored = null;
  try { stored = localStorage.getItem('sc-theme'); } catch (e) {}
  if (stored) root.setAttribute('data-theme', stored);
  var btn = document.getElementById('theme-button');
  function paint(){ btn.textContent = root.getAttribute('data-theme')==='dark' ? 'Light':'Dark'; }
  paint();
  btn.addEventListener('click', function(){
    var next = root.getAttribute('data-theme')==='dark' ? 'light':'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('sc-theme', next); } catch(e) {}
    paint();
  });
})();
"""


def build_page(out: Path) -> Path:
    factors = json.loads((DATA / "value_factors.json").read_text())
    models = json.loads((DATA / "value_models.json").read_text())
    screen = json.loads((DATA / "value_screen.json").read_text())
    surv = json.loads((DATA / "value_survivorship.json").read_text())

    results = [r for r in factors["results"] if r.get("status") == "ok"]
    neutral = [r for r in results if r.get("size_neutral")]
    best_neutral = max(neutral, key=lambda r: r["t_spread"]) if neutral else None
    s = factors["summary"]
    size = surv["size_factor"]

    n_sig = s["n_significant"]
    if n_sig == 0:
        verdict_class, mark = "sc-verdict--none", "✕"
        headline = "No value edge survives once survivorship is accounted for"
        blurb = (
            f"Ranked naively, Graham's metrics look like they work — earnings/price returns "
            f"+3.3% per half-year with a t-statistic near 2.9. But market capitalisation in "
            f"this universe returns {size['mean_spread']:+.1%} per half-year with a "
            f"t-statistic of {size['t_stat']:+.1f} and a "
            f"<strong>{size['hit_rate']:.0%} hit rate across {size['n_periods']} periods</strong>, "
            f"which no real premium achieves — it is the selection rule of a "
            f"current-constituents list. Every price-scaled value ratio inherits part of that "
            f"artifact. Remove it and all {s['n_tests']} factor tests collapse: "
            f"<strong>{n_sig} survive</strong> multiple-testing correction, and the best "
            f"remaining is t = {best_neutral['t_spread']:+.2f}. The four machine-learning "
            f"models go the same way, from t = 4.3 to t = 0.6.")
    else:
        verdict_class, mark = "sc-verdict--some", "✓"
        headline = f"{n_sig} of {s['n_tests']} value factor tests survive correction"
        blurb = (f"{s['raw_hits']} clear a raw 5% threshold and {n_sig} survive "
                 f"Benjamini–Hochberg at 10%, against {s['placebo_significant']} of "
                 f"{s['placebo_tests']} placebo signals.")

    tiles = [
        ("Companies", f"{factors['n_companies']}", "with both price history and SEC filings"),
        ("Rebalances", f"{factors['n_rebalances']}",
         "semi-annual — the real sample size, not the 13,000 stock-periods"),
        ("Factor tests", f"{s['n_tests']}", "10 factors × 3 neutralisation modes"),
        ("Survive correction", f"{n_sig}",
         f"placebo arm: {s['placebo_significant']} of {s['placebo_tests']}"),
        ("Size factor t-stat", f"{size['t_stat']:+.1f}",
         f"{size['hit_rate']:.0%} hit rate — the survivorship artifact"),
        ("Universe vs index", f"{surv['benchmarks'][0]['gap_annual']:+.1%}",
         "per year against the real S&amp;P 600"),
    ]
    tiles_html = "".join(
        f'<div class="sc-tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{sub}</div></div>' for k, v, sub in tiles)

    dates = sorted({p["date"] for r in results for p in r.get("periods", [])})
    generated = dt.datetime.now(dt.UTC)

    html_doc = PAGE.format(
        css=Path("assets/style.css").read_text() + "\n"
            + Path(__file__).with_name("smallcaps.css").read_text() + "\n"
            + CHART_CSS + "\n" + EXTRA_CSS,
        verdict_class=verdict_class, mark=mark,
        headline=esc(headline), blurb=blurb,
        tiles=tiles_html,
        n_companies=factors["n_companies"],
        n_periods=factors["n_rebalances"],
        first_date=esc(dates[0] if dates else ""), last_date=esc(dates[-1] if dates else ""),
        size_evidence=render_size_evidence(surv),
        factors=render_factors(factors),
        models=render_models(models),
        screen=render_screen(screen, factors),
        method=METHOD.format(n_stock=factors["n_stock_periods"],
                             n_periods=factors["n_rebalances"]),
        script=SCRIPT,
        generated=generated.strftime("%d %b %Y %H:%M UTC"),
    )
    out.mkdir(parents=True, exist_ok=True)
    target = out / "value.html"
    target.write_text(html_doc)
    return target


EXTRA_CSS = """
.sc-quints { margin-top: 6px; }
.sc-qrow { display: flex; align-items: center; gap: 10px; margin: 5px 0; font-size: 12.5px; }
.sc-qlab { flex: 0 0 88px; color: var(--text-muted); }
/* The bar's percentage must resolve against a flexible track, not the whole
   row, or a long label pushes it past the viewport. */
.sc-qtrack { flex: 1 1 auto; min-width: 0; display: block; }
.sc-qbar { height: 15px; border-radius: 3px; background: var(--sc-series-1); display: block; }
.sc-qval { color: var(--text-primary); font-variant-numeric: tabular-nums; font-weight: 600; }
.sc-strong { font-weight: 700; color: var(--text-primary); }
.sc-flabel { font-weight: 600; }
.sc-trad, .sc-sector { color: var(--text-muted); font-size: 12px; }
.sc-chk { text-align: center !important; font-size: 12px; }
.sc-factors th { text-align: right; }
.sc-factors thead tr:first-child th { border-bottom: 1px solid var(--border); }
.sc-screen { font-size: 12px; }
.sc-warn { margin: 12px 0 14px; padding: 12px 14px; border-radius: var(--radius);
  border: 1px solid var(--status-critical-line); background: var(--status-critical-bg);
  color: var(--text-secondary); font-size: 13px; line-height: 1.55; }
.sc-warn strong { color: var(--text-primary); }
.sc-bench { padding: 9px 0; border-bottom: 1px solid var(--border); }
.sc-bench:last-child { border-bottom: 0; }
.sc-bench-h { font-size: 12.5px; color: var(--text-secondary); }
.sc-bench-g { font-size: 22px; font-weight: 650; letter-spacing: -0.02em;
              color: var(--text-primary); font-variant-numeric: tabular-nums; }
.sc-bench-g span { font-size: 12px; font-weight: 400; color: var(--text-muted);
                   margin-left: 5px; letter-spacing: 0; }
.sc-bench-d { font-size: 11.5px; color: var(--text-muted); margin-top: 1px; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    target = build_page(args.out)
    print(f"wrote {target} ({target.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
