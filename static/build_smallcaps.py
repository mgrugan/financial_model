#!/usr/bin/env python3
"""Render the small-cap edge study to a standalone page.

Deliberately not part of the Dash single-page app: this is a 500-row table with
its own sorting, filtering and two SVG charts, and it must keep rendering even
if the Bitcoin snapshot build fails. It shares the site's stylesheet, so the
theme toggle and the palette stay common between the two pages.

Run:  python -m static.build_smallcaps --out site
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data")

# Status roles are fixed, never themed, and always ship with a glyph + word.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
# Categorical slots 1 and 2 from the shared palette; validated for CVD in both modes.
SERIES = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt(value: object, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(f):
        return dash
    return format(f, spec)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def histogram_svg(real: list[float], placebo: list[float], lo: float, hi: float,
                  bins: int, x_label: str, chart_id: str,
                  null_line: float | None = None,
                  null_label: str = "") -> str:
    """Real distribution as filled bars, placebo as a stepped outline.

    Two encodings rather than two fills: overlapping translucent histograms turn
    to mud, and the mark difference means the comparison survives a colourblind
    reader and a greyscale print without relying on hue.
    """
    W, H = 520, 210
    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 34
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]

    def counts(values: list[float]) -> list[float]:
        out = [0.0] * bins
        for v in values:
            if v is None or not math.isfinite(v):
                continue
            k = int((v - lo) / (hi - lo) * bins)
            out[min(max(k, 0), bins - 1)] += 1.0
        total = sum(out) or 1.0
        return [c / total * 100.0 for c in out]          # percent of that family

    real_pct, plac_pct = counts(real), counts(placebo)
    ymax = max(max(real_pct or [1]), max(plac_pct or [1])) * 1.14 or 1.0

    def X(v: float) -> float:
        return pad_l + (v - lo) / (hi - lo) * pw

    def Y(v: float) -> float:
        return pad_t + ph - (v / ymax) * ph

    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Distribution of {esc(x_label)}, real small caps versus random walks">'
    ]

    # Recessive grid.
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = Y(ymax * frac)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                     f'class="sc-grid"/>')

    # Real: filled bars with a 2px surface gap between neighbours.
    for i, pct in enumerate(real_pct):
        if pct <= 0:
            continue
        x0, x1 = X(edges[i]), X(edges[i + 1])
        y = Y(pct)
        parts.append(
            f'<rect class="sc-bar" x="{x0+1:.1f}" y="{y:.1f}" width="{max(x1-x0-2,0.5):.1f}" '
            f'height="{pad_t+ph-y:.1f}" rx="2" '
            f'data-tip="{edges[i]:.3f}–{edges[i+1]:.3f} · {pct:.1f}% of real names"/>')

    # Placebo: stepped outline.
    pts: list[str] = []
    for i, pct in enumerate(plac_pct):
        x0, x1 = X(edges[i]), X(edges[i + 1])
        pts += [f"{x0:.1f},{Y(pct):.1f}", f"{x1:.1f},{Y(pct):.1f}"]
    if pts:
        parts.append(f'<polyline class="sc-outline" points="{" ".join(pts)}"/>')

    if null_line is not None:
        x = X(null_line)
        parts.append(f'<line class="sc-null" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{pad_t+ph}"/>')
        if null_label:
            parts.append(f'<text class="sc-axis" x="{x:.1f}" y="{pad_t-2}" '
                         f'text-anchor="middle">{esc(null_label)}</text>')

    # Axes.
    parts.append(f'<line class="sc-axisline" x1="{pad_l}" y1="{pad_t+ph}" '
                 f'x2="{pad_l+pw}" y2="{pad_t+ph}"/>')
    for frac in (0.0, 0.5, 1.0):
        v = lo + (hi - lo) * frac
        parts.append(f'<text class="sc-axis" x="{X(v):.1f}" y="{pad_t+ph+14}" '
                     f'text-anchor="middle">{v:.2f}</text>')
    for frac in (0.5, 1.0):
        parts.append(f'<text class="sc-axis" x="{pad_l-6}" y="{Y(ymax*frac)+3.5:.1f}" '
                     f'text-anchor="end">{ymax*frac:.0f}%</text>')
    parts.append(f'<text class="sc-axis" x="{pad_l+pw/2:.1f}" y="{H-4}" '
                 f'text-anchor="middle">{esc(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
COLUMNS = [
    ("ticker",   "Ticker",     "text"),
    ("name",     "Company",    "text"),
    ("sector",   "Sector",     "text"),
    ("auc_1d",   "AUC 1-day",  "num"),
    ("auc_1w",   "AUC 1-week", "num"),
    ("q_best",   "Best q",     "num"),
    ("cagr",     "Strat CAGR", "num"),
    ("hold",     "Buy & hold", "num"),
    ("verdict",  "Reading",    "text"),
]


def build_ticker_rows(screen: dict, inventory: dict) -> list[dict]:
    meta = {r["ticker"]: r for r in inventory["rows"]}
    by_ticker: dict[str, dict] = {}
    for row in screen["rows"]:
        if row.get("status") != "ok":
            continue
        entry = by_ticker.setdefault(row["ticker"], {})
        entry[row["horizon"]] = row

    out = []
    for ticker, horizons in sorted(by_ticker.items()):
        info = meta.get(ticker, {})
        qs = [h.get("q_value") for h in horizons.values() if h.get("q_value") is not None]
        edge = any(h.get("edge") for h in horizons.values())
        # Report economics from the longer horizon where available: a 1-day
        # strategy on a small cap is dominated by the spread assumption.
        econ = horizons.get("1w") or horizons.get("1d") or {}
        out.append({
            "ticker": ticker,
            "name": info.get("name", ""),
            "sector": info.get("sector", ""),
            "auc_1d": horizons.get("1d", {}).get("auc"),
            "auc_1w": horizons.get("1w", {}).get("auc"),
            "p_1d": horizons.get("1d", {}).get("p_value"),
            "p_1w": horizons.get("1w", {}).get("p_value"),
            "q_best": min(qs) if qs else None,
            "cagr": econ.get("strategy_cagr"),
            "hold": econ.get("buyhold_cagr"),
            "n_eff_1w": horizons.get("1w", {}).get("n_eff"),
            "edge": bool(edge),
        })
    return out


def render_table(rows: list[dict]) -> str:
    head = "".join(
        f'<th data-key="{esc(k)}" data-type="{esc(t)}" scope="col">{esc(label)}</th>'
        for k, label, t in COLUMNS)

    body: list[str] = []
    for r in rows:
        chip = ('<span class="chip chip--edge"><span class="glyph" aria-hidden="true">●</span>EDGE</span>'
                if r["edge"] else
                '<span class="chip chip--none"><span class="glyph" aria-hidden="true">▲</span>NO EDGE</span>')
        body.append(
            f'<tr data-edge="{1 if r["edge"] else 0}" '
            f'data-search="{esc((r["ticker"] + " " + r["name"] + " " + r["sector"]).lower())}">'
            f'<td class="tk">{esc(r["ticker"])}</td>'
            f'<td>{esc(r["name"])}</td>'
            f'<td>{esc(r["sector"])}</td>'
            f'<td data-v="{fmt(r["auc_1d"], ".6f", "")}">{fmt(r["auc_1d"])}</td>'
            f'<td data-v="{fmt(r["auc_1w"], ".6f", "")}">{fmt(r["auc_1w"])}</td>'
            f'<td data-v="{fmt(r["q_best"], ".6f", "")}">{fmt(r["q_best"], ".3f")}</td>'
            f'<td data-v="{fmt(r["cagr"], ".4f", "")}">{fmt(r["cagr"], "+.1f")}%</td>'
            f'<td data-v="{fmt(r["hold"], ".4f", "")}">{fmt(r["hold"], "+.1f")}%</td>'
            f'<td>{chip}</td>'
            "</tr>")

    return (f'<div class="sc-table-wrap"><table class="sc-table" id="sc-table">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def build_page(out: Path) -> Path:
    screen = json.loads((DATA / "smallcap_screen.json").read_text())
    inventory = json.loads((DATA / "smallcap_inventory.json").read_text())
    stage2_path = DATA / "smallcap_stage2.json"
    stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else None

    rows = build_ticker_rows(screen, inventory)
    real_ok = [r for r in screen["rows"] if r.get("status") == "ok"]
    plac_ok = [r for r in screen["placebo_rows"] if r.get("status") == "ok"]
    alpha = screen["fdr_alpha"]

    n_tickers = len(rows)
    n_edge = sum(r["edge"] for r in rows)
    n_tests = len(real_ok)
    raw_hits = sum(1 for r in real_ok if (r["p_value"] or 1) < 0.05)
    plac_edges = sum(1 for r in plac_ok if r.get("edge"))
    plac_raw = sum(1 for r in plac_ok if (r["p_value"] or 1) < 0.05)

    aucs_real = [r["auc"] for r in real_ok]
    aucs_plac = [r["auc"] for r in plac_ok]
    mean_real = sum(aucs_real) / len(aucs_real)
    mean_plac = sum(aucs_plac) / len(aucs_plac)

    # ---- headline ---------------------------------------------------------
    if n_edge == 0:
        verdict_class, mark = "sc-verdict--none", "✕"
        headline = f"No edge found in any of the {n_tickers} small caps"
        blurb = (
            f"{n_tests} independent tests were run — one per ticker per horizon. "
            f"{raw_hits} came back significant at the usual 5% threshold, which sounds "
            f"like a lot until you note that pure chance produces about "
            f"{0.05 * n_tests:.0f}. After Benjamini–Hochberg correction across the whole "
            f"family, <strong>none</strong> survives at a {alpha:.0%} false-discovery rate. "
            f"The identical pipeline run over {len(plac_ok)} synthetic random walks — where "
            f"there is provably nothing to find — produced {plac_raw} raw hits and "
            f"{plac_edges} survivors, so the screen is behaving exactly as it does on noise.")
    else:
        verdict_class, mark = "sc-verdict--some", "✓"
        headline = (f"{n_edge} of {n_tickers} small caps show a signal that survives "
                    f"multiple-testing correction")
        blurb = (
            f"{n_tests} tests were run; {raw_hits} were significant at a raw 5% threshold "
            f"(chance alone gives about {0.05 * n_tests:.0f}), and {n_edge} tickers survive "
            f"Benjamini–Hochberg at a {alpha:.0%} false-discovery rate. The same pipeline over "
            f"{len(plac_ok)} synthetic random walks returned {plac_edges} survivors. "
            f"Surviving this screen means the result is unlikely to be pure noise — it does "
            f"not yet mean the signal is tradeable. Every survivor is re-run below through "
            f"the full eleven-model stack.")

    tiles = [
        ("Small caps tested", f"{n_tickers}", "S&amp;P SmallCap 600 members with enough clean history"),
        ("Tests run", f"{n_tests}", "one per ticker per horizon (1&nbsp;day, 1&nbsp;week)"),
        ("Raw hits at p&lt;0.05", f"{raw_hits}",
         f"chance alone gives ≈{0.05 * n_tests:.0f} — this is why raw p-values cannot be used"),
        (f"Survive BH q&lt;{alpha:g}", f"{n_edge}",
         "the only number that licenses a green reading"),
        ("Placebo survivors", f"{plac_edges}",
         f"of {len(plac_ok)} tests on random walks — the screen's measured error rate"),
        ("Mean AUC", f"{mean_real:.4f}",
         f"random walks scored {mean_plac:.4f}; 0.5000 is a coin flip"),
    ]
    tiles_html = "".join(
        f'<div class="sc-tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in tiles)

    # ---- charts -----------------------------------------------------------
    # Fit the axis to the data rather than a fixed window: values outside a
    # hardcoded range would pile into the edge bins and hide the real spread.
    span = max(abs(a - 0.5) for a in aucs_real + aucs_plac if math.isfinite(a))
    half = max(math.ceil(span * 100) / 100 + 0.01, 0.05)
    auc_chart = histogram_svg(aucs_real, aucs_plac, 0.5 - half, 0.5 + half, 30,
                              "Out-of-sample AUC", "auc",
                              null_line=0.5, null_label="coin flip")
    p_chart = histogram_svg([r["p_value"] for r in real_ok],
                            [r["p_value"] for r in plac_ok], 0.0, 1.0, 20,
                            "p-value", "pval")

    legend = (
        '<div class="sc-legend">'
        f'<span><i class="sc-swatch" style="background:var(--sc-series-1)"></i>'
        f'{n_tickers} real small caps</span>'
        '<span style="color:var(--sc-series-2)"><i class="sc-swatch sc-swatch--line"></i>'
        f'<span style="color:var(--text-secondary)">{len(plac_ok)//2} random walks (placebo)</span></span>'
        "</div>")

    # ---- stage 2 ----------------------------------------------------------
    stage2_html = render_stage2(stage2, n_edge)

    generated = dt.datetime.now(dt.UTC)
    html_doc = PAGE.format(
        css=Path("assets/style.css").read_text() + "\n"
            + Path(__file__).with_name("smallcaps.css").read_text() + "\n" + CHART_CSS,
        verdict_class=verdict_class,
        mark=mark,
        headline=esc(headline),
        blurb=blurb,
        tiles=tiles_html,
        auc_chart=auc_chart,
        p_chart=p_chart,
        legend=legend,
        table=render_table(rows),
        n_tickers=n_tickers,
        n_edge=n_edge,
        n_none=n_tickers - n_edge,
        alpha=f"{alpha:g}",
        stage2=stage2_html,
        method=METHOD.format(alpha=f"{alpha:.0%}", n_tests=n_tests,
                             n_plac=len(plac_ok), cost=20),
        script=SCRIPT,
        generated=generated.strftime("%d %b %Y %H:%M UTC"),
        screened_window=esc(f"{real_ok[0]['start']} → {real_ok[0]['end']}") if real_ok else "",
    )
    out.mkdir(parents=True, exist_ok=True)
    target = out / "smallcaps.html"
    target.write_text(html_doc)
    return target


def render_stage2(stage2: dict | None, n_edge: int) -> str:
    if not stage2:
        if n_edge == 0:
            return (
                '<div class="sc-section"><h2>Stage 2 — the full model stack</h2>'
                '<p class="lede">Stage 2 runs all eleven models (GBM, Merton, GARCH-t, Heston, '
                'regime-switching, gradient boosting, random forest, elastic-net logistic, MLP, '
                'LSTM and the neural-drift hybrid) over any ticker that clears stage 1. '
                'Nothing cleared stage 1, so there was nothing to promote — which is the point '
                'of running a cheap screen first rather than 500 × 11 walk-forwards.</p></div>')
        return ""

    rows = stage2.get("rows", [])
    if not rows:
        return ""
    body = "".join(
        f'<tr><td class="tk">{esc(r["ticker"])}</td><td>{esc(r["model"])}</td>'
        f'<td>{esc(r["horizon"])}</td>'
        f'<td>{fmt(r.get("auc"))}</td>'
        f'<td>{fmt(r.get("auc_ci_low"))} – {fmt(r.get("auc_ci_high"))}</td>'
        f'<td>{fmt(r.get("strategy_sharpe"), ".2f")}</td>'
        f'<td>{"yes" if r.get("confirms") else "no"}</td></tr>'
        for r in rows)
    return (
        '<div class="sc-section"><h2>Stage 2 — the full model stack on the survivors</h2>'
        f'<p class="lede">{esc(stage2.get("note", ""))}</p>'
        '<div class="sc-table-wrap"><table class="sc-table">'
        '<thead><tr><th>Ticker</th><th>Model</th><th>Horizon</th><th>AUC</th>'
        '<th>95% CI (block bootstrap)</th><th>Sharpe</th><th>Confirms?</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></div>")


CHART_CSS = """
:root {
  --sc-series-1: #2a78d6; --sc-series-2: #eb6834;
  --sc-grid: #e1e0d9; --sc-axis: #898781; --sc-baseline: #c3c2b7;
  --status-good: #0ca30c; --status-critical: #d03b3b;
  --status-good-ink: #076b07; --status-good-bg: rgba(12,163,12,.10);
  --status-good-line: rgba(12,163,12,.38);
  --status-critical-ink: #a52a2a; --status-critical-bg: rgba(208,59,59,.09);
  --status-critical-line: rgba(208,59,59,.34);
}
html[data-theme="dark"] {
  --sc-series-1: #3987e5; --sc-series-2: #d95926;
  --sc-grid: #2c2c2a; --sc-axis: #898781; --sc-baseline: #383835;
  --status-good-ink: #4cc94c; --status-good-bg: rgba(12,163,12,.16);
  --status-good-line: rgba(12,163,12,.45);
  --status-critical-ink: #ec7b7b; --status-critical-bg: rgba(208,59,59,.15);
  --status-critical-line: rgba(208,59,59,.42);
}
.sc-grid { stroke: var(--sc-grid); stroke-width: 1; }
.sc-axisline { stroke: var(--sc-baseline); stroke-width: 1; }
.sc-axis { fill: var(--sc-axis); font-size: 10px;
           font-family: var(--sans); }
.sc-bar { fill: var(--sc-series-1); }
.sc-bar:hover { fill: var(--sc-series-2); }
.sc-outline { fill: none; stroke: var(--sc-series-2); stroke-width: 2;
              stroke-dasharray: 5 3; stroke-linejoin: round; }
.sc-null { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; }
"""

METHOD = """
<h3>What each ticker was actually asked</h3>
<p>For every name, an expanding-window walk-forward: train on everything known up to a
quarter boundary, predict each of the next 63 sessions, then roll forward. Roughly six
years of out-of-sample predictions per ticker per horizon. The label is the sign of the
forward return over 1 session and over 5 sessions.</p>

<h3>Why the model is deliberately plain</h3>
<p>Stage 1 uses one L2-regularised logistic regression over the same 40 causal features
as the Bitcoin stack — identically specified for every ticker, with no per-name tuning
and no choosing the best of several models. That restraint is the point. The maximum of
eleven models over 500 tickers has a far wider null distribution than any single model
does, so &ldquo;the best model on the best ticker&rdquo; would clear 0.5 by a wide margin
with nothing real underneath. One pre-specified model keeps the null distribution
something we can actually compute against.</p>

<h3>Multiple testing is the whole problem</h3>
<p>{n_tests} tests at a 5% threshold yield about {n_tests} × 0.05 false positives when
nothing is there. Reading those as discoveries is the standard way a 500-name study
fools its author. Every p-value here goes through Benjamini&ndash;Hochberg across the
entire family, and only the resulting q-value below {alpha} turns a row green.</p>

<h3>The standard error is computed on effective sample size</h3>
<p>A 5-session label overlaps its neighbour on 4 of its 5 days, so 1,500 weekly
predictions are not 1,500 independent observations — average uniqueness is 1/h, giving
n/5. Using the nominal count would understate the standard error by √5 ≈ 2.2×, which is
the entire difference between &ldquo;significant&rdquo; and &ldquo;a coin flip&rdquo;.</p>

<h3>The placebo arm</h3>
<p>{n_plac} tests were run over synthetic random walks with realistic volatility, pushed
through byte-identical code. There is no signal in them by construction, so whatever the
screen reports there is its own false-positive rate — measured rather than asserted.</p>

<h3>What would still be wrong even if a name turned green</h3>
<ul>
<li><strong>Survivorship.</strong> These are today's index members. Companies that were
dropped after collapsing are absent, which flatters every return column on this page.
AUC is largely immune (it measures ranking within a series, not drift), but the
CAGR columns should be read as indicative, not achievable.</li>
<li><strong>Costs.</strong> The strategy columns charge {cost} bps round-trip. Real small-cap
spreads are frequently wider, and wider still in exactly the names where a model looks
most confident.</li>
<li><strong>Point-in-time membership.</strong> Index constituents are as of today, applied
backwards. A name only joined the 600 after it had already performed.</li>
<li><strong>Capacity.</strong> Several names here trade a few million dollars a day. A
signal that is real can still be untradeable at any size that matters.</li>
</ul>
"""

SCRIPT = """
(function () {
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem('sc-theme'); } catch (e) {}
  if (stored) root.setAttribute('data-theme', stored);
  var btn = document.getElementById('theme-button');
  function paint() {
    btn.textContent = root.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark';
  }
  paint();
  btn.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('sc-theme', next); } catch (e) {}
    paint();
  });

  var table = document.getElementById('sc-table');
  var tbody = table.querySelector('tbody');
  var allRows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
  var search = document.getElementById('sc-search');
  var count = document.getElementById('sc-count');
  var filter = 'all';

  function apply() {
    var q = (search.value || '').trim().toLowerCase();
    var shown = 0;
    allRows.forEach(function (tr) {
      var isEdge = tr.getAttribute('data-edge') === '1';
      var ok = (filter === 'all') || (filter === 'edge' && isEdge) ||
               (filter === 'none' && !isEdge);
      if (ok && q) ok = tr.getAttribute('data-search').indexOf(q) !== -1;
      tr.hidden = !ok;
      if (ok) shown++;
    });
    count.textContent = shown + ' of ' + allRows.length + ' shown';
  }

  document.querySelectorAll('.sc-seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      filter = b.getAttribute('data-filter');
      document.querySelectorAll('.sc-seg button').forEach(function (o) {
        o.setAttribute('aria-pressed', String(o === b));
      });
      apply();
    });
  });
  search.addEventListener('input', apply);

  // Sorting. Numeric columns read data-v so "—" and "+12.3%" both sort sanely.
  var sortKey = null, sortDir = 1;
  table.querySelectorAll('thead th').forEach(function (th, idx) {
    th.addEventListener('click', function () {
      var type = th.getAttribute('data-type');
      if (sortKey === idx) { sortDir = -sortDir; } else { sortKey = idx; sortDir = 1; }
      table.querySelectorAll('thead th').forEach(function (o) { o.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', sortDir === 1 ? 'ascending' : 'descending');
      var rows = allRows.slice();
      rows.sort(function (a, b) {
        var ca = a.children[idx], cb = b.children[idx];
        if (type === 'num') {
          var va = parseFloat(ca.getAttribute('data-v'));
          var vb = parseFloat(cb.getAttribute('data-v'));
          var na = isNaN(va), nb = isNaN(vb);
          if (na && nb) return 0;
          if (na) return 1;            // blanks always sink, either direction
          if (nb) return -1;
          return (va - vb) * sortDir;
        }
        return ca.textContent.trim().localeCompare(cb.textContent.trim()) * sortDir;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });
  apply();

  // Shared tooltip for the SVG charts.
  var tip = document.createElement('div');
  tip.className = 'sc-tip';
  document.body.appendChild(tip);
  document.querySelectorAll('svg [data-tip]').forEach(function (el) {
    el.addEventListener('mouseenter', function () {
      tip.textContent = el.getAttribute('data-tip');
      tip.setAttribute('data-show', '1');
    });
    el.addEventListener('mousemove', function (ev) {
      tip.style.left = (ev.clientX + 12) + 'px';
      tip.style.top = (ev.clientY - 30) + 'px';
    });
    el.addEventListener('mouseleave', function () { tip.removeAttribute('data-show'); });
  });
})();
"""


PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Small-Cap Edge Study</title>
<meta name="description" content="Walk-forward directional tests across 500+ S&amp;P SmallCap 600 names, with Benjamini-Hochberg control and a random-walk placebo arm.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23d03b3b'/%3E%3Cpath d='M6 22l6-7 5 4 9-11' stroke='%23fff' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<style>{css}</style>
</head>
<body>
<div class="sc-page">
  <div class="sc-head">
    <div class="sc-crumbs">
      <a href="index.html">&larr; Bitcoin model dashboard</a>
      <span>·</span><span>Rebuilt {generated}</span>
      <button class="btn" id="theme-button" style="margin-left:auto">Light</button>
    </div>
    <h1>Small-cap edge study</h1>
    <p>The same modelling stack, pointed at {n_tickers} S&amp;P SmallCap 600 companies instead of
    Bitcoin. Each name gets its own walk-forward backtest over {screened_window}. The question
    is narrow and the answer is a colour: is there any evidence this ticker's next move is
    predictable, once you account for the fact that testing 500 things guarantees some of them
    look special?</p>
  </div>

  <div class="sc-verdict {verdict_class}">
    <div class="sc-verdict-mark" aria-hidden="true">{mark}</div>
    <div><h2>{headline}</h2><p>{blurb}</p></div>
  </div>

  <div class="sc-tiles">{tiles}</div>

  <div class="sc-section">
    <h2>Real names against pure noise</h2>
    <p class="lede">If the small caps carried signal, their AUC distribution would sit to the
    right of the random walks'. It does not. The p-value chart is the same evidence in a
    sharper form: under a true null, p-values are uniform — a flat histogram. Real signal
    shows up as a spike in the leftmost bin.</p>
    <div class="sc-charts">
      <div class="sc-card">
        <h3>Where the AUCs landed</h3>
        <p class="sub">Share of each family, by out-of-sample AUC</p>
        {auc_chart}{legend}
      </div>
      <div class="sc-card">
        <h3>p-value distribution</h3>
        <p class="sub">Flat = no signal anywhere. A left spike = real discoveries.</p>
        {p_chart}{legend}
      </div>
    </div>
  </div>

  <div class="sc-section">
    <h2>Every ticker</h2>
    <p class="lede">Green means the evidence survived correction for having tested
    {n_tickers} names. Red means it did not. Sort any column; search by ticker, company or
    sector. Currently {n_edge} green and {n_none} red.</p>
    <div class="sc-controls">
      <div class="sc-seg" role="group" aria-label="Filter by reading">
        <button data-filter="all" aria-pressed="true">All</button>
        <button data-filter="edge" aria-pressed="false">Edge only</button>
        <button data-filter="none" aria-pressed="false">No edge</button>
      </div>
      <input type="search" id="sc-search" placeholder="Search ticker, company or sector…"
             aria-label="Search tickers">
      <span class="sc-count" id="sc-count"></span>
    </div>
    {table}
    <p class="sc-note"><strong>How to read a row.</strong> AUC is the probability the model
    ranks a random up-day above a random down-day; 0.500 is a coin flip. <em>Best q</em> is the
    Benjamini&ndash;Hochberg false-discovery rate at which that ticker would first be called a
    discovery — below {alpha} earns green. <em>Strat CAGR</em> is the long/flat strategy net of
    20&nbsp;bps round-trip, on non-overlapping weekly blocks; compare it against buy &amp; hold
    in the next column, not against zero.</p>
  </div>

  {stage2}

  <div class="sc-section">
    <h2>Method, and what would still be wrong</h2>
    <div class="sc-prose">{method}</div>
  </div>

  <footer class="site-footer" style="margin-top:36px">
    Rebuilt {generated} by GitHub Actions · research tool, not financial advice ·
    past backtested performance does not predict future returns
  </footer>
</div>
<script>{script}</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    target = build_page(args.out)
    print(f"wrote {target} ({target.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
