# Bitcoin Model Dashboard

Eleven forecasting models — classical stochastic processes, machine learning,
neural networks and a hybrid — each producing a **full price distribution** for
Bitcoin over the next 24 hours and the next week, each carrying its own
walk-forward track record, and each giving its own read on the live Deribit
options book.

Built to run on [Render](https://render.com). Data comes from Yahoo Finance
(price history) and Deribit's public API (option chain, forward curve).

---

## Why a distribution and not a price

Every model here is required to emit a sample of terminal prices, not a point
forecast. Direction probability, target prices, confidence bands and option
expected values are all read off that single object.

That one design decision is what makes a gradient-boosted classifier and a
Heston diffusion directly comparable — and it is what lets a *classifier* price
an option, which it otherwise could not do.

---

## Research background

The models were chosen to span the ways the Black–Scholes assumptions fail for
crypto, and the ways machine learning has been shown to help.

**Brownian motion and its extensions.** Geometric Brownian motion assumes
constant volatility, which contradicts the volatility clustering visible in
every crypto series. The Merton model adds a Poisson jump component; empirical
work on Bitcoin finds jumps frequent enough that a pure diffusion assigns
near-zero probability to moves that actually happen several times a month
([high-frequency jump analysis of the Bitcoin market](https://arxiv.org/pdf/1704.08175),
[path-dependent Monte Carlo for cryptocurrency prices](https://arxiv.org/html/2405.12988)).
Heston lets volatility follow its own mean-reverting SDE correlated with the
price shock, which is what bends the option smile
([Heston, GARCH and jump-diffusion compared for option pricing](https://arxiv.org/html/2604.06068)).

**Where machine learning actually helps.** The productive direction is not
replacing the SDE but *parameterising it*: neural stochastic volatility models
represent the drift and diffusion coefficients as network outputs, and hybrid
architectures such as GARCH-LSTM embed the stochastic recursion as a layer so
the stylised facts survive
([neural SDE models for financial forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0307904X22005340),
[neural Lévy SDE for state-dependent risk and density forecasting](https://arxiv.org/pdf/2509.01041),
[predicting volatility with neural networks](https://macrosynergy.com/research/predicting-volatility-with-neural-networks/)).
The hybrid model here follows that pattern directly.

**How much directional edge is real.** Headline accuracy figures in this
literature are mostly artefacts of evaluation design. A survey of peer-reviewed
Bitcoin prediction work finds that the majority of papers use methods that
cannot establish genuine predictive ability, and that a heavily cited LSTM study
reported 52% directional accuracy — barely above chance, and *below* the
always-up base rate
([peer-reviewed evidence on Bitcoin price prediction](https://arxiv.org/pdf/2606.00071)).
Work that does use walk-forward protocols with transaction costs reports much
more modest results
([ML-based Bitcoin trading under transaction costs](https://arxiv.org/html/2606.00060v1),
[XGBoost walk-forward analysis for short-term Bitcoin](https://pyquantlab.medium.com/xgboost-for-short-term-bitcoin-prediction-walk-forward-analysis-and-thresholded-performance-b83dc2e677eb)).

That is why the backtest in this project scores against the **always-up base
rate** rather than 50%, and why every live probability carries its
out-of-sample verdict next to it.

**Options.** Crypto implied volatility runs structurally far above equity index
levels, and the greeks are the standard way to decompose that risk
([option greeks in crypto derivatives](https://blog.amberdata.io/options-greeks-explained-managing-risk-in-crypto-derivatives),
[Deribit's introduction to option greeks](https://insights.deribit.com/education/introduction-to-option-greeks/)).

---

## The eleven models

### Stochastic — Brownian motion and its relaxations

| Model | Assumption it drops |
|---|---|
| **Geometric Brownian Motion** | — the baseline: constant drift, constant vol, normal shocks |
| **Merton Jump Diffusion** | continuity of paths — adds Poisson jumps with normal sizes |
| **GARCH(1,1)-t** | constant volatility — variance follows a recursion, shocks are Student-t |
| **Heston** | deterministic volatility — variance gets its own correlated square-root SDE |
| **Markov Regime-Switching** | a single regime — two hidden states, each with its own drift and vol |

### Machine learning — 47 engineered features

| Model | Inductive bias |
|---|---|
| **Gradient-Boosted Trees (XGBoost)** | sequential residual fitting, sharp conditional splits |
| **Random Forest** | bagged decorrelated trees; biased toward the base rate |
| **Elastic-Net Logistic Regression** | linear control — shows how much of the signal is genuinely non-linear |

### Neural networks

| Model | Inductive bias |
|---|---|
| **Feed-Forward (MLP)** | smooth decision surface over the daily snapshot; 5-seed ensemble |
| **LSTM** | reads the last 24 days as an ordered path; NumPy implementation, 3-seed ensemble |

### Hybrid

| Model | What it does |
|---|---|
| **ML-Drift Diffusion** | a diffusion whose drift *and* volatility are learned functions of market state, with bootstrapped fat-tailed shocks — each learned component shrunk by its out-of-sample regression slope |

---

## How a classifier gets a price distribution

A classifier only answers *"up, with probability p"*. To turn that into a
distribution:

1. Draw a zero-drift sample of horizon log-returns by **block-bootstrapping
   standardised GARCH residuals** — this carries Bitcoin's real skew and
   kurtosis rather than a normal approximation.
2. **Shift the whole sample by a constant** until exactly `p` of its mass is
   positive. The shift is exact by construction: `c = −Quantile(sample, 1−p)`.

The learned model supplies the *location*; the stochastic backbone supplies the
*shape*. Every downstream number — target prices, option expected values,
probability of touching a strike — follows from that one object.

---

## Decisions that materially change the numbers

**Probability calibration is fitted on purged walk-forward folds.** Raw
classifier scores are badly over-confident. Platt scaling is fitted on pooled
out-of-sample predictions from several purged folds — a few thousand honest
predictions across several regimes, not a few hundred from one recent window.

**The calibration slope is confined to `[0, 3]`.** A negative slope means "this
model was anti-predictive on held-out data, so invert it." Inverting a weak
model on one contiguous block of evidence is a reliable way to manufacture
nonsense. The honest response to a model with no edge is a slope of **zero**,
which collapses its output to the base rate — and that is what the bound
enforces. It also prevents the Newton iteration diverging when scores separate
the calibration labels perfectly.

**GARCH uses a four-year window with variance targeting.** Fitted on the full
twelve-year history, GARCH(1,1) estimates persistence of *exactly* 1.000 — an
IGARCH degeneracy where the long-run variance does not exist. With a free
intercept the implied unconditional volatility came out near **61%** against
**47%** realised over the same window. Following Engle–Mezrich, omega is pinned
so the model's long-run level equals the sample's by construction. Without this
every distribution on the dashboard would have been too wide.

**The LSTM stops on validation AUC, not log-loss.** The base rate drifts between
the training block and the held-out tail, so validation log-loss is dominated by
a level shift the network cannot know about and rises from epoch one even while
the ranking keeps improving. AUC is invariant to that shift, and the probability
*level* is set afterwards by the calibrator.

**The hybrid shrinks each learned component by its out-of-sample slope.** Pooled
predictions from purged folds are regressed on what actually happened; the
fitted slope is the fraction of the forecast that survived out of sample, and
only that fraction is used. In the current fit the **volatility** component
earns a slope near 1.0 with R² ≈ 0.11–0.15, while the **drift** component's
slope collapses toward 0 at the one-week horizon. The model degrades gracefully
into a well-calibrated diffusion rather than a confident wrong answer — and the
split is itself the finding: ML improves the volatility term far more
convincingly than the drift term.

---

## The backtest protocol

Run with `python scripts/run_backtest.py` (roughly half an hour).

- **Strictly out of sample.** On test day *t* no model has seen anything after
  *t*. Feature causality is verified: a feature row computed on the full history
  is **bit-identical** to one computed on data truncated at that date.
- **Periodic re-estimation, daily state.** Parameters are re-estimated on a
  fixed schedule, as a desk would; fast-moving state (GARCH conditional
  variance, the regime filter, today's features) updates every day.
- **Purged calibration.** Each refit re-derives its calibration from purged
  folds inside its own training window.
- **Costed, non-overlapping trading.** A 7-day signal evaluated daily produces
  seven overlapping bets on the same move, flattering Sharpe by roughly √7. The
  equity curve uses non-overlapping holding periods, net of 10 bps round trip.
- **Scored against the right null.** Bitcoin rose on ~53% of days in this
  sample, so a model that is 52% accurate has *negative* skill. Brier skill and
  log-loss skill are measured against that base rate; zero means no skill.

---

## What the backtest actually found

Across **1,088 walk-forward days** (Sep 2023 – Sep 2026), **no model's AUC
confidence interval excludes 0.5, at either horizon.** Zero of 22 model-horizon
cells. That is the result.

Seven-day labels on daily bars overlap 6/7, so the independent sample is
**n_eff = 155**, not 1,088 — and the interval that follows is wide:

| Model (1 week) | AUC | 95% CI | corr(P(up), drawdown) |
|---|---|---|---|
| GARCH(1,1)-t | 0.469 | [0.377, 0.560] | — |
| Geometric Brownian Motion | 0.468 | [0.376, 0.559] | +0.63 |
| Random Forest | 0.466 | [0.374, 0.557] | +0.33 |
| ML-Drift Diffusion | 0.456 | [0.365, 0.547] | +0.66 |
| LSTM | 0.452 | [0.361, 0.543] | −0.13 |
| XGBoost | 0.451 | [0.360, 0.542] | +0.30 |
| MLP | 0.430 | [0.339, 0.520] | +0.27 |

The mechanism is still visible: `drawdown` (distance below the running all-time
high) drives most of these models' probabilities, and it correlated **−0.129**
with the forward 7-day return over the test window while the models lean
positive on it. They learned "near the highs, keep going up" from the 2017 and
2021 cycles.

**A claim in an earlier version of this file was wrong.** It reported the LSTM
as the one model above 0.5 at a week (AUC 0.523) because it leaned the other way
on drawdown. After the fixes below, the LSTM scores **0.452**. Its train/
validation split had been purging nothing, and a windowed model needs a gap of
`horizon + seq_len − 1` = 30 days, not 7 — training windows overlapped
validation windows by up to 23 days, inflating the statistic used to pick the
stopping epoch. The apparent edge was substantially that leak.

### What the eleven fixes changed

They removed biases and leaks. They did not create edge, and were not expected
to:

| | before | after |
|---|---|---|
| GARCH 24h AUC | 0.488 | **0.523** (Brier skill −0.0016 → **+0.0008**) |
| Heston 1w AUC | 0.430 | 0.443 |
| LSTM 1w AUC | 0.523 | **0.452** (leak removed) |
| MLP 1w AUC | 0.496 | 0.430 (recency weights now applied) |
| Models with AUC CI excluding 0.5 | 0 / 22 | 0 / 22 |

GARCH gained the most, from variance targeting actually running and an Itô term
that is no longer an estimator of infinity. The LSTM and MLP lost the most, from
leaks closing. Everything converged toward 0.5, which is the honest picture.

### Read the trading numbers carefully

The best 1-week Sharpe is **GBM at 1.06, returning +201.3% against buy-and-hold's
+202.5%.** It is long almost always in a market that tripled. That is beta, not
skill — and the same caution applies to the hybrid (1.01, +185.9%). A strategy
that underperforms buy-and-hold on both return and Sharpe has not demonstrated
anything, however good the Sharpe looks in isolation.

### The study is underpowered, and that is a design fact

With n_eff = 155, this window can only resolve an edge of roughly **AUC ≥ 0.625**
at 80% power. Detecting a true edge of 0.52 would need on the order of *decades*
of daily data. So the negative result does not say "these models failed" — it
says this experiment cannot resolve the question either way, and neither can any
three-year daily backtest. Reporting the interval is the finding.

## The small-cap edge study

The same stack pointed at **532 S&P SmallCap 600 companies**, on the theory that
if an edge exists anywhere it should be in the neglected tail rather than in the
most-arbitraged asset on earth. Each name gets its own walk-forward backtest over
roughly six years out of sample, at a 1-day and a 1-week horizon: **1,064 tests**.
Published at `smallcaps.html`.

### The result

| | Real small caps | Placebo (random walks) |
|---|---|---|
| Tests | 1,064 | 500 |
| Mean AUC | 0.5029 | 0.4986 |
| Cross-sectional sd | 0.0231 | 0.0226 |
| Raw hits at p&lt;0.05 | 49 (chance gives ~53) | 17 (chance gives ~25) |
| **Survive BH at 10% FDR** | **0** | **0** |

Not one name clears the bar. The strongest raw signal in the universe is PRK at
one day, AUC 0.5533, p = 3.1e-4 — which sounds decisive until you remember 1,064
tests were run: its BH q-value is 0.327, and its backtested strategy returned
+6.5% a year against +17.8% for simply holding the stock.

### Multiple testing is the entire problem

500 names × 2 horizons at a 5% threshold manufactures ~53 "discoveries" from pure
noise. The real universe produced 49 — *fewer* than chance. Every p-value goes
through Benjamini–Hochberg across the whole family, and only the q-value is
allowed to turn a row green.

### The number that could fool you

The mean AUC across all tests came out at 0.5029. Divided by the textbook
standard error (sd/√n = 0.00071) that is **z = +4.10** — a four-sigma market-wide
finding.

It is not one. The 532 names correlate **0.28** pairwise: they share a market
factor, so the tests are nowhere near independent and the variance of their mean
does not shrink like 1/N — it converges to ρσ². Rather than derive the correction,
`scripts/correlated_null.py` measures it: 14 cohorts of 22 random walks sharing a
factor at the observed ρ, pushed through byte-identical code. Against that null,
**z = +0.67**. Correlated noise wanders that far from 0.5 routinely.

The cross-sectional spread confirms it from the other side. Genuine per-name
predictability would make the real AUCs *more* dispersed than noise. Real sd is
0.0231 against a correlated-null 0.0226 — there is no market-wide edge, and none
hiding underneath the average either.

### Two data problems that do not arise for Bitcoin

Both would have manufactured a fake edge if left alone:

* **Splits and dividends.** A raw close steps −50% on a split date. That is a
  units change, not a return, and it would be the largest "move" in many of these
  histories. We take Yahoo's `adjclose` and rescale open/high/low by the same
  per-bar ratio so range features stay consistent with it.
* **Pre-listing stale quotes and bad ticks.** Several names carry years of
  zero-volume history at a frozen price (non-traded REITs, pre-IPO stubs), and two
  carry outright decimal typos — a 4.50 → 0.45 → 5.00 sequence on no volume. A bad
  tick that fully reverses next bar is *genuinely predictable*, so leaving it in
  would create exactly the false positive the study exists to rule out. Histories
  are trimmed to where sustained liquidity begins; residual spike-reversals are
  flagged, not silently repaired.

### Why two stages

Stage 1 runs **one pre-specified logistic regression** for every ticker — no
per-name tuning, no best-of-eleven. That restraint is load-bearing: the maximum
over model choices has a far wider null distribution than any single choice, so
"the best model on the best ticker" clears 0.5 easily with nothing underneath.
Only names that clear stage 1 go to the full eleven-model stack. Nothing cleared,
so stage 2 cost nothing — which is the point of screening before spending 500 × 11
walk-forwards.

Stage 2 is also explicitly labelled a *robustness check, not confirmation*: it
reuses the window stage 1 already saw, so it can demote a finding but never
promote one.

### What is still wrong even so

* **Survivorship.** These are today's index members; names dropped after
  collapsing are absent. That flatters every return column. AUC is largely immune
  — it measures ranking within a series, not drift — but the CAGR columns are
  indicative, not achievable.
* **Point-in-time membership.** Constituents as of today, applied backwards.
* **Costs.** 20 bps round-trip is charged. Real small-cap spreads are often wider,
  and widest in exactly the names where a model looks most confident.
* **Capacity.** Several names trade a few million dollars a day. A real signal can
  still be untradeable at any size that matters.

## The small-cap value study

The same universe, asked a different question: forget predicting next week, does
**cheap** beat **expensive** over the following six months? Graham's defensive
criteria and Burry's enterprise-value multiples, tested on 528 companies across
28 semi-annual rebalances from 2012 to 2026. Published at `value.html`.

### The finding

Ranked the way a normal screen ranks, value looks like it works:

| Factor | Pooled t | Sector-neutral t | **Sector + size neutral t** |
|---|---|---|---|
| Earnings / price | +2.72 | +2.88 | **+1.23** |
| Graham number / price | +2.61 | +2.36 | **−0.17** |
| Book / price | +1.54 | +2.31 | **−1.56** |
| Revenue / EV | +1.68 | +2.89 | **−1.69** |
| EBITDA / EV *(Burry's metric)* | +1.22 | +0.91 | **−0.93** |
| Free cash flow / EV | +0.33 | +0.14 | **−1.02** |

Across all 36 tests, 8 clear a raw 5% threshold and **0 survive
Benjamini–Hochberg** at a 10% false-discovery rate. The placebo arm of 40 random
signals returns 2 raw hits and 0 survivors.

### Why the third column is the only one that counts

In this universe, **market capitalisation returns +17.8% per half-year with a
t-statistic of +9.2 and is positive in 28 of 28 periods.** No real premium goes
28-for-28. It is the selection rule of a current-constituents list made visible:
to be in the S&P 600 *today*, a company that was small in 2012 must have grown
enough to still qualify, so "was small" mechanically predicts "went up".

Two independent readings agree. The universe beat the actual small-cap indices by
3.6 points a year (vs IJR) and 4.2 points (vs IWM), winning 82–89% of half-years.

Every price-scaled value ratio — earnings/price, book/price, EV/EBITDA — carries
market cap in its denominator, so each one inherits part of that artifact.
Residualising the score on the within-period size rank removes it, and when it
goes, the value effect goes with it.

### The machine-learning models make the same point twice

| Model | As trained | Size-neutralised |
|---|---|---|
| Ridge | +7.03% spread, t = +3.59 | +0.89%, t = +0.58 |
| Random forest | +8.40%, t = +4.34 | +3.04%, t = +1.91 |
| Gradient boosting | +4.26%, t = +2.62 | +0.57%, t = +0.34 |
| Neural net (MLP) | +4.73%, t = +2.62 | −1.80%, t = −1.06 |

Market cap was already removed from the feature list. The models rebuilt the size
bet out of the price-scaled ratios anyway — mean information coefficients turn
slightly *negative* once the prediction itself is neutralised. Nothing was
learned about value that was not really about size.

### Point-in-time fundamentals, from filings

Fundamentals come from SEC EDGAR's XBRL `companyfacts` API rather than a vendor,
because every fact carries the date of the filing that first disclosed it. A
rebalance on date T uses only facts filed by T, and where a period was reported
more than once the **earliest** filing wins. Both halves matter: a fiscal year
ending 31 October is not public on 31 October, and the companies that file latest
are disproportionately the distressed ones a deep-value screen is most interested
in. Vendors also overwrite history with restatements, which makes a company that
later restated earnings downward look as though the market should have known.

Four data problems, each of which would have manufactured a result:

* **Market cap needs unadjusted prices.** An adjusted close is scaled backwards
  through every later split and dividend; pairing it with a point-in-time share
  count understated ABM's 2011 market cap by 27%.
* **Share counts are sometimes tagged in thousands.** One filer reports 25,829
  diluted shares against $902m of equity — a $34,900 book value per share and a
  price-to-book of 0.00, which sorts *straight to the top* of a value screen
  because every ratio is understated by the same three orders of magnitude.
* **Some filings are forward-looking.** Companies emerging from Chapter 11 file
  projected financials and mortgage REITs tag debt maturities as period ends —
  one carries "periods" ending in 2030. Any fact whose period ends after the
  filing reporting it is discarded.
* **Loss-makers are not "expensive".** A negative earnings yield sorts to the
  bottom of the book, silently relabelling loss-making companies as expensive.
  They are opposite kinds of company: that bottom quintile was 83% loss-making
  and returned +14.4%, above the profitable middle. Graham's first defensive test
  is positive earnings, so gating on it is the tradition's own rule.

Sample size is **28**, not 13,230: within one half-year every name shares that
half-year's market move, so all inference runs on the period-level spread series.

### What this does and does not say

It does **not** disprove value investing. The minimum detectable effect here is
5–10% a year depending on the factor, and the historical value premium is roughly
4–5% — this design could not resolve it either way. Size-neutralising is also a
conservative correction that may over-correct, since value and size are genuinely
correlated in the real world.

The honest position is that **a current-constituents universe cannot answer this
question at all**, because value ratios are price-scaled and price is exactly
what the selection operated on. The screen at `value.html` is published as a
description of what is statistically cheap today — a starting point for reading
annual reports, which is what both Graham and Burry actually did — and explicitly
not as a signal.

## Options analytics

Priced with **Black-76 on each expiry's own forward** rather than Black-Scholes
on spot — crypto options are quoted and hedged against the dated future for
their expiry, and using spot introduces a basis error that shows up as a fake
skew. The forward comes from Deribit; the carry rate is implied from the futures
basis.

Greeks are computed analytically and verified against numerical differentiation:
**delta, gamma, theta, vega, rho, vanna, volga, charm**, expressed per 1 BTC of
notional in USD.

Per model you get: expected value versus the market mark, model-implied vol
versus market mark IV, real-world P(ITM) versus the risk-neutral N(d₂), a
day-by-day theta decay schedule with spot and IV held fixed, an ITM / ATM / OTM
suggestion picked by target delta (0.70 / 0.50 / 0.28) subject to open interest,
target prices from the model's own quantiles, and a defined-risk vertical spread
for comparison.

> **Model expected value is not a price.** These are discounted expected payoffs
> under each model's *real-world* distribution — what a contract is worth if
> that model is right — not arbitrage-free prices. Because the models generally
> forecast higher volatility than the market is charging, most contracts show a
> positive expected value: that is **one volatility opinion repeated across a
> surface**, not a list of independent opportunities. The dashboard states the
> model-vs-market volatility gap next to every expected-value figure.

---

## Running locally

```bash
pip install -r requirements.txt

# optional but recommended: generate the walk-forward track record (~30 min)
python scripts/run_backtest.py

python app.py                 # http://localhost:8050
```

The first page load triggers a full refresh — fetching prices, fitting eleven
models and pulling the option chain — which takes about a minute. The dashboard
shows a warming-up state until it completes, then refreshes in the background.

Verify the whole stack against live data:

```bash
python scripts/smoke_test.py
```

## Deploying — free, on GitHub Pages

There is no server. GitHub Actions fits the models on a runner (4 vCPU / 16 GB —
more than most free web tiers give you), pre-renders every view, and publishes
static files. Nothing sleeps, nothing cold-starts, there is no memory ceiling,
and it needs no credit card.

**One-time setup:** in the repository, **Settings → Pages → Source → GitHub
Actions**. Without that the deploy step fails.

Then:

| Workflow | Schedule | What it does |
|---|---|---|
| `dashboard.yml` | hourly + on push | Fits all eleven models, pulls the live option chain, renders every view, deploys to Pages |
| `backtest.yml` | weekly (Sun 05:40 UTC) | Re-runs the 40-minute walk-forward and commits `data/backtest.json` |

Both can be run on demand from the Actions tab. To build the site locally:

```bash
python -m static.build --out site --clean
cd site && python -m http.server 8000     # http://localhost:8000
```

### Why the backtest is a committed file

`data/backtest.json` is checked in on purpose. It takes 40 minutes to
regenerate, a static site has no disk to compute one on, and it barely moves
week to week — it is a three-year walk-forward, so a few extra days at the end
shift the metrics in the third decimal. The weekly workflow refreshes it and
refuses to commit an artefact with too few models, too few test days, or an
implausible AUC.

### How the static build reuses the app

The layout is not rewritten. `static/render_html.py` walks the Dash component
tree that `dashboard/layout.py` already produces and emits HTML, so the served
app and the static site come from one definition and cannot drift apart:
`dcc.Graph` becomes a div plus figure JSON, `DataTable` becomes a `<table>`,
`Dropdown` becomes a `<select>`.

Each view ships as its own JSON payload rather than being inlined. Rendering
every tab × theme × horizon × model into one document is about 8 MB the browser
must parse before first paint; instead the forecast view is inlined (253 KB
including CSS and script) and the other 117 panels are fetched on click and
cached. Figures are pre-rendered per theme rather than recoloured in the
browser, because the palette's light and dark steps are separately validated for
colour-vision separation against their own surface — one is not a transformation
of the other.

### Running it as a live server instead

`app.py` is unchanged and still runs the interactive Dash app:

```bash
python app.py                 # http://localhost:8050
```

It also still deploys to any host that runs Python. A full refresh of all eleven
models peaks at **267 MB** of RSS, so it fits a 512 MB instance. Note that
Render's free tier has no persistent disk — the app handles that (it falls back
to the committed `data/backtest.json`), but free instances also sleep after 15
minutes, and waking one refits eleven models on a shared 0.1 CPU. That is why
this project publishes statically instead.

## Layout

```
app.py                    Dash entry point; exposes `server` for gunicorn
btcmodels/
  config.py               all tunables, environment-overridable
  data.py                 Yahoo Finance client, disk cache, stale-cache fallback
  deribit.py              public Deribit client: index, forwards, option book
  features.py             47 causal features + label construction
  base.py                 ModelForecast contract, MarketContext, GARCH backbone
  stochastic.py           GBM, Merton, GARCH-t, Heston, regime-switching
  supervised.py           calibration, purged folds, recency weighting, tilting
  ml.py                   XGBoost, random forest, elastic net, MLP
  nn.py                   NumPy LSTM (forward, BPTT, Adam) + sequence model
  hybrid.py               ML-drift diffusion
  registry.py             model registry, consensus aggregation
  backtest.py             walk-forward driver and scoring
  options.py              Black-76 pricing, greeks, per-model valuation
  engine.py               snapshot orchestration and background refresh
dashboard/
  theme.py                validated colour tokens
  figures.py              chart builders
  components.py           cards, meters, tables
  layout.py               page structure and tab rendering
  callbacks.py            interactivity
static/
  build.py                renders every view to a static site
  render_html.py          Dash component tree -> HTML
  site.js                 client: tab/model/theme switching, Plotly wiring
  static.css              controls the served app gets from Dash
scripts/
  run_backtest.py         offline walk-forward run
  report.py               terminal report
  smoke_test.py           end-to-end verification against live data
data/
  backtest.json           committed walk-forward track record
.github/workflows/
  dashboard.yml           hourly build + Pages deploy
  backtest.yml            weekly walk-forward refresh
```

---

## Design notes on the charts

Direction uses **blue for up and red for down**, not the conventional
green/red, because green and red are close to indistinguishable for the ~8% of
men with deuteranopia. Direction is also always written out in text, so colour
never carries meaning alone.

The four model-family hues are a set validated for colour-vision separation
against both the light and dark surface. That set is only validated for
*adjacent* pairs, so charts that could place any series beside any other (
overlaid densities, scatter) either cap at three series or use small multiples.

---

## Limitations

- Daily bars. A "24 hour" forecast is a calendar-day-close forecast, not a
  rolling 24-hour one.
- Deribit mark prices are the exchange's marks, not executable quotes. On far
  strikes the bid/ask spread is frequently wider than any modelled edge.
- Deribit BTC options are **inverse** and settle in BTC. The greeks shown are
  standard USD-linear sensitivities per 1 BTC of notional; a delta-hedged book
  also carries the BTC-denominated exposure inverse settlement creates.
- Backtest costs are a flat 10 bps round trip; no slippage, funding or borrow
  cost is modelled for the short side.
- One asset, one three-year test window, and the models were tuned by someone
  who had already seen that history. Out-of-sample in the backtest is not the
  same as out-of-sample in the future.

**This is a modelling and research tool. Nothing here is financial advice.
Options can lose 100% of the premium paid.**
