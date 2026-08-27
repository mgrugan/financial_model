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

## Deploying to Render

The repository contains a `render.yaml` blueprint. Point Render at the repo and
it provisions:

- a **web service** running `gunicorn app:server --workers 1 --threads 4`
- a **nightly cron job** re-running the backtest and writing it to a shared disk

A single gunicorn worker is deliberate: the fitted models live in process
memory, so a second worker would train its own copy of all eleven. Threads
handle request concurrency.

**Plan sizing.** A full refresh fits eleven models and takes about a minute on
two cores. **Standard** (1 CPU / 2 GB) keeps it near that and leaves headroom for
the tree ensembles. Starter (0.5 CPU / 512 MB) will run, but refreshes take
several minutes and memory is tight. Free additionally spins the instance down
after 15 minutes idle, discarding every fitted model and making the next visitor
sit through a cold rebuild.

**The backtest tab is empty until a backtest exists.** The nightly cron job
fills it, so a fresh deploy shows a "no backtest loaded" notice until 04:20 UTC.
To fill it immediately, either run `python scripts/run_backtest.py` once from
the Render Shell, or set `BTC_BACKTEST_ON_START=1` so the web service generates
one in a background thread on first boot when the disk is empty — that competes
with request serving for CPU, so only enable it where there is headroom.

### Configuration

| Variable | Default | What it does |
|---|---|---|
| `BTC_CACHE_DIR` | `.cache` | where price data and backtest artefacts live |
| `BTC_REFRESH_SECONDS` | `300` | background refresh interval |
| `BTC_N_PATHS` | `40000` | Monte Carlo paths per model per horizon |
| `BTC_BACKTEST_DAYS` | `1095` | walk-forward test window |
| `BTC_REFIT_EVERY` | `45` | re-estimation cadence, in days |
| `BTC_COST_BPS` | `10` | round-trip transaction cost in the backtest |
| `BTC_OPT_MAX_DTE` | `10` | expiry window for the options page |
| `BTC_N_JOBS` | `2` | threads for the tree models |
| `BTC_BACKTEST_ON_START` | `0` | generate a backtest on first boot if none is cached |

---

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
scripts/
  run_backtest.py         offline walk-forward run
  smoke_test.py           end-to-end verification against live data
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
