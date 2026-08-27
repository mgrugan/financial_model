"""Page structure and per-tab rendering."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from dash import dcc, html

from btcmodels.config import HORIZONS, HORIZON_LABELS, OPTIONS_MAX_DTE
from btcmodels.registry import FAMILY_BLURB, FAMILY_ORDER, MODEL_ORDER, model_catalogue

from . import figures as fg
from .components import callout, data_table, model_card, pill, section, stat
from .theme import direction_color, family_color, tokens

TABS = [
    ("forecast", "Forecasts"),
    ("models", "Model detail"),
    ("backtest", "Backtest"),
    ("options", "Options"),
    ("method", "Method"),
]


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
def shell() -> html.Div:
    """Static page frame.

    The controls live here rather than inside the callback-rendered masthead.
    Putting them in regenerated markup would make the horizon toggle both an
    input to, and an output of, the same dependency chain, and would recreate
    the component underneath the user on every poll.

    Selections are held in per-session ``dcc.Store`` components rather than in a
    module-level dict, so two people looking at the dashboard at once do not
    move each other's dropdowns.
    """
    return html.Div([
        dcc.Store(id="theme-store", data="dark", storage_type="local"),
        dcc.Store(id="horizon-store", data="1w", storage_type="local"),
        dcc.Store(id="model-store", data="gbm", storage_type="session"),
        dcc.Store(id="backtest-store", data="gbm", storage_type="session"),
        dcc.Store(id="options-store", data="gbm", storage_type="session"),
        dcc.Store(id="greek-store", data="theta", storage_type="session"),
        # Bumped only when a genuinely new snapshot exists, so the expensive tab
        # render does not re-run on every poll tick.
        dcc.Store(id="snapshot-id", data=""),
        dcc.Interval(id="poll", interval=20_000, n_intervals=0),
        html.Div(id="theme-sink", style={"display": "none"}),

        html.Div([
            html.Div(id="masthead"),
            html.Div([
                dcc.RadioItems(
                    id="horizon-toggle",
                    options=[{"label": f" {HORIZON_LABELS[k]}", "value": k} for k in HORIZONS],
                    value="1w", inline=True, className="horizon-toggle",
                    inputStyle={"marginRight": "5px", "marginLeft": "12px"},
                ),
                html.Button("Light", id="theme-button", n_clicks=0, className="btn"),
                html.Button("Refresh", id="refresh-button", n_clicks=0, className="btn"),
            ], className="controls shell-controls"),
        ], className="masthead-wrap"),

        html.Div(dcc.Tabs(
            id="tabs", value="forecast", className="tab-bar",
            children=[dcc.Tab(label=label, value=key, className="tab",
                              selected_className="tab--selected")
                      for key, label in TABS],
        )),
        dcc.Loading(html.Div(id="tab-content"), type="default", color="#3987e5"),
    ], className="app")


def render_masthead(snapshot, theme: str, horizon: str, status: str) -> html.Div:
    t = tokens(theme)
    if snapshot is None:
        return html.Header([
            html.Div([
                html.Div([
                    html.H1("Bitcoin Model Dashboard"),
                    html.P("Warming up: fetching market data and training models…"),
                ], className="brand"),
            ], className="masthead-top"),
            html.Div(f"status: {status}", className="status-line"),
        ], className="masthead")

    market = snapshot.market
    price = market.get("price", snapshot.context.spot)

    def change_stat(label: str, key: str) -> Any:
        value = market.get(key)
        if value is None:
            return stat(label, "—")
        color = t["up"] if value >= 0 else t["down"]
        return stat(label, f"{value:+.2f}%", color=color)

    age = snapshot.age_seconds
    dot = t["up"] if age < 900 else t["warn"]

    return html.Header([
        html.Div([
            html.Div([
                html.H1("Bitcoin Model Dashboard"),
                html.P("Eleven forecasting models — stochastic, machine-learned and "
                       "hybrid — each producing a full price distribution, a "
                       "walk-forward track record, and its own read on the options book."),
            ], className="brand"),
            html.Div([
                html.Div([
                    html.Div(f"${price:,.0f}", className="price-now"),
                    html.Div("BTC-USD", className="stat-sub"),
                ]),
                change_stat("24 hours", "change_24h"),
                change_stat("7 days", "change_7d"),
                change_stat("30 days", "change_30d"),
                stat("Realised vol 30d",
                     f"{market.get('realised_vol_30d', float('nan')):.0f}%",
                     sub="annualised"),
            ], className="price-strip"),
        ], className="masthead-top"),
        html.Div([
            html.Span(className="status-dot", style={"background": dot}),
            f"models rebuilt {_ago(age)} in {snapshot.duration_seconds:.0f}s · "
            f"price data through {_date(market.get('as_of'))} · "
            f"{market.get('n_bars', 0):,} daily bars since {_date(market.get('history_start'))}"
            + (f" · errors: {', '.join(snapshot.errors)}" if snapshot.errors else ""),
        ], className="status-line"),
    ], className="masthead")


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def _date(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value[:10]
    return pd.Timestamp(value).strftime("%d %b %Y")


# ---------------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------------
def render_forecast(snapshot, theme: str, horizon: str) -> html.Div:
    t = tokens(theme)
    forecasts = snapshot.bundle.by_horizon(horizon)
    consensus = snapshot.bundle.consensus.get(horizon, {})
    horizon_days = HORIZONS[horizon]
    spot = snapshot.context.spot

    if not forecasts:
        return html.Div(callout("No model produced a forecast for this horizon.", "bad"))

    color = direction_color(consensus.get("p_up", 0.5), theme)
    skilled = [f for f in forecasts
               if (snapshot.reliability(f.model_key, horizon) or {}).get("has_skill")]

    consensus_panel = html.Div([
        html.Div([
            html.Div([
                html.Div(consensus["direction"], className="consensus-direction",
                         style={"color": color}),
                html.Div(f"{consensus['p_up'] * 100:.1f}% probability higher",
                         className="consensus-prob"),
                html.Div(
                    f"{consensus['n_up']} of {consensus['n_models']} models point up. "
                    f"Individual estimates span {consensus['prob_min'] * 100:.0f}–"
                    f"{consensus['prob_max'] * 100:.0f}%.",
                    className="consensus-note"),
            ], className="consensus-headline"),
            html.Div([
                html.Div([
                    stat("Median target", f"${consensus['median_price']:,.0f}",
                         sub=f"{(consensus['median_price'] / spot - 1) * 100:+.2f}% vs now"),
                    stat("50% band",
                         f"${consensus['q25']:,.0f} – ${consensus['q75']:,.0f}",
                         sub="one in two outcomes land here"),
                    stat("90% band",
                         f"${consensus['q05']:,.0f} – ${consensus['q95']:,.0f}",
                         sub="nine in ten outcomes land here"),
                    stat("Model agreement", f"{consensus['agreement'] * 100:.0f}%",
                         sub=f"spread of ±{consensus['prob_std'] * 100:.1f} pts"),
                ], className="stat-row"),
                dcc.Graph(
                    figure=fg.price_history_figure(snapshot.context.daily, consensus,
                                                   horizon_days, theme),
                    config={"displayModeBar": False}, style={"marginTop": "10px"}),
            ]),
        ], className="consensus"),
    ], className="panel")

    if skilled:
        names = ", ".join(f.model_name for f in skilled)
        evidence = callout(
            f"{len(skilled)} of {len(forecasts)} models beat the always-up base rate "
            f"out of sample over the walk-forward window: {names}. Weight their "
            f"readings above the rest.", "good", "Which models have earned trust")
    elif snapshot.backtest.get("available"):
        evidence = callout(
            "No model beat the always-up base rate out of sample at this horizon in "
            "the walk-forward test. Treat every probability below as a description "
            "of the distribution, not as a tradeable directional edge — the Backtest "
            "tab shows the evidence.", "warn", "Which models have earned trust")
    else:
        evidence = callout(
            "The walk-forward backtest has not been run yet, so none of these "
            "probabilities has a track record attached. Run "
            "`python scripts/run_backtest.py` to generate one.", "warn",
            "No track record loaded")

    return html.Div([
        section(f"Consensus — {HORIZON_LABELS[horizon].lower()}",
                "Probabilities are pooled in log-odds space, so a unanimous panel "
                "reads as more certain than its average member rather than less.",
                consensus_panel),
        section("Every model, side by side",
                "Bars run from the 50% coin-flip line; whiskers are each model's own "
                "uncertainty about its estimate. Direction is written out, so the "
                "colour is redundant rather than load-bearing.",
                html.Div([evidence,
                          dcc.Graph(figure=fg.probability_bars(forecasts, theme, consensus),
                                    config={"displayModeBar": False})], className="panel")),
        section("Model cards",
                "Each card carries its own out-of-sample verdict. A confident "
                "probability from a model with no demonstrated edge is not evidence.",
                html.Div([model_card(f, theme, snapshot.reliability(f.model_key, horizon))
                          for f in sorted(forecasts, key=lambda x: (
                              FAMILY_ORDER.index(x.family), -x.p_up))],
                         className="card-grid")),
        section("Where each family lands",
                "Small multiples rather than eleven overlaid curves: the four family "
                "hues are only separation-safe when same-family marks sit together.",
                html.Div(dcc.Graph(
                    figure=fg.family_distribution_facets(forecasts, theme, spot),
                    config={"displayModeBar": False}), className="panel")),
    ])


# ---------------------------------------------------------------------------
# Model detail tab
# ---------------------------------------------------------------------------
def render_models(snapshot, theme: str, horizon: str, model_key: str) -> html.Div:
    forecasts = {f.model_key: f for f in snapshot.bundle.by_horizon(horizon)}
    if model_key not in forecasts:
        model_key = next(iter(forecasts), "")
    if not model_key:
        return html.Div(callout("No models available.", "bad"))

    forecast = forecasts[model_key]
    spot = snapshot.context.spot
    reliability = snapshot.reliability(model_key, horizon)
    q = forecast.quantile_prices()

    diagnostics = pd.DataFrame(
        [{"parameter": k, "value": _fmt_value(v)}
         for k, v in forecast.diagnostics.items()
         if not isinstance(v, (list, dict, np.ndarray))])

    quantile_rows = pd.DataFrame([
        {"quantile": label, "price": f"${q[key]:,.0f}",
         "move": f"{(q[key] / spot - 1) * 100:+.2f}%"}
        for key, label in (("q05", "5th percentile"), ("q25", "25th percentile"),
                           ("q50", "median"), ("q75", "75th percentile"),
                           ("q95", "95th percentile"))])

    selector = dcc.Dropdown(
        id="model-select", value=model_key, clearable=False,
        style={"minWidth": "320px"},
        options=[{"label": f"{forecasts[k].model_name}  ·  {forecasts[k].family}",
                  "value": k} for k in MODEL_ORDER if k in forecasts])

    return html.Div([
        section(forecast.model_name, forecast.notes,
                html.Div([
                    html.Div([
                        stat("Direction", forecast.direction,
                             color=direction_color(forecast.p_up, theme)),
                        stat("P(up)", f"{forecast.p_up * 100:.2f}%",
                             sub=f"±{forecast.p_up_stderr * 100:.2f} pts"),
                        stat("Expected return", f"{forecast.expected_return:+.2f}%"),
                        stat("Median target", f"${q['q50']:,.0f}"),
                        stat("Implied vol", f"{forecast.annualised_vol:.1f}%",
                             sub="annualised"),
                        stat("P(move > 5%)", f"{forecast.prob_move_beyond(5) * 100:.0f}%"),
                        stat("Worst 5% average", f"{forecast.expected_shortfall(0.05):+.2f}%",
                             sub="expected shortfall"),
                    ], className="stat-row"),
                    dcc.Graph(figure=fg.distribution_figure(forecast, theme, spot),
                              config={"displayModeBar": False}),
                ], className="panel"),
                right=selector),
        html.Div([
            html.Div([
                html.H3("Fitted parameters & diagnostics", className="section-title"),
                html.P("What the model actually estimated from the data on this run.",
                       className="section-subtitle"),
                data_table(diagnostics,
                           [{"name": "Parameter", "id": "parameter"},
                            {"name": "Value", "id": "value"}],
                           theme, "diag-table", page_size=18),
            ], className="panel"),
            html.Div([
                html.H3("Target prices", className="section-title"),
                html.P("Read straight off this model's simulated terminal distribution.",
                       className="section-subtitle"),
                data_table(quantile_rows,
                           [{"name": "Quantile", "id": "quantile"},
                            {"name": "Price", "id": "price"},
                            {"name": "Move", "id": "move"}],
                           theme, "quantile-table", page_size=6),
                html.Div(_reliability_block(reliability, horizon), style={"marginTop": "14px"}),
            ], className="panel"),
        ], className="grid-2", style={"marginTop": "14px"}),
        _importance_section(snapshot, theme, model_key, HORIZONS[horizon], forecast.model_name),
    ])


def _importance_section(snapshot, theme: str, model_key: str, horizon_days: int,
                        model_name: str):
    importance = snapshot.feature_importance(model_key, horizon_days, top=14)
    if importance is None or importance.empty:
        return html.Div()
    return section(
        "What is driving this model",
        "Relative contribution of each engineered feature. For the tree models this "
        "is split-gain share; for the linear model it is the size of the "
        "standardised coefficient.",
        html.Div(dcc.Graph(
            figure=fg.feature_importance_figure(importance, theme, model_name),
            config={"displayModeBar": False}), className="panel"))


def _reliability_block(reliability: dict | None, horizon: str) -> Any:
    if reliability is None:
        return callout("No walk-forward result for this model yet.", "warn")
    tone = "good" if reliability["has_skill"] else "warn"
    verdict = ("beat the always-up base rate out of sample"
               if reliability["has_skill"] else
               "did not beat the always-up base rate out of sample")
    return callout(
        f"Over {reliability['n']} walk-forward days it {verdict}: "
        f"{reliability['accuracy_pct']:.1f}% directional accuracy against "
        f"{reliability['always_up_accuracy_pct']:.1f}% for always predicting up, "
        f"AUC {reliability['auc']:.3f}, Brier skill {reliability['brier_skill']:+.4f}, "
        f"strategy Sharpe {reliability['sharpe']:.2f}.",
        tone, f"Track record — {HORIZON_LABELS[horizon].lower()}")


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        if abs(value) >= 1e-4 or value == 0:
            return f"{value:.6f}"
        return f"{value:.3e}"
    return str(value)


# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------
BACKTEST_COLUMNS = [
    {"name": "Model", "id": "model"},
    {"name": "Family", "id": "family"},
    {"name": "Days", "id": "n"},
    {"name": "Accuracy %", "id": "accuracy_pct", "type": "numeric",
     "format": {"specifier": ".1f"}},
    {"name": "Always-up %", "id": "always_up_accuracy_pct", "type": "numeric",
     "format": {"specifier": ".1f"}},
    {"name": "Edge pp", "id": "edge_vs_always_up_pp", "type": "numeric",
     "format": {"specifier": "+.2f"}},
    {"name": "AUC", "id": "auc", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Brier skill", "id": "brier_skill", "type": "numeric",
     "format": {"specifier": "+.4f"}},
    {"name": "Log-loss skill", "id": "log_loss_skill", "type": "numeric",
     "format": {"specifier": "+.4f"}},
    {"name": "Sharpe", "id": "sharpe", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Return %", "id": "total_return_pct", "type": "numeric",
     "format": {"specifier": "+.1f"}},
    {"name": "Max DD %", "id": "max_drawdown_pct", "type": "numeric",
     "format": {"specifier": ".1f"}},
]


def render_backtest(snapshot, theme: str, horizon: str, model_key: str) -> html.Div:
    backtest = snapshot.backtest
    if not backtest.get("available"):
        return html.Div([
            section("Walk-forward backtest", None,
                    html.Div(callout(
                        f"{backtest.get('reason', 'Not available.')} Run "
                        "`python scripts/run_backtest.py` to generate one — it "
                        "re-estimates every model on a rolling schedule across roughly "
                        "three years of history and takes about half an hour.",
                        "warn", "No backtest loaded"), className="panel")),
        ])

    results = snapshot.backtest_results()
    summary = snapshot.backtest_summary()
    block = summary[summary["horizon"] == horizon].copy()
    if block.empty:
        return html.Div(callout("No results for this horizon.", "warn"))

    if model_key not in results or horizon not in results.get(model_key, {}):
        model_key = next((k for k in MODEL_ORDER if horizon in results.get(k, {})), "")
    result = results[model_key][horizon]

    n_skilled = int(((block["brier_skill"] > 0) & (block["auc"] > 0.5)).sum())
    scored = block.dropna(subset=["brier_skill"])
    best_line = ""
    if not scored.empty:
        best = scored.loc[scored["brier_skill"].idxmax()]
        best_line = (f"Best by Brier skill: {best['model']} at "
                     f"{best['brier_skill']:+.4f} with AUC {best['auc']:.3f}. ")

    verdict = callout(
        f"{n_skilled} of {len(block)} models cleared both bars at this horizon "
        f"(positive Brier skill and AUC above 0.5). " + best_line +
        f"Bitcoin rose on {block['base_rate_pct'].iloc[0]:.1f}% of days in this window, "
        f"so any model below {block['always_up_accuracy_pct'].iloc[0]:.1f}% accuracy is "
        f"losing to a strategy that never thinks.",
        "good" if n_skilled else "warn",
        "Out-of-sample verdict")

    selector = dcc.Dropdown(
        id="backtest-model-select", value=model_key, clearable=False,
        style={"minWidth": "320px"},
        options=[{"label": results[k][horizon].model_name, "value": k}
                 for k in MODEL_ORDER if horizon in results.get(k, {})])

    meta = (f"{backtest.get('backtest_days', '?')} test days · re-estimated every "
            f"{backtest.get('refit_every', '?')} days · generated "
            f"{backtest.get('age_hours', 0):.1f} h ago · data through "
            f"{backtest.get('data_end', '?')} · run took "
            f"{backtest.get('elapsed_seconds', 0) / 60:.0f} min")

    return html.Div([
        section(f"Walk-forward results — {HORIZON_LABELS[horizon].lower()}",
                "Every prediction below was made with data available on the day, by a "
                "model re-estimated on a fixed schedule and calibrated on purged folds "
                "inside its own training window. " + meta,
                html.Div([verdict,
                          dcc.Graph(figure=fg.skill_scatter(summary, theme, horizon),
                                    config={"displayModeBar": False})], className="panel")),
        section("Scoreboard",
                "Accuracy alone is misleading in a market that rises more often than "
                "it falls, so it sits next to the always-up baseline it must beat. "
                "Brier skill and log-loss skill are measured against that same baseline; "
                "zero means no skill and negative means worse than not modelling at all.",
                html.Div(data_table(
                    block.sort_values("brier_skill", ascending=False),
                    BACKTEST_COLUMNS, theme, "backtest-table", page_size=12,
                ), className="panel")),
        section("One model in detail", None,
                html.Div([
                    html.Div([
                        html.H3("Equity curve", className="section-title"),
                        html.P("Long or short on the model's own signal, on "
                               "non-overlapping holding periods, net of "
                               f"{result.strategy.get('cost_bps', 0):.0f} bps round-trip cost. "
                               "Overlapping daily bets on a 7-day signal would flatter "
                               "the Sharpe by roughly sqrt(7).",
                               className="section-subtitle"),
                        dcc.Graph(figure=fg.equity_figure(result, theme),
                                  config={"displayModeBar": False}),
                        html.Div([
                            stat("Sharpe", f"{result.strategy.get('sharpe', float('nan')):.2f}"),
                            stat("Total return",
                                 f"{result.strategy.get('total_return_pct', float('nan')):+.1f}%"),
                            stat("Buy & hold",
                                 f"{result.strategy.get('buy_hold_return_pct', float('nan')):+.1f}%"),
                            stat("Max drawdown",
                                 f"{result.strategy.get('max_drawdown_pct', float('nan')):.1f}%"),
                            stat("Hit rate",
                                 f"{result.strategy.get('hit_rate_pct', float('nan')):.1f}%"),
                            stat("Trades", f"{result.strategy.get('n_trades', 0)}"),
                        ], className="stat-row", style={"marginTop": "10px"}),
                    ], className="panel"),
                    html.Div([
                        html.H3("Calibration", className="section-title"),
                        html.P("Do the probabilities mean what they say? Points on the "
                               "diagonal are honest; points above it mean the model is "
                               "under-confident, below it over-confident. Marker size is "
                               "the number of days in each bin.",
                               className="section-subtitle"),
                        dcc.Graph(figure=fg.calibration_figure(result, theme),
                                  config={"displayModeBar": False}),
                        html.Div([
                            stat("Confident third",
                                 f"{result.metrics.get('confident_third_accuracy_pct', float('nan')):.1f}%",
                                 sub="accuracy when the model is most sure"),
                            stat("Mean P(up)", f"{result.metrics['mean_p_up'] * 100:.1f}%",
                                 sub=f"actual {result.metrics['base_rate_pct']:.1f}%"),
                            stat("Spread of P(up)", f"±{result.metrics['p_up_std'] * 100:.1f} pts",
                                 sub="how much it moves day to day"),
                        ], className="stat-row", style={"marginTop": "10px"}),
                    ], className="panel"),
                ], className="grid-2"),
                right=selector),
    ])


# ---------------------------------------------------------------------------
# Options tab
# ---------------------------------------------------------------------------
CHAIN_COLUMNS = [
    {"name": "Instrument", "id": "instrument"},
    {"name": "Type", "id": "option_type"},
    {"name": "Strike", "id": "strike", "type": "numeric", "format": {"specifier": ",.0f"}},
    {"name": "Mark $", "id": "mark_usd", "type": "numeric", "format": {"specifier": ",.0f"}},
    {"name": "IV %", "id": "iv_pct", "type": "numeric", "format": {"specifier": ".1f"}},
    {"name": "Delta", "id": "delta", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "Gamma", "id": "gamma_x1e6", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Theta $/day", "id": "theta", "type": "numeric", "format": {"specifier": ",.1f"}},
    {"name": "Theta %/day", "id": "theta_pct_of_premium", "type": "numeric",
     "format": {"specifier": ".2f"}},
    {"name": "Vega $", "id": "vega", "type": "numeric", "format": {"specifier": ",.1f"}},
    {"name": "Rho $", "id": "rho", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Vanna", "id": "vanna", "type": "numeric", "format": {"specifier": ".4f"}},
    {"name": "Charm", "id": "charm", "type": "numeric", "format": {"specifier": ".4f"}},
    {"name": "Mkt P(ITM) %", "id": "rn_itm_pct", "type": "numeric", "format": {"specifier": ".1f"}},
    {"name": "Model P(ITM) %", "id": "model_itm_pct", "type": "numeric",
     "format": {"specifier": ".1f"}},
    {"name": "Model edge %", "id": "model_edge_pct", "type": "numeric",
     "format": {"specifier": "+.0f"}},
    {"name": "OI", "id": "open_interest", "type": "numeric", "format": {"specifier": ",.0f"}},
]


def _chain_table_frame(valued: pd.DataFrame, expiry) -> pd.DataFrame:
    block = valued[valued["expiry"] == expiry].copy()
    block["iv_pct"] = block["mark_iv"] * 100.0
    block["gamma_x1e6"] = block["gamma"] * 1e6
    block["rn_itm_pct"] = block["rn_prob_itm"] * 100.0
    block["model_itm_pct"] = block["model_p_itm"] * 100.0
    return block.sort_values(["option_type", "strike"])


def _suggestion_card(suggestion: dict, theme: str) -> html.Div:
    t = tokens(theme)
    edge_tone = "good" if suggestion["model_edge_usd"] > 0 else "bad"
    rows = [
        ("Breakeven", f"${suggestion['breakeven']:,.0f}  "
                      f"({suggestion['breakeven_move_pct']:+.2f}%)"),
        ("Delta", f"{suggestion['delta']:+.3f}"),
        ("Gamma", f"{suggestion['gamma'] * 1e6:.2f} per $1m"),
        ("Theta", f"${suggestion['theta_usd_per_day']:,.0f}/day "
                  f"({suggestion['theta_pct_per_day']:.1f}%/day)"),
        ("Vega", f"${suggestion['vega']:,.1f} per vol point"),
        ("Rho", f"${suggestion['rho']:,.2f} per 1% rate"),
        ("Vanna / Charm", f"{suggestion['vanna']:.4f} / {suggestion['charm']:.4f}"),
        ("Market IV", f"{suggestion['market_iv_pct']:.1f}%"),
        ("Model IV", f"{suggestion['model_iv_pct']:.1f}%"
                     f"  ({suggestion['iv_gap_pts']:+.1f} pts)"),
        ("P(ITM) — model", f"{suggestion['model_prob_itm_pct']:.1f}%"),
        ("P(ITM) — market", f"{suggestion['rn_prob_itm_pct']:.1f}%"),
        ("P(profit) — model", f"{suggestion['model_prob_profit_pct']:.1f}%"),
        ("Model fair value", f"${suggestion['model_ev_usd']:,.0f}"),
        ("Max loss", f"${suggestion['max_loss_usd']:,.0f} (premium)"),
        ("Target — median", f"${suggestion['target_price_median']:,.0f}"),
        ("Target — stretch", f"${suggestion['target_price_stretch']:,.0f}"),
        ("Open interest", f"{suggestion['open_interest']:,.0f} BTC"),
    ]
    return html.Div([
        html.Div([
            html.Div([
                html.Div(suggestion["moneyness_label"], className="suggest-label"),
                html.Div(suggestion["instrument"], className="suggest-instrument"),
            ]),
            pill(f"{suggestion['model_edge_pct']:+.0f}% edge", edge_tone,
                 title="Model expected value versus the market mark"),
        ], className="suggest-head"),
        html.Div(f"${suggestion['premium_usd']:,.0f}", className="suggest-price"),
        html.Div(f"{suggestion['premium_btc']:.4f} BTC · "
                 f"{suggestion['premium_pct_of_spot']:.2f}% of spot",
                 className="stat-sub"),
        html.Div([html.Div([html.Span(k, className="kv-k"), html.Span(v, className="kv-v")],
                           className="kv") for k, v in rows], className="suggest-rows"),
    ], className="suggest-card")


def render_options(snapshot, theme: str, model_key: str, greek: str = "theta") -> html.Div:
    view = snapshot.option_view
    if not view.get("available"):
        return html.Div(section(
            "Options", None,
            html.Div(callout(view.get("error", "Option data unavailable."), "bad",
                             "Live option chain not loaded"), className="panel")))

    per_model = view["per_model"]
    if model_key not in per_model or "error" in per_model.get(model_key, {}):
        model_key = next((k for k in MODEL_ORDER if k in per_model
                          and "error" not in per_model[k]), "")
    if not model_key:
        return html.Div(callout("No model could value the chain.", "bad"))

    entry = per_model[model_key]
    forecast = snapshot.bundle.forecasts[model_key][view["horizon_key"]]
    valued = entry["valued"]
    expiry = view["target_expiry"]
    spot = view["index_price"]
    vol = entry["vol_context"]
    suggestions = entry["suggestions"]

    reliability = snapshot.reliability(model_key, view["horizon_key"])
    trust = callout(
        (f"{entry['model_name']} beat the always-up base rate out of sample "
         f"(AUC {reliability['auc']:.3f}). Its edge numbers carry evidence."
         if reliability and reliability["has_skill"] else
         f"{entry['model_name']} did not beat the always-up base rate out of sample"
         + (f" (AUC {reliability['auc']:.3f})." if reliability else " — no backtest loaded.")
         + " Every expected-value figure below inherits that weakness: treat them as "
           "what the model believes, not as a measured edge."),
        "good" if reliability and reliability["has_skill"] else "warn",
        "How much weight this model has earned")

    gap = vol["vol_gap_pts"]
    if not np.isfinite(gap):
        vol_note = callout(
            "Market at-the-money implied volatility could not be read for this expiry, "
            "so the expected values below cannot be separated into a directional view "
            "and a volatility view. Treat them with extra caution.", "warn",
            "Read this before the expected values")
    else:
        vol_note = callout(
        f"This model's annualised volatility is {vol['model_vol_pct']:.1f}% against a "
        f"market at-the-money implied of {vol['market_atm_iv_pct']:.1f}% — a gap of "
        f"{vol['vol_gap_pts']:+.1f} points. "
        + ("Because the model expects more movement than the market is charging for, "
           "options look cheap almost everywhere on the surface: the positive expected "
           "values below are substantially one volatility opinion, not many independent "
           "findings."
           if vol["vol_gap_pts"] > 2 else
           "Because the model expects less movement than the market is charging for, "
           "long premium looks expensive here and the edge, if any, is in selling it."
           if vol["vol_gap_pts"] < -2 else
           "Model and market broadly agree on volatility, so the expected values below "
           "reflect the model's directional view rather than a volatility bet."),
        "warn" if abs(gap) > 2 else "info",
        "Read this before the expected values")

    selector = dcc.Dropdown(
        id="options-model-select", value=model_key, clearable=False,
        style={"minWidth": "320px"},
        options=[{"label": per_model[k]["model_name"], "value": k}
                 for k in MODEL_ORDER if k in per_model and "error" not in per_model[k]])

    greek_selector = dcc.Dropdown(
        id="greek-select", value=greek, clearable=False, style={"minWidth": "170px"},
        options=[{"label": g.title(), "value": g}
                 for g in ("delta", "gamma", "theta", "vega", "vanna", "charm")])

    table_frame = _chain_table_frame(valued, expiry)
    atm = next((s for s in suggestions if s["moneyness_label"] == "ATM"),
               suggestions[0] if suggestions else None)
    decay_source = None
    if atm is not None:
        pick = valued[valued["instrument"] == atm["instrument"]]
        if not pick.empty:
            decay_source = pick.iloc[0]

    decay_block = html.Div()
    if decay_source is not None:
        from btcmodels.options import theta_decay_schedule
        schedule = theta_decay_schedule(decay_source, view["rate"], spot, n_points=14)
        decay_block = html.Div([
            html.H3("Time decay on the at-the-money contract", className="section-title"),
            html.P("Spot and implied volatility held fixed, so the only thing moving is "
                   "the clock. Decay is not linear: it accelerates into expiry, and on a "
                   "contract this close to the date most of the premium is gone in the "
                   "final days.", className="section-subtitle"),
            dcc.Graph(figure=fg.theta_decay_figure(schedule, theme,
                                                   str(decay_source["instrument"])),
                      config={"displayModeBar": False}),
        ], className="panel")

    payoff_block = html.Div()
    if atm is not None:
        payoff_block = html.Div([
            html.H3("Payoff against the model's own distribution", className="section-title"),
            html.P("The grey shape is where this model thinks price actually lands at "
                   "expiry. Profit requires finishing beyond the dashed breakeven, "
                   "which is further out than the strike by the premium paid.",
                   className="section-subtitle"),
            dcc.Graph(figure=fg.payoff_figure(atm, forecast, theme, spot),
                      config={"displayModeBar": False}),
        ], className="panel")

    return html.Div([
        section(f"Options — expiry {pd.Timestamp(expiry).strftime('%d %b %Y')}"
                f"  ({view['target_dte']:.1f} days)",
                f"Live Deribit book: {view['n_contracts']} contracts inside "
                f"{OPTIONS_MAX_DTE} days that pass a tradeable-delta screen. Priced with "
                f"Black-76 on each expiry's own forward, carry rate "
                f"{view['rate'] * 100:.2f}% implied by the futures basis.",
                html.Div([
                    html.Div([
                        stat("BTC index", f"${spot:,.0f}"),
                        stat("ATM implied vol",
                             f"{view['atm_iv_pct']:.1f}%"
                             if np.isfinite(view["atm_iv_pct"]) else "—"),
                        stat("Model vol", f"{vol['model_vol_pct']:.1f}%",
                             sub=f"{gap:+.1f} pts vs market" if np.isfinite(gap) else None),
                        stat("Model direction", forecast.direction,
                             sub=f"P(up) {forecast.p_up * 100:.1f}%",
                             color=direction_color(forecast.p_up, theme)),
                        stat("Contracts", f"{view['n_contracts']}"),
                    ], className="stat-row"),
                    trust,
                    vol_note,
                ], className="panel"),
                right=selector),

        section("Suggested structures",
                "One contract at each moneyness, in the direction this model favours, "
                "picked by closest match to a target delta (0.70 / 0.50 / 0.28) among "
                "contracts with real open interest. Max loss on a long option is the "
                "premium; the target prices come from this model's own distribution.",
                html.Div([_suggestion_card(s, theme) for s in suggestions],
                         className="suggest-grid")
                if suggestions else
                html.Div(callout("No liquid contracts matched the delta targets.", "warn"),
                         className="panel")),

        _spread_block(entry.get("spread"), theme),

        html.Div([decay_block, payoff_block], className="grid-2",
                 style={"marginTop": "14px"}),

        section("Volatility surface and greeks", None,
                html.Div([
                    html.Div([
                        html.H3("Implied volatility by strike", className="section-title"),
                        html.P("Market marks against the volatility this model's own "
                               "distribution implies at each strike. Where the dashed "
                               "line sits above the market, the model thinks that strike "
                               "is underpriced.", className="section-subtitle"),
                        dcc.Graph(figure=fg.volatility_smile(valued, theme, expiry, spot,
                                                             model_iv=valued),
                                  config={"displayModeBar": False}),
                    ], className="panel"),
                    html.Div([
                        html.Div([
                            html.Div([html.H3("Greek across the strike ladder",
                                              className="section-title"),
                                      html.P("How the exposure changes as you move up "
                                             "and down the strikes.",
                                             className="section-subtitle")],
                                     className="section-head-text"),
                            html.Div(greek_selector, className="section-head-right"),
                        ], className="section-head"),
                        dcc.Graph(figure=fg.greeks_profile(valued, theme, expiry, greek, spot),
                                  config={"displayModeBar": False}),
                    ], className="panel"),
                ], className="grid-2")),

        section("Full chain with greeks",
                "Every tradeable contract at this expiry. Gamma is scaled per $1m of "
                "spot move so it is readable; theta is dollars of premium lost per "
                "calendar day, and the percentage next to it is what that costs "
                "relative to the premium you paid.",
                html.Div(data_table(table_frame, CHAIN_COLUMNS, theme, "chain-table",
                                    page_size=16), className="panel")),
    ])


def _spread_block(spread: dict | None, theme: str):
    if not spread:
        return html.Div()
    rows = [
        ("Long", f"{spread['long_instrument']} @ ${spread['long_strike']:,.0f}"),
        ("Short", f"{spread['short_instrument']} @ ${spread['short_strike']:,.0f}"),
        ("Net debit", f"${spread['net_debit_usd']:,.0f}"),
        ("Max profit", f"${spread['max_profit_usd']:,.0f}"),
        ("Max loss", f"${spread['max_loss_usd']:,.0f}"),
        ("Risk / reward", f"1 : {spread['risk_reward']:.2f}"),
        ("Breakeven", f"${spread['breakeven']:,.0f}"),
        ("Net delta", f"{spread['net_delta']:+.3f}"),
        ("Net gamma", f"{spread['net_gamma'] * 1e6:.2f} per $1m"),
        ("Net theta", f"${spread['net_theta_usd_per_day']:,.1f}/day"),
        ("Net vega", f"${spread['net_vega']:,.1f}"),
        ("Model fair value", f"${spread['model_ev_usd']:,.0f}"),
        ("Model edge", f"${spread['model_edge_usd']:+,.0f}"),
        ("P(profit) — model", f"{spread['model_prob_profit_pct']:.1f}%"),
    ]
    return section(
        f"Defined-risk alternative — {spread['structure']}",
        "Selling a further-out strike against the long one caps the upside but cuts "
        "both the premium and the daily theta bill. With every expiry in this window "
        "inside ten days, decay is the dominant cost of a long option, which is what "
        "makes the spread worth comparing rather than an afterthought.",
        html.Div(html.Div([
            html.Div([html.Span(k, className="kv-k"), html.Span(v, className="kv-v")],
                     className="kv") for k, v in rows
        ], className="suggest-rows"), className="panel"))


# ---------------------------------------------------------------------------
# Method tab
# ---------------------------------------------------------------------------
METHOD_NOTES = [
    ("Why a distribution rather than a number",
     "Every model here is required to emit a full sample of terminal prices, not a "
     "point forecast. Direction probability, target prices, confidence bands and "
     "option expected values are all read off that one object, which is what makes "
     "a gradient-boosted classifier and a Heston diffusion directly comparable. It "
     "also removes the temptation to quote a single price for an asset whose "
     "one-week 90% band is routinely twenty per cent wide."),
    ("How a classifier gets a price distribution",
     "A classifier only answers 'up, with probability p'. To turn that into a "
     "distribution, a zero-drift sample of horizon returns is drawn by block-"
     "bootstrapping standardised GARCH residuals — which carries the real skew and "
     "kurtosis of Bitcoin returns rather than a normal approximation — and then "
     "shifted by a constant until exactly p of its mass is positive. The shift is "
     "exact by construction. The learned model supplies the location; the stochastic "
     "backbone supplies the shape."),
    ("Probability calibration",
     "Raw classifier scores on financial data are badly over-confident, so every "
     "learned model is recalibrated with Platt scaling fitted on pooled out-of-sample "
     "predictions from purged walk-forward folds — a few thousand honest predictions "
     "spanning several regimes, rather than a few hundred from one recent window. The "
     "calibration slope is confined to [0, 3]: a negative slope would mean 'this model "
     "was anti-predictive on held-out data, so invert it', and inverting a weak model "
     "on one block of evidence is a reliable way to manufacture nonsense. The honest "
     "response to a model with no edge is a slope of zero, which collapses its output "
     "to the base rate."),
    ("Variance targeting in the volatility backbone",
     "Fitted on the full twelve-year history, GARCH(1,1) estimates a persistence of "
     "exactly 1.000 — an IGARCH degeneracy in which the long-run variance does not "
     "exist — and the level is dragged up by the 2017-2021 regime. The window is "
     "therefore four years, and following Engle-Mezrich the intercept is pinned so "
     "that the model's unconditional volatility equals the sample's by construction "
     "rather than by luck. Without it the implied long-run volatility came out near "
     "61% against 47% realised over the same window, and every distribution on this "
     "page would have been too wide."),
    ("What the backtest does and does not prove",
     "Predictions are strictly out of sample: features are causal (verified to be "
     "bit-identical whether computed on the full history or truncated at the "
     "prediction date), labels never enter a training set that produced their own "
     "prediction, parameters are re-estimated on a fixed schedule while fast state "
     "updates daily, and the equity curve uses non-overlapping holding periods net of "
     "costs. What it cannot prove is that any edge survives into the future: it is one "
     "asset over one three-year window, and the models were selected and tuned by "
     "someone who had already seen that history."),
    ('What the backtest actually found',
     "Across 1,088 walk-forward days, essentially no model demonstrated a reliable directional edge, and the diagnosis is specific rather than vague. The dominant driver of the learned models' probabilities was `drawdown` -- distance below the running all-time high -- and the sign of its relationship to forward returns inverted between the training history and the test window. The tree models carry a correlation of about +0.3 between their probability and that feature, having learned 'near the highs, keep going up' from the 2017 and 2021 bull runs; over the test window the same feature correlated -0.12 with forward returns. Every model that leaned that way scored below 0.5 AUC, and the LSTM -- the only model whose probability leaned the other way (correlation -0.20) -- was the only one above 0.5 at the one-week horizon. That is not a bug in the pipeline; it is what a genuine regime change looks like from inside a model, and it is the single best argument for why a live probability should never be shown without its track record beside it."),
    ("Model expected value is not a price",
     "Option fair values on this page are discounted expected payoffs under each "
     "model's real-world distribution. That is not an arbitrage-free price and does "
     "not imply a riskless trade — it is what a contract is worth if that model's view "
     "of the world is right. Because the models generally forecast higher volatility "
     "than the market is charging, most contracts show a positive expected value; that "
     "is one volatility opinion repeated across a surface, not a list of independent "
     "opportunities, which is why the volatility gap is stated next to every figure."),
    ("Colour choices",
     "Direction uses blue for up and red for down rather than the conventional "
     "green/red, because green and red are close to indistinguishable for the roughly "
     "eight per cent of men with deuteranopia. Direction is also always written out in "
     "text, so colour never carries meaning on its own. The four family hues are a set "
     "validated for colour-vision separation against both the light and dark surface; "
     "charts that could place any series beside any other cap at three series or use "
     "small multiples, because the four-hue set is only validated for adjacent pairs."),
]

LIMITATIONS = [
    "Daily bars from Yahoo Finance; a 24-hour forecast is a calendar-day-close forecast, "
    "not a rolling-24-hour one.",
    "Deribit option data is unauthenticated public book data. Mark prices are the "
    "exchange's marks, not executable quotes, and the bid/ask spread on far strikes is "
    "frequently wider than any modelled edge.",
    "Deribit BTC options are inverse and settle in BTC. The greeks shown are standard "
    "USD-linear sensitivities per 1 BTC of notional; a delta-hedged book also carries "
    "the BTC-denominated exposure that inverse settlement creates.",
    "Transaction costs in the backtest are a flat 10 bps round trip and no slippage, "
    "funding, or borrow cost is modelled for the short side.",
    "The models are re-estimated on a schedule, not continuously, and the backtest "
    "reflects that same schedule.",
    "This is a modelling and research tool. Nothing here is financial advice, and "
    "options can lose 100% of the premium paid.",
]


def render_method(theme: str) -> html.Div:
    catalogue = model_catalogue()
    family_blocks = []
    for family in FAMILY_ORDER:
        entries = [e for e in catalogue if e["family"] == family]
        if not entries:
            continue
        family_blocks.append(html.Div([
            html.Div([
                html.Span(className="family-dot",
                          style={"background": family_color(family, theme)}),
                html.H3(family, className="section-title",
                        style={"display": "inline-block", "marginLeft": "8px"}),
            ]),
            html.P(FAMILY_BLURB[family], className="section-subtitle"),
            *[html.Div([
                html.H4(entry["name"]),
                html.Div(entry["method"], className="meta"),
                html.P(entry["description"]),
            ], className="model-doc") for entry in entries],
        ], className="panel"))

    return html.Div([
        section("The eleven models",
                "Each one relaxes a different assumption or brings a different "
                "inductive bias. Agreement between models that fail in different ways "
                "is informative; agreement between eleven variations of one idea is not.",
                *family_blocks),
        section("Methodology notes",
                "The decisions that materially change the numbers, and why they were "
                "made that way.",
                html.Div([html.Div([html.H4(title), html.P(body)], className="model-doc")
                          for title, body in METHOD_NOTES], className="panel")),
        section("Limitations",
                "What this tool does not do, stated plainly.",
                html.Div(html.Ul([html.Li(item, style={"marginBottom": "6px"})
                                  for item in LIMITATIONS],
                                 style={"paddingLeft": "20px", "fontSize": "12.5px"}),
                         className="panel")),
    ])
