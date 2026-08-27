"""Chart builders.

Every figure here follows the same rules: recessive grid and axes, thin marks,
direct labels wherever a reader would otherwise have to trace back to a legend,
and no dual axes anywhere.  Where a form could put arbitrary series next to each
other, the series count is capped at three or the data is faceted, because the
four-hue family palette is only validated for adjacent pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from btcmodels.registry import FAMILY_ORDER

from .theme import BLUE_RAMP, FONT_STACK, direction_color, family_color, tokens

TRANSPARENT = "rgba(0,0,0,0)"


def base_layout(theme: str, height: int = 320, **kwargs) -> dict:
    t = tokens(theme)
    layout = dict(
        height=height,
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(family=FONT_STACK, size=12, color=t["text_secondary"]),
        margin=dict(l=56, r=20, t=32, b=44),
        hoverlabel=dict(
            bgcolor=t["surface_raised"], bordercolor=t["border_strong"],
            font=dict(family=FONT_STACK, size=12, color=t["text_primary"]),
        ),
        xaxis=dict(gridcolor=t["grid"], zerolinecolor=t["border"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"])),
        yaxis=dict(gridcolor=t["grid"], zerolinecolor=t["border"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=t["text_secondary"], size=11),
                    bgcolor=TRANSPARENT),
        showlegend=False,
    )
    layout.update(kwargs)
    return layout


def empty_figure(theme: str, message: str, height: int = 240) -> go.Figure:
    t = tokens(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(family=FONT_STACK, size=13, color=t["text_muted"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(**base_layout(theme, height=height,
                                    xaxis=dict(visible=False), yaxis=dict(visible=False)))
    return fig


# ---------------------------------------------------------------------------
# Probability comparison
# ---------------------------------------------------------------------------
def probability_bars(forecasts: list, theme: str, consensus: dict | None = None) -> go.Figure:
    """P(up) per model as a bar from the 50% coin-flip line.

    The baseline is 50%, not zero: what matters is the distance from a coin
    flip, and anchoring at zero would make an 0.52 and an 0.65 look nearly
    identical.  Direction is spelled out in the label so colour is never the
    only carrier.
    """
    t = tokens(theme)
    if not forecasts:
        return empty_figure(theme, "No forecasts available")

    ordered = sorted(forecasts, key=lambda f: (FAMILY_ORDER.index(f.family), -f.p_up))
    names = [f.model_name for f in ordered]
    probs = [f.p_up for f in ordered]
    errors = [f.p_up_stderr for f in ordered]
    colors = [family_color(f.family, theme) for f in ordered]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=[p - 0.5 for p in probs], base=0.5, orientation="h",
        marker=dict(color=colors, line=dict(color=t["surface"], width=2)),
        error_x=dict(type="data", array=errors, color=t["text_muted"],
                     thickness=1.2, width=3),
        text=[f"{p * 100:.1f}%  {'UP' if p >= 0.5 else 'DOWN'}" for p in probs],
        textposition="outside",
        textfont=dict(color=t["text_secondary"], size=11),
        hovertemplate=("<b>%{y}</b><br>P(up) = %{x:.1%}"
                       "<extra></extra>"),
        customdata=probs,
        showlegend=False,
    ))
    fig.add_vline(x=0.5, line=dict(color=t["border_strong"], width=1.5, dash="dot"))
    if consensus:
        fig.add_vline(x=consensus["p_up"],
                      line=dict(color=t["text_muted"], width=1.5),
                      annotation_text=f"consensus {consensus['p_up']:.1%}",
                      annotation_font=dict(color=t["text_muted"], size=10),
                      annotation_position="top")

    lo = min(0.42, min(probs) - 0.04)
    hi = max(0.58, max(probs) + 0.10)
    fig.update_layout(**base_layout(
        theme, height=max(280, 30 * len(names) + 70),
        margin=dict(l=210, r=40, t=34, b=40),
        xaxis=dict(range=[lo, hi], tickformat=".0%", gridcolor=t["grid"],
                   title=dict(text="Probability the price is higher at the horizon",
                              font=dict(size=11, color=t["text_muted"])),
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"])),
        yaxis=dict(autorange="reversed", gridcolor=TRANSPARENT,
                   linecolor=TRANSPARENT,
                   tickfont=dict(color=t["text_secondary"], size=11)),
    ))
    return fig


# ---------------------------------------------------------------------------
# Terminal price distribution
# ---------------------------------------------------------------------------
def _density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with a Silverman bandwidth, evaluated on a fixed grid."""
    sample = samples if samples.size <= 6000 else np.random.default_rng(0).choice(
        samples, 6000, replace=False)
    sd = np.std(sample, ddof=1)
    if sd <= 0:
        return np.zeros_like(grid)
    bw = 1.06 * sd * sample.size ** (-1 / 5)
    diff = (grid[:, None] - sample[None, :]) / bw
    return np.exp(-0.5 * diff**2).sum(axis=1) / (sample.size * bw * np.sqrt(2 * np.pi))


def distribution_figure(forecast, theme: str, spot: float) -> go.Figure:
    """One model's terminal price distribution with its quantile band."""
    t = tokens(theme)
    samples = forecast.terminal_prices
    lo, hi = np.quantile(samples, [0.005, 0.995])
    grid = np.linspace(lo, hi, 240)
    dens = _density(samples, grid)
    q = forecast.quantile_prices()

    fig = go.Figure()
    up_mask = grid >= spot
    fig.add_trace(go.Scatter(
        x=grid[~up_mask], y=dens[~up_mask], mode="lines", fill="tozeroy",
        line=dict(color=t["down"], width=2),
        fillcolor=_alpha(t["down"], 0.16), name="Below today",
        hovertemplate="$%{x:,.0f}<extra>below today</extra>"))
    fig.add_trace(go.Scatter(
        x=grid[up_mask], y=dens[up_mask], mode="lines", fill="tozeroy",
        line=dict(color=t["up"], width=2),
        fillcolor=_alpha(t["up"], 0.16), name="Above today",
        hovertemplate="$%{x:,.0f}<extra>above today</extra>"))

    fig.add_vline(x=spot, line=dict(color=t["text_primary"], width=1.5),
                  annotation_text=f"now ${spot:,.0f}",
                  annotation_font=dict(color=t["text_primary"], size=10),
                  annotation_position="top")
    for key, label in (("q05", "5%"), ("q50", "median"), ("q95", "95%")):
        fig.add_vline(x=q[key], line=dict(color=t["text_muted"], width=1, dash="dot"),
                      annotation_text=f"{label} ${q[key]:,.0f}",
                      annotation_font=dict(color=t["text_muted"], size=9),
                      annotation_position="bottom")

    fig.update_layout(**base_layout(
        theme, height=300, showlegend=True,
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text=f"BTC price in {forecast.horizon_days} day(s)",
                              font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(visible=False),
    ))
    return fig


def family_distribution_facets(forecasts: list, theme: str, spot: float) -> go.Figure:
    """Small multiples by family -- avoids overlaying eleven series in one frame."""
    from plotly.subplots import make_subplots

    t = tokens(theme)
    if not forecasts:
        return empty_figure(theme, "No forecasts available")

    families = [f for f in FAMILY_ORDER if any(x.family == f for x in forecasts)]
    fig = make_subplots(rows=1, cols=len(families), subplot_titles=families,
                        shared_yaxes=True, horizontal_spacing=0.03)

    all_samples = np.concatenate([f.terminal_prices for f in forecasts])
    lo, hi = np.quantile(all_samples, [0.01, 0.99])
    grid = np.linspace(lo, hi, 200)

    for col, family in enumerate(families, start=1):
        color = family_color(family, theme)
        for forecast in [f for f in forecasts if f.family == family]:
            dens = _density(forecast.terminal_prices, grid)
            fig.add_trace(go.Scatter(
                x=grid, y=dens, mode="lines",
                line=dict(color=color, width=1.6),
                name=forecast.model_name, showlegend=False,
                hovertemplate=f"<b>{forecast.model_name}</b><br>$%{{x:,.0f}}<extra></extra>",
            ), row=1, col=col)
        fig.add_vline(x=spot, line=dict(color=t["text_muted"], width=1, dash="dot"),
                      row=1, col=col)

    fig.update_layout(**base_layout(theme, height=250,
                                    margin=dict(l=40, r=16, t=44, b=40)))
    fig.update_xaxes(tickprefix="$", tickformat=".2s", gridcolor=t["grid"],
                     linecolor=t["border"], tickfont=dict(color=t["text_muted"], size=10))
    fig.update_yaxes(visible=False)
    for annotation in fig.layout.annotations:
        annotation.font.update(size=11, color=t["text_secondary"])
    return fig


def _alpha(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Price history with forecast cone
# ---------------------------------------------------------------------------
def price_history_figure(daily: pd.DataFrame, consensus: dict, horizon_days: int,
                         theme: str, lookback: int = 180) -> go.Figure:
    t = tokens(theme)
    frame = daily.iloc[-lookback:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["close"], mode="lines",
        line=dict(color=t["text_primary"], width=1.8), name="BTC-USD",
        hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))

    if consensus:
        last_date = frame.index[-1]
        future = last_date + pd.Timedelta(days=horizon_days)
        spot = float(frame["close"].iloc[-1])
        for lo_key, hi_key, alpha, label in (("q05", "q95", 0.10, "90% band"),
                                             ("q25", "q75", 0.18, "50% band")):
            fig.add_trace(go.Scatter(
                x=[last_date, future, future, last_date],
                y=[spot, consensus[hi_key], consensus[lo_key], spot],
                fill="toself", mode="lines",
                line=dict(width=0), fillcolor=_alpha(t["accent"], alpha),
                name=label, hoverinfo="skip", showlegend=True))
        fig.add_trace(go.Scatter(
            x=[last_date, future], y=[spot, consensus["median_price"]],
            mode="lines+markers", line=dict(color=t["accent"], width=2, dash="dot"),
            marker=dict(size=8), name="Consensus median",
            hovertemplate="$%{y:,.0f}<extra>consensus median</extra>"))

    fig.update_layout(**base_layout(
        theme, height=320, showlegend=True,
        margin=dict(l=64, r=24, t=34, b=40),
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"])),
        hovermode="x unified",
    ))
    return fig


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def skill_scatter(summary: pd.DataFrame, theme: str, horizon_key: str) -> go.Figure:
    """Out-of-sample discrimination against out-of-sample calibration.

    AUC on one axis, Brier skill on the other, with the null point marked.  Only
    the upper-right quadrant contains models that both rank days correctly *and*
    produce probabilities worth acting on; everything else is noise dressed up
    as a forecast.  A bar chart of accuracy would hide that distinction, because
    accuracy alone is beaten by always predicting "up".
    """
    t = tokens(theme)
    block = summary[summary["horizon"] == horizon_key]
    if block.empty:
        return empty_figure(theme, "No backtest results yet")

    fig = go.Figure()
    x0, x1 = 0.5, max(0.60, float(block["auc"].max()) + 0.02)
    y0, y1 = min(-0.02, float(block["brier_skill"].min()) - 0.01), max(0.02, float(block["brier_skill"].max()) + 0.01)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=y1, layer="below",
                  fillcolor=_alpha(t["up"], 0.07), line=dict(width=0))
    fig.add_annotation(x=(x0 + x1) / 2, y=y1, yanchor="top", showarrow=False,
                       text="skill: ranks correctly and is calibrated",
                       font=dict(size=10, color=t["text_muted"]))

    for family in FAMILY_ORDER:
        sub = block[block["family"] == family]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["auc"], y=sub["brier_skill"], mode="markers+text",
            marker=dict(size=11, color=family_color(family, theme),
                        line=dict(color=t["surface"], width=2)),
            text=sub["model"], textposition="middle right",
            textfont=dict(size=10, color=t["text_secondary"]),
            name=family,
            hovertemplate=("<b>%{text}</b><br>AUC %{x:.3f}<br>"
                           "Brier skill %{y:+.4f}<extra></extra>"),
        ))

    fig.add_vline(x=0.5, line=dict(color=t["border_strong"], width=1.5, dash="dot"))
    fig.add_hline(y=0.0, line=dict(color=t["border_strong"], width=1.5, dash="dot"))
    fig.update_layout(**base_layout(
        theme, height=380, showlegend=True,
        margin=dict(l=70, r=150, t=40, b=52),
        xaxis=dict(title=dict(text="AUC  (0.5 = no ranking ability)",
                              font=dict(size=11, color=t["text_muted"])),
                   gridcolor=t["grid"], linecolor=t["border"],
                   tickfont=dict(color=t["text_muted"])),
        yaxis=dict(title=dict(text="Brier skill vs. base rate  (0 = no skill)",
                              font=dict(size=11, color=t["text_muted"])),
                   gridcolor=t["grid"], linecolor=t["border"],
                   tickformat="+.3f", tickfont=dict(color=t["text_muted"])),
    ))
    return fig


def equity_figure(result, theme: str) -> go.Figure:
    """Strategy equity against buy-and-hold -- two series, so no palette cap."""
    t = tokens(theme)
    strategy = result.strategy
    if not strategy or "equity_curve" not in strategy:
        return empty_figure(theme, "No trades in this window")

    equity = np.array(strategy["equity_curve"])
    hold = np.array(strategy["buy_hold_curve"])
    idx = np.arange(0, len(result.dates), result.horizon_days)[:len(equity)]
    dates = result.dates[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=hold, mode="lines", name="Buy & hold",
                             line=dict(color=t["text_muted"], width=1.6, dash="dot"),
                             hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}x<extra>buy & hold</extra>"))
    final = equity[-1]
    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines", name="Model strategy",
        line=dict(color=t["up"] if final >= 1 else t["down"], width=2.2),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}x<extra>model</extra>"))
    fig.add_hline(y=1.0, line=dict(color=t["border_strong"], width=1, dash="dot"))

    fig.update_layout(**base_layout(
        theme, height=300, showlegend=True, hovermode="x unified",
        yaxis=dict(title=dict(text="Growth of $1 (net of costs)",
                              font=dict(size=11, color=t["text_muted"])),
                   gridcolor=t["grid"], linecolor=t["border"],
                   tickformat=".2f", tickfont=dict(color=t["text_muted"])),
    ))
    return fig


def calibration_figure(result, theme: str) -> go.Figure:
    """Predicted probability against what actually happened."""
    t = tokens(theme)
    rows = result.calibration
    if not rows:
        return empty_figure(theme, "Not enough observations to bin")

    predicted = [r["predicted"] for r in rows]
    observed = [r["observed"] for r in rows]
    counts = [r["n"] for r in rows]
    lo = min(min(predicted), min(observed)) - 0.05
    hi = max(max(predicted), max(observed)) + 0.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect",
                             line=dict(color=t["border_strong"], width=1.5, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=predicted, y=observed, mode="markers+lines", name="Observed",
        line=dict(color=t["accent"], width=2),
        marker=dict(size=[max(8, min(22, n / 12)) for n in counts],
                    color=t["accent"], line=dict(color=t["surface"], width=2)),
        customdata=counts,
        hovertemplate=("predicted %{x:.1%}<br>observed %{y:.1%}"
                       "<br>%{customdata} days<extra></extra>")))

    fig.update_layout(**base_layout(
        theme, height=300, showlegend=True,
        xaxis=dict(range=[lo, hi], tickformat=".0%", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Predicted P(up)", font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(range=[lo, hi], tickformat=".0%", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Actual frequency of up",
                              font=dict(size=11, color=t["text_muted"]))),
    ))
    return fig


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def volatility_smile(chain: pd.DataFrame, theme: str, expiry, spot: float,
                     model_iv: pd.DataFrame | None = None) -> go.Figure:
    """Market mark IV by strike, optionally against a model's implied surface."""
    t = tokens(theme)
    block = chain[chain["expiry"] == expiry]
    if block.empty:
        return empty_figure(theme, "No contracts for this expiry")

    fig = go.Figure()
    for option_type, dash, name in (("call", None, "Calls"), ("put", "dot", "Puts")):
        sub = block[block["option_type"] == option_type].sort_values("strike")
        if sub.empty:
            continue
        color = t["up"] if option_type == "call" else t["down"]
        fig.add_trace(go.Scatter(
            x=sub["strike"], y=sub["mark_iv"] * 100, mode="lines+markers",
            name=f"{name} (market)", line=dict(color=color, width=2, dash=dash),
            marker=dict(size=6, line=dict(color=t["surface"], width=1.5)),
            hovertemplate=("K $%{x:,.0f}<br>mark IV %{y:.1f}%"
                           f"<extra>{name}</extra>")))

    if model_iv is not None and not model_iv.empty:
        sub = model_iv[(model_iv["expiry"] == expiry) &
                       (model_iv["option_type"] == "call")].sort_values("strike")
        sub = sub[np.isfinite(sub["model_iv"])]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["strike"], y=sub["model_iv"] * 100, mode="lines",
                name="Model-implied", line=dict(color=t["neutral"], width=2, dash="dash"),
                hovertemplate="K $%{x:,.0f}<br>model IV %{y:.1f}%<extra>model</extra>"))

    fig.add_vline(x=spot, line=dict(color=t["text_primary"], width=1.5),
                  annotation_text="spot", annotation_position="top",
                  annotation_font=dict(color=t["text_primary"], size=10))
    fig.update_layout(**base_layout(
        theme, height=320, showlegend=True,
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Strike", font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(ticksuffix="%", gridcolor=t["grid"], linecolor=t["border"],
                   tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Implied volatility",
                              font=dict(size=11, color=t["text_muted"]))),
    ))
    return fig


def greeks_profile(chain: pd.DataFrame, theme: str, expiry, greek: str,
                   spot: float) -> go.Figure:
    """One greek across the strike ladder, calls and puts."""
    t = tokens(theme)
    block = chain[chain["expiry"] == expiry]
    if block.empty:
        return empty_figure(theme, "No contracts for this expiry")

    labels = {
        "delta": ("Delta", "USD change per $1 of spot", ".3f"),
        "gamma": ("Gamma", "Delta change per $1 of spot", ".2e"),
        "theta": ("Theta", "USD lost per calendar day", ",.1f"),
        "vega": ("Vega", "USD per 1 volatility point", ",.1f"),
        "vanna": ("Vanna", "Delta change per volatility point", ".4f"),
        "charm": ("Charm", "Delta lost per day", ".4f"),
    }
    title, axis, fmt = labels.get(greek, (greek.title(), "", ".3f"))

    fig = go.Figure()
    for option_type, dash, name in (("call", None, "Calls"), ("put", "dot", "Puts")):
        sub = block[block["option_type"] == option_type].sort_values("strike")
        if sub.empty:
            continue
        color = t["up"] if option_type == "call" else t["down"]
        fig.add_trace(go.Scatter(
            x=sub["strike"], y=sub[greek], mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=6, line=dict(color=t["surface"], width=1.5)),
            hovertemplate=f"K $%{{x:,.0f}}<br>{title} %{{y:{fmt}}}<extra>{name}</extra>"))

    fig.add_vline(x=spot, line=dict(color=t["text_primary"], width=1.5),
                  annotation_text="spot", annotation_position="top",
                  annotation_font=dict(color=t["text_primary"], size=10))
    fig.add_hline(y=0, line=dict(color=t["border_strong"], width=1, dash="dot"))
    fig.update_layout(**base_layout(
        theme, height=320, showlegend=True,
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Strike", font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(gridcolor=t["grid"], linecolor=t["border"],
                   tickfont=dict(color=t["text_muted"]),
                   title=dict(text=axis, font=dict(size=11, color=t["text_muted"]))),
    ))
    return fig


def theta_decay_figure(schedule: pd.DataFrame, theme: str, label: str) -> go.Figure:
    """Premium erosion to expiry with spot and volatility held still."""
    t = tokens(theme)
    if schedule.empty:
        return empty_figure(theme, "No decay schedule")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=schedule["days_to_expiry"], y=schedule["value"], mode="lines+markers",
        name="Option value", line=dict(color=t["accent"], width=2.2),
        marker=dict(size=7, line=dict(color=t["surface"], width=1.5)),
        fill="tozeroy", fillcolor=_alpha(t["accent"], 0.12),
        hovertemplate=("%{x:.1f} days left<br>$%{y:,.0f}"
                       "<br>%{customdata:.1f}% of today<extra></extra>"),
        customdata=schedule["pct_of_today"]))

    start = float(schedule["value"].iloc[0])
    end = float(schedule["value"].iloc[-1])
    fig.add_annotation(
        x=float(schedule["days_to_expiry"].iloc[-1]), y=end, xanchor="left",
        text=f"${end:,.0f} at expiry<br>({(end / start - 1) * 100:+.1f}% vs today)",
        showarrow=False, font=dict(size=10, color=t["text_secondary"]))

    fig.update_layout(**base_layout(
        theme, height=280,
        xaxis=dict(autorange="reversed", gridcolor=t["grid"], linecolor=t["border"],
                   tickfont=dict(color=t["text_muted"]),
                   title=dict(text=f"Days to expiry — {label}",
                              font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Value, spot & IV held fixed",
                              font=dict(size=11, color=t["text_muted"]))),
    ))
    return fig


def payoff_figure(suggestion: dict, forecast, theme: str, spot: float) -> go.Figure:
    """Expiry payoff of a suggested contract over the model's price distribution."""
    t = tokens(theme)
    strike = suggestion["strike"]
    premium = suggestion["premium_usd"]
    is_call = suggestion["option_type"] == "call"

    lo = min(spot * 0.85, strike * 0.9)
    hi = max(spot * 1.15, strike * 1.1)
    grid = np.linspace(lo, hi, 200)
    payoff = (np.maximum(grid - strike, 0) if is_call else np.maximum(strike - grid, 0)) - premium

    from btcmodels.options import scale_distribution
    samples = scale_distribution(forecast, suggestion["dte"])
    dens = _density(samples, grid)
    scaled = dens / dens.max() * (np.nanmax(payoff) - np.nanmin(payoff)) * 0.35

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=scaled + np.nanmin(payoff), mode="lines", name="Model price distribution",
        line=dict(color=t["neutral"], width=1.4),
        fill="tozeroy", fillcolor=_alpha(t["neutral"], 0.10), hoverinfo="skip"))
    profit = np.where(payoff >= 0, payoff, np.nan)
    loss = np.where(payoff < 0, payoff, np.nan)
    fig.add_trace(go.Scatter(x=grid, y=loss, mode="lines", name="Loss",
                             line=dict(color=t["down"], width=2.4),
                             hovertemplate="$%{x:,.0f}<br>P&L $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=profit, mode="lines", name="Profit",
                             line=dict(color=t["up"], width=2.4),
                             hovertemplate="$%{x:,.0f}<br>P&L $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["border_strong"], width=1))
    fig.add_vline(x=spot, line=dict(color=t["text_primary"], width=1.4),
                  annotation_text="now", annotation_position="top left",
                  annotation_font=dict(color=t["text_primary"], size=10))
    fig.add_vline(x=suggestion["breakeven"], line=dict(color=t["warn"], width=1.4, dash="dash"),
                  annotation_text=f"breakeven ${suggestion['breakeven']:,.0f}",
                  annotation_position="top right",
                  annotation_font=dict(color=t["warn"], size=10))

    fig.update_layout(**base_layout(
        theme, height=300, showlegend=True,
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="BTC price at expiry",
                              font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=t["grid"],
                   linecolor=t["border"], tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Profit / loss per contract",
                              font=dict(size=11, color=t["text_muted"]))),
    ))
    return fig


def feature_importance_figure(series: pd.Series, theme: str, model_name: str) -> go.Figure:
    """Relative importance of the top features for a learned model."""
    t = tokens(theme)
    if series is None or series.empty:
        return empty_figure(theme, "This model has no feature importances")

    ordered = series.sort_values()
    fig = go.Figure(go.Bar(
        x=ordered.values, y=ordered.index, orientation="h",
        marker=dict(color=t["accent"], line=dict(color=t["surface"], width=2)),
        text=[f"{v * 100:.1f}%" for v in ordered.values],
        textposition="outside", textfont=dict(size=10, color=t["text_secondary"]),
        hovertemplate="<b>%{y}</b><br>%{x:.1%} of total importance<extra></extra>"))
    fig.update_layout(**base_layout(
        theme, height=max(240, 22 * len(ordered) + 60),
        margin=dict(l=150, r=64, t=24, b=40),
        xaxis=dict(tickformat=".0%", gridcolor=t["grid"], linecolor=t["border"],
                   tickfont=dict(color=t["text_muted"]),
                   title=dict(text="Share of total importance",
                              font=dict(size=11, color=t["text_muted"]))),
        yaxis=dict(gridcolor=TRANSPARENT, linecolor=TRANSPARENT,
                   tickfont=dict(color=t["text_secondary"], size=10.5)),
    ))
    return fig
