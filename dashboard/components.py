"""Reusable presentational pieces.

Kept free of any computation: everything here takes already-computed numbers and
turns them into markup, so the display layer can be re-rendered on a theme
change without touching a model.
"""

from __future__ import annotations

import math
from typing import Any

from dash import dash_table, dcc, html

from .theme import direction_color, family_color, tokens


def _fmt(value: Any, spec: str = ",.2f", dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def stat(label: str, value: str, sub: str | None = None,
         color: str | None = None, title: str | None = None) -> html.Div:
    """A single labelled number. Not a chart -- a chart would add nothing here."""
    children = [
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value",
                 style={"color": color} if color else None),
    ]
    if sub:
        children.append(html.Div(sub, className="stat-sub"))
    return html.Div(children, className="stat", title=title or "")


def section(title: str, subtitle: str | None, *children, right=None) -> html.Section:
    head = [html.H2(title, className="section-title")]
    if subtitle:
        head.append(html.P(subtitle, className="section-subtitle"))
    header = html.Div([
        html.Div(head, className="section-head-text"),
        html.Div(right or [], className="section-head-right"),
    ], className="section-head")
    return html.Section([header, *children], className="section")


def callout(text, tone: str = "info", title: str | None = None) -> html.Div:
    icons = {"info": "ℹ", "warn": "⚠", "good": "✓", "bad": "✕"}
    return html.Div([
        html.Span(icons.get(tone, "ℹ"), className="callout-icon"),
        html.Div([
            html.Strong(title, className="callout-title") if title else None,
            html.Div(text, className="callout-body"),
        ]),
    ], className=f"callout callout-{tone}")


def pill(text: str, tone: str = "neutral", title: str = "") -> html.Span:
    return html.Span(text, className=f"pill pill-{tone}", title=title)


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def probability_meter(p_up: float, theme: str, stderr: float = 0.0) -> html.Div:
    """A one-dimensional meter: position on a 0-100% track, 50% marked.

    A gauge or donut would use area to encode a single scalar, which reads as
    more information than is there.  The uncertainty band is drawn on the same
    track so the width of the estimate is visible, not just its centre.
    """
    color = direction_color(p_up, theme)
    left = max(0.0, min(1.0, p_up - 1.96 * stderr)) * 100
    width = max(1.2, min(100.0, (min(1.0, p_up + 1.96 * stderr) * 100) - left))
    return html.Div([
        html.Div([
            html.Div(className="meter-track"),
            html.Div(className="meter-midline"),
            html.Div(className="meter-band",
                     style={"left": f"{left}%", "width": f"{width}%",
                            "background": color, "opacity": 0.25}),
            html.Div(className="meter-marker",
                     style={"left": f"{p_up * 100:.2f}%", "background": color}),
        ], className="meter"),
        html.Div([
            html.Span("0%", className="meter-tick"),
            html.Span("50%", className="meter-tick meter-tick-mid"),
            html.Span("100%", className="meter-tick"),
        ], className="meter-ticks"),
    ], className="meter-wrap")


def reliability_badge(reliability: dict | None) -> html.Div:
    """Backtest verdict shown on the live prediction, not buried on another tab."""
    if reliability is None:
        return html.Div([pill("no backtest yet", "neutral")], className="reliability")

    has_skill = reliability["has_skill"]
    tone = "good" if has_skill else "bad"
    label = "beat the base rate" if has_skill else "no out-of-sample edge"
    return html.Div([
        pill(label, tone,
             title="Brier skill > 0 and AUC > 0.5 over the walk-forward test window"),
        html.Span(
            f"{reliability['accuracy_pct']:.1f}% acc vs {reliability['always_up_accuracy_pct']:.1f}% always-up"
            f" · AUC {reliability['auc']:.3f} · n={reliability['n']}",
            className="reliability-detail"),
    ], className="reliability")


def model_card(forecast, theme: str, reliability: dict | None,
               selected: bool = False) -> html.Div:
    t = tokens(theme)
    color = direction_color(forecast.p_up, theme)
    q = forecast.quantile_prices()
    accent = family_color(forecast.family, theme)

    return html.Div([
        html.Div([
            html.Div([
                html.Span(className="family-dot", style={"background": accent}),
                html.Span(forecast.family, className="card-family"),
            ], className="card-family-row"),
            html.H3(forecast.model_name, className="card-title"),
        ], className="card-head"),

        html.Div([
            html.Div([
                html.Span(forecast.direction, className="card-direction",
                          style={"color": color}),
                html.Span(f"{forecast.p_up * 100:.1f}%", className="card-prob"),
            ], className="card-direction-row"),
            html.Div(f"{forecast.confidence_label} signal · ±{forecast.p_up_stderr * 100:.1f} pts",
                     className="card-confidence"),
        ], className="card-headline"),

        probability_meter(forecast.p_up, theme, forecast.p_up_stderr),

        html.Div([
            html.Div([html.Span("Expected", className="kv-k"),
                      html.Span(f"{forecast.expected_return:+.2f}%", className="kv-v")], className="kv"),
            html.Div([html.Span("Median", className="kv-k"),
                      html.Span(f"${q['q50']:,.0f}", className="kv-v")], className="kv"),
            html.Div([html.Span("90% band", className="kv-k"),
                      html.Span(f"${q['q05']:,.0f} – ${q['q95']:,.0f}", className="kv-v")], className="kv"),
            html.Div([html.Span("Implied vol", className="kv-k"),
                      html.Span(f"{forecast.annualised_vol:.0f}%", className="kv-v")], className="kv"),
        ], className="card-kv"),

        reliability_badge(reliability),
    ], className="model-card" + (" model-card-selected" if selected else ""),
        style={"borderTopColor": accent})


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def data_table(df, columns: list[dict], theme: str, table_id: str,
               page_size: int = 14, sort_by: list | None = None) -> dash_table.DataTable:
    t = tokens(theme)
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=columns,
        page_size=page_size,
        sort_action="native",
        sort_by=sort_by or [],
        filter_action="none",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": t["surface_sunken"],
            "color": t["text_muted"],
            "fontWeight": "600",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
            "border": "none",
            "borderBottom": f"1px solid {t['border']}",
            "padding": "10px 12px",
        },
        style_cell={
            "backgroundColor": "transparent",
            "color": t["text_secondary"],
            "border": "none",
            "borderBottom": f"1px solid {t['border']}",
            "padding": "9px 12px",
            "fontSize": "12.5px",
            "fontFamily": "var(--mono)",
            "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left", "fontFamily": "var(--sans)"}
            for c in ("model", "family", "instrument", "moneyness_label",
                      "option_type", "horizon", "expiry", "structure")
        ],
        style_data_conditional=[
            {"if": {"state": "active"},
             "backgroundColor": t["surface_sunken"], "border": f"1px solid {t['accent']}"},
        ],
    )
